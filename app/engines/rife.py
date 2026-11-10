from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.engines.model_manager import ModelManager, ProgressCallback

logger = logging.getLogger(__name__)

RIFE_REPO = "MonsterMMORPG/Practical-RIFE"
RIFE_REVISION = "4bbe4c19fb5363333cca83b6e82bb65ee02d741b"  # immutable v4.25 model revision
RIFE_MODEL_URL = (
    "https://huggingface.co/MonsterMMORPG/Practical-RIFE/resolve/"
    f"{RIFE_REVISION}/train_log/flownet.pkl?download=true"
)


class RIFEError(RuntimeError):
    pass


class RIFEEngine:
    """Practical-RIFE v4.25 CUDA inference adapter.

    Inputs/outputs are RGB uint8 H×W×3 arrays. Model weights are downloaded on
    first use; inference is performed on the selected CUDA device.
    """

    def __init__(
        self,
        device_index: int = 0,
        fp16: bool = True,
        inference_scale: float | None = None,
        model_root: Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise RIFEError("PyTorch is not installed. Install the CUDA-enabled AI dependencies.") from exc

        if not torch.cuda.is_available():
            raise RIFEError("PyTorch cannot access CUDA. Check the NVIDIA driver and CUDA-enabled PyTorch build.")
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise RIFEError(f"CUDA device index {device_index} is not available.")

        self.torch = torch
        self.functional = F
        self.device = torch.device(f"cuda:{device_index}")
        self.device_index = device_index
        capability = torch.cuda.get_device_capability(device_index)
        self.use_fp16 = bool(fp16 and capability[0] >= 7)
        total_vram = torch.cuda.get_device_properties(device_index).total_memory
        vram_gb = total_vram / (1024**3)
        self.inference_scale = (
            inference_scale if inference_scale is not None
            else (0.5 if vram_gb <= 8.0 else 1.0)
        )
        if self.inference_scale not in (0.25, 0.5, 1.0):
            raise ValueError("RIFE inference_scale must be 0.25, 0.5 or 1.0.")

        self.model_path = ModelManager(model_root).ensure_file(
            relative_path="rife/v4.25/flownet.pkl",
            url=RIFE_MODEL_URL,
            min_bytes=24_636_301,
            expected_sha256=(
                "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2"
            ),
            progress=progress,
        )
        # Fixed input shapes benefit from cuDNN autotuning. The model is
        # inference-only and the chosen shape remains stable for a video job.
        torch.backends.cudnn.benchmark = True
        self.model = self._load_model()

    def _load_model(self):
        torch = self.torch
        from app.engines.rife_vendor.ifnet_hdv3 import IFNet

        network = IFNet()
        try:
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as exc:
            raise RIFEError(
                "This RIFE adapter requires a PyTorch build supporting safe weights_only loading."
            ) from exc
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict) or not checkpoint:
            raise RIFEError("RIFE checkpoint does not contain a valid state dictionary.")
        # Practical-RIFE v4.25 checkpoints also contain training-only weights.
        # The official inference module deliberately omits these two modules:
        # `teacher` and `caltime`. Filter only those known prefixes. All other
        # keys are passed to strict=True below so inference mismatches still fail.
        state = {}
        for key, value in checkpoint.items():
            if not isinstance(key, str):
                continue
            normalized_key = key.removeprefix("module.")
            if normalized_key.startswith(("teacher.", "caltime.")):
                continue
            state[normalized_key] = value

        model_keys = set(network.state_dict())
        checkpoint_keys = set(state)
        missing_keys = sorted(model_keys - checkpoint_keys)
        unexpected_keys = sorted(checkpoint_keys - model_keys)
        if missing_keys or unexpected_keys:
            details = []
            if missing_keys:
                details.append("missing inference keys: " + ", ".join(missing_keys[:8]))
            if unexpected_keys:
                details.append("unexpected keys: " + ", ".join(unexpected_keys[:8]))
            raise RIFEError(
                "RIFE v4.25 checkpoint does not match the bundled inference architecture ("
                + "; ".join(details)
                + "). The known training-only teacher/caltime weights were excluded."
            )

        try:
            network.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RIFEError(
                "RIFE checkpoint architecture mismatch after excluding the documented "
                "training-only teacher and caltime weights."
            ) from exc
        network.eval().to(self.device)
        if self.use_fp16:
            network.half()
        return network

    def interpolate_tensor(self, frame0, frame1, timestep: float):
        """Interpolate two normalized N×3×H×W CUDA tensors at time t in (0,1)."""
        if not 0.0 < timestep < 1.0:
            raise ValueError("RIFE timestep must be strictly between 0 and 1.")
        if frame0.shape != frame1.shape or frame0.ndim != 4 or frame0.shape[1] != 3:
            raise ValueError("RIFE frames must have identical N×3×H×W shapes.")
        if frame0.device != self.device or frame1.device != self.device:
            raise ValueError("RIFE tensors must already reside on the selected CUDA device.")

        torch = self.torch
        F = self.functional
        original_h, original_w = frame0.shape[-2:]
        scale = self.inference_scale
        last_error: RuntimeError | None = None

        while scale >= 0.25:
            alignment = max(128, int(128 / scale))
            padded_h = ((original_h + alignment - 1) // alignment) * alignment
            padded_w = ((original_w + alignment - 1) // alignment) * alignment
            pad = (0, padded_w - original_w, 0, padded_h - original_h)
            input0 = F.pad(frame0, pad)
            input1 = F.pad(frame1, pad)
            if self.use_fp16:
                input0, input1 = input0.half(), input1.half()
            else:
                input0, input1 = input0.float(), input1.float()
            scale_list = (16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale)

            try:
                with torch.inference_mode():
                    output = self.model.interpolate(input0, input1, timestep, scale_list)
                # Preserve FP16 output when enabled so Real-ESRGAN can consume
                # it directly without a full-frame FP32 allocation.
                output = output[..., :original_h, :original_w].clamp_(0.0, 1.0)
                self.inference_scale = scale
                return output
            except RuntimeError as exc:
                oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
                is_oom = (oom_type is not None and isinstance(exc, oom_type)) or "out of memory" in str(exc).lower()
                if not is_oom:
                    raise
                last_error = exc
                logger.warning("RIFE CUDA out of memory at inference scale %s; retrying smaller.", scale)
                del input0, input1
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                scale *= 0.5

        raise RIFEError(
            "RIFE ran out of GPU memory even at 0.25 inference scale. Close other GPU applications or use a smaller source resolution."
        ) from last_error

    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        if not isinstance(frame0, np.ndarray) or not isinstance(frame1, np.ndarray):
            raise TypeError("RIFE frame inputs must be NumPy arrays.")
        if frame0.dtype != np.uint8 or frame1.dtype != np.uint8:
            raise ValueError("RIFE expects uint8 RGB frames.")
        if frame0.shape != frame1.shape or frame0.ndim != 3 or frame0.shape[2] != 3:
            raise ValueError("RIFE frames must have identical H×W×3 RGB shapes.")

        torch = self.torch
        tensor0 = torch.from_numpy(np.ascontiguousarray(frame0)).permute(2, 0, 1).unsqueeze(0)
        tensor1 = torch.from_numpy(np.ascontiguousarray(frame1)).permute(2, 0, 1).unsqueeze(0)
        tensor0 = tensor0.to(device=self.device, dtype=torch.float32).div_(255.0)
        tensor1 = tensor1.to(device=self.device, dtype=torch.float32).div_(255.0)
        result = self.interpolate_tensor(tensor0, tensor1, timestep)
        return (
            result[0].mul(255.0).round().byte().permute(1, 2, 0).contiguous().cpu().numpy()
        )

    def close(self) -> None:
        del self.model
        with self.torch.cuda.device(self.device):
            self.torch.cuda.empty_cache()
