from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.engines.model_manager import ModelManager, ProgressCallback

logger = logging.getLogger(__name__)

REALESRGAN_MODELS = {
    2: {
        "name": "RealESRGAN_x2plus",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.2.1/RealESRGAN_x2plus.pth"
        ),
        "min_bytes": 67_061_725,
    },
    4: {
        "name": "RealESRGAN_x4plus",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth"
        ),
        "min_bytes": 67_040_989,
    },
}


class RealESRGANError(RuntimeError):
    pass


class RealESRGANEngine:
    """Official Real-ESRGAN x2plus/x4plus CUDA inference with tiled assembly."""

    native_scale = 4

    def __init__(
        self,
        device_index: int = 0,
        fp16: bool = True,
        tile_size: int = 0,
        tile_pad: int = 16,
        model_scale: int = 4,
        model_root: Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RealESRGANError("PyTorch is not installed. Install the CUDA-enabled AI dependencies.") from exc

        if not torch.cuda.is_available():
            raise RealESRGANError("PyTorch cannot access CUDA. Check the NVIDIA driver and CUDA-enabled PyTorch build.")
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise RealESRGANError(f"CUDA device index {device_index} is not available.")

        self.torch = torch
        self.device = torch.device(f"cuda:{device_index}")
        props = torch.cuda.get_device_properties(device_index)
        self.vram_gb = props.total_memory / (1024**3)
        capability = torch.cuda.get_device_capability(device_index)
        self.use_fp16 = bool(fp16 and capability[0] >= 7)
        self.tile_size = tile_size or (128 if self.vram_gb <= 8 else 256)
        if self.tile_size < 32:
            raise ValueError("Real-ESRGAN tile size must be at least 32 pixels.")
        if tile_pad < 0 or tile_pad > 128:
            raise ValueError("Real-ESRGAN tile padding must be between 0 and 128 pixels.")
        self.tile_pad = tile_pad
        if model_scale not in REALESRGAN_MODELS:
            raise ValueError("Real-ESRGAN model_scale must be 2 or 4.")
        self.model_scale = model_scale
        self.native_scale = model_scale
        model_info = REALESRGAN_MODELS[model_scale]
        self.model_name = str(model_info["name"])

        self.model_path = ModelManager(model_root).ensure_file(
            relative_path=f"realesrgan/{self.model_name}.pth",
            url=str(model_info["url"]),
            min_bytes=int(model_info["min_bytes"]),
            progress=progress,
        )
        self.model = self._load_model()

    def _load_model(self):
        torch = self.torch
        # Import the architecture only after a CUDA-capable PyTorch runtime has
        # been confirmed. This keeps the non-AI desktop UI importable without it.
        from app.engines.realesrgan_arch import RRDBNet

        network = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.model_scale,
        )
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RealESRGANError(
                "This Real-ESRGAN adapter requires PyTorch safe weights_only checkpoint loading."
            ) from exc
        if isinstance(checkpoint, dict):
            state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        else:
            state = checkpoint
        if not isinstance(state, dict) or not state:
            raise RealESRGANError("Real-ESRGAN checkpoint does not contain a valid state dictionary.")
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
            if isinstance(key, str)
        }
        try:
            network.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RealESRGANError(
                f"Real-ESRGAN checkpoint architecture mismatch for {self.model_name}. "
                "The checkpoint and network definition must match."
            ) from exc
        network.eval().to(self.device)
        if self.use_fp16:
            network.half()
        return network

    @staticmethod
    def _validate_rgb(frame: np.ndarray) -> tuple[int, int]:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError("Real-ESRGAN expects a uint8 NumPy RGB frame.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Real-ESRGAN expects an H×W×3 RGB frame.")
        return int(frame.shape[0]), int(frame.shape[1])

    def _upscale_tiles_tensor(self, tensor, tile_size: int):
        """Upscale one normalized RGB CUDA tensor using bounded GPU tiles.

        The full output is kept as uint8 on the GPU. Only the final frame is
        copied to host memory, so an 8K output does not need a float32 VRAM
        allocation and tiles do not shuttle individually over PCIe.
        """
        torch = self.torch
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError("Real-ESRGAN tensor input must have shape 1×3×H×W.")
        if tensor.device != self.device:
            raise ValueError("Real-ESRGAN tensor must reside on the selected CUDA device.")

        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
        out_scale = self.model_scale
        output = torch.empty(
            (1, 3, height * out_scale, width * out_scale),
            dtype=torch.uint8,
            device=self.device,
        )
        dtype = torch.float16 if self.use_fp16 else torch.float32

        with torch.inference_mode():
            for y0 in range(0, height, tile_size):
                y1 = min(y0 + tile_size, height)
                for x0 in range(0, width, tile_size):
                    x1 = min(x0 + tile_size, width)
                    iy0 = max(0, y0 - self.tile_pad)
                    iy1 = min(height, y1 + self.tile_pad)
                    ix0 = max(0, x0 - self.tile_pad)
                    ix1 = min(width, x1 + self.tile_pad)

                    tile = tensor[:, :, iy0:iy1, ix0:ix1].to(dtype=dtype)
                    # x2plus uses pixel_unshuffle(2), so odd tile dimensions
                    # need a replicated edge pixel before entering the model.
                    if self.model_scale == 2:
                        pad_right = int(tile.shape[-1] % 2)
                        pad_bottom = int(tile.shape[-2] % 2)
                        if pad_right or pad_bottom:
                            tile = torch.nn.functional.pad(
                                tile, (0, pad_right, 0, pad_bottom), mode="replicate"
                            )

                    result = self.model(tile).float().clamp_(0.0, 1.0)
                    crop_top = (y0 - iy0) * out_scale
                    crop_bottom = crop_top + (y1 - y0) * out_scale
                    crop_left = (x0 - ix0) * out_scale
                    crop_right = crop_left + (x1 - x0) * out_scale
                    result = result[:, :, crop_top:crop_bottom, crop_left:crop_right]
                    tile_out = result.mul(255.0).round().to(dtype=torch.uint8)
                    output[
                        :, :, y0 * out_scale:y1 * out_scale,
                        x0 * out_scale:x1 * out_scale,
                    ] = tile_out
                    del tile, result, tile_out

        return output

    def _upscale_tiles(self, frame: np.ndarray, tile_size: int) -> np.ndarray:
        """Convert one host RGB frame once, then process its tiles on CUDA."""
        tensor = self.torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.torch.float32).div_(255.0)
        try:
            output = self._upscale_tiles_tensor(tensor, tile_size)
            return output[0].permute(1, 2, 0).contiguous().cpu().numpy()
        finally:
            del tensor

    def _run_adaptive_tiles(self, tensor, initial_tile_size: int):
        tile_size = initial_tile_size
        last_error: RuntimeError | None = None
        while tile_size >= 32:
            try:
                output = self._upscale_tiles_tensor(tensor, tile_size)
                self.tile_size = tile_size
                return output
            except RuntimeError as exc:
                oom_type = getattr(self.torch.cuda, "OutOfMemoryError", None)
                is_oom = (
                    (oom_type is not None and isinstance(exc, oom_type))
                    or "out of memory" in str(exc).lower()
                )
                if not is_oom:
                    raise
                last_error = exc
                tile_size //= 2
                logger.warning("Real-ESRGAN CUDA OOM; retrying with tile size %s.", tile_size)
                with self.torch.cuda.device(self.device):
                    self.torch.cuda.empty_cache()
        raise RealESRGANError(
            "Real-ESRGAN ran out of GPU memory even with 32-pixel tiles. "
            "Close other GPU applications or lower source resolution."
        ) from last_error

    def upscale_tensor(
        self,
        tensor,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Upscale a normalized 1x3xHxW CUDA tensor and return final RGB uint8."""
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError("Real-ESRGAN tensor input must have shape 1×3×H×W.")
        if tensor.device != self.device:
            raise ValueError("Real-ESRGAN tensor must reside on the selected CUDA device.")
        output = self._run_adaptive_tiles(tensor, self.tile_size)
        if output_size is not None:
            target_width, target_height = output_size
            if target_width <= 0 or target_height <= 0:
                raise ValueError("Output dimensions must be positive.")
            output = self._fit_tensor_to_canvas(output, target_width, target_height)
        return output[0].permute(1, 2, 0).contiguous().cpu().numpy()

    def _fit_tensor_to_canvas(self, tensor, target_width: int, target_height: int):
        """Resize an NCHW uint8 CUDA tensor into a centered target canvas."""
        import torch.nn.functional as F

        torch = self.torch

        source_height, source_width = int(tensor.shape[-2]), int(tensor.shape[-1])
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, min(target_width, round(source_width * scale)))
        resized_height = max(1, min(target_height, round(source_height * scale)))
        resized = tensor
        if (resized_width, resized_height) != (source_width, source_height):
            resized = F.interpolate(
                tensor.float(),
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).round().clamp_(0, 255).to(dtype=torch.uint8)
        canvas = torch.zeros(
            (1, 3, target_height, target_width),
            dtype=torch.uint8,
            device=self.device,
        )
        top = (target_height - resized_height) // 2
        left = (target_width - resized_width) // 2
        canvas[:, :, top:top + resized_height, left:left + resized_width] = resized
        return canvas

    def upscale(
        self,
        frame: np.ndarray,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """AI-upscale one RGB uint8 frame; output_size is (width, height)."""
        input_height, input_width = self._validate_rgb(frame)
        if output_size is not None and output_size == (input_width, input_height):
            return frame.copy()
        tensor = self.torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.torch.float32).div_(255.0)
        try:
            return self.upscale_tensor(tensor, output_size)
        finally:
            del tensor

    def _fit_to_canvas(self, frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Compatibility helper for NumPy RGB frames; production uses GPU canvas fitting."""
        tensor = self.torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.torch.uint8)
        try:
            output = self._fit_tensor_to_canvas(tensor, target_width, target_height)
            return output[0].permute(1, 2, 0).contiguous().cpu().numpy()
        finally:
            del tensor

    def close(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
        with self.torch.cuda.device(self.device):
            self.torch.cuda.empty_cache()
