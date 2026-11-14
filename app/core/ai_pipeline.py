from __future__ import annotations

import logging
from typing import Self

import numpy as np

logger = logging.getLogger(__name__)



class AIFramePipeline:
    """Connect RIFE interpolation and Real-ESRGAN super-resolution per frame pair."""

    def __init__(
        self,
        *,
        interpolate: bool,
        upscale: bool,
        output_size: tuple[int, int],
        device_index: int = 0,
        fp16: bool = True,
        rife_scale: float | None = None,
        upscale_tile_size: int = 0,
        upscale_model_scale: int = 4,
        model_root=None,
        progress=None,
    ) -> None:
        # Both engines may be skipped when the source already matches the
        # requested FPS/resolution. In that case this coordinator performs only
        # timestamp selection and/or a CUDA resize, without downloading models.
        self.output_size = output_size
        self.device_index = device_index
        self.rife = None
        self.upscaler = None
        try:
            if interpolate:
                from app.engines.rife import RIFEEngine

                self.rife = RIFEEngine(
                    device_index=device_index,
                    fp16=fp16,
                    inference_scale=rife_scale,
                    model_root=model_root,
                    progress=progress,
                )
            if upscale:
                from app.engines.realesrgan import RealESRGANEngine

                self.upscaler = RealESRGANEngine(
                    device_index=device_index,
                    fp16=fp16,
                    tile_size=upscale_tile_size,
                    model_scale=upscale_model_scale,
                    model_root=model_root,
                    progress=progress,
                )
        except Exception:
            # If the second engine fails (for example, model download or strict
            # checkpoint validation), release the first engine's GPU memory.
            logger.exception("CUDA AI pipeline initialization failed; releasing initialized engines.")
            self.close()
            raise

    def prepare_source_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale or GPU-resize a source frame to the configured output size."""
        if self.upscaler is not None:
            return self.upscaler.upscale(frame, self.output_size)
        return self._resize_cuda(frame, self.output_size, self.device_index)

    def process_pair(self, frame0: np.ndarray, frame1: np.ndarray, timestep: float) -> np.ndarray:
        """Interpolate a timestamp, then upscale without copying RIFE output to CPU."""
        if self.rife is None:
            selected = frame0 if timestep < 0.5 else frame1
            if self.upscaler is not None:
                return self.upscaler.upscale(selected, self.output_size)
            return self._resize_cuda(selected, self.output_size, self.device_index)

        import torch

        device = torch.device(f"cuda:{self.device_index}")
        tensor0 = torch.from_numpy(np.ascontiguousarray(frame0)).permute(2, 0, 1).unsqueeze(0)
        tensor1 = torch.from_numpy(np.ascontiguousarray(frame1)).permute(2, 0, 1).unsqueeze(0)
        tensor0 = tensor0.to(device=device, dtype=torch.float32).div_(255.0)
        tensor1 = tensor1.to(device=device, dtype=torch.float32).div_(255.0)
        try:
            interpolated = self.rife.interpolate_tensor(tensor0, tensor1, timestep)
            if self.upscaler is not None:
                # The interpolated image and tiled output assembly stay on CUDA;
                # only the completed upscaled frame is copied to host memory.
                return self.upscaler.upscale_tensor(interpolated, self.output_size)
            return self._tensor_to_canvas(interpolated, self.output_size)
        finally:
            del tensor0, tensor1

    @staticmethod
    def _tensor_to_canvas(tensor, size: tuple[int, int]) -> np.ndarray:
        """Resize a normalized CUDA image into a centered output canvas."""
        import torch
        import torch.nn.functional as F

        width, height = size
        source_height, source_width = int(tensor.shape[-2]), int(tensor.shape[-1])
        scale = min(width / source_width, height / source_height)
        resized_width = max(1, min(width, round(source_width * scale)))
        resized_height = max(1, min(height, round(source_height * scale)))
        resized = tensor
        if (resized_width, resized_height) != (source_width, source_height):
            resized = F.interpolate(
                tensor.float(), size=(resized_height, resized_width),
                mode="bicubic", align_corners=False, antialias=True,
            )
        canvas = torch.zeros(
            (1, 3, height, width), dtype=torch.float32, device=tensor.device
        )
        top = (height - resized_height) // 2
        left = (width - resized_width) // 2
        canvas[:, :, top:top + resized_height, left:left + resized_width] = resized
        return (
            canvas[0].clamp_(0.0, 1.0).mul_(255.0).round().byte()
            .permute(1, 2, 0).contiguous().cpu().numpy()
        )

    @staticmethod
    def _resize_cuda(frame: np.ndarray, size: tuple[int, int], device_index: int) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        width, height = size
        source_height, source_width = frame.shape[:2]
        scale = min(width / source_width, height / source_height)
        resized_width = max(1, min(width, round(source_width * scale)))
        resized_height = max(1, min(height, round(source_height * scale)))
        if (resized_width, resized_height) == (source_width, source_height):
            resized = frame
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for AI Upscaler Studio frame processing.")
            if device_index < 0 or device_index >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device index {device_index} is unavailable.")
            device = torch.device(f"cuda:{device_index}")
            tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(device=device, dtype=torch.float32)
            tensor = F.interpolate(
                tensor, size=(resized_height, resized_width), mode="bicubic",
                align_corners=False, antialias=True,
            )
            resized = (
                tensor[0].clamp_(0, 255).round().byte()
                .permute(1, 2, 0).contiguous().cpu().numpy()
            )
            del tensor
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        top = (height - resized_height) // 2
        left = (width - resized_width) // 2
        canvas[top:top + resized_height, left:left + resized_width] = resized
        return canvas

    def close(self) -> None:
        rife, upscaler = self.rife, self.upscaler
        self.rife = None
        self.upscaler = None
        try:
            if rife is not None:
                rife.close()
        finally:
            if upscaler is not None:
                upscaler.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
