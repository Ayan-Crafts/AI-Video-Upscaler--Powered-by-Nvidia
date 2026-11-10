from __future__ import annotations

from app.core.config import ProcessingConfig
from app.engines.availability import realesrgan_status, rife_status
from app.media.ffmpeg import discover_ffmpeg


class PipelineNotReadyError(RuntimeError):
    pass


def validate_ai_pipeline(config: ProcessingConfig) -> None:
    """Fail closed unless each requested AI stage has a visible CUDA device."""
    if not config.interpolate and not config.upscale:
        raise PipelineNotReadyError(
            "No AI stage is enabled. Use the standard FFmpeg export for non-AI conversion."
        )

    missing: list[str] = []
    if config.interpolate:
        status = rife_status(config.gpu_index)
        if not status.available:
            missing.append(f"RIFE CUDA interpolation: {status.detail}")
    if config.upscale:
        status = realesrgan_status(config.gpu_index)
        if not status.available:
            missing.append(f"Real-ESRGAN CUDA upscaling: {status.detail}")
    if missing:
        raise PipelineNotReadyError("Requested AI stages are unavailable: " + "; ".join(missing))

    ffmpeg = discover_ffmpeg()
    if not ffmpeg.available:
        raise PipelineNotReadyError(ffmpeg.error or "FFmpeg and FFprobe are required.")
    if config.video_codec.endswith("_nvenc") and config.video_codec not in ffmpeg.nvenc_encoders:
        raise PipelineNotReadyError(
            f"FFmpeg does not expose {config.video_codec}. Select a supported encoder or install an NVENC-enabled FFmpeg build."
        )
