from __future__ import annotations

from app.core.config import ProcessingConfig
from app.engines.availability import realesrgan_status, rife_status


class PipelineNotReadyError(RuntimeError):
    pass


def validate_ai_pipeline(config: ProcessingConfig) -> None:
    missing: list[str] = []
    if config.interpolate and not rife_status().available:
        missing.append("RIFE CUDA interpolation")
    if config.upscale and not realesrgan_status().available:
        missing.append("Real-ESRGAN CUDA upscaling")
    if missing:
        raise PipelineNotReadyError("Requested AI stages unavailable: " + ", ".join(missing))
    raise PipelineNotReadyError("No production AI pipeline has been registered yet.")
