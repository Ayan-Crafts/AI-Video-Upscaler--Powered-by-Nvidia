from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_CONTAINERS = {"mp4", "mkv", "mov", "webm"}
CONTAINER_CODECS = {
    "mp4": {"h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264", "libx265", "libaom-av1"},
    "mkv": {"h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264", "libx265", "libaom-av1", "libvpx-vp9"},
    "mov": {"h264_nvenc", "hevc_nvenc", "prores_ks", "libx264", "libx265"},
    "webm": {"libvpx-vp9", "libaom-av1", "av1_nvenc"},
}
VALID_SAMPLE_RATES = {44_100, 48_000, 96_000}


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    input_path: Path
    output_path: Path
    target_fps: float = 60.0
    output_width: int = 3840
    output_height: int = 2160
    audio_sample_rate: int = 48_000
    container: str = "mp4"
    video_codec: str = "hevc_nvenc"
    interpolate: bool = True
    upscale: bool = True
    resample_audio: bool = True
    overwrite: bool = False
    gpu_index: int = 0
    fp16: bool = True
    rife_scale: float | None = None
    upscale_tile_size: int = 0


def validate_config(config: ProcessingConfig) -> list[str]:
    errors: list[str] = []
    if not config.input_path.exists() or not config.input_path.is_file():
        errors.append("Input video does not exist or is not a file.")
    if config.container not in SUPPORTED_CONTAINERS:
        errors.append(f"Unsupported output container: {config.container}")
    elif config.video_codec not in CONTAINER_CODECS[config.container]:
        errors.append(f"Codec {config.video_codec} is not configured for {config.container}.")
    if config.output_path.suffix.lower().lstrip(".") != config.container:
        errors.append("Output filename extension must match the selected container.")
    if not 1.0 <= config.target_fps <= 240.0:
        errors.append("Target FPS must be between 1 and 240.")
    if config.output_width <= 0 or config.output_height <= 0:
        errors.append("Output dimensions must be positive.")
    elif config.output_width % 2 or config.output_height % 2:
        errors.append("Output width and height must be even for the current yuv420p output path.")
    if config.output_width > 7680 or config.output_height > 4320:
        errors.append("Output dimensions exceed the supported 8K canvas (7680 × 4320).")
    if config.audio_sample_rate not in VALID_SAMPLE_RATES:
        errors.append("Audio sample rate must be 44100, 48000 or 96000 Hz.")
    if config.gpu_index < 0:
        errors.append("GPU index must be zero or greater.")
    if config.rife_scale is not None and config.rife_scale not in {0.25, 0.5, 1.0}:
        errors.append("RIFE inference scale must be 0.25, 0.5 or 1.0.")
    if config.upscale_tile_size < 0 or (0 < config.upscale_tile_size < 32):
        errors.append("Upscaling tile size must be 0 (automatic) or at least 32 pixels.")
    if not (config.interpolate or config.upscale or config.resample_audio):
        errors.append("At least one processing operation must be enabled.")
    try:
        if config.output_path.resolve() == config.input_path.resolve():
            errors.append("Output path must not overwrite the source video.")
    except OSError:
        errors.append("Input or output path cannot be resolved.")
    if config.output_path.exists() and not config.overwrite:
        errors.append("Output already exists; enable overwrite or choose another path.")
    return errors
