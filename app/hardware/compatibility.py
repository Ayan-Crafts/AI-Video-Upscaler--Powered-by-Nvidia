from __future__ import annotations

from dataclasses import asdict, dataclass

from app.hardware.gpu import detect_gpus
from app.media.ffmpeg import discover_ffmpeg


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    gpu_detected: bool
    gpu_name: str
    driver_version: str | None
    vram_total_mb: int | None
    compute_capability: str | None
    cuda_available: bool
    nvenc_encoders: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_compatibility_report(gpu_index: int = 0) -> CompatibilityReport:
    gpu = next((item for item in detect_gpus() if item.index == gpu_index), None)
    warnings: list[str] = []
    capability: str | None = None
    cuda_available = False
    if gpu is None or not gpu.available:
        warnings.append("No NVIDIA GPU could be queried; check the NVIDIA display driver.")
        name, driver, vram = "Unavailable", None, None
    else:
        name, driver, vram = gpu.name, gpu.driver_version, gpu.vram_total_mb
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available() and gpu_index < torch.cuda.device_count())
            if cuda_available:
                major, minor = torch.cuda.get_device_capability(gpu_index)
                capability = f"{major}.{minor}"
            else:
                warnings.append("NVIDIA GPU is visible, but PyTorch cannot access CUDA.")
        except ImportError:
            warnings.append("PyTorch is not installed; CUDA inference self-test was not run.")
        except (OSError, RuntimeError) as exc:
            warnings.append(f"CUDA capability check failed: {exc}")
    ffmpeg = discover_ffmpeg()
    if not ffmpeg.available:
        warnings.append(ffmpeg.error or "FFmpeg/FFprobe not available.")
    if not ffmpeg.nvenc_encoders:
        warnings.append("No NVENC encoders detected in the available FFmpeg build.")
    return CompatibilityReport(bool(gpu and gpu.available), name, driver, vram, capability,
                               cuda_available, ffmpeg.nvenc_encoders, tuple(warnings))
