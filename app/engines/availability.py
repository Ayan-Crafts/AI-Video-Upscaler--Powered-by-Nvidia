from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineAvailability:
    name: str
    available: bool
    detail: str


def torch_cuda_status(device_index: int = 0) -> EngineAvailability:
    if importlib.util.find_spec("torch") is None:
        return EngineAvailability("PyTorch CUDA", False, "PyTorch is not installed.")
    try:
        import torch

        if not torch.cuda.is_available():
            return EngineAvailability("PyTorch CUDA", False, "PyTorch cannot access a CUDA device.")
        if device_index < 0 or device_index >= torch.cuda.device_count():
            return EngineAvailability("PyTorch CUDA", False, f"CUDA device {device_index} is unavailable.")
        name = torch.cuda.get_device_name(device_index)
        runtime = torch.version.cuda or "unknown"
        capability = torch.cuda.get_device_capability(device_index)
        memory = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
        return EngineAvailability(
            "PyTorch CUDA",
            True,
            f"{name} · CUDA {runtime} · CC {capability[0]}.{capability[1]} · {memory:.1f} GB VRAM",
        )
    except (ImportError, OSError, RuntimeError) as exc:
        return EngineAvailability("PyTorch CUDA", False, f"CUDA initialization failed: {exc}")


def rife_status(device_index: int = 0) -> EngineAvailability:
    cuda = torch_cuda_status(device_index)
    return EngineAvailability(
        "RIFE CUDA",
        cuda.available,
        "Practical-RIFE v4.25 · model downloads on first use" if cuda.available else cuda.detail,
    )


def realesrgan_status(device_index: int = 0) -> EngineAvailability:
    cuda = torch_cuda_status(device_index)
    return EngineAvailability(
        "Real-ESRGAN CUDA",
        cuda.available,
        "RealESRGAN_x4plus · tiled FP16 inference · model downloads on first use"
        if cuda.available else cuda.detail,
    )
