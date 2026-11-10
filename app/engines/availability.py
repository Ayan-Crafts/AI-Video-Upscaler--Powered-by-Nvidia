from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineAvailability:
    name: str
    available: bool
    detail: str


def torch_cuda_status() -> EngineAvailability:
    if importlib.util.find_spec("torch") is None:
        return EngineAvailability("PyTorch CUDA", False, "PyTorch is not installed.")
    try:
        import torch
        if not torch.cuda.is_available():
            return EngineAvailability("PyTorch CUDA", False, "PyTorch cannot access a CUDA device.")
        return EngineAvailability("PyTorch CUDA", True,
                                  f"{torch.cuda.get_device_name(0)} · runtime {torch.version.cuda}")
    except (ImportError, OSError, RuntimeError) as exc:
        return EngineAvailability("PyTorch CUDA", False, f"CUDA initialization failed: {exc}")


def rife_status() -> EngineAvailability:
    return EngineAvailability("RIFE CUDA", False,
                              "Not integrated in this development build; no AI interpolation is performed yet.")


def realesrgan_status() -> EngineAvailability:
    return EngineAvailability("Real-ESRGAN CUDA", False,
                              "Not integrated in this development build; no AI super-resolution is performed yet.")
