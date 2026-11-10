from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any
import logging

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class GPUInfo:
    available: bool = False
    index: int = 0
    name: str = "No NVIDIA GPU detected"
    driver_version: str | None = None
    compute_capability: str | None = None
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    temperature_c: int | None = None
    utilization_percent: int | None = None
    backend: str = "unavailable"
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _detect_with_nvml() -> list[GPUInfo]:
    import pynvml  # type: ignore[import-not-found]
    pynvml.nvmlInit()
    devices: list[GPUInfo] = []
    try:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            if isinstance(driver, bytes):
                driver = driver.decode(errors="replace")
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            devices.append(GPUInfo(
                available=True, index=index, name=str(name), driver_version=str(driver),
                vram_total_mb=int(memory.total // (1024 * 1024)),
                vram_free_mb=int(memory.free // (1024 * 1024)),
                temperature_c=int(temp), utilization_percent=int(util.gpu), backend="NVML",
            ))
    finally:
        pynvml.nvmlShutdown()
    return devices


def detect_gpus() -> list[GPUInfo]:
    """Discover all visible NVIDIA GPUs, without matching marketing model names."""
    try:
        devices = _detect_with_nvml()
        if devices:
            return devices
    except Exception:
        pass
    executable = shutil.which("nvidia-smi")
    if not executable:
        return [GPUInfo(warning="NVIDIA driver utilities unavailable.")]
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,name,driver_version,memory.total,memory.free,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=8, check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return [GPUInfo(warning=result.stderr.strip() or "nvidia-smi could not query a GPU.")]
        devices = []
        for line in result.stdout.splitlines():
            row = [part.strip() for part in line.split(",")]
            if len(row) < 7:
                continue
            devices.append(GPUInfo(
                available=True, index=int(row[0]), name=row[1], driver_version=row[2],
                vram_total_mb=int(float(row[3])), vram_free_mb=int(float(row[4])),
                temperature_c=int(float(row[5])), utilization_percent=int(float(row[6])),
                backend="nvidia-smi",
            ))
        return devices or [GPUInfo(warning="Unexpected nvidia-smi response.")]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return [GPUInfo(warning=f"GPU detection failed: {exc}")]


def detect_gpu(index: int = 0) -> GPUInfo:
    return next((gpu for gpu in detect_gpus() if gpu.index == index),
                GPUInfo(warning=f"NVIDIA GPU index {index} was not found."))
