
from __future__ import annotations

import csv
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GPUInfo:
    """Information and telemetry for one NVIDIA GPU."""

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


def _optional_int(value: str | None) -> int | None:
    """Convert an NVIDIA telemetry value into an integer safely.

    NVIDIA may return N/A for unsupported sensors or unavailable metrics.
    """
    if value is None:
        return None

    try:
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return None


def _detect_with_nvml() -> list[GPUInfo]:
    """Discover NVIDIA GPUs using NVML.

    NVML provides efficient hardware telemetry without repeatedly launching
    nvidia-smi subprocesses.
    """
    import pynvml  # type: ignore[import-not-found]

    pynvml.nvmlInit()

    devices: list[GPUInfo] = []

    try:
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode(errors="replace")
        except pynvml.NVMLError as exc:
            logger.warning("Unable to retrieve NVIDIA driver version: %s", exc)
            driver = None

        gpu_count = pynvml.nvmlDeviceGetCount()

        for index in range(gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except pynvml.NVMLError as exc:
                logger.warning("Unable to access NVIDIA GPU %s: %s", index, exc)
                continue

            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode(errors="replace")
            except pynvml.NVMLError as exc:
                logger.warning("Unable to retrieve GPU %s name: %s", index, exc)
                name = f"NVIDIA GPU {index}"

            # Memory information
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory = int(memory.total // (1024 * 1024))
                free_memory = int(memory.free // (1024 * 1024))
            except pynvml.NVMLError as exc:
                logger.warning("Unable to retrieve GPU %s memory: %s", index, exc)
                total_memory = None
                free_memory = None

            # GPU utilization
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = int(utilization.gpu)
            except pynvml.NVMLError as exc:
                logger.debug("GPU %s utilization unavailable: %s", index, exc)
                gpu_utilization = None

            # GPU temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
                temperature_c = int(temperature)
            except pynvml.NVMLError as exc:
                logger.debug("GPU %s temperature unavailable: %s", index, exc)
                temperature_c = None

            devices.append(
                GPUInfo(
                    available=True,
                    index=index,
                    name=str(name),
                    driver_version=str(driver) if driver else None,
                    vram_total_mb=total_memory,
                    vram_free_mb=free_memory,
                    temperature_c=temperature_c,
                    utilization_percent=gpu_utilization,
                    backend="NVML",
                )
            )

    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as exc:
            logger.debug("NVML shutdown returned an error: %s", exc)

    return devices


def _detect_with_nvidia_smi() -> list[GPUInfo]:
    """Fallback NVIDIA GPU detection using the nvidia-smi executable."""

    executable = shutil.which("nvidia-smi")

    if not executable:
        return [
            GPUInfo(
                warning=(
                    "NVIDIA GPU utilities are unavailable. "
                    "Check the NVIDIA driver installation."
                )
            )
        ]

    try:
        result = subprocess.run(
            [
                executable,
                (
                    "--query-gpu=index,name,driver_version,"
                    "memory.total,memory.free,"
                    "temperature.gpu,utilization.gpu"
                ),
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )

        if result.returncode != 0:
            return [
                GPUInfo(
                    warning=(
                        result.stderr.strip()
                        or "nvidia-smi could not query NVIDIA GPUs."
                    )
                )
            ]

        if not result.stdout.strip():
            return [
                GPUInfo(
                    warning="nvidia-smi returned no NVIDIA GPU devices."
                )
            ]

        devices: list[GPUInfo] = []

        for row in csv.reader(
            result.stdout.splitlines(),
            skipinitialspace=True,
        ):
            if len(row) < 7:
                logger.warning("Ignoring malformed nvidia-smi row: %r", row)
                continue

            index = _optional_int(row[0])

            if index is None:
                logger.warning("Invalid NVIDIA GPU index: %r", row)
                continue

            devices.append(
                GPUInfo(
                    available=True,
                    index=index,
                    name=row[1].strip(),
                    driver_version=row[2].strip() or None,
                    vram_total_mb=_optional_int(row[3]),
                    vram_free_mb=_optional_int(row[4]),
                    temperature_c=_optional_int(row[5]),
                    utilization_percent=_optional_int(row[6]),
                    backend="nvidia-smi",
                )
            )

        if not devices:
            return [
                GPUInfo(
                    warning="No parseable NVIDIA GPU records were returned."
                )
            ]

        return devices

    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.exception("nvidia-smi GPU detection failed")

        return [
            GPUInfo(
                warning=f"GPU detection failed: {exc}"
            )
        ]


def detect_gpus() -> list[GPUInfo]:
    """Discover all visible NVIDIA GPUs.

    Detection order:
        1. NVML
        2. nvidia-smi fallback

    GPU discovery is independent of CUDA, TensorRT and AI model availability.
    Those capabilities must be tested separately.
    """

    try:
        import pynvml  # type: ignore[import-not-found]

    except ImportError:
        logger.info(
            "pynvml is unavailable; using nvidia-smi for GPU detection."
        )

    else:
        try:
            devices = _detect_with_nvml()

            if devices:
                return devices

            logger.warning(
                "NVML returned no NVIDIA GPUs; trying nvidia-smi."
            )

        except (
            pynvml.NVMLError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            logger.warning(
                "NVML detection failed; falling back to nvidia-smi: %s",
                exc,
            )

    return _detect_with_nvidia_smi()


def detect_gpu(index: int = 0) -> GPUInfo:
    """Return a particular NVIDIA GPU by its device index."""

    devices = detect_gpus()

    return next(
        (
            gpu
            for gpu in devices
            if gpu.index == index
        ),
        GPUInfo(
            warning=f"NVIDIA GPU index {index} was not found."
        ),
    )
