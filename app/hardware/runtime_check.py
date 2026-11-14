"""Startup environment and NVIDIA hardware checks.

The goal is to turn obscure CUDA/DLL failures into actionable messages. The
checks never modify the system and never install or replace drivers.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from app.hardware.gpu import detect_gpus
from app.media.ffmpeg import discover_ffmpeg, probe_encoder
from app.paths import is_frozen, model_dir
from app.version import APP_VERSION, NVIDIA_DRIVER_URL

logger = logging.getLogger(__name__)

OK = "ok"
WARNING = "warning"
ERROR = "error"
_RANK = {OK: 0, WARNING: 1, ERROR: 2}

# NVIDIA driver requirements for the CUDA 12.6 runtime bundled with
# torch 2.14.1+cu126 (https://docs.nvidia.com/deploy/cuda-compatibility/).
#   minimum: CUDA 12.x minor-version compatibility (CUDA 12.0 GA driver).
#   recommended: the CUDA 12.6 GA driver.
DRIVER_REQUIREMENTS = {
    "12.6": {
        "windows": ("527.41", "560.76"),
        "linux": ("525.60.13", "560.28.03"),
    },
}

# FP16 inference is enabled from compute capability 7.0 (Volta/Turing and
# newer, which have Tensor Cores). This mirrors the engines' own rule.
FP16_MIN_CAPABILITY = (7, 0)
LOW_VRAM_GB = 4.0
REDUCED_TILE_VRAM_GB = 8.0


@dataclass(slots=True)
class CheckItem:
    name: str
    status: str
    detail: str


@dataclass(slots=True)
class DeviceReport:
    index: int
    name: str
    compute_capability: str
    vram_gb: float
    architecture_supported: bool
    fp16: bool
    rife_inference_scale: float
    upscale_tile_size: int


@dataclass(slots=True)
class RuntimeReport:
    app_version: str = APP_VERSION
    frozen: bool = False
    platform: str = sys.platform
    python_version: str = sys.version.split()[0]
    driver_version: str | None = None
    torch_version: str | None = None
    torch_cuda_runtime: str | None = None
    torch_arch_list: list[str] = field(default_factory=list)
    cuda_available: bool = False
    devices: list[DeviceReport] = field(default_factory=list)
    ffmpeg_path: str | None = None
    ffmpeg_source: str | None = None
    ffmpeg_version: str | None = None
    nvenc_compiled: list[str] = field(default_factory=list)
    nvenc_usable: list[str] = field(default_factory=list)
    model_cache: str = ""
    items: list[CheckItem] = field(default_factory=list)

    @property
    def level(self) -> str:
        return max((item.status for item in self.items), key=_RANK.__getitem__, default=OK)

    @property
    def gpu_ready(self) -> bool:
        return self.cuda_available and any(device.architecture_supported for device in self.devices)

    def add(self, name: str, status: str, detail: str) -> None:
        self.items.append(CheckItem(name, status, detail))

    def problems(self) -> list[CheckItem]:
        return [item for item in self.items if item.status != OK]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["level"] = self.level
        payload["gpu_ready"] = self.gpu_ready
        return payload

    def as_text(self) -> str:
        symbols = {OK: "✔", WARNING: "⚠", ERROR: "✖"}
        lines = [f"{symbols[item.status]} {item.name}: {item.detail}" for item in self.items]
        return "\n".join(lines)


def parse_version(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    parts: list[int] = []
    for piece in text.strip().split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def driver_requirement(cuda_runtime: str | None, platform: str | None = None) -> tuple[str, str] | None:
    if not cuda_runtime:
        return None
    key = ".".join(cuda_runtime.split(".")[:2])
    platform_key = "windows" if (platform or sys.platform).startswith("win") else "linux"
    entry = DRIVER_REQUIREMENTS.get(key)
    return entry[platform_key] if entry else None


def architecture_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """Whether the PyTorch build ships kernels runnable on this compute capability.

    ``sm_XY`` binaries run on devices of the same major version with an equal
    or higher minor version; ``compute_XY`` PTX can be JIT-compiled for any
    device with capability >= X.Y.
    """
    if not arch_list:
        return False
    major, minor = capability
    for arch in arch_list:
        kind, _, number = arch.partition("_")
        if not number.isdigit() or len(number) < 2:
            continue
        arch_major, arch_minor = int(number[:-1]), int(number[-1])
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            return True
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return True
    return False


def default_tuning(vram_gb: float) -> tuple[float, int]:
    """Defaults used by the engines: (RIFE inference scale, Real-ESRGAN tile)."""
    if vram_gb <= REDUCED_TILE_VRAM_GB:
        return 0.5, 128
    return 1.0, 256


def run_runtime_checks(*, probe_nvenc: bool = True) -> RuntimeReport:
    report = RuntimeReport(frozen=is_frozen(), model_cache=str(model_dir()))

    # 1. NVIDIA driver / GPU visibility (NVML, then nvidia-smi).
    gpus = [gpu for gpu in detect_gpus() if gpu.available]
    if gpus:
        report.driver_version = gpus[0].driver_version
        names = ", ".join(f"{gpu.name} ({gpu.vram_total_mb or '?'} MB)" for gpu in gpus)
        report.add("NVIDIA GPU", OK, f"{names} · driver {report.driver_version or 'unknown'}")
    else:
        report.add(
            "NVIDIA GPU",
            ERROR,
            "No NVIDIA GPU or NVIDIA display driver was detected. AI processing needs an NVIDIA "
            f"GPU with a current driver: {NVIDIA_DRIVER_URL}",
        )

    # 2. Bundled PyTorch runtime.
    torch = None
    try:
        import torch as torch_module

        torch = torch_module
        report.torch_version = torch.__version__
        report.torch_cuda_runtime = torch.version.cuda
        try:
            report.torch_arch_list = list(torch.cuda.get_arch_list())
        except (RuntimeError, AttributeError):
            report.torch_arch_list = []
    except (ImportError, OSError) as exc:
        hint = (
            "The bundled PyTorch runtime failed to load. Reinstall AI Upscaler Studio."
            if is_frozen()
            else "PyTorch is not installed or failed to load in this environment."
        )
        if os.name == "nt" and isinstance(exc, OSError):
            hint += " A missing or blocked DLL is the usual cause (see docs/INSTALLATION.md)."
        report.add("PyTorch runtime", ERROR, f"{hint} ({exc})")

    if torch is not None:
        if not report.torch_cuda_runtime:
            report.add(
                "PyTorch runtime",
                ERROR,
                f"PyTorch {report.torch_version} is a CPU-only build; CUDA inference is impossible.",
            )
        else:
            report.add(
                "PyTorch runtime",
                OK,
                f"PyTorch {report.torch_version} · bundled CUDA runtime {report.torch_cuda_runtime}",
            )

    # 3. Driver version against the bundled CUDA runtime.
    requirement = driver_requirement(report.torch_cuda_runtime)
    driver = parse_version(report.driver_version)
    if requirement and driver:
        minimum, recommended = requirement
        if driver < parse_version(minimum):
            report.add(
                "Driver compatibility",
                ERROR,
                f"NVIDIA driver {report.driver_version} is too old for the bundled CUDA "
                f"{report.torch_cuda_runtime} runtime (minimum {minimum}, recommended {recommended}+). "
                f"Update the driver: {NVIDIA_DRIVER_URL}",
            )
        elif driver < parse_version(recommended):
            report.add(
                "Driver compatibility",
                WARNING,
                f"Driver {report.driver_version} works through CUDA minor-version compatibility; "
                f"{recommended}+ is recommended for CUDA {report.torch_cuda_runtime}.",
            )
        else:
            report.add(
                "Driver compatibility",
                OK,
                f"Driver {report.driver_version} meets CUDA {report.torch_cuda_runtime} requirements.",
            )

    # 4. CUDA initialization and per-device capability.
    if torch is not None and report.torch_cuda_runtime:
        try:
            report.cuda_available = bool(torch.cuda.is_available())
            if not report.cuda_available and gpus:
                torch.cuda.init()  # raises with the driver's real reason
        except (RuntimeError, OSError, AssertionError) as exc:
            report.cuda_available = False
            report.add(
                "CUDA initialization",
                ERROR,
                "CUDA could not be initialized. This usually means the NVIDIA driver is missing, "
                f"too old or not loaded. Update it from {NVIDIA_DRIVER_URL} ({exc})",
            )
        if report.cuda_available:
            for index in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(index)
                    capability = (int(props.major), int(props.minor))
                    vram_gb = props.total_memory / (1024**3)
                    supported = architecture_supported(capability, report.torch_arch_list)
                    fp16 = capability >= FP16_MIN_CAPABILITY
                    scale, tile = default_tuning(vram_gb)
                    report.devices.append(
                        DeviceReport(index, props.name, f"{capability[0]}.{capability[1]}",
                                     round(vram_gb, 2), supported, fp16, scale, tile)
                    )
                except (RuntimeError, AssertionError) as exc:
                    report.add(f"CUDA device {index}", ERROR, f"Could not query device: {exc}")
            if not report.devices:
                report.add("CUDA initialization", ERROR, "CUDA initialized but reported no devices.")
            for device in report.devices:
                label = f"CUDA device {device.index}"
                summary = (
                    f"{device.name} · compute capability {device.compute_capability} · "
                    f"{device.vram_gb:.1f} GB · {'FP16' if device.fp16 else 'FP32'} · "
                    f"RIFE scale {device.rife_inference_scale} · tile {device.upscale_tile_size}px"
                )
                if not device.architecture_supported:
                    report.add(
                        label,
                        ERROR,
                        f"{summary}. This GPU architecture is not supported by the bundled PyTorch "
                        f"CUDA {report.torch_cuda_runtime} build (supported: "
                        f"{' '.join(report.torch_arch_list)}).",
                    )
                elif device.vram_gb < LOW_VRAM_GB:
                    report.add(label, WARNING, f"{summary}. Less than {LOW_VRAM_GB:.0f} GB VRAM: "
                               "expect small tiles, slow processing and possible out-of-memory errors.")
                elif not device.fp16:
                    report.add(label, WARNING, f"{summary}. FP16 is disabled on this older "
                               "architecture; FP32 inference is slower and uses more memory.")
                else:
                    report.add(label, OK, summary)
        elif not any(item.name == "CUDA initialization" for item in report.items):
            status = ERROR if gpus else WARNING
            report.add("CUDA initialization", status, "PyTorch cannot access a CUDA device.")

    # 5. FFmpeg / FFprobe and NVENC.
    tools = discover_ffmpeg(refresh=True)
    report.ffmpeg_path = tools.ffmpeg
    report.ffmpeg_source = tools.source
    report.ffmpeg_version = tools.ffmpeg_version
    report.nvenc_compiled = list(tools.nvenc_encoders)
    if not tools.available or tools.error:
        report.add("FFmpeg", ERROR, tools.error or "FFmpeg/FFprobe unavailable.")
    else:
        report.add("FFmpeg", OK, f"{tools.ffmpeg_version} ({tools.source})")
        if not tools.nvenc_encoders:
            report.add("NVENC", WARNING, "This FFmpeg build has no NVENC encoders; use software codecs.")
        elif probe_nvenc and gpus:
            failures: list[str] = []
            for codec in tools.nvenc_encoders:
                ok, detail = probe_encoder(codec, tools.ffmpeg)
                if ok:
                    report.nvenc_usable.append(codec)
                else:
                    failures.append(f"{codec}: {detail}")
            if report.nvenc_usable:
                note = f" Unavailable: {'; '.join(failures)}" if failures else ""
                report.add("NVENC", OK, f"Hardware encode verified: {', '.join(report.nvenc_usable)}.{note}")
            else:
                report.add("NVENC", WARNING, "NVENC encoders are compiled in but failed an encode test "
                           f"on this GPU/driver; use software codecs. {'; '.join(failures)}")
        else:
            report.add("NVENC", WARNING if not gpus else OK,
                       f"Compiled in: {', '.join(tools.nvenc_encoders)} (not tested; no NVIDIA GPU)."
                       if not gpus else f"Compiled in: {', '.join(tools.nvenc_encoders)}.")

    logger.info("Runtime check level=%s\n%s", report.level, report.as_text())
    return report
