from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.subprocess_utils import run_tool
from app.paths import ENV_FFMPEG_DIR, bundled_ffmpeg_dir, executable_name

logger = logging.getLogger(__name__)

NVENC_ENCODERS = ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
SOFTWARE_ENCODERS = ("libx264", "libx265", "libvpx-vp9", "libaom-av1", "prores_ks", "aac", "libopus")

SOURCE_BUNDLED = "bundled"
SOURCE_CONFIGURED = "configured"
SOURCE_PATH = "system PATH"


@dataclass(slots=True)
class ToolStatus:
    ffmpeg: str | None
    ffprobe: str | None
    ffmpeg_version: str | None = None
    nvenc_encoders: tuple[str, ...] = ()
    error: str | None = None
    source: str | None = None
    software_encoders: tuple[str, ...] = ()
    build_configuration: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.ffmpeg and self.ffprobe)


def _run(args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return run_tool(args, timeout=timeout)


def find_tool(name: str) -> tuple[str | None, str | None]:
    """Locate ``ffmpeg``/``ffprobe``.

    Search order:
        1. Executables bundled with the packaged application.
        2. A directory configured through ``AI_UPSCALER_STUDIO_FFMPEG_DIR``.
        3. The system PATH (development fallback).
    """
    filename = executable_name(name)
    bundled = bundled_ffmpeg_dir() / filename
    if bundled.is_file():
        return str(bundled), SOURCE_BUNDLED
    configured_dir = os.environ.get(ENV_FFMPEG_DIR)
    if configured_dir:
        configured = Path(configured_dir) / filename
        if configured.is_file():
            return str(configured), SOURCE_CONFIGURED
        logger.warning("%s is set but %s was not found there.", ENV_FFMPEG_DIR, filename)
    on_path = shutil.which(name)
    if on_path:
        return on_path, SOURCE_PATH
    return None, None


def _parse_encoders(output: str) -> set[str]:
    names: set[str] = set()
    for line in output.splitlines():
        parts = line.split()
        # Encoder rows look like " V....D h264_nvenc  NVIDIA NVENC H.264 encoder".
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS" and parts[1] != "=":
            names.add(parts[1])
    return names


_cache_lock = threading.Lock()
_cached_status: ToolStatus | None = None


def discover_ffmpeg(refresh: bool = False) -> ToolStatus:
    """Return FFmpeg/FFprobe locations and compiled-in encoders (cached).

    A listed NVENC encoder only means FFmpeg was compiled with it; use
    :func:`probe_encoder` to confirm the GPU and driver can actually encode.
    """
    global _cached_status
    with _cache_lock:
        if _cached_status is not None and not refresh:
            return _cached_status
        _cached_status = _discover_uncached()
        return _cached_status


def _discover_uncached() -> ToolStatus:
    ffmpeg, source = find_tool("ffmpeg")
    ffprobe, probe_source = find_tool("ffprobe")
    status = ToolStatus(ffmpeg=ffmpeg, ffprobe=ffprobe, source=source)
    if not status.available:
        missing = [name for name, path in (("FFmpeg", ffmpeg), ("FFprobe", ffprobe)) if not path]
        status.error = (
            f"{' and '.join(missing)} could not be found. The packaged application bundles both; "
            "reinstall it, or for source builds put FFmpeg on PATH or set "
            f"{ENV_FFMPEG_DIR}."
        )
        return status
    if source != probe_source:
        logger.warning("FFmpeg (%s) and FFprobe (%s) come from different locations.", source, probe_source)
    try:
        version = _run([ffmpeg, "-hide_banner", "-version"])
        lines = version.stdout.splitlines()
        status.ffmpeg_version = lines[0] if lines else None
        status.build_configuration = next(
            (line.split(":", 1)[1].strip() for line in lines if line.startswith("configuration:")),
            None,
        )
        encoders = _parse_encoders(_run([ffmpeg, "-hide_banner", "-encoders"]).stdout)
        status.nvenc_encoders = tuple(name for name in NVENC_ENCODERS if name in encoders)
        status.software_encoders = tuple(name for name in SOFTWARE_ENCODERS if name in encoders)
    except (OSError, subprocess.SubprocessError) as exc:
        status.error = f"FFmpeg could not be executed: {exc}"
    return status


def probe_encoder(codec: str, ffmpeg_path: str | None = None, timeout: float = 20.0) -> tuple[bool, str]:
    """Encode a few synthetic frames to prove an encoder works on this machine."""
    executable = ffmpeg_path or discover_ffmpeg().ffmpeg
    if not executable:
        return False, "FFmpeg not found"
    args = [
        executable, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi", "-i", "testsrc2=size=256x144:rate=30:duration=0.2",
        "-c:v", codec, "-pix_fmt", "yuv420p", "-f", "null", "-",
    ]
    try:
        result = _run(args, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, "encode test passed"
    detail = (result.stderr.strip().splitlines() or [f"exit code {result.returncode}"])[-1]
    return False, detail


def probe_media(path: Path, ffprobe_path: str | None = None) -> dict[str, Any]:
    executable = ffprobe_path or find_tool("ffprobe")[0]
    if not executable:
        raise RuntimeError(
            "FFprobe was not found. Reinstall AI Upscaler Studio, or for source builds "
            f"install FFmpeg or set {ENV_FFMPEG_DIR}."
        )
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    result = _run(
        [executable, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        timeout=60.0,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFprobe failed.")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("FFprobe returned invalid JSON.") from exc
    payload["path"] = str(path.resolve())
    return payload


def parse_ffmpeg_progress(payload: dict[str, str], duration_seconds: float | None) -> float | None:
    """Parse FFmpeg -progress output; out_time_us is expressed in microseconds."""
    if not duration_seconds or duration_seconds <= 0:
        return None
    raw = payload.get("out_time_us") or payload.get("out_time_ms")
    if raw is None:
        return None
    try:
        elapsed = int(raw) / 1_000_000
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, elapsed / duration_seconds * 100.0))
