from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ToolStatus:
    ffmpeg: str | None
    ffprobe: str | None
    ffmpeg_version: str | None = None
    nvenc_encoders: tuple[str, ...] = ()
    error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.ffmpeg and self.ffprobe)


def _run(args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, check=False)


def discover_ffmpeg() -> ToolStatus:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    status = ToolStatus(ffmpeg=ffmpeg, ffprobe=ffprobe)
    if not status.available:
        status.error = "FFmpeg and FFprobe must be available on PATH or bundled with the application."
        return status
    try:
        version = _run([ffmpeg, "-version"])
        status.ffmpeg_version = version.stdout.splitlines()[0] if version.stdout else None
        encoders = _run([ffmpeg, "-hide_banner", "-encoders"])
        status.nvenc_encoders = tuple(
            name for name in ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
            if name in encoders.stdout
        )
    except (OSError, subprocess.SubprocessError) as exc:
        status.error = str(exc)
    return status


def probe_media(path: Path, ffprobe_path: str | None = None) -> dict[str, Any]:
    executable = ffprobe_path or shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("FFprobe was not found. Install FFmpeg or configure its executable path.")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    result = _run([executable, "-v", "error", "-show_format", "-show_streams",
                   "-of", "json", str(path)], timeout=60.0)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFprobe failed.")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("FFprobe returned invalid JSON.") from exc
    payload["path"] = str(path.resolve())
    return payload


def parse_ffmpeg_progress(payload: dict[str, str], duration_seconds: float | None) -> float | None:
    """Parse FFmpeg -progress output; out_time_us is measured in microseconds."""
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
