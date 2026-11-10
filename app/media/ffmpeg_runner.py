from __future__ import annotations

import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from app.core.config import ProcessingConfig, validate_config
from app.engines.ffmpeg_command import build_ffmpeg_transcode_command
from app.media.ffmpeg import parse_ffmpeg_progress


class FFmpegExecutionError(RuntimeError):
    pass


class FFmpegRunner:
    """Cancellable FFmpeg runner for a clearly labelled non-AI fallback export."""

    def __init__(self, ffmpeg_path: str) -> None:
        self.ffmpeg_path = ffmpeg_path
        self._process: subprocess.Popen[str] | None = None
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        self._cancel_requested.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()

    def run(self, config: ProcessingConfig, duration_seconds: float,
            on_progress: Callable[[float, str], None]) -> Path:
        errors = validate_config(config)
        if errors:
            raise FFmpegExecutionError("Invalid processing configuration: " + "; ".join(errors))
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_ffmpeg_transcode_command(self.ffmpeg_path, config)
        self._cancel_requested.clear()
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_log:
            self._process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=stderr_log, text=True,
                encoding="utf-8", errors="replace", bufsize=1,
            )
            assert self._process.stdout is not None
            record: dict[str, str] = {}
            for line in self._process.stdout:
                if self._cancel_requested.is_set():
                    break
                line = line.strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                record[key] = value
                if key == "progress":
                    percent = parse_ffmpeg_progress(record, duration_seconds)
                    if percent is not None:
                        on_progress(percent, "FFmpeg standard export (not AI)")
                    record.clear()
            return_code = self._process.wait()
            stderr_log.seek(0)
            error_text = stderr_log.read()[-8000:]
            self._process = None
            if self._cancel_requested.is_set():
                config.output_path.unlink(missing_ok=True)
                raise FFmpegExecutionError("Processing was cancelled.")
            if return_code != 0:
                config.output_path.unlink(missing_ok=True)
                raise FFmpegExecutionError(error_text or f"FFmpeg exited with code {return_code}.")
            if not config.output_path.exists() or config.output_path.stat().st_size == 0:
                config.output_path.unlink(missing_ok=True)
                raise FFmpegExecutionError("FFmpeg reported success but produced no output file.")
            on_progress(100.0, "Completed")
            return config.output_path
