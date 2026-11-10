from __future__ import annotations

import shutil

from PySide6.QtCore import QObject, Signal, Slot

from app.core.config import ProcessingConfig
from app.media.ffmpeg_runner import FFmpegRunner


class TranscodeWorker(QObject):
    progress = Signal(float, str)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, config: ProcessingConfig, duration_seconds: float) -> None:
        super().__init__()
        self.config = config
        self.duration_seconds = duration_seconds
        self.runner: FFmpegRunner | None = None

    @Slot()
    def run(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.failed.emit("FFmpeg was not found on PATH in this development build.")
            return
        try:
            self.runner = FFmpegRunner(ffmpeg)
            output = self.runner.run(self.config, self.duration_seconds,
                                     lambda pct, stage: self.progress.emit(pct, stage))
            self.succeeded.emit(str(output))
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot()
    def cancel(self) -> None:
        if self.runner:
            self.runner.cancel()
