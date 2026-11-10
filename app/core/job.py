from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from uuid import uuid4


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROBING = "probing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Job:
    input_path: Path
    output_path: Path
    id: str = field(default_factory=lambda: str(uuid4()))
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    stage: str = "Queued"
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def set_progress(self, value: float, stage: str) -> None:
        if not 0.0 <= value <= 100.0:
            raise ValueError("Progress must be between 0 and 100.")
        self.progress = value
        self.stage = stage
