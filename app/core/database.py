from __future__ import annotations

import sqlite3
from pathlib import Path

from app.core.job import Job


class JobDatabase:
    """Persist job metadata only; never copy source media into SQLite."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, input_path TEXT NOT NULL, output_path TEXT NOT NULL,
                status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, stage TEXT NOT NULL,
                error TEXT, created_at TEXT NOT NULL
            )""")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def upsert_job(self, job: Job) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, progress=excluded.progress, stage=excluded.stage,
                error=excluded.error""",
                (job.id, str(job.input_path), str(job.output_path), job.status.value,
                 job.progress, job.stage, job.error, job.created_at),
            )

    def list_jobs(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            )]
