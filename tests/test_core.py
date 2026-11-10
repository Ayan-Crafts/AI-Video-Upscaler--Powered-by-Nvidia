from pathlib import Path

import pytest

from app.core.config import ProcessingConfig, validate_config
from app.core.database import JobDatabase
from app.core.job import Job, JobStatus


def source_file(tmp_path: Path) -> Path:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    return source


def test_valid_configuration(tmp_path: Path) -> None:
    assert validate_config(ProcessingConfig(source_file(tmp_path), tmp_path / "result.mp4")) == []


def test_reject_source_overwrite(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    assert any("must not overwrite" in e for e in validate_config(ProcessingConfig(source, source)))


def test_reject_bad_fps_and_sample_rate(tmp_path: Path) -> None:
    config = ProcessingConfig(source_file(tmp_path), tmp_path / "out.mp4", target_fps=0.1,
                              audio_sample_rate=32000)
    errors = validate_config(config)
    assert any("FPS" in e for e in errors)
    assert any("sample rate" in e for e in errors)


def test_container_codec_mismatch(tmp_path: Path) -> None:
    config = ProcessingConfig(source_file(tmp_path), tmp_path / "out.webm", container="webm",
                              video_codec="hevc_nvenc")
    assert any("not configured" in e for e in validate_config(config))


def test_8k_limit(tmp_path: Path) -> None:
    ok = ProcessingConfig(source_file(tmp_path), tmp_path / "out.mp4", output_width=7680, output_height=4320)
    bad = ProcessingConfig(ok.input_path, tmp_path / "bad.mp4", output_width=8192, output_height=4320)
    assert validate_config(ok) == []
    assert any("8K" in e for e in validate_config(bad))


def test_existing_output_is_protected(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    output = tmp_path / "out.mp4"
    output.write_text("keep")
    assert any("already exists" in e for e in validate_config(ProcessingConfig(source, output)))


def test_job_progress_and_state() -> None:
    job = Job(Path("in.mp4"), Path("out.mp4"))
    assert job.status == JobStatus.QUEUED
    job.set_progress(70, "Upscaling")
    assert job.progress == 70
    with pytest.raises(ValueError):
        job.set_progress(101, "bad")


def test_job_database_round_trip(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "jobs.sqlite3")
    job = Job(Path("input.mp4"), Path("output.mp4"))
    db.upsert_job(job)
    job.set_progress(25, "Interpolation")
    db.upsert_job(job)
    rows = db.list_jobs()
    assert len(rows) == 1 and rows[0]["id"] == job.id and rows[0]["progress"] == 25
