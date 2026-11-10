from pathlib import Path

from app.core.config import ProcessingConfig
from app.engines.ffmpeg_command import build_ffmpeg_transcode_command
from app.media.ffmpeg import ToolStatus, parse_ffmpeg_progress


def test_ffmpeg_needs_both_tools() -> None:
    assert not ToolStatus("ffmpeg", None).available
    assert ToolStatus("ffmpeg", "ffprobe").available


def test_progress_percent() -> None:
    assert parse_ffmpeg_progress({"out_time_us": "5000000"}, 10) == 50
    assert parse_ffmpeg_progress({"out_time_us": "12000000"}, 10) == 100
    assert parse_ffmpeg_progress({"out_time_us": "not-a-number"}, 10) is None
    assert parse_ffmpeg_progress({"out_time_us": "100"}, None) is None


def test_command_requests_nvenc_and_48khz(tmp_path: Path) -> None:
    cfg = ProcessingConfig(tmp_path / "in.mp4", tmp_path / "out.mp4")
    cmd = build_ffmpeg_transcode_command("ffmpeg", cfg)
    assert "hevc_nvenc" in cmd
    assert "48000" in cmd
    assert "-progress" in cmd
    assert any("lanczos" in arg for arg in cmd)


def test_conventional_command_is_not_labelled_ai(tmp_path: Path) -> None:
    cfg = ProcessingConfig(tmp_path / "in.mp4", tmp_path / "out.mp4")
    cmd = build_ffmpeg_transcode_command("ffmpeg", cfg)
    assert not any("rife" in arg.lower() or "realesrgan" in arg.lower() for arg in cmd)
