from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.core.config import ProcessingConfig
from app.media.ffmpeg import probe_media
from app.media.ffmpeg_runner import FFmpegRunner


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="FFmpeg/FFprobe not installed")
def test_short_real_ffmpeg_transcode(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    subprocess.run([
        shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "testsrc2=size=320x240:rate=24:duration=1", "-f", "lavfi", "-i",
        "sine=frequency=1000:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(source),
    ], check=True, capture_output=True)
    config = ProcessingConfig(source, output, target_fps=30, output_width=640,
                              output_height=480, video_codec="libx264")
    progress: list[float] = []
    FFmpegRunner(shutil.which("ffmpeg")).run(config, 1.0,
        lambda percentage, _stage: progress.append(percentage))
    metadata = probe_media(output)
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    audio = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
    assert output.exists() and output.stat().st_size > 0
    assert video["width"] == 640 and video["height"] == 480
    assert video["r_frame_rate"] == "30/1"
    assert audio["sample_rate"] == "48000"
    assert progress[-1] == 100.0
