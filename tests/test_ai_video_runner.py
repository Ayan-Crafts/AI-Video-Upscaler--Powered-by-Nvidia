from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.core.config import ProcessingConfig
from app.media.ai_video_runner import AIVideoRunner
from app.media.ffmpeg import probe_media


class _FakeFramePipeline:
    """Deterministic CPU test double; the hardware models are tested separately."""

    def __init__(self, *, output_size: tuple[int, int], **_kwargs) -> None:
        self.width, self.height = output_size

    def prepare_source_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame.copy()

    def process_pair(
        self, frame0: np.ndarray, frame1: np.ndarray, timestep: float
    ) -> np.ndarray:
        blended = (
            frame0.astype(np.float32) * (1.0 - timestep)
            + frame1.astype(np.float32) * timestep
        )
        return np.clip(blended, 0, 255).round().astype(np.uint8)

    def close(self) -> None:
        pass


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and FFprobe are required for the integration test",
)
def test_ai_video_runner_cfr_29_44_to_60_with_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.media.ai_video_runner as runner_module

    monkeypatch.setattr(runner_module, "AIFramePipeline", _FakeFramePipeline)

    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=736/25:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=44100:duration=2.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    config = ProcessingConfig(
        input_path=source,
        output_path=output,
        target_fps=60,
        output_width=320,
        output_height=180,
        audio_sample_rate=48_000,
        container="mp4",
        video_codec="libx264",
        interpolate=True,
        upscale=False,
        resample_audio=True,
    )
    progress: list[tuple[float, str]] = []
    result = AIVideoRunner().run(config, lambda pct, stage: progress.append((pct, stage)))

    assert result == output
    assert output.is_file() and output.stat().st_size > 0
    source_payload = probe_media(source)
    source_video = next(stream for stream in source_payload["streams"] if stream["codec_type"] == "video")
    payload = probe_media(output)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in payload["streams"] if stream["codec_type"] == "audio")
    assert (int(video["width"]), int(video["height"])) == (320, 180)
    assert video["avg_frame_rate"] == "60/1"
    expected_frames = round(float(source_video["duration"]) * 60)
    assert int(video["nb_frames"]) == expected_frames
    assert audio["sample_rate"] == "48000"
    assert float(payload["format"]["duration"]) > float(video["duration"])
    assert progress[-1][0] == 100.0


class _PartialWriter:
    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.data = bytearray()

    def write(self, value: memoryview) -> int:
        count = min(self.chunk_size, len(value))
        self.data.extend(value[:count])
        return count


def test_write_all_handles_partial_writes() -> None:
    from app.media.ai_video_runner import _write_all

    writer = _PartialWriter(chunk_size=3)
    payload = bytes(range(32))
    _write_all(writer, payload)
    assert bytes(writer.data) == payload


def test_read_exact_handles_short_reads() -> None:
    from io import BytesIO

    from app.media.ai_video_runner import _read_exact

    class ShortReader(BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(min(size, 3))

    assert _read_exact(ShortReader(b"0123456789"), 10) == b"0123456789"
