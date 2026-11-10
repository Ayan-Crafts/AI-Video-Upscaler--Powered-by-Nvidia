from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pytest

from app.engines.model_manager import ModelDownloadError, ModelManager


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self._stream = io.BytesIO(content)
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self._stream.close()

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def test_model_manager_downloads_atomically_and_records_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"model-weights-content"
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_args, **_kwargs: _FakeResponse(payload)
    )
    progress: list[float] = []
    destination = ModelManager(tmp_path).ensure_file(
        relative_path="models/test.pth",
        url="https://models.invalid/test.pth",
        min_bytes=8,
        expected_sha256=expected,
        progress=lambda percent, _stage: progress.append(percent),
    )

    assert destination.read_bytes() == payload
    metadata = json.loads(destination.with_suffix(".pth.json").read_text())
    assert metadata["sha256"] == expected
    assert metadata["checksum_status"] == "verified"
    assert progress[-1] == 100.0
    assert not list(destination.parent.glob("*.part"))


def test_model_manager_rejects_cached_checksum_mismatch(tmp_path: Path) -> None:
    destination = tmp_path / "models" / "test.pth"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"incorrect model payload")
    manager = ModelManager(tmp_path)

    with pytest.raises(ModelDownloadError, match="checksum mismatch"):
        manager.ensure_file(
            relative_path="models/test.pth",
            url="https://models.invalid/test.pth",
            min_bytes=1,
            expected_sha256="0" * 64,
        )

    assert not destination.exists()
