"""Packaging, path, FFmpeg-discovery and runtime-check tests (no GPU required)."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

import pytest

from app import paths
from app.core import subprocess_utils
from app.engines.model_manager import ModelDownloadError, ModelManager
from app.hardware import runtime_check
from app.media import ffmpeg as ffmpeg_module
from app.version import APP_CODENAME, APP_NAME, APP_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging" / "scripts"))


# --------------------------------------------------------------------- paths
def test_linux_paths_follow_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths, "is_windows", lambda: False)
    monkeypatch.delenv(paths.ENV_MODEL_DIR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    assert paths.model_dir() == tmp_path / "cache" / "ai-upscaler-studio" / "models"
    assert paths.log_dir() == tmp_path / "state" / "ai-upscaler-studio" / "logs"
    assert paths.user_config_dir() == tmp_path / "config" / "ai-upscaler-studio"
    assert paths.executable_name("ffmpeg") == "ffmpeg"


def test_windows_paths_use_localappdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths, "is_windows", lambda: True)
    monkeypatch.delenv(paths.ENV_MODEL_DIR, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local AppData ü"))
    assert paths.model_dir() == tmp_path / "Local AppData ü" / "AI Upscaler Studio" / "models"
    assert paths.log_dir() == tmp_path / "Local AppData ü" / "AI Upscaler Studio" / "logs"
    assert paths.legacy_model_dirs() == [tmp_path / "Local AppData ü" / "NVIDIA AI Video Studio" / "models"]
    assert paths.executable_name("ffprobe") == "ffprobe.exe"


def test_model_dir_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.ENV_MODEL_DIR, str(tmp_path / "custom models"))
    assert paths.model_dir() == tmp_path / "custom models"


def test_frozen_resource_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert paths.resource_root() == tmp_path
    assert paths.bundled_ffmpeg_dir() == tmp_path / "ffmpeg"


def test_source_resource_root_is_repository() -> None:
    assert (paths.resource_root() / "app" / "paths.py").is_file()
    assert paths.icon_path().is_file()


def test_no_window_flag_only_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_utils.os, "name", "posix")
    assert subprocess_utils.no_window_kwargs() == {}
    monkeypatch.setattr(subprocess_utils.os, "name", "nt")
    assert subprocess_utils.no_window_kwargs()["creationflags"] == 0x08000000


# ------------------------------------------------------------ FFmpeg discovery
def _fake_tool(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / paths.executable_name(name)
    tool.write_text("stub")
    return tool


def test_bundled_ffmpeg_has_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundled = _fake_tool(tmp_path / "bundle", "ffmpeg")
    configured = _fake_tool(tmp_path / "configured", "ffmpeg")
    monkeypatch.setattr(ffmpeg_module, "bundled_ffmpeg_dir", lambda: tmp_path / "bundle")
    monkeypatch.setenv(paths.ENV_FFMPEG_DIR, str(configured.parent))
    assert ffmpeg_module.find_tool("ffmpeg") == (str(bundled), ffmpeg_module.SOURCE_BUNDLED)


def test_configured_ffmpeg_before_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configured = _fake_tool(tmp_path / "configured dir", "ffprobe")
    monkeypatch.setattr(ffmpeg_module, "bundled_ffmpeg_dir", lambda: tmp_path / "missing")
    monkeypatch.setenv(paths.ENV_FFMPEG_DIR, str(configured.parent))
    assert ffmpeg_module.find_tool("ffprobe") == (str(configured), ffmpeg_module.SOURCE_CONFIGURED)


def test_path_fallback_and_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ffmpeg_module, "bundled_ffmpeg_dir", lambda: tmp_path / "missing")
    monkeypatch.delenv(paths.ENV_FFMPEG_DIR, raising=False)
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert ffmpeg_module.find_tool("ffmpeg") == ("/usr/bin/ffmpeg", ffmpeg_module.SOURCE_PATH)
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: None)
    assert ffmpeg_module.find_tool("ffmpeg") == (None, None)
    status = ffmpeg_module.discover_ffmpeg(refresh=True)
    assert not status.available and "could not be found" in (status.error or "")
    ffmpeg_module.discover_ffmpeg(refresh=True)


def test_encoder_listing_parser() -> None:
    listing = """Encoders:
 V..... = Video
 ------
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V....D libx264              libx264 H.264 / AVC (codec h264)
 A....D aac                  AAC (Advanced Audio Coding)
"""
    assert ffmpeg_module._parse_encoders(listing) == {"h264_nvenc", "libx264", "aac"}


def test_verify_ffmpeg_licence_classification_and_parser() -> None:
    import verify_ffmpeg

    assert verify_ffmpeg.classify("--enable-gpl --enable-version3") == "GPL-3.0-or-later"
    assert verify_ffmpeg.classify("--enable-gpl") == "GPL-2.0-or-later"
    assert verify_ffmpeg.classify("--enable-version3") == "LGPL-3.0-or-later"
    assert verify_ffmpeg.classify("").startswith("LGPL-2.1")
    assert "NONFREE" in verify_ffmpeg.classify("--enable-gpl --enable-nonfree")
    filters = " .. scale             V->V       Scale the input video size\n TSC fps  V->V  x\n"
    assert verify_ffmpeg.names(filters) == {"scale", "fps"}


# -------------------------------------------------------------- runtime check
def test_architecture_support_rules() -> None:
    cu126 = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    assert runtime_check.architecture_supported((8, 9), cu126)  # RTX 40 (Ada) via sm_86
    assert runtime_check.architecture_supported((8, 6), cu126)  # RTX 30 (Ampere)
    assert runtime_check.architecture_supported((7, 5), cu126)  # RTX 20 / GTX 16 (Turing)
    assert not runtime_check.architecture_supported((12, 0), cu126)  # RTX 50 (Blackwell)
    assert not runtime_check.architecture_supported((3, 7), cu126)  # Kepler
    assert runtime_check.architecture_supported((12, 0), [*cu126, "compute_90"])  # PTX JIT
    assert not runtime_check.architecture_supported((8, 9), [])


def test_driver_requirements_and_versions() -> None:
    assert runtime_check.parse_version("560.94") == (560, 94)
    assert runtime_check.parse_version("525.60.13") == (525, 60, 13)
    assert runtime_check.parse_version(None) is None
    assert runtime_check.driver_requirement("12.6", "win32") == ("527.41", "560.76")
    assert runtime_check.driver_requirement("12.6", "linux") == ("525.60.13", "560.28.03")
    assert runtime_check.driver_requirement(None) is None
    assert runtime_check.default_tuning(6.0) == (0.5, 128)
    assert runtime_check.default_tuning(12.0) == (1.0, 256)


def test_runtime_check_reports_missing_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_check, "detect_gpus", list)
    report = runtime_check.run_runtime_checks(probe_nvenc=False)
    gpu_item = next(item for item in report.items if item.name == "NVIDIA GPU")
    assert gpu_item.status == runtime_check.ERROR
    assert "nvidia.com" in gpu_item.detail
    assert report.level == runtime_check.ERROR
    assert json.dumps(report.to_dict())


def test_runtime_check_flags_old_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.hardware.gpu import GPUInfo

    pytest.importorskip("torch")
    import torch

    if not torch.version.cuda:
        pytest.skip("CUDA build of PyTorch required for the driver comparison")
    old = GPUInfo(available=True, name="Test GPU", driver_version="470.10", vram_total_mb=8192)
    monkeypatch.setattr(runtime_check, "detect_gpus", lambda: [old])
    report = runtime_check.run_runtime_checks(probe_nvenc=False)
    item = next(item for item in report.items if item.name == "Driver compatibility")
    assert item.status == runtime_check.ERROR and "too old" in item.detail


# -------------------------------------------------------------- model manager
class _Response:
    def __init__(self, content: bytes, length: int | None = None) -> None:
        self._stream = io.BytesIO(content)
        self.headers = {"Content-Length": str(len(content) if length is None else length)}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self._stream.close()

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def test_interrupted_download_leaves_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _Response(b"partial", length=100))
    manager = ModelManager(tmp_path, retries=2, retry_delay=0)
    with pytest.raises(ModelDownloadError, match="interrupted"):
        manager.ensure_file(relative_path="m/model.pth", url="https://x.invalid/m", min_bytes=1)
    assert not (tmp_path / "m" / "model.pth").exists()
    assert not list((tmp_path / "m").glob("*.part"))


def test_download_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    attempts = {"count": 0}

    def flaky(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.URLError("temporary failure")
        return _Response(b"good-model-bytes")

    import urllib.error

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    path = ModelManager(tmp_path, retries=3, retry_delay=0).ensure_file(
        relative_path="model.pth", url="https://x.invalid/m", min_bytes=4, expected_bytes=16)
    assert path.read_bytes() == b"good-model-bytes" and attempts["count"] == 2


def test_exact_size_mismatch_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _Response(b"12345678"))
    with pytest.raises(ModelDownloadError, match="expected exactly"):
        ModelManager(tmp_path, retries=1).ensure_file(
            relative_path="model.pth", url="https://x.invalid/m", min_bytes=1, expected_bytes=9)


def test_cached_model_reused_without_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def no_network(*_args, **_kwargs):
        raise AssertionError("network must not be used for a valid cached model")

    import hashlib

    payload = b"cached-model"
    (tmp_path / "model.pth").write_bytes(payload)
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    manager = ModelManager(tmp_path)
    for _ in range(2):
        manager.ensure_file(relative_path="model.pth", url="https://x.invalid/m", min_bytes=1,
                            expected_sha256=hashlib.sha256(payload).hexdigest())


def test_legacy_cache_is_reused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    legacy = tmp_path / "NVIDIA AI Video Studio" / "models"
    (legacy / "rife" / "v4.25").mkdir(parents=True)
    (legacy / "rife" / "v4.25" / "flownet.pkl").write_bytes(b"legacy-weights")

    def no_network(*_args, **_kwargs):
        raise AssertionError("legacy checkpoint should have been reused")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    manager = ModelManager(tmp_path / "new", legacy_roots=[legacy])
    path = manager.ensure_file(relative_path="rife/v4.25/flownet.pkl", url="https://x.invalid", min_bytes=4)
    assert path.read_bytes() == b"legacy-weights"
    assert (legacy / "rife" / "v4.25" / "flownet.pkl").exists()  # never moved or deleted


def test_stale_partial_downloads_are_removed(tmp_path: Path) -> None:
    import os
    import time

    stale = tmp_path / "model.pth.abc123.part"
    fresh = tmp_path / "model.pth.def456.part"
    stale.write_bytes(b"x")
    fresh.write_bytes(b"y")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    ModelManager._remove_stale_parts(tmp_path / "model.pth")
    assert not stale.exists() and fresh.exists()


def test_production_models_pin_size_and_checksum() -> None:
    from app.engines.realesrgan import REALESRGAN_MODELS

    for info in REALESRGAN_MODELS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", str(info["sha256"]))
        assert int(info["min_bytes"]) > 60_000_000


# ------------------------------------------------------- packaging consistency
def test_versions_are_consistent() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == APP_VERSION == "1.0.0"
    assert APP_NAME == "AI Upscaler Studio" and APP_CODENAME == "Imaginoromancophilia"
    versions = dict(line.split("=", 1) for line in
                    (ROOT / "packaging/common/versions.env").read_text().splitlines()
                    if "=" in line and not line.startswith("#"))
    assert versions["APP_VERSION"] == APP_VERSION
    iss = (ROOT / "packaging/windows/installer.iss").read_text(encoding="utf-8")
    assert f'#define AppVersion "{APP_VERSION}"' in iss
    assert "PrivilegesRequired=lowest" in iss
    assert "OutputBaseFilename=AI-Upscaler-Studio-v{#AppVersion}-Windows-x64-Setup" in iss
    assert iss.isascii(), "Inno Setup reads BOM-less scripts as ANSI"


def test_runtime_requirements_pin_cuda_pytorch() -> None:
    lines = (ROOT / "packaging/requirements/runtime.txt").read_text().splitlines()
    pins = [line for line in lines if line and not line.startswith(("#", "--"))]
    assert "torch==2.14.1+cu126" in pins
    assert all("==" in line for line in pins), "every runtime dependency must be pinned"
    assert any("download.pytorch.org/whl/cu126" in line for line in lines)
    for dev_only in ("pytest", "ruff", "pyinstaller"):
        assert not any(line.lower().startswith(dev_only) for line in pins)


def test_external_binaries_are_pinned() -> None:
    entries = json.loads((ROOT / "packaging/common/external_binaries.json").read_text())
    for key, entry in entries.items():
        if key.startswith("_"):
            continue
        assert entry["url"].startswith("https://"), key
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), key


def test_linux_packaging_files() -> None:
    desktop = (ROOT / "packaging/linux/desktop/ai-upscaler-studio.desktop").read_text()
    assert "Name=AI Upscaler Studio" in desktop and "Exec=ai-upscaler-studio" in desktop
    assert "Icon=ai-upscaler-studio" in desktop
    control = (ROOT / "packaging/linux/debian/control.in").read_text()
    assert "Package: ai-upscaler-studio" in control and "Architecture: amd64" in control
    for script in ("build.sh", "docker-build.sh", "test-packages.sh", "appimage/AppRun",
                   "debian/postinst", "debian/postrm"):
        data = (ROOT / "packaging/linux" / script).read_bytes()
        assert b"\r\n" not in data, f"{script} must use LF line endings"
        assert data.startswith(b"#!"), script


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    from app.main import main

    assert main(["--version"]) == 0
    assert APP_VERSION in capsys.readouterr().out


def test_spec_never_enables_upx() -> None:
    spec = (ROOT / "packaging/pyinstaller/ai_upscaler_studio.spec").read_text(encoding="utf-8")
    assert "upx=True" not in spec and "console=False" in spec


def test_checksum_generator(tmp_path: Path) -> None:
    (tmp_path / "AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb").write_bytes(b"deb")
    (tmp_path / "notes.txt").write_text("ignored")
    subprocess.run([sys.executable, str(ROOT / "packaging/scripts/generate_checksums.py"), str(tmp_path)],
                   check=True, capture_output=True)
    sums = (tmp_path / "SHA256SUMS.txt").read_text()
    assert "AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb" in sums and "notes.txt" not in sums


def test_cuda_cache_release_never_masks_errors() -> None:
    from app.engines.cuda_utils import release_cuda_cache

    class _Device:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    class _Cuda:
        calls = 0

        def device(self, _device):
            return _Device()

        def empty_cache(self) -> None:
            _Cuda.calls += 1
            raise RuntimeError("CUDA error: out of memory")

    class _Torch:
        cuda = _Cuda()

    release_cuda_cache(_Torch(), "cuda:0")  # must not raise
    assert _Cuda.calls == 1
