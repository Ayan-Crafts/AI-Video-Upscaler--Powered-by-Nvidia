"""Headless self-test for packaged builds (``ai-upscaler-studio --self-test``).

Checks, in order: bundled runtime and NVIDIA hardware (see
:mod:`app.hardware.runtime_check`), the Qt GUI stack (platform plugins and the
real main window), bundled FFmpeg/FFprobe decoding, and optionally (``--e2e``)
model download, strict checkpoint loading, CUDA inference and a short
end-to-end video job with audio. Work files use a directory containing spaces
and non-ASCII characters to exercise path handling.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from app.core.subprocess_utils import run_tool
from app.paths import is_frozen, resource_root
from app.version import APP_VERSION

logger = logging.getLogger(__name__)


def _check_qt(result: dict[str, Any]) -> bool:
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6 import __version__ as pyside_version
        from PySide6.QtCore import QLibraryInfo
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([sys.argv[0]])
        plugins = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
        platform_plugins = sorted(p.name for p in (plugins / "platforms").glob("*")) if plugins.is_dir() else []

        from app.ui.main_window import MainWindow

        window = MainWindow(run_startup_checks=False)
        window.show()
        app.processEvents()
        title = window.windowTitle()
        window.timer.stop()
        window.close()
        app.processEvents()
        result["qt"] = {
            "ok": True,
            "pyside6": pyside_version,
            "platform": app.platformName(),
            "plugins_path": str(plugins),
            "platform_plugins": platform_plugins,
            "window_title": title,
        }
        return True
    except Exception as exc:  # noqa: BLE001 - report any GUI failure
        result["qt"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()}
        return False


def _make_sample(ffmpeg: str, path: Path, size: str = "320x180", rate: int = 24,
                 seconds: float = 2.0) -> None:
    result = run_tool([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}:sample_rate=44100",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg could not create the sample video.")


def _check_ffmpeg(result: dict[str, Any], work: Path) -> Path | None:
    from app.media.ffmpeg import discover_ffmpeg, probe_media

    tools = discover_ffmpeg(refresh=True)
    entry: dict[str, Any] = {
        "ffmpeg": tools.ffmpeg, "ffprobe": tools.ffprobe, "source": tools.source,
        "version": tools.ffmpeg_version, "nvenc_compiled": list(tools.nvenc_encoders),
        "software_encoders": list(tools.software_encoders),
    }
    result["ffmpeg"] = entry
    if not tools.available:
        entry.update(ok=False, error=tools.error)
        return None
    if is_frozen() and tools.source != "bundled":
        entry.update(ok=False, error=f"Packaged build is using {tools.source} FFmpeg instead of the bundled copy.")
        return None
    try:
        sample = work / "sample source ✓.mp4"
        _make_sample(tools.ffmpeg, sample)
        metadata = probe_media(sample, tools.ffprobe)
        streams = {s.get("codec_type") for s in metadata.get("streams", [])}
        entry.update(ok={"video", "audio"} <= streams, sample=str(sample), streams=sorted(streams))
        return sample if entry["ok"] else None
    except Exception as exc:  # noqa: BLE001
        entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        return None


def _check_models_and_inference(result: dict[str, Any], device_index: int, fp16: bool) -> bool:
    import numpy as np

    entry: dict[str, Any] = {}
    result["inference"] = entry
    ok = True
    rng = np.random.default_rng(1234)
    frame0 = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
    frame1 = np.roll(frame0, 4, axis=1)
    try:
        from app.engines.rife import RIFEEngine

        start = time.perf_counter()
        rife = RIFEEngine(device_index=device_index, fp16=fp16)
        middle = rife.interpolate(frame0, frame1, 0.5)
        entry["rife"] = {"ok": middle.shape == frame0.shape, "fp16": rife.use_fp16,
                         "model": str(rife.model_path), "seconds": round(time.perf_counter() - start, 2)}
        rife.close()
    except Exception as exc:  # noqa: BLE001
        entry["rife"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    ok &= bool(entry["rife"].get("ok"))
    from app.engines.realesrgan import RealESRGANEngine

    for scale in (2, 4):
        key = f"realesrgan_x{scale}plus"
        try:
            start = time.perf_counter()
            engine = RealESRGANEngine(device_index=device_index, fp16=fp16, model_scale=scale)
            output = engine.upscale(frame0[:64, :96])
            entry[key] = {"ok": output.shape == (64 * scale, 96 * scale, 3), "fp16": engine.use_fp16,
                          "tile": engine.tile_size, "model": str(engine.model_path),
                          "seconds": round(time.perf_counter() - start, 2)}
            engine.close()
        except Exception as exc:  # noqa: BLE001
            entry[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        ok &= bool(entry[key].get("ok"))
    entry["ok"] = ok
    return ok


def _check_end_to_end(result: dict[str, Any], sample: Path, work: Path, device_index: int,
                      fp16: bool, nvenc_usable: list[str]) -> bool:
    from app.core.config import ProcessingConfig
    from app.media.ai_video_runner import AIVideoRunner
    from app.media.ffmpeg import probe_media

    codec = "h264_nvenc" if "h264_nvenc" in nvenc_usable else "libx264"
    output = work / "enhanced output ✓.mp4"
    entry: dict[str, Any] = {"codec": codec, "output": str(output)}
    result["end_to_end"] = entry
    try:
        config = ProcessingConfig(
            input_path=sample, output_path=output, target_fps=48.0, output_width=1280,
            output_height=720, audio_sample_rate=48_000, container="mp4", video_codec=codec,
            interpolate=True, upscale=True, overwrite=True, gpu_index=device_index, fp16=fp16,
        )
        stages: list[str] = []
        start = time.perf_counter()
        AIVideoRunner().run(config, lambda pct, stage: stages.append(stage),
                            device_index=device_index, fp16=fp16)
        metadata = probe_media(output)
        video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
        audio = [s for s in metadata["streams"] if s["codec_type"] == "audio"]
        entry.update(
            ok=bool(audio) and video["width"] == 1280 and video["height"] == 720,
            seconds=round(time.perf_counter() - start, 1),
            video={k: video.get(k) for k in ("codec_name", "width", "height", "avg_frame_rate", "nb_frames")},
            audio=[{k: s.get(k) for k in ("codec_name", "sample_rate", "channels")} for s in audio],
            final_stage=stages[-1] if stages else None,
        )
    except Exception as exc:  # noqa: BLE001
        entry.update(ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    return bool(entry.get("ok"))


def run_self_test(*, e2e: bool = False, require_gpu: bool = False,
                  report_path: Path | None = None) -> int:
    from app.hardware.runtime_check import run_runtime_checks

    result: dict[str, Any] = {
        "app_version": APP_VERSION, "frozen": is_frozen(), "resource_root": str(resource_root()),
        "executable": sys.executable,
    }
    work = Path(tempfile.mkdtemp(prefix="ai upscaler self-test ü "))
    failures: list[str] = []
    try:
        runtime = run_runtime_checks(probe_nvenc=True)
        result["runtime"] = runtime.to_dict()
        if not runtime.torch_version:
            failures.append("PyTorch did not load")
        elif not runtime.torch_cuda_runtime:
            failures.append("PyTorch is a CPU-only build")
        if require_gpu and not runtime.gpu_ready:
            failures.append("CUDA GPU inference unavailable")

        if not _check_qt(result):
            failures.append("Qt GUI failed")
        sample = _check_ffmpeg(result, work)
        if sample is None:
            failures.append("FFmpeg/FFprobe check failed")

        if e2e:
            if not runtime.gpu_ready:
                result["inference"] = {"ok": False, "skipped": "no usable CUDA GPU"}
                failures.append("end-to-end skipped: no CUDA GPU")
            else:
                device = runtime.devices[0]
                if not _check_models_and_inference(result, device.index, device.fp16):
                    failures.append("model download/inference failed")
                if sample is not None and not _check_end_to_end(
                    result, sample, work, device.index, device.fp16, runtime.nvenc_usable
                ):
                    failures.append("end-to-end video processing failed")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"self-test crashed: {exc}")
        result["crash"] = traceback.format_exc()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    result["failures"] = failures
    result["passed"] = not failures
    text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
    if sys.stdout is not None:
        try:
            print(text)
        except (OSError, UnicodeEncodeError):
            print(text.encode("ascii", "replace").decode("ascii"))
    logger.info("Self-test %s: %s", "PASSED" if not failures else "FAILED", failures or "all checks passed")
    return 0 if not failures else 1
