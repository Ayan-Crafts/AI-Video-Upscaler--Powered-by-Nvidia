# NVIDIA AI Video Studio

Windows-first desktop application for NVIDIA-accelerated AI interpolation, AI super-resolution, audio resampling and hardware encoding.

> **Status: development baseline v0.1.0.** This branch contains a premium PySide6 UI foundation, media probing, configuration validation, NVIDIA/system diagnostics, an explicit non-AI FFmpeg export path, SQLite job metadata and tests. The production RIFE CUDA and Real-ESRGAN CUDA adapters are **not yet integrated**. AI jobs fail closed rather than silently substituting ordinary FPS conversion/scaling and claiming it is AI.

## Current functionality

- PySide6 Dashboard and Process Video screens.
- FFprobe media inspection.
- Target FPS, up-to-8K canvas, audio rate, container and codec controls.
- NVIDIA telemetry via NVML with `nvidia-smi` fallback; CPU/RAM telemetry.
- FFmpeg/FFprobe discovery and NVENC encoder enumeration.
- A clearly labelled conventional FFmpeg fallback export with progress and background worker; it is **not** AI interpolation/upscaling.
- SQLite job metadata foundation, config validation and automated tests.

## Not yet complete

- RIFE CUDA model integration and model manager.
- Real-ESRGAN CUDA tiled inference and model manager.
- True AI streaming pipeline with timestamp-aware arbitrary FPS output.
- Persistent interactive queue, resume/recovery, preview and ETA.
- Full single-installer bundle with PyTorch CUDA runtime and verified models.
- Physical NVIDIA GPU/NVENC multi-generation testing.

## Developer setup (Windows 10/11 x64)

Python 3.11 or 3.12 recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,nvidia]"
python -m app
```

FFmpeg and FFprobe must be on PATH for this development build. The eventual installer should bundle a license-compliant FFmpeg build. For optional CUDA diagnostics/inference dependencies, install PyTorch using the official selector: https://pytorch.org/get-started/locally/ . A compatible NVIDIA driver is always required and is not bundled.

## Checks

```powershell
python -m pytest -q
ruff check .
python -m compileall -q app tests
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md).

## Hardware portability

Feature selection is capability-based, not hardcoded to one RTX model. CUDA runtime, compute capability, VRAM, PyTorch/model-kernel support, driver version and NVENC support are independent checks. TensorRT is optional. Older GTX cards are supported only where their runtime and model kernels remain compatible.

## License

Application code is intended to be GPL-3.0-or-later. Third-party code, binaries and model weights keep their own licenses. No model weights are included until redistribution terms are confirmed. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Push this working copy to your GitHub repository

If this source archive was downloaded instead of cloned, extract it, open PowerShell in the project folder, and run:

```powershell
.\PUSH_TO_GITHUB.ps1
```

GitHub authentication is handled by your local Git credential manager. This script only pushes the local `main` branch; it does not store or request a GitHub token.
