# Apply the CUDA AI pipeline overlay

This overlay adds Practical-RIFE v4.25 CUDA interpolation, Real-ESRGAN x2/x4 CUDA tiled super-resolution, a CFR streaming video runner, the CUDA tensor handoff, tests and the Process Video UI patch.

## Before applying

1. Close NVIDIA AI Video Studio.
2. Commit or back up your current local changes.
3. Make sure this is the repository root (the folder containing `pyproject.toml`).
4. Extract/merge this archive into that folder and allow it to replace the files included in the overlay. It intentionally does not replace `app/ui/main_window.py`; the patch script edits that existing file in place.

## Apply and install

Keep your existing CUDA-enabled PyTorch installation. Do **not** install the generic `.[ml]` extra, because the default PyPI Torch wheel may replace a CUDA build with a CPU build.

```powershell
# from the repository root, with .venv active
python scripts/patch_ai_ui.py
python -m pip install -e ".[ai,dev,nvidia]"
```

The `ai` extra installs NumPy. The existing PyTorch 2.14.1+cu126 installation is deliberately not declared in that extra and will be left in place.

## Run static checks and tests

```powershell
ruff check .
python -m compileall -q app tests scripts
python -m pytest -q
```

The CUDA-only integration test is skipped on machines without an available CUDA device. On the RTX 4070 SUPER it should run rather than skip.

## Run real model inference benchmarks

First run downloads the checkpoints into `%LOCALAPPDATA%\NVIDIA AI Video Studio\models` (about 25 MB for RIFE, 67 MB for Real-ESRGAN x2plus; x4plus is another 67 MB if requested).

```powershell
python scripts/benchmark_cuda_engines.py --device 0 --width 640 --height 360
```

This performs real RIFE inference and Real-ESRGAN x2plus inference on the selected CUDA GPU. It also exercises the CUDA tensor handoff from RIFE into the upscaler. Add `--include-x4` to test x4plus too. The benchmark reports latency, effective FPS, precision, selected RIFE inference scale, tile-size adaptation and peak allocated VRAM.

Run this benchmark on both your RTX 4070 SUPER 12 GB desktop and RTX 4050 6 GB laptop. Close other GPU-heavy applications first.

## First end-to-end video test

Use a short **constant-frame-rate (CFR)** video, ideally 5–15 seconds, at 720p or 1080p. For an actual interpolation test, make sure the target FPS differs from the source FPS (for example 30 → 60). For an actual AI upscaling test, choose a larger output resolution (for example 1080p → 4K).

1. Launch `python -m app`.
2. Open Process Video and select the NVIDIA GPU.
3. Enable AI Interpolation and choose a target FPS different from the source.
4. Enable AI Upscaling and choose a larger resolution.
5. Select MP4 + HEVC NVENC for the RTX 40-series test, and audio 48 kHz.
6. Choose a new output filename and start AI processing.
7. Verify the output with FFprobe:

```powershell
ffprobe -v error -show_streams -show_format -of json "path\to\output.mp4"
```

Confirm video width/height, `avg_frame_rate`, video duration, audio stream count and 48000 Hz audio. Watch the complete output for visible interpolation or upscaling artifacts.

## Current limitations

- The video runner currently supports CFR input and rejects obvious VFR footage; full per-frame PTS/VFR support is not implemented.
- FFmpeg decodes to CPU RGB frames. RIFE and Real-ESRGAN run on CUDA; the RIFE→Real-ESRGAN intermediate remains on CUDA. Dedicated NVDEC/PyNvVideoCodec zero-copy decode is a later optimization.
- NVENC is used when selected and exposed by the installed FFmpeg build. Codec availability still depends on the GPU generation and driver.
- The first implementation outputs 8-bit SDR; HDR/10-bit preservation is not yet implemented.
- TensorRT, a full model manager UI, persistent queue/recovery and the final installer are later milestones.
- CUDA, model checkpoint strict-load, NVENC, and performance must be verified on the physical Windows NVIDIA machines. Passing CPU tests alone does not certify those hardware paths.
