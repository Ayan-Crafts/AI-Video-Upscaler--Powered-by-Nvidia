# NVIDIA AI Video Studio

Windows-first desktop application for NVIDIA-accelerated AI frame interpolation, AI super-resolution, audio resampling and hardware video encoding.

> **Development status: v0.2.0-dev1.** This development integration adds Practical-RIFE v4.25 CUDA inference, Real-ESRGAN x2/x4 CUDA tiled inference, rational frame scheduling, a streaming FFmpeg video runner, optional NVENC encoding, and a Process Video UI integration patch. It is not yet a release-ready installer. Real checkpoint loading, full CUDA video processing, and performance must be validated on the target Windows NVIDIA machines.

## Current functionality

- Premium PySide6 Dashboard and Process Video screens.
- FFprobe media inspection and FFmpeg capability discovery.
- NVIDIA telemetry through NVML with `nvidia-smi` fallback.
- CPU/RAM monitoring.
- Conventional FFmpeg conversion path, clearly labelled as non-AI.
- RIFE v4.25 architecture and CUDA inference adapter; checkpoint downloads on first use.
- Real-ESRGAN x2plus and x4plus RRDBNet architectures and CUDA tiled inference; checkpoint downloads on first use.
- Adaptive RIFE inference scale and Real-ESRGAN tile-size reduction after CUDA OOM.
- CFR timestamp scheduler using rational arithmetic; it does not merely change FPS metadata.
- Streaming FFmpeg decode/encode runner, optional NVENC video encoding and audio resampling.
- Output verification before replacing the destination file.
- Tests for model network shapes, frame scheduling, and a real FFmpeg CFR/audio pipeline using a deterministic fake inference stage.

## Important limitations

- The model checkpoints are downloaded when first used. Their source URLs and computed SHA-256 hashes are recorded. Where upstream does not publish an expected digest, the calculated hash is an audit record, not an independent authenticity check.
- This first video runner supports constant-frame-rate (CFR) inputs. It rejects obvious VFR cases using a conservative `avg_frame_rate` / `r_frame_rate` comparison; this is not a complete frame-PTS VFR detector.
- FFmpeg performs video decode to bounded raw RGB frames. CUDA performs RIFE and Real-ESRGAN inference. NVENC is used only when the selected encoder is an available NVENC codec. Dedicated NVDEC/PyNvVideoCodec GPU-resident decoding is not integrated yet.
- The first implementation preserves and re-encodes source audio streams, but does not preserve subtitles, chapters, alternate data streams or every container metadata field.
- The Real-ESRGAN x2plus model is selected for requested scale ratios up to 2× (including 4K→8K); x4plus is selected above 2×. A compact uint8 output canvas and final canvas fitting remain on the GPU; the completed RGB frame is copied to host memory for FFmpeg encoding. Very large 8K jobs can still use substantial VRAM and take a long time.
- TensorRT acceleration, GPU-resident decode/encode interop, VFR preservation, installer bundling and multi-generation physical GPU validation remain future/release-gate work.
- The NVIDIA display driver is an external prerequisite and is never installed or replaced by this app.

## Windows developer setup

Use Python 3.11 or 3.12 and a compatible NVIDIA driver. Create/activate a virtual environment first.

Install a CUDA-enabled PyTorch build explicitly. For the currently used Python 3.11 / PyTorch 2.14.1 / CUDA 12.6 setup:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

Then install the project AI dependencies without replacing PyTorch:

```powershell
python -m pip install -e ".[ai,dev,nvidia]"
```

Install FFmpeg and FFprobe and ensure both are on PATH for this development build. Verify the NVIDIA runtime:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU')"
ffmpeg -version
ffprobe -version
```

Run the app:

```powershell
python -m app
```

## Apply the AI UI integration patch

When applying the development overlay, merge its files into the repository root and follow `APPLY_AI_PIPELINE.md`. The UI patch can be run with:

```powershell
python scripts/patch_ai_ui.py
```

This connects the Start AI Processing button to the background `AIProcessingWorker`, adds CUDA GPU selection and an FP16 toggle. The patch script is safe to re-run.

## Tests

```powershell
ruff check .
python -m pytest -q
python -m compileall -q app tests
```

The CPU-side tests validate network architecture shapes, tiled x2 edge handling, aspect-preserving canvas sizing, model-download integrity and video scheduling. The FFmpeg integration test generates a small synthetic source, uses a deterministic fake frame processor, then verifies 29.44→60 FPS timing, frame count, resolution and 48 kHz audio. It does not prove RIFE/Real-ESRGAN checkpoint compatibility or CUDA operation.

## Validate both CUDA model engines

After installing the project and CUDA-enabled PyTorch, run the dedicated hardware smoke test:

```powershell
python scripts/benchmark_cuda_engines.py --device 0 --width 640 --height 360
```

Add `--include-x4` to download and benchmark the x4plus checkpoint as well. The first run downloads model weights. The script reports actual RIFE/Real-ESRGAN inference latency, effective FPS, selected precision, adaptive tile/scale settings and peak allocated VRAM. Run it separately on the RTX 4050 laptop and RTX 4070 SUPER desktop; do not compare results while other GPU-heavy tasks are running.

## First AI run

The first AI job downloads two checkpoints to:

- `%LOCALAPPDATA%\NVIDIA AI Video Studio\models\rife\v4.25\flownet.pkl`
- `%LOCALAPPDATA%\NVIDIA AI Video Studio\models\realesrgan\RealESRGAN_x2plus.pth` or `RealESRGAN_x4plus.pth`

Subsequent runs use the cached files. Keep an internet connection for the first run. Check the application log if model download or strict checkpoint loading fails.

## Licensing

Application code is intended to be GPL-3.0-or-later. RIFE model code is MIT; Real-ESRGAN source is BSD-3-Clause; the adapted BasicSR RRDBNet architecture is Apache-2.0. Model checkpoints have their own terms and are fetched from upstream rather than redistributed in this repository. See `app/engines/vendor_licenses/` and `THIRD_PARTY_NOTICES.md`.
