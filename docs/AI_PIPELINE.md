# CUDA AI frame-processing integration

## Components

- **RIFE:** Practical-RIFE HDv3 / v4.25 IFNet implementation, pinned checkpoint revision, PyTorch CUDA inference and optional FP16. It accepts arbitrary temporal positions between a pair of decoded frames.
- **Real-ESRGAN:** official x2plus and x4plus RRDBNet checkpoint architectures. Tile inference and uint8 output-tile assembly run on the selected CUDA GPU. When paired with RIFE, the interpolated tensor stays on CUDA until tiled super-resolution and final canvas fitting complete; only the completed frame is copied to host memory. Tile size is reduced on CUDA out-of-memory.
- **Video coordinator:** FFmpeg streams decoded RGB frames to the Python coordinator. RIFE creates intermediate frames where the output timestamp falls between source frames. Real-ESRGAN upscales each output frame. FFmpeg encodes the generated frame stream and maps the source audio.

## Model selection

- For requested scale ratios up to 2×, use `RealESRGAN_x2plus` to avoid allocating a full 4× intermediate (especially relevant to 4K→8K).
- Above 2×, use `RealESRGAN_x4plus` and resize the model result to the requested canvas if necessary.
- If the target resolution does not exceed the source, the Real-ESRGAN stage is skipped and CUDA bicubic resizing is used only if a dimension still differs.
- RIFE uses a conservative default inference scale of 0.5 for GPUs with up to 8 GB VRAM and 1.0 above 8 GB; CUDA OOM retries at a smaller scale down to 0.25.
- Real-ESRGAN starts with 128-pixel tiles up to 8 GB and 256-pixel tiles above 8 GB. OOM triggers tile-size halving down to 32 pixels.

## GPU engine smoke test

Run `python scripts/benchmark_cuda_engines.py --device 0 --width 640 --height 360` after installing the CUDA-enabled PyTorch wheel. Add `--include-x4` to test both Real-ESRGAN model scales. This downloads checkpoints, performs real inference on the selected CUDA device, and reports latency/throughput and peak allocated VRAM. Run the same command on each supported machine to build the compatibility/performance matrix.

## Model cache and security

The first run downloads model files into the per-user model cache: `%LOCALAPPDATA%\AI Upscaler Studio\models` on Windows and `~/.cache/ai-upscaler-studio/models` on Linux (override with `AI_UPSCALER_STUDIO_MODEL_DIR`; checkpoints from the pre-1.0 `NVIDIA AI Video Studio` cache are reused). Files are written to `.part` then atomically renamed; interrupted downloads are discarded and retried, and stale partial files are cleaned up. The pinned Practical-RIFE v4.25 checkpoint is fetched from immutable revision `4bbe4c19fb5363333cca83b6e82bb65ee02d741b` and validated against SHA-256 `6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2` (the hash Hugging Face publishes for that LFS object) and its exact size. The Real-ESRGAN x2plus and x4plus release assets do not have upstream-published checksums; V1.0.0 enforces their exact sizes and maintainer-pinned SHA-256 digests taken from the official release assets (`49fafd45…6abb` and `4fa0d389…82f1`). Checkpoints are loaded with `torch.load(weights_only=True)` and strict state-dictionary matching.

## Install into an existing CUDA environment

Install the appropriate CUDA-enabled PyTorch wheel first, using the official PyTorch selector. For the currently used Python 3.11 / PyTorch 2.14.1 / CUDA 12.6 setup:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

Install remaining optional dependencies without reinstalling torch:

```powershell
python -m pip install -e ".[ai,dev,nvidia]"
```

## Video-processing semantics and limitations

- Output frames are scheduled on a rational target-FPS grid. This is actual RIFE interpolation when RIFE is enabled and required; it is not metadata relabeling.
- The first runner supports CFR input. It rejects obvious VFR files using an average-vs-nominal FPS heuristic. That heuristic cannot prove every frame timestamp is CFR; full VFR timestamp handling is future work.
- All source audio streams mapped by FFmpeg are preserved as encoded audio (AAC for MP4/MKV/MOV and Opus for WebM) and resampled to the requested rate when `resample_audio=True`. Subtitles, chapters and arbitrary data streams are not currently preserved.
- FFmpeg decodes to bounded raw RGB frames. CUDA runs the AI models. RIFE and Real-ESRGAN share CUDA-resident intermediate tensors; only decoded input frames and final upscaled tiles cross the CPU/GPU boundary. NVENC is used when the selected output codec is an available NVENC encoder. Dedicated NVDEC/PyNvVideoCodec zero-copy decode integration remains future work.
- The first release targets 8-bit SDR `yuv420p` output. HDR/10-bit preservation and 4:2:2/4:4:4 output are not implemented yet.
- The x2/x4 networks keep a compact uint8 output canvas in VRAM while processing bounded float16/float32 tiles. A final completed frame is copied to host memory for FFmpeg encoding. 8K output is supported by dimensions, but throughput and memory depend on source resolution, model scale, tile size and GPU VRAM.
- TensorRT is not required and is not yet wired into this integration. PyTorch CUDA is the current inference backend.
