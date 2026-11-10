# Test plan and release gates

## Automated tests

- Processing configuration validation: FPS, dimensions including 8K, audio sample rate, container/codec, overwrite and source/output safety.
- Job progress and SQLite metadata round-trip.
- NVIDIA diagnostics fallback and system telemetry contract.
- RIFE IFNet CPU forward shape and finite-output checks.
- Real-ESRGAN x2 and x4 RRDBNet CPU forward shape checks.
- Exact CFR timestamp scheduling and no duplicate source-pair endpoint.
- FFmpeg integration: synthetic CFR input at 29.44 FPS with audio, fake deterministic frame processor, exact 60 FPS output, 48 kHz audio and FFprobe verification.
- Pipeline must fail closed when CUDA is unavailable or requested AI engines are not usable.

## Local NVIDIA hardware tests (release-blocking)

On the RTX 4070 SUPER 12 GB and RTX 4050 Laptop 6 GB:

1. Verify `torch.cuda.is_available()` and selected GPU identity.
2. Download RIFE v4.25 model; verify strict checkpoint load with no missing/unexpected state keys.
3. Run RIFE on two real 1440p frames at timesteps 0.25, 0.5 and 0.75; verify finite RGB output and measure inference latency/VRAM.
4. Download Real-ESRGAN x2plus and x4plus weights; verify strict checkpoint load.
5. Run tiled x2 and x4 upscaling on representative frames; verify output dimensions, absence of tile seams, VRAM behavior and OOM backoff.
6. Convert a short 1440p CFR clip from 29.44 to 60 FPS, upscale to 4K, resample audio to 48 kHz and encode HEVC NVENC.
7. Verify exact output FPS/frame count, duration, audio sample rate, playback and A/V synchronization.
8. Repeat on the RTX 4050 with 6 GB and record tile size, RIFE scale, throughput and peak VRAM.
9. Test cancellation, disk-full behavior, model download interruption and partial-output cleanup.

## Known untested behavior

The generic development environment has no physical NVIDIA GPU and cannot validate CUDA checkpoint loading, FP16 kernels, NVENC driver interop or performance. CPU network-shape tests and fake-pipeline FFmpeg tests are not substitutes for these tests.

## Installer release gate

On a clean Windows x64 PC with no Python, Git, FFmpeg or CUDA Toolkit: install from one app installer, verify driver diagnostics, model checksum policy, application launch and offline processing after model download. The NVIDIA display driver remains external. Do not call the installer release-ready until this passes.
