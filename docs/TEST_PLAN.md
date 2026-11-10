# Test plan and release gates

## Automated tests

- FPS, dimensions (including 8K), sample-rate and container/codec validation.
- Source/output collision and overwrite protection.
- Job progress invariants and SQLite metadata round-trip.
- FFmpeg progress parsing and command construction.
- GPU fallback/default behavior and system telemetry contract.
- AI pipeline fails closed while AI adapters are unavailable.
- Optional integration test generates a short synthetic clip, runs FFmpeg, then verifies resolution, FPS, audio sample rate and progress with FFprobe.

## Windows CI

Run Ruff, compileall and unit tests on Windows with Python 3.11 and 3.12. CI does not certify real CUDA/NVENC hardware behavior.

## NVIDIA hardware matrix

Physically test representative Turing, Ampere, Ada and Blackwell GPUs, and selected legacy GTX cards if legacy support is claimed. Record GPU, driver, CUDA/PyTorch versions and VRAM. Verify actual RIFE CUDA inference, Real-ESRGAN CUDA inference, NVENC initialization, 29.44→60 FPS duration/audio sync, 1080p→4K and 4K→8K dimensions, VRAM recovery, cancellation and clean partial-output handling.

## Installer release gate

On a clean Windows x64 PC with no Python, Git, FFmpeg or CUDA Toolkit: install from one app installer, verify NVIDIA driver detection, model checksum verification, launch and offline processing after model download. Do not claim release-ready until this passes. The driver itself is not bundled.
