# Architecture

- **UI:** PySide6 only; long-running work must run outside the GUI thread.
- **Core:** validated immutable processing configuration, job state and SQLite history.
- **Media:** FFprobe inspection, FFmpeg capability checks, cancellable subprocess runner.
- **Hardware:** NVML first, `nvidia-smi` fallback, optional PyTorch CUDA capability detection, psutil for CPU/RAM.
- **Engines:** separate interpolation, super-resolution, audio and encoding adapters.

## GPU portability

Never select a runtime from a marketing GPU name alone. Inspect driver version, CUDA availability, compute capability, VRAM, model-kernel requirements and NVENC capabilities. CUDA is the primary inference backend. TensorRT is optional; PyTorch CUDA is the fallback. Legacy CUDA GPUs are supported only when the selected PyTorch build and model kernels still support them. NVENC is detected independently.

## Installer

V1.0.0 ships PyInstaller one-directory bundles wrapped as a Windows Inno Setup installer, a portable ZIP, an Ubuntu AppImage and a `.deb` (see [BUILD_AND_DISTRIBUTION.md](BUILD_AND_DISTRIBUTION.md)). They bundle the app runtime, PySide6, the CUDA 12.6 PyTorch runtime, FFmpeg/FFprobe and redistributable dependencies. The NVIDIA display driver remains an external prerequisite: `app/hardware/runtime_check.py` detects a missing or incompatible driver, unsupported GPU architectures, VRAM and working NVENC encoders, and guides users to NVIDIA's official driver page. Nothing installs or replaces display drivers. `app/paths.py` keeps all writable data (models, logs, config) in per-user directories, never in the read-only install/AppImage location.

## Memory and timestamps

The production pipeline must use bounded frame queues and tiled super-resolution; it must not decode an entire long video into RAM/VRAM. Recover from CUDA out-of-memory by reducing tile/batch size, otherwise fail cleanly. Output frames must be scheduled on a target presentation timestamp grid, preserving source duration and audio synchronization. Do not relabel FPS metadata and call that interpolation.

## AI correctness gate

The current AI engine adapters report unavailable. AI jobs fail closed until real RIFE and Real-ESRGAN implementations, models, provenance and hardware tests exist. FFmpeg `fps`, `minterpolate` and `scale` filters are conventional filters, not AI inference.
