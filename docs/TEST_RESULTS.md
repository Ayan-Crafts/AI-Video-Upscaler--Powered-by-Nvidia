# AI pipeline overlay validation record

Validation performed in the current development container (Linux, Python 3.13, CPU-only PyTorch; no physical NVIDIA GPU or PySide6 GUI runtime).

| Check | Result | Scope |
|---|---|---|
| `python -m compileall -q app tests scripts` | PASS | All overlay Python modules compile. |
| `python -m pytest -q` | PASS — 18 passed, 1 skipped | Includes CPU IFNet/RRDBNet shape tests, tile-edge test, aspect-ratio tests, no-op upscaler test, model download/checksum tests, frame scheduler tests and real FFmpeg CFR/audio integration. |
| CUDA RIFE → Real-ESRGAN tensor-handoff test | SKIPPED | Requires a physical CUDA device; run on the RTX 4070 SUPER and RTX 4050. |
| RIFE checkpoint strict load | NOT RUN | Checkpoint download and real model initialization require external network access. |
| Real-ESRGAN checkpoint strict load | NOT RUN | Release asset download and real model initialization require external network access. |
| CUDA FP16 inference / VRAM benchmark | NOT RUN | No NVIDIA GPU is attached to this container. |
| NVENC hardware output | NOT RUN | No NVIDIA GPU is attached to this container. |
| PySide6 UI patch/launch | NOT RUN | The container does not include the Windows Qt desktop runtime. |
| Ruff | NOT RUN | Ruff is not installed in this container and package downloads are unavailable. Run `ruff check .` on Windows. |

The FFmpeg integration test synthesizes a short CFR video with audio longer than the video stream, routes it through a deterministic fake frame pipeline, and verifies frame count, 60 FPS, output resolution and 48 kHz audio. It tests the media runner and synchronization contract, not neural inference.

**Release gate:** do not call the AI pipeline hardware-validated until the CUDA benchmark passes on the RTX 4070 SUPER and RTX 4050, real checkpoint state dictionaries strict-load, the end-to-end UI job completes, and FFprobe/playback confirms expected output.
