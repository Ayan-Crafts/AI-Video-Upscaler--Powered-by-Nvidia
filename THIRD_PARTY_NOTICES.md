# Third-party notices

AI Upscaler Studio is licensed under GPL-3.0-or-later. It uses, adapts or
redistributes the following projects; their licences apply to the respective
portions. The packaged applications contain these notices and the collected
licence files under `_internal/licenses/` (Windows, portable ZIP, AppImage) or
`/opt/ai-upscaler-studio/_internal/licenses/` (.deb).

## Source code adapted in this repository

Licences reproduced under `app/engines/vendor_licenses/`.

- **Practical-RIFE / RIFE HDv3 inference code** — MIT License, copyright hzwer. The v4.25 checkpoint is downloaded from the pinned `MonsterMMORPG/Practical-RIFE` Hugging Face revision `4bbe4c19fb5363333cca83b6e82bb65ee02d741b`; checkpoint publisher terms remain applicable.
- **Real-ESRGAN** — BSD 3-Clause License, copyright Xintao Wang and contributors.
- **BasicSR RRDBNet architecture** — Apache License 2.0, copyright the BasicSR contributors.

## Components redistributed in the V1.0.0 packages

| Component | Version | Licence | Notes |
|---|---|---|---|
| CPython | 3.12.13 | PSF License 2.0 | python-build-standalone distribution via uv |
| PyTorch | 2.14.1+cu126 | BSD-3-Clause (with third-party notices in the wheel) | |
| NVIDIA CUDA runtime libraries (cudart, cuBLAS, cuBLASLt, cuFFT, cuSPARSE, cuSOLVER, cuRAND, NVRTC, nvJitLink; NCCL on Linux) | CUDA 12.6 | NVIDIA CUDA Toolkit EULA (redistributable components, Attachment A) / NCCL BSD-3-Clause | Shipped inside the official PyTorch CUDA 12.6 wheels; redistributed unmodified |
| NVIDIA cuDNN | 9.10.2 | NVIDIA cuDNN Software License Agreement (redistribution permitted with applications) | Shipped inside the PyTorch wheels |
| PySide6-Essentials / shiboken6 (Qt for Python, Qt 6) | 6.12.0 | LGPL-3.0 (or commercial) | Dynamically linked Qt libraries, unmodified; users may replace them in `_internal/PySide6` |
| NumPy | 2.5.4 | BSD-3-Clause (bundled OpenBLAS: BSD-3-Clause) | |
| psutil | 7.2.2 | BSD-3-Clause | |
| nvidia-ml-py | 13.615.71 | BSD-3-Clause | Python bindings only; NVML itself is part of the NVIDIA driver |
| sympy, mpmath, networkx, filelock, fsspec, Jinja2, MarkupSafe, typing_extensions, setuptools | as pinned in `packaging/requirements/runtime.txt` | BSD / MIT / PSF | PyTorch dependencies |
| Microsoft Visual C++ runtime (Windows) | from the build machine | Microsoft Visual C++ Redistributable licence | `msvcp140.dll`, `vcruntime140*.dll`, UCRT, collected by PyInstaller |
| PyInstaller bootloader | 6.22.3 | GPL-2.0 with bootloader exception | Permits distribution of the bundled application under its own licence |
| **FFmpeg / FFprobe (Windows)** | 9.0.2, gyan.dev "essentials" build | **GPL-3.0-or-later** (`--enable-gpl --enable-version3`) | Includes libx264, libx265 (GPL), libvpx, libaom, libopus and others; NVENC via ffnvcodec headers. Not built with `--enable-nonfree`. |
| **FFmpeg / FFprobe (Linux)** | n9.0.2-39-g7adfbe40dc, BtbN linux64-gpl build | **GPL-3.0-or-later** (`--enable-gpl --enable-version3`) | Same classification; exact configuration in `licenses/ffmpeg/BUILD_CONFIGURATION.json` |
| AppImage type-2 runtime (AppImage only) | 20251108 | MIT | |

### FFmpeg licensing

The bundled FFmpeg builds are **GPL**, not LGPL-only, because they enable GPL
components such as libx264 and libx265. This is compatible with the
application's GPL-3.0-or-later licence. The exact `configure` line, enabled
libraries and licence classification of each redistributed binary are recorded
in `BUILD_MANIFEST.json` and in `licenses/ffmpeg/BUILD_CONFIGURATION.json`.
Corresponding source: see `licenses/ffmpeg/SOURCE-OFFER.txt` (FFmpeg source
release, the builders' public build scripts, and a written offer). References:
<https://www.ffmpeg.org/legal.html>.

### NVIDIA driver

The NVIDIA display/compute driver (including `nvcuda`/`libcuda`, NVML and the
NVENC runtime) is **not** redistributed. Users install it from NVIDIA.

## Model weights (not redistributed)

Model weights are **not** included in the installers or in this repository. They
are downloaded from upstream at first use:

| Checkpoint | Source | Integrity |
|---|---|---|
| Practical-RIFE v4.25 `flownet.pkl` | Hugging Face `MonsterMMORPG/Practical-RIFE` @ `4bbe4c19…` (third-party mirror of hzwer's release) | SHA-256 `6615790e…6de2` (matches Hugging Face's LFS hash) |
| `RealESRGAN_x2plus.pth` | GitHub release `xinntao/Real-ESRGAN` v0.2.1 | Size 67,061,725 B; maintainer-pinned SHA-256 `49fafd45…6abb` |
| `RealESRGAN_x4plus.pth` | GitHub release `xinntao/Real-ESRGAN` v0.1.0 | Size 67,040,989 B; maintainer-pinned SHA-256 `4fa0d389…82f1` |

Bundling was rejected for V1: the RIFE checkpoint is served from a third-party
mirror without explicit redistribution terms for the weights, and the Real-ESRGAN
release assets do not state separate weight licence terms. First-run download
keeps the application compliant without that review. Where upstream does not
publish a checksum, the pinned digest is a maintainer record of the official
asset, not an upstream authenticity guarantee.

## Trademarks

NVIDIA, GeForce, RTX, CUDA and NVENC are trademarks or registered trademarks of
NVIDIA Corporation. AI Upscaler Studio is an independent project and is not
affiliated with, endorsed by or sponsored by NVIDIA Corporation. The application
icon is original artwork and contains no NVIDIA logo.
