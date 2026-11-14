# AI Upscaler Studio
### Powered by NVIDIA

**V1.0.0 — Imaginoromancophilia**

An experimental, Windows-first desktop application for NVIDIA GPU-accelerated AI video upscaling and frame interpolation.

AI Upscaler Studio combines AI super-resolution, frame interpolation, CUDA inference and FFmpeg video processing into one configurable desktop workflow.

V1 is the first functional end-to-end release, validated on an NVIDIA GeForce RTX 4070 SUPER 12 GB desktop GPU.

> **Version:** V1.0.0 — Imaginoromancophilia
> **Platforms:** Windows 10/11 x64 (installer + portable) · Ubuntu 22.04+ x86_64 (AppImage + .deb, **still under performance verification**)
> **Validated GPU:** NVIDIA GeForce RTX 4070 SUPER 12 GB (V1 source build) · NVIDIA GeForce RTX 4050 Laptop 6 GB (V1.0.0 packaged Windows build)
> **Application licence:** GPL-3.0-or-later

### ⬇ Download V1 — Imaginoromancophilia

**[⬇ Get AI Upscaler Studio V1.0.0 from the Releases page](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/tag/v1.0.0)** — a self-contained Windows installer ([direct download](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/download/v1.0.0/AI-Upscaler-Studio-v1.0.0-Windows-x64-Setup.exe)), a portable ZIP, and Ubuntu AppImage/.deb packages. No Python, PyTorch, CUDA Toolkit or FFmpeg installation required; you only need a supported NVIDIA GPU and driver. See [Download and install](#download-and-install).

---

## Contents

- [What's new in V1.0.0](#whats-new-in-v100)
- [Introduction](#introduction)
- [Why AI Upscaler Studio?](#why-ai-upscaler-studio)
- [V1 demonstration](#v1-demonstration)
- [Features and technology](#features-and-technology)
- [Processing workflow](#processing-workflow)
- [V1 results](#v1-results)
- [Performance](#performance)
- [Known limitations](#known-limitations)
- [Hardware compatibility](#hardware-compatibility)
- [V2 development](#v2-development)
- [Technology stack and credits](#technology-stack-and-credits)
- [Download and install](#download-and-install)
- [Installation from source (developers)](#installation-from-source-developers)
- [Project structure](#project-structure)
- [Building the packages](#building-the-packages)
- [Troubleshooting](#troubleshooting)
- [Validation](#validation)
- [Release notes](#release-notes--v100-imaginoromancophilia)
- [Licensing and disclaimer](#licensing-and-disclaimer)

---

## What's new in V1.0.0

V1.0.0 *Imaginoromancophilia* turns the working V1 pipeline into a downloadable desktop application. The AI pipeline itself (RIFE v4.25, Real-ESRGAN x2plus/x4plus, FP16 CUDA inference, NVENC encoding) is unchanged.

| Area | What changed |
|---|---|
| **Distribution** | Self-contained Windows installer (per-user, no admin rights) and portable ZIP; Ubuntu AppImage and `.deb`. Python, CUDA 12.6 PyTorch, cuDNN, PySide6, NumPy and FFmpeg are bundled. |
| **Startup hardware check** | Detects the NVIDIA GPU, driver version, CUDA runtime, compute capability, VRAM, FP16 support and unsupported architectures, and test-encodes with each NVENC encoder. Problems are explained in plain language with a link to NVIDIA's driver page. |
| **FFmpeg discovery** | Uses the bundled FFmpeg/FFprobe first, then `AI_UPSCALER_STUDIO_FFMPEG_DIR`, then `PATH` (development fallback). NVENC is reported only when it actually works. |
| **Model management** | First-run downloads with progress, exact-size and SHA-256 verification, atomic writes, retries, cleanup of interrupted downloads, and reuse of models from pre-1.0 developer builds. |
| **Platform paths** | Models, logs and settings live in per-user folders (`%LOCALAPPDATA%` on Windows, XDG directories on Linux); nothing is written to the install folder or AppImage. |
| **Reliability** | No console windows flash for FFmpeg in the packaged Windows app; closing during a job asks for confirmation and cancels cleanly; out-of-memory cleanup no longer hides the real error; orderly shutdown with CUDA + Qt; crash log for native faults. |
| **Settings page** | Shows the full environment/hardware report, model cache and log locations, and an About panel. |
| **Self-test** | `--self-test` (and `--self-test --e2e`) produces a JSON report of the runtime, GPU, Qt, FFmpeg, model downloads and a short end-to-end CUDA job. |
| **Reproducible builds** | Pinned versions, SHA-256-verified third-party binaries, build scripts for Windows and Ubuntu, GitHub Actions workflows, `SHA256SUMS.txt` and `BUILD_MANIFEST.json`. |

---

## Introduction

AI Upscaler Studio is a locally executed AI video-processing application designed to combine two important video-enhancement techniques:

- **AI Super-Resolution:** Increase video resolution using neural networks.
- **AI Frame Interpolation:** Generate intermediate frames to increase frame rate and improve motion smoothness.

The application integrates:

- Practical-RIFE v4.25 for frame interpolation.
- Real-ESRGAN x2plus and x4plus for super-resolution.
- PyTorch and CUDA for NVIDIA GPU inference.
- FFmpeg and FFprobe for video and audio processing.
- NVIDIA NVML for GPU telemetry.
- PySide6 for the desktop interface.

The objective is to provide an inspectable, configurable and locally executed video-enhancement pipeline.

Videos are processed on the user's machine rather than being uploaded to a remote enhancement service. An internet connection is required for initial model downloads unless the models are already cached.

---

## Why AI Upscaler Studio?

There are already excellent video-enhancement applications. This project does not claim to be the first, fastest or highest-quality video upscaler.

Instead, its goal is to provide a different combination of openness, control and NVIDIA-focused engineering.

### How does it compare?

| Existing solution | What it provides | Our distinction |
|---|---|---|
| [Topaz Video](https://www.topazlabs.com/topaz-video) | Professional AI video enhancement, restoration, interpolation and upscaling. | AI Upscaler Studio focuses on an inspectable open-source processing pipeline rather than a commercial model ecosystem. |
| [NVIDIA RTX Video](https://nvidia.custhelp.com/app/answers/detail/a_id/5448) | Real-time AI video enhancements for supported playback applications. | Our application produces a new, configurable output video file rather than enhancing playback in supported players. |
| [Video2X](https://github.com/k4yt3x/video2x) | Open-source video super-resolution and frame interpolation. | We provide an independent Windows-first PySide6 application with integrated NVIDIA hardware telemetry and our own CUDA processing workflow. |
| [REAL Video Enhancer](https://github.com/TNTwise/REAL-Video-Enhancer) | Open-source interpolation, upscaling, decompression and denoising with multiple backends. | We focus on a dedicated PyTorch CUDA pipeline and a configurable NVIDIA-oriented desktop workflow. |

These projects are established alternatives and may offer more mature functionality, additional models, better performance or broader compatibility.

### What makes this project different?

**1. Local video processing**

Your video stays on your computer during processing. There is no requirement to upload large personal recordings or gameplay footage to an online enhancement service.

**2. An inspectable AI pipeline**

The application combines independently developed AI models and open-source media tools. Developers can examine how interpolation, super-resolution, CUDA inference and encoding work together.

**3. Configurable output**

V1 exposes important processing choices:

- Target frame rate.
- Output resolution.
- AI interpolation.
- AI upscaling.
- Audio sample rate.
- Video container.
- Video codec.
- NVIDIA GPU selection.
- FP16 inference.

**4. NVIDIA-focused GPU integration**

The application uses CUDA-enabled PyTorch inference, FP16 where supported, GPU-aware tiling and NVIDIA telemetry. NVENC encoder options are available when supported by the installed FFmpeg build and driver.

**5. No proprietary enhancement backend required**

The core AI pipeline is built around publicly available model implementations and open-source libraries. This makes it useful as an engineering and experimentation platform, rather than a black-box enhancement service.

**Our actual advantage is transparency and control, not an unproven claim of superior image quality or speed.**

---

## V1 demonstration

The following videos use the same gameplay recording to demonstrate the original and enhanced output.

| Original gameplay recording | AI enhanced output |
|:---:|:---:|
| [![Original gameplay](https://img.youtube.com/vi/t4xxRYU7Q90/hqdefault.jpg)](https://www.youtube.com/watch?v=t4xxRYU7Q90) | [![Enhanced gameplay](https://img.youtube.com/vi/4B6UuXTA_NI/hqdefault.jpg)](https://www.youtube.com/watch?v=4B6UuXTA_NI) |
| [▶ Watch original](https://www.youtube.com/watch?v=t4xxRYU7Q90) | [▶ Watch enhanced](https://www.youtube.com/watch?v=4B6UuXTA_NI) |
| **1920 × 1080** | **2560 × 1440** |
| Variable frame rate | 60 FPS |
| Average: ~29.29 FPS | Target: 60 FPS |
| Duration: ~58.20 seconds | Duration: ~58 seconds |
| AAC · Stereo · 48 kHz | AAC · Stereo · 48 kHz |
| Original gameplay capture | RIFE + Real-ESRGAN |

### Technical comparison

| Parameter | Original | Enhanced |
|---|---:|---:|
| Resolution | 1920 × 1080 | 2560 × 1440 |
| Frame width | 1920 | 2560 |
| Frame height | 1080 | 1440 |
| Frame rate | VFR (average ~29.29 FPS) | 60 FPS |
| Duration | ~58.20 seconds | ~58 seconds |
| Audio codec | AAC | AAC |
| Audio channels | Stereo | Stereo |
| Audio sample rate | 48 kHz | 48 kHz |
| AI interpolation | No | RIFE v4.25 |
| AI upscaling | No | Real-ESRGAN x2plus |

### What changed?

- Resolution increased from 1080p to 1440p.
- Output frame rate increased to 60 FPS.
- Real-ESRGAN performed AI-based spatial super-resolution.
- RIFE generated intermediate frames for temporal interpolation.
- Stereo audio was retained at 48 kHz.

**Preprocessing note:** The original Microsoft Game DVR recording was variable frame rate. V1 requires constant-frame-rate input, so FFmpeg was used to create a full-resolution 1920 × 1080, 30 FPS CFR intermediate before processing. The original footage was not downscaled.

The enhanced video was played back and visually reviewed. The output appeared smoother and visually improved, with no obvious ghosting noticed during the author's review of this particular clip. Results can vary with source quality, motion and scene complexity.

---

## Application screenshots

### Dashboard — NVIDIA hardware monitoring

![AI Upscaler Studio Dashboard](docs/assets/v1/dashboard.png)

Displays GPU utilization, VRAM usage, CPU utilization, system memory, GPU temperature and available NVIDIA encoder capabilities.

### Process Video — configuration

![AI Upscaler Studio Process Video](docs/assets/v1/process-video.png)

The demonstrated configuration:

- Target FPS: 60
- Resolution: 2560 × 1440
- Audio: 48,000 Hz
- Container: MP4
- Video codec: H.264 NVENC
- AI interpolation: Enabled
- AI upscaling: Enabled
- FP16 inference: Enabled

### FFmpeg — source preparation

![FFmpeg CFR input](docs/assets/v1/ffmpeg-cfr-input.png)

The original recording was variable frame rate and required conversion into a constant-frame-rate intermediate.

### FFmpeg — conversion completion

![FFmpeg CFR conversion complete](docs/assets/v1/ffmpeg-cfr-complete.png)

The intermediate was created at full 1920 × 1080 resolution and 30 FPS.

### Original video properties

![Original video properties](docs/assets/v1/original-video-properties.png)

1920 × 1080, 30 FPS CFR, stereo audio at 48 kHz.

### Enhanced video properties

![Enhanced video properties](docs/assets/v1/enhanced-video-properties.png)

2560 × 1440, 60 FPS, stereo audio at 48 kHz.

---

## Features and technology

### AI frame interpolation — RIFE v4.25

RIFE estimates intermediate frames between existing frames to generate a higher-frame-rate output.

V1 uses Practical-RIFE v4.25 with CUDA inference.

### AI super-resolution — Real-ESRGAN

The application integrates:

- Real-ESRGAN x2plus.
- Real-ESRGAN x4plus.

Tiled inference helps manage GPU memory. Adaptive tile sizing is used when CUDA out-of-memory conditions occur.

### CUDA and FP16 inference

PyTorch executes neural-network inference on the selected NVIDIA GPU.

The RIFE-to-Real-ESRGAN intermediate remains CUDA-resident between those stages. Completed frames are transferred back for FFmpeg encoding.

### NVIDIA telemetry

GPU utilization, temperature, VRAM and available encoder capabilities are displayed alongside CPU and system memory usage.

### FFmpeg integration

FFmpeg and FFprobe provide media inspection, decoding, encoding, audio mapping and resampling.

NVENC can be selected when the installed FFmpeg build and NVIDIA driver expose the requested encoder.

### Startup environment and hardware check

On every launch the application checks, in the background:

- NVIDIA GPU and driver detection (NVML, with an `nvidia-smi` fallback);
- the bundled PyTorch build and its CUDA runtime version (a CPU-only build is reported as an error);
- driver compatibility with the CUDA 12.6 runtime (minimum and recommended versions);
- CUDA initialisation, compute capability and whether the GPU architecture is supported by the bundled kernels;
- VRAM, FP16 eligibility, and the default RIFE scale and Real-ESRGAN tile size chosen for that VRAM;
- the FFmpeg/FFprobe location, plus a real test encode with each NVENC encoder.

The result appears on the **Settings** page. If AI processing is not possible, for example because no NVIDIA driver is installed or the GPU is an unsupported RTX 50-series card, a clear dialog explains why and links to NVIDIA's official driver page. The application never installs or modifies drivers.

---

## Processing workflow

The V1 demonstration followed this sequence:

1. Read the original Microsoft Game DVR recording.
2. Convert VFR footage into a 30 FPS CFR intermediate using FFmpeg.
3. Decode frames for processing.
4. Use RIFE v4.25 to generate intermediate frames.
5. Use Real-ESRGAN x2plus for AI super-resolution.
6. Fit the output to 2560 × 1440.
7. Encode the resulting video and map the source audio.

Native VFR processing is not supported in V1.

---

## V1 results

| Measurement | Original CFR intermediate | Enhanced output |
|---|---:|---:|
| Resolution | 1920 × 1080 | 2560 × 1440 |
| Frame rate | 30 FPS | 60 FPS |
| Duration | ~58 seconds | ~58 seconds |
| Audio | Stereo | Stereo |
| Audio sample rate | 48 kHz | 48 kHz |

The enhanced output contains approximately **1.78× as many pixels per frame** as the original 1080p intermediate, with twice the frame rate.

This is one successful end-to-end demonstration, not a universal quality guarantee.

---

## Performance

**Validated hardware:** NVIDIA GeForce RTX 4070 SUPER — 12 GB VRAM.

The complete 58-second video took approximately **two hours** to process.

Separate 640 × 360 CUDA engine measurements:

| Engine | Latency | Throughput | Peak VRAM |
|---|---:|---:|---:|
| RIFE v4.25 | ~302 ms/pair | 3.31 pairs/s | 0.29 GB |
| Real-ESRGAN x2plus | ~1.63 s/frame | 0.61 frames/s | 5.52 GB |
| Real-ESRGAN x4plus | ~2.48 s/frame | 0.40 frames/s | 9.70 GB |

The upscaler measurements used FP16 and 256-pixel tiles.

These are model-engine smoke-test measurements, not full-video throughput measurements. Actual processing time depends on resolution, model scale, GPU memory and encoding configuration.

---

## Known limitations

V1 is an early release. The following limitations are known and documented.

| Area | V1 limitation |
|---|---|
| VFR input | Requires CFR preprocessing. Native per-frame timestamp handling is not implemented. |
| Performance | The 58-second demonstration took approximately two hours. |
| Hardware support | End-to-end validation is limited to the RTX 4070 SUPER 12 GB. |
| RTX 4050 | The RTX 4050 6 GB laptop has not completed end-to-end validation. |
| 4K / 8K | Not validated through a complete end-to-end video demonstration. |
| HDR / 10-bit | V1 targets 8-bit SDR output. |
| Audio and metadata | Subtitles, chapters, arbitrary data streams and every alternate audio stream are not guaranteed to be preserved. |
| GPU decoding | Dedicated NVDEC/PyNvVideoCodec zero-copy decoding is not integrated. |
| Interface | The UI and progress reporting are still at an early stage. |
| Model downloads | Initial model downloads require an internet connection. |
| Packaging | The Windows and Linux packages are unsigned. Downloads are large (about 2–3 GB) because the CUDA 12.6 PyTorch runtime is bundled. |
| RTX 50-series | Blackwell GPUs are not supported by the bundled CUDA 12.6 PyTorch build. |
| Linux | The Ubuntu packages are **still under performance verification**. Automated install and launch checks pass; CUDA performance on physical Ubuntu NVIDIA hardware is still being verified. |

V1 is intended for experimentation and offline processing. It is not a real-time video-enhancement application.

---

## Hardware compatibility

| Status | Hardware |
|---|---|
| **Tested** | RTX 4070 SUPER 12 GB (V1 source build, full 58 s demonstration) · RTX 4050 Laptop 6 GB (V1.0.0 packaged Windows build: installer, CUDA FP16 inference, NVENC, short end-to-end job) |
| **Expected to work (untested)** | Ada (RTX 40), Ampere (RTX 30, A-series), Turing (RTX 20, GTX 16), Volta and Hopper with FP16; Pascal (GTX 10) and Maxwell (GTX 900) in slower FP32 |
| **Not supported** | RTX 50-series / Blackwell (no `sm_100`/`sm_120` kernels in the bundled CUDA 12.6 PyTorch build) · Kepler and older · AMD/Intel GPUs |
| **Under performance verification** | Ubuntu with a physical NVIDIA GPU and the native proprietary driver; NVENC on Linux |

Driver: **560.76+** (Windows) / **560.28.03+** (Linux) recommended; 527.41 / 525.60.13 is the absolute minimum. AV1 NVENC requires an RTX 40-series or newer GPU. Details: [docs/HARDWARE_COMPATIBILITY.md](docs/HARDWARE_COMPATIBILITY.md).

---

## V2 development

**V2 is currently in active development.**

We are working on the limitations identified during V1 implementation and real-world testing.

V1 will remain preserved as its own release snapshot. Its documented capabilities and limitations will remain specific to that version.

Further development will build on the lessons learned from the first working release.

---

## Technology stack and credits

This project is made possible by the research, engineering and open-source contributions of the following developers and communities.

AI Upscaler Studio integrates these technologies; it does not claim authorship of their underlying models or libraries.

| Technology | Contribution | Official source |
|---|---|---|
| Practical-RIFE / RIFE v4.25 | AI frame interpolation | [GitHub](https://github.com/hzwer/Practical-RIFE) |
| RIFE v4.25 checkpoint | Interpolation model weights | [Hugging Face](https://huggingface.co/MonsterMMORPG/Practical-RIFE/tree/4bbe4c19fb5363333cca83b6e82bb65ee02d741b) |
| Real-ESRGAN | AI super-resolution | [GitHub](https://github.com/xinntao/Real-ESRGAN) |
| BasicSR | RRDBNet architecture | [GitHub](https://github.com/XPixelGroup/BasicSR) |
| PyTorch | Deep learning and CUDA inference | [Website](https://pytorch.org/) |
| Torchvision | PyTorch vision ecosystem | [Documentation](https://pytorch.org/vision/) |
| FFmpeg | Video and audio processing | [Website](https://ffmpeg.org/) |
| FFprobe | Media stream inspection | [Documentation](https://ffmpeg.org/ffprobe.html) |
| PySide6 / Qt for Python | Desktop application interface | [Documentation](https://doc.qt.io/qtforpython-6/) |
| NVIDIA Management Library Python bindings | GPU telemetry | [PyPI](https://pypi.org/project/nvidia-ml-py/) |
| NumPy | Numerical operations | [Website](https://numpy.org/) |
| psutil | CPU and memory monitoring | [GitHub](https://github.com/giampaolo/psutil) |
| Python | Application runtime | [Website](https://www.python.org/) |
| pytest | Automated testing | [Documentation](https://docs.pytest.org/) |
| pytest-qt | Qt testing | [Documentation](https://pytest-qt.readthedocs.io/) |
| Ruff | Linting and code quality | [Documentation](https://docs.astral.sh/ruff/) |

### Licensing acknowledgements

The upstream project notices identify the following licences for important components:

- Practical-RIFE inference code: MIT.
- Real-ESRGAN: BSD 3-Clause.
- BasicSR RRDBNet architecture: Apache License 2.0.
- PySide6 / Qt: LGPLv3 or commercial terms.
- FFmpeg: LGPL/GPL depending on build configuration.
- psutil: BSD-3-Clause.
- NumPy: BSD-style licence.

Model checkpoints have separate terms from their source-code repositories. Review their redistribution conditions independently.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for additional information.

---

## Download and install

V1.0.0 is distributed as self-contained packages on the **[V1 — Imaginoromancophilia release page](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/tag/v1.0.0)**. **You do not need to install Python, pip, PyTorch, the CUDA Toolkit or FFmpeg**: they are bundled.

### Click to download

| Platform | Package | Download |
|---|---|---|
| **Windows 10/11 x64** | Installer (recommended, per-user, no admin rights) · 1.77 GB | **[⬇ AI-Upscaler-Studio-v1.0.0-Windows-x64-Setup.exe](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/download/v1.0.0/AI-Upscaler-Studio-v1.0.0-Windows-x64-Setup.exe)** |
| Windows 10/11 x64 | Portable ZIP, no installation · 2.75 GB in 2 parts | [⬇ Part 1](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/download/v1.0.0/AI-Upscaler-Studio-v1.0.0-Windows-x64-Portable.zip.part1) · [⬇ Part 2](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/download/v1.0.0/AI-Upscaler-Studio-v1.0.0-Windows-x64-Portable.zip.part2) |
| **Ubuntu 22.04+ x86_64** | AppImage, single file · 3.31 GB in 2 parts | [⬇ Part 1](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/download/v1.0.0/AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage.part1) · [⬇ Part 2](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/download/v1.0.0/AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage.part2) |
| Ubuntu 22.04+ x86_64 | Debian package (`apt install`) · 2.71 GB in 2 parts | [⬇ Part 1](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/download/v1.0.0/AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb.part1) · [⬇ Part 2](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/download/v1.0.0/AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb.part2) |
| All | Checksums | [⬇ SHA256SUMS.txt](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/download/v1.0.0/SHA256SUMS.txt) |
| All | Exact component versions and build details | [⬇ BUILD_MANIFEST.json](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/download/v1.0.0/BUILD_MANIFEST.json) |

### Joining the split downloads

GitHub limits each release file to 2 GiB, but the bundled CUDA runtime makes the portable ZIP and the Ubuntu packages larger than that. These are uploaded in two parts. Download **both** parts into the same folder, then join them:

```bat
:: Windows (Command Prompt)
copy /b AI-Upscaler-Studio-v1.0.0-Windows-x64-Portable.zip.part1 + AI-Upscaler-Studio-v1.0.0-Windows-x64-Portable.zip.part2 AI-Upscaler-Studio-v1.0.0-Windows-x64-Portable.zip
```

```bash
# Ubuntu
cat AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage.part1 AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage.part2 > AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage
chmod +x AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage

cat AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb.part1 AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb.part2 > AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb
sudo apt install ./AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb
```

The Windows installer is a single file and needs no joining.

### What is bundled

Python 3.12.13 · PyTorch 2.14.1 with the CUDA 12.6 runtime and cuDNN 9.10 · PySide6 6.12 (Qt 6) · NumPy 2.5.4 · psutil · nvidia-ml-py · FFmpeg and FFprobe 9.0.2 (GPL builds with NVENC) · the RIFE and Real-ESRGAN inference code.

### What you provide

- A supported **NVIDIA GPU**: Maxwell through Ada Lovelace/Hopper (GTX 900 series to RTX 40-series). **RTX 50-series (Blackwell) is not supported by V1.** 6 GB VRAM is the smallest tested size.
- A current **NVIDIA display driver**: 560.76+ on Windows, 560.28.03+ on Ubuntu (proprietary driver) recommended. The installers never install or modify drivers. Get drivers from [NVIDIA](https://www.nvidia.com/en-us/drivers/).
- An internet connection the first time AI processing runs.

### Install

- **Windows installer:** run `Setup.exe`. It installs for the current user without administrator rights by default, creates a Start Menu entry and optional desktop shortcut, and registers an uninstaller. The packages are unsigned, so if SmartScreen warns, choose *More info → Run anyway* only after checking the SHA-256.
- **Windows portable:** join the parts, extract the ZIP with Explorer or 7-Zip, and run `AI Upscaler Studio\AI Upscaler Studio.exe`.
- **Ubuntu AppImage:** join the parts, `chmod +x`, then run it. `libfuse2` is not required.
- **Ubuntu .deb:** join the parts, then `sudo apt install ./AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb` and launch *AI Upscaler Studio* from the app grid or with `ai-upscaler-studio`.

### First run

1. Launch **AI Upscaler Studio**. A startup check detects the GPU, driver, CUDA runtime, VRAM, compute capability, FP16 support, FFmpeg and *working* NVENC encoders, and explains any problem in plain language (Settings page).
2. The first AI job downloads the model weights (RIFE v4.25 ≈ 25 MB, Real-ESRGAN x2plus/x4plus ≈ 67 MB each). Each file is size- and SHA-256-verified, written atomically and reused afterwards.
3. Choose a constant-frame-rate video and start processing.

### Where files are stored

| | Windows | Ubuntu |
|---|---|---|
| Models | `%LOCALAPPDATA%\AI Upscaler Studio\models` | `~/.cache/ai-upscaler-studio/models` |
| Logs | `%LOCALAPPDATA%\AI Upscaler Studio\logs` | `~/.local/state/ai-upscaler-studio/logs` |

Models already downloaded by a pre-1.0 developer build (`%LOCALAPPDATA%\NVIDIA AI Video Studio\models`) are reused automatically.

### Verify your download

Compare the SHA-256 of each file with [SHA256SUMS.txt](https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia/releases/download/v1.0.0/SHA256SUMS.txt). It lists both the individual parts and the joined files.

```powershell
Get-FileHash .\AI-Upscaler-Studio-v1.0.0-Windows-x64-Setup.exe -Algorithm SHA256
```

```bash
sha256sum --check --ignore-missing SHA256SUMS.txt
```

> **Ubuntu status: still under performance verification.** The AppImage and `.deb` are built on Ubuntu 22.04. The `.deb` passed automated install, launch, Qt (offscreen and xcb), FFmpeg, shared-library and removal checks in a fresh Ubuntu 22.04 container. Performance and CUDA inference on a physical Ubuntu machine with the native NVIDIA driver are still being verified, so treat the Linux packages as a preview for now.

Full instructions and troubleshooting: [docs/INSTALLATION.md](docs/INSTALLATION.md). Maintainers: [docs/BUILD_AND_DISTRIBUTION.md](docs/BUILD_AND_DISTRIBUTION.md).

---

## Installation from source (developers)

### Requirements

- Windows 10/11 64-bit or Ubuntu 22.04+.
- Python 3.12 (the packaged builds use 3.12.13; the pinned NumPy 2.5.4 requires 3.12+).
- Compatible NVIDIA GPU and driver.
- CUDA-enabled PyTorch.
- FFmpeg and FFprobe on PATH (or `AI_UPSCALER_STUDIO_FFMPEG_DIR` pointing at their folder).

### Clone the repository

```powershell
git clone https://github.com/Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia.git
cd AI-Video-Upscaler--Powered-by-Nvidia
```

### Create a virtual environment and install the pinned V1 runtime

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r packaging/requirements/runtime.txt -r packaging/requirements/test.txt
python -m pip install -e . --no-deps
```

`packaging/requirements/runtime.txt` pins PyTorch 2.14.1 with CUDA 12.6 from the official PyTorch index. Avoid installing the generic `.[ml]` extra over an existing CUDA environment without checking the PyTorch wheel source.

### Verify dependencies

```powershell
nvidia-smi
ffmpeg -version
ffprobe -version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

### Launch

```powershell
python -m app                 # GUI
python -m app --self-test     # headless environment and hardware report
```

Model checkpoints are downloaded on first use into the per-user model cache listed above.

---

## Project structure

```text
app/                        Application package
  main.py                   Entry point (GUI, --version, --self-test)
  version.py                Product name, version and codename
  paths.py                  Platform-aware resource, model, log and config locations
  selftest.py               Headless self-test used by packaging verification
  core/                     Configuration, job model, AI frame pipeline, subprocess helpers
  engines/                  RIFE and Real-ESRGAN CUDA engines, model manager, vendored model code
  hardware/                 NVML/nvidia-smi telemetry and the startup runtime check
  media/                    FFmpeg/FFprobe discovery, probing, AI video runner, FFmpeg runner
  ui/                       PySide6 main window, workers and theme
packaging/
  requirements/             Pinned runtime, build and test dependencies
  common/                   Versions, pinned third-party binaries, icons, licence notices
  pyinstaller/              Shared PyInstaller spec (Windows and Linux)
  windows/                  build.ps1 and the Inno Setup installer script
  linux/                    build.sh, Docker build/test wrappers, AppImage and Debian files
  scripts/                  Download verification, FFmpeg/CUDA/bundle verification, checksums, manifest
.github/workflows/          Windows, Ubuntu and release-candidate build workflows (artifacts only)
docs/                       Pipeline, architecture, installation, hardware, build and limitation guides
tests/                      Unit, integration and packaging tests
```

---

## Building the packages

Maintainers can reproduce every download:

```powershell
# Windows (needs uv and Inno Setup 6; -E2E also runs a CUDA end-to-end test on the build machine)
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1 -E2E
```

```bash
# Ubuntu packages, built in a clean ubuntu:22.04 container and tested in fresh 22.04 + 24.04 containers
bash packaging/linux/docker-build.sh
# or natively on Ubuntu 22.04
bash packaging/linux/build.sh --install-deps
```

The build refuses CPU-only PyTorch, verifies every downloaded binary against a pinned SHA-256, checks that FFmpeg exposes every encoder offered in the UI and classifies its licence, then runs the packaged app's self-test with a stripped `PATH` so a developer's Python or FFmpeg cannot hide a missing dependency. Outputs go to `dist/release/` with `SHA256SUMS.txt` and `BUILD_MANIFEST.json`. Full guide: [docs/BUILD_AND_DISTRIBUTION.md](docs/BUILD_AND_DISTRIBUTION.md).

---

## Troubleshooting

| Symptom | What to do |
|---|---|
| "No NVIDIA GPU or NVIDIA display driver was detected" | Install the driver from [NVIDIA](https://www.nvidia.com/en-us/drivers/); on Ubuntu use the proprietary driver (`sudo ubuntu-drivers install`) and check `nvidia-smi`. |
| "Driver … is too old for the bundled CUDA 12.6 runtime" | Update to 560.76+ (Windows) or 560.28.03+ (Linux) and reboot. |
| "GPU architecture is not supported" | RTX 50-series (Blackwell) GPUs are not supported by V1. |
| CUDA out of memory | Close other GPU applications, use a lower output resolution, or disable one AI stage. |
| "Variable-frame-rate input is not supported" | Convert to CFR first, for example `ffmpeg -i in.mp4 -fps_mode cfr -r 30 -c:v libx264 -crf 16 -c:a copy cfr.mp4`. |
| NVENC encoder unavailable | Update the driver or choose libx264/libx265. AV1 NVENC needs an RTX 40-series GPU. |
| Qt "xcb" plugin / shared-library errors on Ubuntu | Install the `.deb` with `apt`, or install the libraries listed in the installation guide. |
| Windows DLL errors / antivirus warnings | Reinstall, and exclude the installation folder if antivirus quarantined CUDA DLLs. The packages are unsigned. |

Logs: `%LOCALAPPDATA%\AI Upscaler Studio\logs` (Windows) · `~/.local/state/ai-upscaler-studio/logs` (Ubuntu). Full guide: [docs/INSTALLATION.md](docs/INSTALLATION.md#troubleshooting).

---

## Validation

The V1 development checks included:

```powershell
ruff check .
python -m pytest -q
python -m compileall -q app tests scripts
```

The recorded test run completed with **35 tests passing**.

CUDA engine smoke test:

```powershell
python scripts/benchmark_cuda_engines.py --device 0 --width 640 --height 360
```

Include x4plus:

```powershell
python scripts/benchmark_cuda_engines.py --device 0 --width 640 --height 360 --include-x4
```

These checks do not certify compatibility with every NVIDIA GPU.

### V1.0.0 packaging validation

| Check | Result |
|---|---|
| `ruff check .` · `pytest` · `compileall` (clean Python 3.12.13 + CUDA PyTorch build environment) | Passed — 65 tests on Windows (CUDA tests ran); 62 passed / 3 GPU-only skipped in the Ubuntu 22.04 build container |
| Windows installer | Silent per-user install and upgrade, Start Menu and uninstaller registration, clean uninstall (user models kept) |
| Installed Windows app, `PATH` reduced to System32 (RTX 4050 Laptop 6 GB, driver 621.24) | Bundled CUDA 12.6 PyTorch and FFmpeg used; NVENC H.264/HEVC/AV1 test encodes passed; RIFE + Real-ESRGAN x2plus/x4plus downloaded, verified, FP16 inference; 2 s synthetic clip 320×180 @ 24 fps → 1280×720 @ 48 fps with AAC 48 kHz audio retained |
| Windows GUI | Launches with the bundled Qt platform plugin; 8/8 launch/close cycles exited cleanly |
| Portable ZIP | Extracted to a path with spaces and non-ASCII characters; self-test passed with CUDA |
| Ubuntu packages | **Still under performance verification.** Built in `ubuntu:22.04`. The `.deb` passed install, files, permissions, `ldd`, `--version`, offscreen and xcb self-tests, `desktop-file-validate` and clean removal in a fresh 22.04 container. AppImage, Ubuntu 24.04 and GPU performance runs are in progress |

Linux is **still under performance verification**: CUDA inference and performance on a physical Ubuntu machine have not been confirmed yet. See [docs/PACKAGING_LIMITATIONS.md](docs/PACKAGING_LIMITATIONS.md).

---

## Release notes — V1.0.0 Imaginoromancophilia

- First downloadable, self-contained release for Windows 10/11 x64 and Ubuntu 22.04+ x86_64.
- Bundles Python 3.12.13, PyTorch 2.14.1 (CUDA 12.6, cuDNN 9.10.2), PySide6 6.12.0, NumPy 2.5.4, psutil 7.2.2, nvidia-ml-py 13.615.71 and FFmpeg/FFprobe 9.0.2.
- Startup hardware/driver/CUDA/NVENC check with actionable messages; Settings diagnostics page.
- Production model manager: verified, atomic, resumable-by-retry first-run downloads; reuse of earlier caches.
- Packaged builds are unsigned. Model weights are downloaded on first use, not bundled.
- Known limitations (CFR-only input, 8-bit SDR output, processing speed, limited hardware validation, RTX 50-series unsupported) are listed in [Known limitations](#known-limitations) and [docs/PACKAGING_LIMITATIONS.md](docs/PACKAGING_LIMITATIONS.md).

---

## Licensing and disclaimer

The application package metadata declares **GPL-3.0-or-later**.

Third-party libraries, model implementations, checkpoints and FFmpeg builds retain their own licensing requirements. The packaged builds redistribute GPL-3.0-or-later FFmpeg binaries; model weights are downloaded at first use and are not redistributed. Review the corresponding upstream licence files and `THIRD_PARTY_NOTICES.md` before redistribution.

AI Upscaler Studio is an independent project. It is not affiliated with, endorsed by or sponsored by NVIDIA Corporation.

NVIDIA, GeForce, CUDA and NVENC are trademarks or registered trademarks of NVIDIA Corporation.

AI-generated frames are model estimates. Enhancement does not guarantee recovery of original ground-truth detail and may introduce artifacts.

The gameplay footage is used for technical demonstration. Game content and associated intellectual property belong to their respective rights holders.

---

## Created and developed by

**Sanjay Kumar S.**

[GitHub](https://github.com/Sanjay1712KSK) · [Portfolio](https://sanjay1712ksk.github.io/)

AI Upscaler Studio — Powered by NVIDIA

*Built to experiment. Open to inspect. Improved through real-world testing.*
