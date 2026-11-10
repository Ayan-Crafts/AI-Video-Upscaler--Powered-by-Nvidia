# AI Upscaler Studio
### Powered by NVIDIA

**V1.0.0 — Imaginoromancophilia**

An experimental, Windows-first desktop application for NVIDIA GPU-accelerated AI video upscaling and frame interpolation.

AI Upscaler Studio combines AI super-resolution, frame interpolation, CUDA inference and FFmpeg video processing into one configurable desktop workflow.

V1 is the first functional end-to-end release, validated on an NVIDIA GeForce RTX 4070 SUPER 12 GB desktop GPU.

> **Version:** V1.0.0 — Imaginoromancophilia  
> **Platform:** Windows  
> **Validated GPU:** NVIDIA GeForce RTX 4070 SUPER 12 GB  
> **Application licence:** GPL-3.0-or-later

---

## Contents

- [Introduction](#introduction)
- [Why AI Upscaler Studio?](#why-ai-upscaler-studio)
- [V1 demonstration](#v1-demonstration)
- [Features and technology](#features-and-technology)
- [Processing workflow](#processing-workflow)
- [V1 results](#v1-results)
- [Performance](#performance)
- [Known limitations](#known-limitations)
- [V2 development](#v2-development)
- [Technology stack and credits](#technology-stack-and-credits)
- [Installation](#installation)
- [Validation](#validation)
- [Licensing and disclaimer](#licensing-and-disclaimer)

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

V1 is intended for experimentation and offline processing. It is not a real-time video-enhancement application.

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

## Installation

V1 is a source-based Windows application. Do not assume a standalone installer is available unless a validated packaged build has been published.

### Requirements

- Windows 10/11 64-bit.
- Python 3.11 or 3.12.
- Compatible NVIDIA GPU and driver.
- CUDA-enabled PyTorch.
- FFmpeg and FFprobe on PATH.

### Clone the repository

```powershell
git clone https://github.com/Ayan-Crafts/Nvidia-AI-Video-Upscaler.git
cd Nvidia-AI-Video-Upscaler
```

### Create a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Install CUDA-enabled PyTorch

The validated development environment used PyTorch 2.14.1 with CUDA 12.6:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

For other configurations, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

### Install project dependencies

```powershell
python -m pip install -e ".[ai,dev,nvidia]"
```

Avoid installing the generic `.[ml]` extra over an existing CUDA environment without checking the PyTorch wheel source.

### Verify dependencies

```powershell
nvidia-smi
ffmpeg -version
ffprobe -version
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

### Launch

```powershell
python -m app
```

Model checkpoints are downloaded on first use and cached under:

```text
%LOCALAPPDATA%\NVIDIA AI Video Studio\models\
```

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

---

## Licensing and disclaimer

The application package metadata declares **GPL-3.0-or-later**.

Third-party libraries, model implementations, checkpoints and FFmpeg builds retain their own licensing requirements. Review the corresponding upstream licence files and `THIRD_PARTY_NOTICES.md` before redistribution.

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
