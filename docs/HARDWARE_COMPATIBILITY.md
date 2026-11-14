# Hardware compatibility — AI Upscaler Studio V1.0.0

This document separates hardware that has **actually been tested** from hardware
that is **expected to work** because the bundled runtime supports it, and hardware
that is **not supported or not validated**. Do not read "expected" as "tested".

## What the packaged application bundles

| Component | Version | Notes |
|---|---|---|
| PyTorch | 2.14.1+cu126 | Official CUDA 12.6 wheel from download.pytorch.org |
| CUDA runtime | 12.6 (cuBLAS, cuFFT, cuSPARSE, cuSOLVER, cuRAND, NVRTC) | Redistributable libraries shipped inside the PyTorch wheels |
| cuDNN | 9.10.2 | Shipped inside the PyTorch wheels |
| CUDA kernels (arch list) | `sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90` | No PTX is included, so GPUs newer than `sm_90` cannot JIT-compile kernels |

The NVIDIA **display driver is not bundled** and is never installed or modified
by the installers. The CUDA Toolkit is **not** required.

## NVIDIA driver requirements

From NVIDIA's [CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/):

| Platform | Absolute minimum (CUDA 12.x minor-version compatibility) | Recommended (CUDA 12.6 GA or newer) |
|---|---|---|
| Windows 10/11 | 527.41 | **560.76 or newer** |
| Ubuntu (proprietary driver) | 525.60.13 | **560.28.03 or newer** |

The startup check compares the installed driver against these values and shows
an actionable message (with a link to <https://www.nvidia.com/en-us/drivers/>)
instead of an obscure CUDA initialization error. On Ubuntu use the proprietary
NVIDIA driver (`ubuntu-drivers` / "Additional Drivers"); the open-source
Nouveau driver does not provide CUDA.

## How the application adapts to the GPU

Detected at startup (Settings → *Environment and hardware check*) and when a job starts:

| Property | Rule (unchanged from the V1 engines) |
|---|---|
| FP16 inference | Enabled when compute capability ≥ 7.0 and the FP16 toggle is on; FP32 otherwise |
| RIFE inference scale | 0.5 for ≤ 8 GB VRAM, 1.0 above; halves automatically on CUDA out-of-memory down to 0.25 |
| Real-ESRGAN tile size | 128 px for ≤ 8 GB VRAM, 256 px above; halves on out-of-memory down to 32 px |
| Architecture support | Device compute capability is checked against the bundled kernel list |
| NVENC | Each NVENC encoder is proven with a real test encode; encoders that fail are reported, not assumed |

## 1. Hardware actually tested

| GPU | VRAM | OS / driver | Build tested | Result |
|---|---|---|---|---|
| NVIDIA GeForce RTX 4070 SUPER (Ada, CC 8.9) | 12 GB | Windows | V1 **source** build (PyTorch 2.14.1+cu126) | Full 58 s end-to-end demonstration (see README). **The packaged installer has not yet been validated on this GPU.** |
| NVIDIA GeForce RTX 4050 Laptop GPU (Ada, CC 8.9) | 6 GB | Windows 11, driver 621.24 | **Packaged** Windows build (installer and portable ZIP), November 2026 | Packaged self-test passed: CUDA detected, RIFE v4.25 + Real-ESRGAN x2plus + x4plus downloaded, SHA-256 verified, strict-loaded, FP16 inference; NVENC H.264/HEVC/AV1 encode tests passed; short end-to-end job (320×180 @ 24 fps → 1280×720 @ 48 fps, H.264 NVENC, AAC 48 kHz audio retained). This is a short synthetic test, not a long real-world video. |
| NVIDIA GeForce RTX 4050 Laptop GPU | 6 GB | Ubuntu 22.04 **container** on Windows (Docker Desktop + WSL2 GPU paravirtualisation) | Packaged Linux `.deb` and AppImage | Under performance verification; see [Linux status](#linux-status). This is **not** a native Ubuntu NVIDIA driver test. |

## 2. Expected to work (not tested)

These GPUs are covered by the bundled CUDA kernels and a supported driver, but no
one has run AI Upscaler Studio on them. Performance and memory headroom vary widely.

| Architecture | Compute capability | Example GPUs | Precision | Notes |
|---|---|---|---|---|
| Ada Lovelace | 8.9 | RTX 4060 – 4090, RTX 2000–6000 Ada | FP16 | Same architecture as the tested GPUs |
| Ampere | 8.6 / 8.0 | RTX 3050 – 3090, RTX A2000 – A6000, A100 | FP16 | |
| Turing | 7.5 | RTX 2060 – 2080 Ti, GTX 1650/1660, Quadro RTX | FP16 | GTX 16-series have no Tensor Cores; FP16 still works |
| Volta | 7.0 | Titan V, V100 | FP16 | |
| Hopper | 9.0 | H100 | FP16 | Data-centre GPU; NVENC not present |
| Pascal | 6.1 / 6.0 | GTX 1050 – 1080 Ti, Titan Xp, P100 | **FP32** | Slow; FP16 is disabled below CC 7.0 |
| Maxwell | 5.0 / 5.2 | GTX 750 Ti, GTX 900 series | **FP32** | Very slow; ≤ 4 GB VRAM models will struggle |

**VRAM guidance:** 6 GB is the smallest tested size. GPUs with less than 4 GB
are flagged with a warning; expect very small tiles, slow processing and possible
out-of-memory failures, especially for x4plus and large outputs.

**NVENC:** H.264 needs Kepler or newer, HEVC needs Maxwell (2nd generation) or
newer, and **AV1 NVENC needs an RTX 40-series (Ada) or newer GPU**. Data-centre
GPUs (for example A100 and H100) have no NVENC. The application tests each
encoder at runtime; choose a software encoder (libx264/libx265) when NVENC is
unavailable.

## 3. Not supported / not validated

| Hardware | Status |
|---|---|
| **GeForce RTX 50-series (Blackwell, CC 12.0) and other Blackwell GPUs (CC 10.x)** | **Not supported by V1.** The bundled CUDA 12.6 PyTorch build contains no `sm_100`/`sm_120` kernels and no PTX. The startup check reports this explicitly. Support requires a newer CUDA (12.8+) PyTorch build, which is out of scope for the V1 package. |
| Kepler (GTX 600/700, CC 3.x) and older | Not supported by the bundled PyTorch build |
| AMD and Intel GPUs | Not supported (CUDA only). The standard FFmpeg export still works. |
| Systems without an NVIDIA GPU | The application starts and explains the requirement; AI processing is unavailable. The non-AI "standard FFmpeg export" still works. |
| Multi-GPU systems | GPU selection is available but has not been tested |
| Windows on ARM, Linux on ARM (aarch64), macOS | No packages are produced |

## Linux status

The Ubuntu AppImage and `.deb` were built on an Ubuntu 22.04 userland (glibc
2.35). The `.deb` passed automated checks in a fresh Ubuntu 22.04 container
(installation, files and permissions, shared-library resolution, Qt xcb plugin
under Xvfb, bundled FFmpeg, packaged self-test, removal). AppImage, Ubuntu 24.04
and GPU performance runs are still in progress. See
[PACKAGING_LIMITATIONS.md](PACKAGING_LIMITATIONS.md) for exact results.

> **Linux is still under performance verification.** Container GPU testing uses
> WSL2 GPU paravirtualisation, which relies on the Windows NVIDIA driver, not the
> native Linux driver. NVENC is not
> available through that path. A run on a physical Ubuntu machine with the
> proprietary NVIDIA driver is still required before claiming Linux GPU support.

## Reporting results

Run the built-in self-test and attach the JSON report to an issue:

```text
Windows:  "%LOCALAPPDATA%\Programs\AI Upscaler Studio\AI Upscaler Studio.exe" --self-test --e2e --report "%USERPROFILE%\Desktop\aus-report.json"
Ubuntu:   ai-upscaler-studio --self-test --e2e --report ~/aus-report.json
AppImage: ./AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage --self-test --e2e --report ~/aus-report.json
```

`--e2e` downloads the models (about 160 MB) and processes a two-second synthetic
clip; it takes one to a few minutes on a supported GPU.
