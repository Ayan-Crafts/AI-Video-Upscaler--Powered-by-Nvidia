# Packaging status and limitations — V1.0.0

Packaging V1 did **not** change V1's processing limitations (speed, CFR-only
input, 8-bit SDR output, limited hardware validation, early-stage interface).
Those remain as documented in the README. This page covers what packaging
achieved, how it was verified, and what is still open.

## Artifact status (November 2026 build)

| Artifact | Generated | Size | Verification performed |
|---|---|---|---|
| `AI-Upscaler-Studio-v1.0.0-Windows-x64-Setup.exe` | Yes | 1.77 GB (1,768,631,080 B) | Silent per-user install, Start Menu/uninstall registration, installed-app self-test with real CUDA inference on an RTX 4050 Laptop GPU, GUI launch/close cycles, uninstall |
| `AI-Upscaler-Studio-v1.0.0-Windows-x64-Portable.zip` | Yes | 2.75 GB (2,747,930,808 B) | Same PyInstaller tree as the installer; bundle self-test with CUDA end-to-end in a stripped environment |
| `AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage` | Yes | 3.31 GB (3,314,084,344 B) | Built in `ubuntu:22.04`. **Still under performance verification**: AppImage, 24.04 and GPU runs in progress |
| `AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb` | Yes | 2.71 GB (2,713,961,506 B) | Install/remove with `apt`, files, permissions, `ldd`, desktop entry, self-test (offscreen + xcb) passed in a fresh 22.04 container. **Still under performance verification** (24.04 and GPU) |

Exact versions and SHA-256 values: `dist/release/BUILD_MANIFEST.json` and `SHA256SUMS.txt`.

## Verified on real hardware (Windows)

On an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB, driver 621.24, Windows 11), the
**installed** application, run with `PATH` reduced to `System32` (no Python or
FFmpeg reachable):

- loaded the bundled CUDA 12.6 PyTorch and detected the GPU (CC 8.9, FP16 enabled);
- used the **bundled** FFmpeg 9.0.2 and passed real NVENC test encodes (H.264, HEVC, AV1);
- downloaded RIFE v4.25, Real-ESRGAN x2plus and x4plus into
  `%LOCALAPPDATA%\AI Upscaler Studio\models`, verified size + SHA-256, strict-loaded them
  and ran FP16 CUDA inference;
- processed a 2-second synthetic CFR clip with audio (320×180 @ 24 fps → 1280×720 @ 48 fps,
  H.264 NVENC) and kept the audio (AAC, resampled to 48 kHz);
- launched the GUI (Qt `windows` platform plugin, correct title/icon, startup check green).

## Still to be validated by the maintainer

| Item | Why it is open |
|---|---|
| Packaged build on the **RTX 4070 SUPER** reference machine | Not available in this build session |
| A real, longer CFR video end-to-end in the installed app (quality review, playback) | Only a synthetic 2-second clip was processed automatically |
| Launch after a **Windows restart** | The build machine was not rebooted |
| **Ubuntu on physical NVIDIA hardware** with the native proprietary driver | No Ubuntu NVIDIA machine available. Linux CUDA ran only through WSL2 GPU paravirtualisation (Windows driver) |
| **NVENC on Linux** | WSL2 containers do not expose `libnvidia-encode`; the Linux e2e test fell back to libx264 |
| Wayland-native session behaviour | Tested with X11 (Xvfb) and offscreen only |
| GitHub Actions workflows | Written and pinned; not yet executed on GitHub's runners (the release artifacts were built locally) |

## Known limitations and open issues

1. **Download size / GitHub Release limit.** The CUDA 12.6 PyTorch runtime (cuDNN, cuBLAS,
   cuFFT, cuSPARSE, NCCL …) is ~4 GB on Windows and ~5.5 GB on Linux uncompressed.
   GitHub Release assets are limited to **2 GiB per file**: the portable ZIP, the AppImage and the .deb exceed it, so the
   V1.0.0 release publishes them as two parts each (`.part1`, `.part2`), created by
   `packaging/scripts/split_for_release.py`. Users join them with `copy /b` (Windows) or `cat`
   (Linux); `SHA256SUMS.txt` lists both the parts and the joined files. The CUDA libraries are hard link-time dependencies of
   `torch_cuda`, so they cannot be removed safely.
2. **Unsigned.** No code-signing certificate was available. SmartScreen and some antivirus
   products may warn about or quarantine the installer or large CUDA DLLs.
3. **RTX 50-series (Blackwell) unsupported.** The pinned V1 PyTorch build has kernels only up
   to `sm_90` and no PTX. The app reports this clearly. Supporting it needs a CUDA 12.8+
   PyTorch build (V2 scope).
4. **NVIDIA driver required.** Minimum 527.41 (Windows) / 525.60.13 (Linux) through CUDA
   minor-version compatibility; 560.76 / 560.28.03 recommended.
5. **First-run internet access** is required to download the model weights (~160 MB total);
   weights are not redistributed (see THIRD_PARTY_NOTICES.md).
6. **Python 3.12 instead of 3.11.** The pinned `numpy 2.5.4` requires Python ≥ 3.12; all
   other package versions are identical to the V1 environment.
7. **Shutdown handling.** With CUDA PyTorch and Qt loaded together, interpreter teardown could
   intermittently abort (`0xC0000409`) after the window closed. V1.0.0 now cancels jobs, joins
   threads, flushes logs and exits the process directly. Closing during a job asks for confirmation.
8. **FFmpeg GPL compliance.** The bundled FFmpeg binaries are GPL-3.0-or-later. A written source
   offer and links to the exact sources ship in `licenses/ffmpeg/SOURCE-OFFER.txt`. For the
   strictest compliance, attach the FFmpeg source tarball to the release alongside the binaries.
9. **No `LICENSE` file at the repository root.** `pyproject.toml` declares GPL-3.0-or-later and the
   packages ship the GPL text (via FFmpeg's licence file) plus a licence notice. Consider adding
   the full GPLv3 text as `LICENSE` in the repository.
10. **`.deb` maintainer field** uses a GitHub URL instead of an e-mail address (no address was
    added to public metadata). Set one in `packaging/linux/debian/control.in` if wanted.
11. **Build provenance.** Artifacts built from an uncommitted working tree record
    `git_worktree_dirty: true` in the manifest. Rebuild after committing for a clean manifest.
12. **Portable mode** still stores models and logs in the per-user directory (by design: the
    application never writes next to its executable).
13. **Linux desktop libraries.** The AppImage relies on the standard Ubuntu Desktop graphics/X11
    libraries (`libGL`, `libEGL`, `libxcb-*`, `libxkbcommon`), which are present on Ubuntu Desktop
    but not on minimal/server installs; the `.deb` declares them as dependencies.
14. **Old repository URL inside V1.0.0 binaries.** The repository was renamed to
    `Ayan-Crafts/AI-Video-Upscaler--Powered-by-Nvidia` after the packages were built. The About
    panel, installer metadata and licence notices still show the previous URL, which GitHub
    redirects permanently. Update `app/version.py`, `packaging/windows/installer.iss`,
    `packaging/linux/debian/control.in` and `packaging/common/*.txt` before the next build.
15. **AppImage extract-and-run in memory-constrained containers.** In Docker's 8 GB VM the
    `--appimage-extract-and-run` mode (used only when FUSE is unavailable) once failed with
    `sqfs_read_range error: Out of memory` while other tests ran in parallel; repeated isolated
    runs passed. Normal desktop use mounts the AppImage with FUSE and does not extract it.
