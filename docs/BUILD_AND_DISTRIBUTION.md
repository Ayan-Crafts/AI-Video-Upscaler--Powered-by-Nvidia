# Building and distributing AI Upscaler Studio V1.0.0

Maintainer guide for producing the Windows and Ubuntu downloads. End-user
instructions are in [INSTALLATION.md](INSTALLATION.md).

## Artifacts

| File | Built by |
|---|---|
| `AI-Upscaler-Studio-v1.0.0-Windows-x64-Setup.exe` | `packaging/windows/build.ps1` (Inno Setup) |
| `AI-Upscaler-Studio-v1.0.0-Windows-x64-Portable.zip` | `packaging/windows/build.ps1` |
| `AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage` | `packaging/linux/build.sh` (appimagetool) |
| `AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb` | `packaging/linux/build.sh` (dpkg-deb) |
| `SHA256SUMS.txt` | `packaging/scripts/generate_checksums.py` |
| `BUILD_MANIFEST.json` (+ per-platform `BUILD_MANIFEST-*.json`) | `packaging/scripts/write_manifest.py` |

All outputs go to `dist/release/`; intermediates go to `build/`. Both are ignored by Git.

## Packaging approach and why

**PyInstaller 6.22.3, one-directory mode**, wrapped by an installer/ZIP/AppImage/.deb.

| Option | Assessment |
|---|---|
| **PyInstaller (chosen)** | Mature hooks for PyTorch (CUDA DLLs on Windows; `nvidia-*` wheel libraries on Linux) and for PySide6 Qt plugins via `pyinstaller-hooks-contrib`. Bundles the VC++ runtime on Windows. Builds in minutes. Same spec file for both platforms. |
| Nuitka / `pyside6-deploy` | `pyside6-deploy` is Nuitka-based. Compiling PyTorch's very large Python surface with Nuitka is slow (tens of minutes to hours) and PyTorch support relies on many manual data/DLL inclusions; no benefit for a GPU-bound app whose hot path is in CUDA kernels. Rejected for V1 reliability. |
| One-file executables | Rejected. A 4+ GB CUDA runtime would be extracted to a temp folder on **every** launch (slow start, fills `%TEMP%`, antivirus scans). One-dir + installer starts instantly. |

**Windows installer: Inno Setup 6.7.3.** Per-user install without administrator
rights by default (`PrivilegesRequired=lowest`, with an "all users" option), Start
Menu and optional desktop shortcuts, uninstaller registration, LZMA2 compression
of the 4 GB tree into one EXE. WiX/MSI was not chosen: MSI handles multi-GB
payloads poorly (cab size limits) and adds no value here.

**Linux: AppImage (appimagetool 1.9.1 + static type-2 runtime 20251108)** — no
`libfuse2` dependency — and a **`.deb`** installing to `/opt/ai-upscaler-studio`
with declared Qt/X11 dependencies. Both are built on an **Ubuntu 22.04 userland
(glibc 2.35)** so they run on 22.04 and newer.

## Pinned inputs

| Input | Pin | Where |
|---|---|---|
| Python | CPython 3.12.13 (uv-managed python-build-standalone) | `packaging/common/versions.env` |
| uv | 0.11.28 | `versions.env`, `external_binaries.json` (Linux tarball SHA-256) |
| Runtime packages | torch 2.14.1+cu126, numpy 2.5.4, PySide6-Essentials 6.12.0, shiboken6 6.12.0, psutil 7.2.2, nvidia-ml-py 13.615.71 (+ torch's own deps) | `packaging/requirements/runtime.txt` |
| Build tools | pyinstaller 6.22.3, pyinstaller-hooks-contrib 2026.10 | `packaging/requirements/build.txt` |
| Test tools | pytest 9.2.0, pytest-qt 4.5.0, ruff 0.17.4 | `packaging/requirements/test.txt` |
| FFmpeg (Windows) | gyan.dev 9.0.2 essentials, SHA-256 pinned | `packaging/common/external_binaries.json` |
| FFmpeg (Linux) | BtbN n9.0.2-39-g7adfbe40dc linux64-gpl (month-end autobuild), SHA-256 pinned | `external_binaries.json` |
| appimagetool / runtime | 1.9.1 / 20251108, SHA-256 pinned | `external_binaries.json` |
| GitHub Actions | pinned to commit SHAs | `.github/workflows/*.yml` |

Why Python 3.12 rather than 3.11: the V1 environment's pinned `numpy 2.5.4`
requires Python ≥ 3.12. 3.12 keeps every package at the V1 version and is within
the README's documented 3.11/3.12 range.

`torchvision` is not bundled: the application never imports it. `triton` is
excluded on Linux for the same reason (no `torch.compile`). Only
`PySide6-Essentials` is installed (the app uses QtCore/QtGui/QtWidgets), which
avoids ~1 GB of QtWebEngine and add-on modules.

Every downloaded binary is verified against its pinned SHA-256 by
`packaging/scripts/fetch_binary.py` before use; a mismatch stops the build.
Downloads are never executed except the pinned appimagetool.

## Building on Windows

Prerequisites: Windows 10/11 x64, [uv](https://docs.astral.sh/uv/) on `PATH`,
[Inno Setup 6](https://jrsoftware.org/isdl.php) (`winget install JRSoftware.InnoSetup`), internet access.
An NVIDIA GPU is only needed for `-E2E`.

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1 -E2E
```

Options: `-SkipTests`, `-SkipInstaller`, `-E2E` (frozen end-to-end CUDA test),
`-InnoSetupCompiler <path>`.

Steps performed:
1. Clean venv `build\venv-win` with Python 3.12.13; install pinned requirements.
2. `verify_cuda_runtime.py` — **fails if torch is a CPU build** or not CUDA 12.6.
3. `ruff check .`, `pytest`, `compileall`.
4. Download and verify FFmpeg; `verify_ffmpeg.py` checks every encoder the UI offers
   (NVENC H.264/HEVC/AV1, libx264, libx265, libvpx-vp9, libaom-av1, prores_ks, AAC, Opus),
   required decoders/filters, and classifies the licence (fails on `--enable-nonfree`).
5. `stage_licenses.py` collects licence files; PyInstaller builds `build\windows\pyinstaller-dist\AI-Upscaler-Studio`.
6. `verify_bundle.py` checks the bundle (launcher, FFmpeg, Qt platform plugin, CUDA
   DLLs, no dev files) and runs `--self-test` with a **stripped environment**
   (`PATH` = System32 only, no `PYTHONPATH`), so a developer's Python/FFmpeg cannot
   mask a missing dependency. Model downloads use `build\windows\verify-models`, never the user cache.
7. ZIP64 portable archive, Inno Setup installer, manifest, `SHA256SUMS.txt`.

## Building for Ubuntu

**Without an Ubuntu machine (recommended locally):** Docker Desktop (WSL2) on Windows, or Docker on Linux:

```bash
bash packaging/linux/docker-build.sh            # build in ubuntu:22.04, test in fresh 22.04 + 24.04
bash packaging/linux/docker-build.sh --e2e-gpu  # also run CUDA e2e in a --gpus all container
```

The repository is copied into the container's native filesystem; caches live in
the `ai-upscaler-studio-build-cache` Docker volume.

**On an Ubuntu 22.04 machine or runner:**

```bash
bash packaging/linux/build.sh --install-deps [--e2e]
```

`build.sh` mirrors the Windows steps (pinned uv/Python/requirements, CPU-torch
guard, tests, FFmpeg verification, licences, PyInstaller, isolated self-test),
then builds the AppDir/AppImage and the `.deb`.

`test-packages.sh` runs inside a **fresh** container and checks: `.deb` install with
dependency resolution, launcher symlink, executable bits, desktop entry
(`desktop-file-validate`), icon, root ownership and no world-writable files, `ldd`
resolution of every bundled ELF (driver libraries excepted), `--version`, the
packaged self-test with the offscreen and the real **xcb** Qt platform (Xvfb),
clean removal, and the AppImage self-test.

## GitHub Actions

| Workflow | Trigger | What it does |
|---|---|---|
| `build-ubuntu.yml` | manual, PRs touching app/packaging, `workflow_call` | Builds on `ubuntu-22.04`; tests packages in `ubuntu:22.04` and `ubuntu:24.04` containers; uploads artifacts |
| `build-windows.yml` | manual, PRs, `workflow_call` | Builds installer + ZIP on `windows-2022`; uploads artifacts |
| `release-builds.yml` | manual only | Runs both, merges manifests, writes `SHA256SUMS.txt`, uploads one release-candidate artifact |

No workflow creates tags or GitHub Releases. GitHub-hosted runners have no NVIDIA
GPU, so CI verifies packaging, not CUDA inference.

## Release checklist (manual)

1. Build both platforms from a clean commit (`BUILD_MANIFEST.json` → `git_worktree_dirty: false`).
2. Run the Windows installer on the reference GPU (RTX 4070 SUPER): install, launch, Settings report,
   `--self-test --e2e`, a real CFR clip end-to-end with audio, reboot, relaunch, uninstall.
3. Run the `.deb` and AppImage on a physical Ubuntu NVIDIA machine (pending for V1).
4. GitHub Release assets are limited to 2 GiB per file. Prepare the upload folder with
   `python packaging/scripts/split_for_release.py dist/release dist/upload`: files over the limit
   become `.part1`/`.part2`, and `dist/upload/SHA256SUMS.txt` lists the parts and the joined files.
   Upload everything in `dist/upload/` and include the join commands in the release notes.
5. Attach or link the FFmpeg corresponding source (see `licenses/ffmpeg/SOURCE-OFFER.txt`).

## Updating a pinned component

Change the pin, rebuild, rerun all verification, and update `THIRD_PARTY_NOTICES.md`.
For a new FFmpeg/appimagetool: update URL **and** SHA-256 in `external_binaries.json`
(take the digest GitHub publishes for the asset). Do not change PyTorch/CUDA for a
V1 rebuild; a newer CUDA build (e.g. for RTX 50-series) is V2 scope.
