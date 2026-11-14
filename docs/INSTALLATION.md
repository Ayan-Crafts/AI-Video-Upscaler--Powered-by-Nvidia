# Installing AI Upscaler Studio V1.0.0

AI Upscaler Studio — Powered by NVIDIA · V1.0.0 *Imaginoromancophilia*

The downloads are self-contained. They bundle Python 3.12, PyTorch 2.14.1 with
the CUDA 12.6 runtime and cuDNN, Qt for Python (PySide6), NumPy, psutil,
nvidia-ml-py, FFmpeg and FFprobe. **You do not install Python, pip, PyTorch, the
CUDA Toolkit or FFmpeg.**

The one prerequisite you provide is a **supported NVIDIA GPU with a current
NVIDIA display driver** (see [HARDWARE_COMPATIBILITY.md](HARDWARE_COMPATIBILITY.md)).
RTX 50-series (Blackwell) GPUs are not supported by V1.

| Download | Platform |
|---|---|
| `AI-Upscaler-Studio-v1.0.0-Windows-x64-Setup.exe` | Windows 10 (1809+) / 11, x64 — installer |
| `AI-Upscaler-Studio-v1.0.0-Windows-x64-Portable.zip` | Windows 10 / 11, x64 — no installation |
| `AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage` | Ubuntu 22.04+ x86_64 — single file |
| `AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb` | Ubuntu 22.04+ x86_64 — system package |
| `SHA256SUMS.txt`, `BUILD_MANIFEST.json` | Checksums and exact component versions |

## Verify your download

The installers are **not code-signed**. Check the SHA-256 against `SHA256SUMS.txt`:

```powershell
# Windows (PowerShell)
Get-FileHash .\AI-Upscaler-Studio-v1.0.0-Windows-x64-Setup.exe -Algorithm SHA256
```

```bash
# Ubuntu (in the folder containing the downloads and SHA256SUMS.txt)
sha256sum --check --ignore-missing SHA256SUMS.txt
```

## Windows

### Installer

1. Install or update the NVIDIA driver from <https://www.nvidia.com/en-us/drivers/> (560.76 or newer recommended).
2. Run `AI-Upscaler-Studio-v1.0.0-Windows-x64-Setup.exe`. Because it is unsigned,
   SmartScreen may show *Windows protected your PC*; choose **More info → Run anyway**
   only if the checksum matched.
3. By default it installs **for the current user, without administrator rights**, into
   `%LOCALAPPDATA%\Programs\AI Upscaler Studio`. Choose *Install for all users* in the
   first dialog to install into `C:\Program Files\AI Upscaler Studio` (requires administrator).
4. Optionally create a desktop shortcut. A Start Menu entry is always created.
5. Launch **AI Upscaler Studio**. The startup check runs automatically
   (Settings page shows the full report).

Uninstall from **Settings → Apps → Installed apps → AI Upscaler Studio**, or the
*Uninstall AI Upscaler Studio* Start Menu entry. Downloaded models and logs in
`%LOCALAPPDATA%\AI Upscaler Studio` are kept; delete that folder to remove them.

### Portable ZIP

Extract the ZIP anywhere you can write to (for example `D:\Apps`) and run
`AI Upscaler Studio\AI Upscaler Studio.exe`. Use **7-Zip or Windows Explorer's
"Extract all"**; do not run the app from inside the ZIP. The portable build still
stores models and logs in `%LOCALAPPDATA%\AI Upscaler Studio`, never next to the executable.

## Ubuntu

Install the proprietary NVIDIA driver first (Ubuntu: *Software & Updates →
Additional Drivers*, or `sudo ubuntu-drivers install`) and reboot. Check it with
`nvidia-smi`.

### .deb package

```bash
sudo apt install ./AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.deb
ai-upscaler-studio            # or launch "AI Upscaler Studio" from the app grid
```

`apt` pulls in the small set of standard desktop libraries Qt needs (all present
on a normal Ubuntu Desktop). The application is installed in
`/opt/ai-upscaler-studio`. Remove it with `sudo apt remove ai-upscaler-studio`.

### AppImage

```bash
chmod +x AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage
./AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage
```

The AppImage uses the static type-2 runtime, so `libfuse2` is **not** required,
but the kernel FUSE module must be available (it is on Ubuntu Desktop). Without
FUSE (containers, some VMs) run it with `--appimage-extract-and-run`.

> **Linux GPU status:** the Linux packages passed automated build, install and
> launch checks, but **CUDA inference has not yet been validated on a physical
> Ubuntu machine with the native NVIDIA driver**.

## First launch and model downloads

1. The startup check detects the GPU, driver, CUDA runtime, VRAM, compute
   capability, FP16 support, FFmpeg and working NVENC encoders. Problems are shown
   in plain language with a link to NVIDIA's driver page.
2. AI model weights are **not** included in the downloads. The first AI job
   downloads what it needs (internet required once):

   | Model | Size | Integrity check |
   |---|---|---|
   | Practical-RIFE v4.25 `flownet.pkl` | 24.6 MB | SHA-256 published by Hugging Face for the pinned revision |
   | Real-ESRGAN `RealESRGAN_x2plus.pth` | 67.1 MB | Exact size + maintainer-pinned SHA-256 |
   | Real-ESRGAN `RealESRGAN_x4plus.pth` | 67.0 MB | Exact size + maintainer-pinned SHA-256 |

   Downloads go to a temporary `.part` file, are verified, then atomically moved
   into the cache. Interrupted downloads are discarded and retried; cached models
   are reused on every later launch.
3. Select a **constant-frame-rate** video, choose the target FPS/resolution/codec
   and start AI processing.

## Where files are stored

| Data | Windows | Ubuntu |
|---|---|---|
| Model cache | `%LOCALAPPDATA%\AI Upscaler Studio\models` | `~/.cache/ai-upscaler-studio/models` |
| Logs | `%LOCALAPPDATA%\AI Upscaler Studio\logs\ai-upscaler-studio.log` | `~/.local/state/ai-upscaler-studio/logs/ai-upscaler-studio.log` |
| Temporary files | `%TEMP%` | `$TMPDIR` or `/tmp` |
| Output videos | Wherever you choose | Wherever you choose |

Nothing is written to the installation directory, the portable folder or the
AppImage mount. Set `AI_UPSCALER_STUDIO_MODEL_DIR` to use a different model
folder. Models from pre-1.0 developer builds
(`%LOCALAPPDATA%\NVIDIA AI Video Studio\models`) are copied automatically on
first use, so they are not downloaded again.

## Self-test

Every package contains a headless self-test that writes a JSON report:

```text
"<install dir>\AI Upscaler Studio.exe" --self-test --report report.json          (Windows)
ai-upscaler-studio --self-test --report report.json                              (Ubuntu .deb)
./AI-Upscaler-Studio-v1.0.0-Ubuntu-x86_64.AppImage --self-test --report report.json
```

Add `--e2e` to also download the models and process a short synthetic clip on the GPU.

---

# Troubleshooting

Always check the **Settings** page and the log file first; both show the exact reason.

### NVIDIA driver not detected
*"No NVIDIA GPU or NVIDIA display driver was detected."*
- Windows: install the driver from <https://www.nvidia.com/en-us/drivers/>. Laptops
  with switchable graphics: make sure the NVIDIA GPU is enabled in the BIOS/MUX settings.
- Ubuntu: `nvidia-smi` must work. Install the proprietary driver (`sudo ubuntu-drivers
  install`), reboot, and check Secure Boot MOK enrolment if the module does not load.
  Nouveau does not support CUDA.

### CUDA runtime initialization fails
*"CUDA could not be initialized …"* or *"driver … is too old for the bundled CUDA 12.6 runtime"*
- Update to driver 560.76+ (Windows) / 560.28.03+ (Linux). Reboot after updating.
- RTX 50-series (Blackwell) GPUs are not supported by V1 — the check says so explicitly.
- Remote desktop sessions and some virtual machines hide the GPU from CUDA.
- Linux: if `nvidia-smi` works but CUDA fails, check `dmesg` for `NVRM` errors and that
  `/dev/nvidia*` device nodes exist and are accessible to your user.

### Insufficient VRAM / out of memory
The engines automatically shrink RIFE scale and Real-ESRGAN tiles on out-of-memory.
If it still fails: close games/browsers using the GPU, choose a lower output
resolution (x2 is used for ≤ 2× upscales and needs far less memory than x4), or disable
one AI stage. GPUs under 4 GB are not recommended.

### Missing Qt plugins ("could not load the Qt platform plugin xcb")
- Windows: reinstall; the plugins are in `_internal\PySide6\plugins`. Do not move files
  out of the installation folder.
- Ubuntu `.deb`: run `sudo apt install -f` to restore dependencies.
- AppImage on a minimal system: `sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libegl1 libgl1`.
- Run with `QT_DEBUG_PLUGINS=1 ai-upscaler-studio` to see which library is missing.
- Wayland sessions use XWayland automatically through the bundled xcb plugin.

### FFmpeg not found
The packages bundle FFmpeg in `_internal/ffmpeg`. If the Settings page reports FFmpeg
from "system PATH" or "missing" in a packaged build, the installation is incomplete —
reinstall. Source checkouts use FFmpeg from `PATH` or `AI_UPSCALER_STUDIO_FFMPEG_DIR`.

### NVENC encoder unavailable
The startup check test-encodes with each NVENC encoder. If one fails, update the
driver, close other applications that are encoding (consumer GPUs limit concurrent
sessions), or pick libx264/libx265. AV1 NVENC needs an RTX 40-series or newer GPU.

### Corrupted model checkpoint
Cached models are re-hashed before use; a mismatch deletes the file and downloads
it again automatically. To force a clean download, close the app and delete the
model cache folder listed above. Behind a proxy or firewall, allow access to
`huggingface.co` and `github.com` / `objects.githubusercontent.com`.

### AppImage permission issues
- *Permission denied*: `chmod +x` the file, and make sure it is not on a filesystem
  mounted `noexec` (some USB drives, `/tmp` on hardened systems).
- *FUSE errors*: run with `--appimage-extract-and-run`, or `sudo apt install fuse3`.
- Ubuntu 24.04 AppArmor restrictions on unprivileged user namespaces do not affect this
  AppImage (it does not use a Chromium-style sandbox).

### Ubuntu shared-library errors
*"error while loading shared libraries: libxcb-cursor.so.0"* or similar:
install the `.deb` with `apt` (it resolves dependencies), or for the AppImage
`sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1
libxcb-render-util0 libxcb-image0 libxcb-shape0 libegl1 libgl1`. Only Ubuntu 22.04
and newer (glibc ≥ 2.35) are supported.

### Windows DLL loading errors
*"OSError: [WinError 126] … torch\lib\…dll"* or *"The code execution cannot proceed"*:
- Antivirus software sometimes quarantines large CUDA DLLs (cuDNN, cuBLAS). Restore
  them / add an exclusion for the installation folder and reinstall.
- Do not copy only the `.exe`; the whole folder (including `_internal`) is required.
- Paths longer than 260 characters can break DLL loading — install to the default
  location or a short path.
- The Visual C++ runtime is bundled; installing the Microsoft VC++ redistributable
  is not required.

### Video-specific errors
- *"Variable-frame-rate input is not supported"*: V1 needs CFR input. Convert first, e.g.
  `ffmpeg -i in.mp4 -fps_mode cfr -r 30 -c:v libx264 -crf 16 -c:a copy cfr.mp4`.
- Processing is slow: this is a V1 limitation (minutes per second of video), not an
  installation problem.
