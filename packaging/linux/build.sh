#!/usr/bin/env bash
# Build AI Upscaler Studio for Ubuntu x86_64: PyInstaller app directory,
# AppImage and .deb, with verification.
#
# Must run on native Linux with an Ubuntu 22.04 userland (glibc 2.35) so the
# result runs on Ubuntu 22.04 and newer: a GitHub Actions ubuntu-22.04 runner,
# or the ubuntu:22.04 container used by packaging/linux/docker-build.sh.
#
#   bash packaging/linux/build.sh [--install-deps] [--skip-tests] [--e2e]
#
# Outputs: dist/release/AI-Upscaler-Studio-v<ver>-Ubuntu-x86_64.{AppImage,deb},
#          BUILD_MANIFEST-ubuntu-x86_64.json, SHA256SUMS.txt
set -euo pipefail

INSTALL_DEPS=0
SKIP_TESTS=0
E2E=0
for arg in "$@"; do
  case "$arg" in
    --install-deps) INSTALL_DEPS=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --e2e) E2E=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source packaging/common/versions.env
BUILD="$REPO/build/linux"
VENV="${AUS_VENV:-$REPO/build/venv-linux}"
STAGE="$BUILD/stage"
OUT="$REPO/dist/release"
APP_DIR="$BUILD/pyinstaller-dist/AI-Upscaler-Studio"
BASE="AI-Upscaler-Studio-v${APP_VERSION}-Ubuntu-x86_64"
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"
step() { printf '\n==> %s\n' "$*"; }
mkdir -p "$BUILD" "$OUT"

if [ "$INSTALL_DEPS" -eq 1 ]; then
  step "Install system build dependencies"
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  # python3 only runs the helper scripts before the pinned venv exists. The Qt/X11
  # libraries let PyInstaller resolve and bundle the xcb platform plugin's
  # dependencies and let the offscreen self-test run.
  $SUDO apt-get install -y -qq --no-install-recommends \
    ca-certificates curl git file binutils patchelf xz-utils zstd dpkg-dev desktop-file-utils \
    python3 libgl1 libegl1 libopengl0 libglib2.0-0 libdbus-1-3 libfontconfig1 libfreetype6 \
    libxkbcommon0 libxkbcommon-x11-0 libx11-6 libx11-xcb1 libxcb1 libxcb-cursor0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
    libxcb-xinerama0 libxcb-xkb1 libxcb-xfixes0 libxcb-sync1 libxrender1 libxi6 libsm6 libice6 \
    > /dev/null
fi

HOST_PY="$(command -v python3)"
fetch() { "$HOST_PY" packaging/scripts/fetch_binary.py "$@"; }

# 1-2. Pinned uv, Python and dependencies.
if ! command -v uv > /dev/null || [ "$(uv --version | awk '{print $2}')" != "$UV_VERSION" ]; then
  step "Install uv $UV_VERSION (pinned SHA-256)"
  UV_TAR="$(fetch uv_linux)"
  mkdir -p "$BUILD/uv"
  tar -xzf "$UV_TAR" -C "$BUILD/uv" --strip-components=1
  export PATH="$BUILD/uv:$PATH"
fi
step "Python $PYTHON_VERSION build environment"
uv python install "$PYTHON_VERSION"
[ -x "$VENV/bin/python" ] || uv venv --python "$PYTHON_VERSION" --seed "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --disable-pip-version-check -q \
  -r packaging/requirements/runtime.txt -r packaging/requirements/build.txt -r packaging/requirements/test.txt

step "Verify CUDA PyTorch build (refuse CPU-only wheels)"
"$PY" packaging/scripts/verify_cuda_runtime.py --expected-cuda 12.6 --json "$BUILD/cuda-environment.json"

if [ "$SKIP_TESTS" -eq 0 ]; then
  step "ruff / pytest / compileall"
  "$PY" -m ruff check .
  QT_QPA_PLATFORM=offscreen "$PY" -m pytest -q -p no:cacheprovider --basetemp "$BUILD/pytest-tmp"
  "$PY" -m compileall -q app tests scripts packaging/scripts
fi

# 3. FFmpeg.
step "Download and verify FFmpeg"
FF_EXTRACT="$(fetch ffmpeg_linux --extract "$BUILD/ffmpeg-extract")"
FF_ROOT="$(find "$FF_EXTRACT" -mindepth 1 -maxdepth 1 -type d | head -n1)"
rm -rf "$STAGE"
mkdir -p "$STAGE/ffmpeg"
install -m 0755 "$FF_ROOT/bin/ffmpeg" "$FF_ROOT/bin/ffprobe" "$STAGE/ffmpeg/"
"$PY" packaging/scripts/verify_ffmpeg.py "$STAGE/ffmpeg" --json "$BUILD/ffmpeg-report.json"

# 4. Licences and PyInstaller.
step "Stage licences"
"$PY" packaging/scripts/stage_licenses.py "$STAGE/licenses" --ffmpeg-dir "$FF_ROOT" \
  --ffmpeg-report "$BUILD/ffmpeg-report.json"
step "PyInstaller"
AUS_REPO_ROOT="$REPO" AUS_STAGE_DIR="$STAGE" "$PY" -m PyInstaller --noconfirm --clean --log-level WARN \
  --distpath "$BUILD/pyinstaller-dist" --workpath "$BUILD/pyinstaller-work" \
  packaging/pyinstaller/ai_upscaler_studio.spec
chmod 0755 "$APP_DIR/ai-upscaler-studio" "$APP_DIR/_internal/ffmpeg/ffmpeg" "$APP_DIR/_internal/ffmpeg/ffprobe"

step "Verify bundle and run self-test (isolated environment)"
VERIFY=("$APP_DIR" --json "$BUILD/verify-bundle.json")
[ "$E2E" -eq 1 ] && VERIFY+=(--e2e --require-gpu)
AI_UPSCALER_STUDIO_MODEL_DIR="$BUILD/verify-models" "$PY" packaging/scripts/verify_bundle.py "${VERIFY[@]}"

rm -f "$OUT/$BASE".*

# 5. AppImage.
step "AppImage"
APPIMAGETOOL="$(fetch appimagetool)"
RUNTIME="$(fetch appimage_runtime)"
chmod +x "$APPIMAGETOOL"
APPDIR="$BUILD/AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/lib" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -a "$APP_DIR" "$APPDIR/usr/lib/ai-upscaler-studio"
install -m 0755 packaging/linux/appimage/AppRun "$APPDIR/AppRun"
install -m 0644 packaging/linux/desktop/ai-upscaler-studio.desktop "$APPDIR/ai-upscaler-studio.desktop"
install -m 0644 packaging/linux/desktop/ai-upscaler-studio.desktop "$APPDIR/usr/share/applications/"
install -m 0644 packaging/common/icons/ai-upscaler-studio-256.png "$APPDIR/ai-upscaler-studio.png"
install -m 0644 packaging/common/icons/ai-upscaler-studio-256.png \
  "$APPDIR/usr/share/icons/hicolor/256x256/apps/ai-upscaler-studio.png"
ln -sf ai-upscaler-studio.png "$APPDIR/.DirIcon"
desktop-file-validate "$APPDIR/ai-upscaler-studio.desktop"
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 VERSION="$APP_VERSION" "$APPIMAGETOOL" --no-appstream \
  --runtime-file "$RUNTIME" --comp zstd --mksquashfs-opt -Xcompression-level --mksquashfs-opt 19 \
  "$APPDIR" "$OUT/$BASE.AppImage"
chmod 0755 "$OUT/$BASE.AppImage"

# 6. Debian package (installs to /opt, launcher symlink in /usr/bin).
step ".deb"
DEBROOT="$BUILD/debroot"
rm -rf "$DEBROOT"
mkdir -p "$DEBROOT/DEBIAN" "$DEBROOT/opt" "$DEBROOT/usr/bin" "$DEBROOT/usr/share/applications" \
  "$DEBROOT/usr/share/doc/ai-upscaler-studio"
cp -a "$APP_DIR" "$DEBROOT/opt/ai-upscaler-studio"
ln -s /opt/ai-upscaler-studio/ai-upscaler-studio "$DEBROOT/usr/bin/ai-upscaler-studio"
install -m 0644 packaging/linux/desktop/ai-upscaler-studio.desktop "$DEBROOT/usr/share/applications/"
for size in 16 24 32 48 64 128 256 512; do
  install -D -m 0644 "packaging/common/icons/ai-upscaler-studio-$size.png" \
    "$DEBROOT/usr/share/icons/hicolor/${size}x${size}/apps/ai-upscaler-studio.png"
done
install -m 0644 packaging/common/LICENSE-NOTICE.txt "$DEBROOT/usr/share/doc/ai-upscaler-studio/copyright"
install -m 0644 THIRD_PARTY_NOTICES.md "$DEBROOT/usr/share/doc/ai-upscaler-studio/"
find "$DEBROOT" -type d -exec chmod 0755 {} +
find "$DEBROOT/opt" -type f -exec chmod u+rw,go+r {} +
INSTALLED_SIZE="$(du -sk --exclude=DEBIAN "$DEBROOT" | cut -f1)"
sed -e "s/@VERSION@/$APP_VERSION/g" -e "s/@INSTALLED_SIZE@/$INSTALLED_SIZE/" \
  packaging/linux/debian/control.in > "$DEBROOT/DEBIAN/control"
install -m 0755 packaging/linux/debian/postinst "$DEBROOT/DEBIAN/postinst"
install -m 0755 packaging/linux/debian/postrm "$DEBROOT/DEBIAN/postrm"
dpkg-deb --root-owner-group -Zzstd -z9 --build "$DEBROOT" "$OUT/$BASE.deb"
dpkg-deb --info "$OUT/$BASE.deb" | sed -n '1,12p'

# 7. Manifest and checksums.
step "Manifest and checksums"
PYI_VERSION="$("$PY" -c 'import PyInstaller; print(PyInstaller.__version__)')"
. /etc/os-release
"$PY" packaging/scripts/write_manifest.py --platform ubuntu-x86_64 \
  --environment "$PRETTY_NAME / glibc $(ldd --version | head -n1 | awk '{print $NF}') / uv CPython $PYTHON_VERSION" \
  --artifacts "$OUT" --ffmpeg-report "$BUILD/ffmpeg-report.json" \
  --tool "PyInstaller=$PYI_VERSION" --tool "appimagetool=1.9.1" --tool "AppImage type2-runtime=20251108" \
  --tool "dpkg-deb=$(dpkg-deb --version | head -n1 | grep -oE '[0-9]+.[0-9]+(.[0-9]+)?' | head -n1)" \
  --output "$OUT/BUILD_MANIFEST-ubuntu-x86_64.json"
"$PY" packaging/scripts/write_manifest.py --merge "$OUT"/BUILD_MANIFEST-*.json --output "$OUT/BUILD_MANIFEST.json"
"$PY" packaging/scripts/generate_checksums.py "$OUT"
ls -l "$OUT"
