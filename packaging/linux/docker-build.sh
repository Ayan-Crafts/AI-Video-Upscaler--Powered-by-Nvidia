#!/usr/bin/env bash
# Build the Ubuntu packages inside a clean ubuntu:22.04 container, then test the
# .deb and AppImage in fresh ubuntu:22.04 and ubuntu:24.04 containers.
#
# Works from Linux or from Git Bash on Windows with Docker Desktop (WSL2).
#   bash packaging/linux/docker-build.sh [--e2e-gpu] [build.sh options...]
#
# --e2e-gpu additionally runs the packaged self-test with model downloads and
# CUDA inference inside a container started with --gpus all. On Windows this
# uses WSL2 GPU paravirtualisation; it is NOT a native Ubuntu driver test.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
IMAGE_BUILD="ubuntu:22.04"
GPU_TEST=0
BUILD_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --e2e-gpu) GPU_TEST=1 ;;
    *) BUILD_ARGS+=("$arg") ;;
  esac
done

HOST_REPO="$REPO"
if command -v cygpath > /dev/null; then
  HOST_REPO="$(cygpath -w "$REPO")"
  export MSYS_NO_PATHCONV=1
fi
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=0
[ -n "$(git status --porcelain 2>/dev/null)" ] && GIT_DIRTY=1
mkdir -p dist/release build/linux

# The repository is copied onto the container's native filesystem: building on
# a Windows bind mount is very slow and loses Unix permissions.
docker run --rm \
  -v "$HOST_REPO:/src" \
  -v ai-upscaler-studio-build-cache:/cache \
  -v ai-upscaler-studio-work:/work \
  -e AUS_GIT_COMMIT="$GIT_COMMIT" -e AUS_GIT_BRANCH="$GIT_BRANCH" -e AUS_GIT_DIRTY="$GIT_DIRTY" \
  -e AUS_VENV=/cache/venv-linux -e AUS_DOWNLOAD_CACHE=/cache/downloads \
  -e XDG_CACHE_HOME=/cache/xdg -e UV_CACHE_DIR=/cache/uv -e UV_PYTHON_INSTALL_DIR=/cache/uv-python \
  -e PIP_CACHE_DIR=/cache/pip \
  "$IMAGE_BUILD" bash -euo pipefail -c '
    apt-get update -qq && apt-get install -y -qq --no-install-recommends tar > /dev/null
    # /work is a named volume, kept after the run for inspection; start clean.
    find /work -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    tar -C /src --exclude=./build --exclude=./dist --exclude=./.venv --exclude=./.git \
        --exclude=./nvidia_ai_video_studio.egg-info -cf - . | tar -C /work -xf -
    cd /work
    # Windows bind mounts report every file as executable; restore Git-like modes.
    find . -type d -exec chmod 0755 {} +
    find . -type f -exec chmod 0644 {} +
    chmod 0755 packaging/linux/*.sh packaging/linux/appimage/AppRun packaging/linux/debian/postinst \
        packaging/linux/debian/postrm
    bash packaging/linux/build.sh --install-deps '"${BUILD_ARGS[*]:-}"'
    cp -f dist/release/*Ubuntu* dist/release/BUILD_MANIFEST-ubuntu-x86_64.json /src/dist/release/
    cp -f build/linux/*.json /src/build/linux/
  '

# Package tests in fresh containers (no build tools, no Python, no FFmpeg).
for image in ubuntu:22.04 ubuntu:24.04; do
  echo "==> Package tests on $image"
  docker run --rm -v "$HOST_REPO:/src:ro" -v "$HOST_REPO/build/linux:/results" "$image" \
    bash /src/packaging/linux/test-packages.sh "/results/package-tests-${image//[:.]/-}.json"
done

if [ "$GPU_TEST" -eq 1 ]; then
  echo "==> GPU end-to-end self-test (docker --gpus all)"
  docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v "$HOST_REPO:/src:ro" -v "$HOST_REPO/build/linux:/results" ubuntu:22.04 \
    bash /src/packaging/linux/test-packages.sh /results/package-tests-gpu.json --gpu-e2e
fi

# Refresh combined checksums and manifest on the host.
PY="$(command -v python3 || command -v python)"
if [ -f build/venv-win/Scripts/python.exe ]; then PY=build/venv-win/Scripts/python.exe; fi
"$PY" packaging/scripts/write_manifest.py --merge dist/release/BUILD_MANIFEST-*.json --output dist/release/BUILD_MANIFEST.json
"$PY" packaging/scripts/generate_checksums.py dist/release
