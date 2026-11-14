#!/usr/bin/env bash
# Test the built .deb and AppImage inside a FRESH Ubuntu container (run as root).
# The image has no Python, FFmpeg or build tools, so a missing bundled
# dependency cannot be satisfied by the host.
#
#   bash test-packages.sh <result.json> [--gpu-e2e]
#
# Without a GPU these checks cover installation, files, permissions, shared
# library resolution, the Qt xcb platform plugin (under Xvfb), the bundled
# FFmpeg and the packaged self-test. --gpu-e2e additionally requires CUDA and
# runs model downloads, inference and a short end-to-end video job.
set -uo pipefail

RESULT="${1:?result json path}"
MODE="${2:-}"
SRC=/src
# shellcheck disable=SC1091
. "$SRC/packaging/common/versions.env"
DEB="$SRC/dist/release/AI-Upscaler-Studio-v${APP_VERSION}-Ubuntu-x86_64.deb"
APPIMAGE="$SRC/dist/release/AI-Upscaler-Studio-v${APP_VERSION}-Ubuntu-x86_64.AppImage"
. /etc/os-release
CHECKS="$(mktemp)"
FAILED=0
export DEBIAN_FRONTEND=noninteractive

record() {  # name status detail
  local detail="${3//\\/\\\\}"
  detail="${detail//\"/\\\"}"
  detail="$(printf '%s' "$detail" | tr '\n\t' '  ' | cut -c1-1500)"
  printf '{"name":"%s","status":"%s","detail":"%s"}\n' "$1" "$2" "$detail" >> "$CHECKS"
  printf '[%s] %s: %s\n' "$2" "$1" "$(printf '%s' "$3" | head -c 300)"
  [ "$2" = "FAIL" ] && FAILED=1
  return 0
}
check() {  # name command...
  local name="$1"; shift
  local output
  if output="$("$@" 2>&1)"; then record "$name" PASS "$output"; else record "$name" FAIL "$output"; fi
}
self_test() {  # name launcher [extra args]
  local name="$1" launcher="$2"; shift 2
  local home report
  home="$(mktemp -d)"; report="$home/report.json"
  if HOME="$home" PATH=/usr/bin:/bin "$@" "$launcher" --self-test --report "$report" > "$home/out.txt" 2>&1; then
    record "$name" PASS "$(grep -E '"(passed|platform|source|window_title)"' "$report" | tr -s ' ' | head -c 600)"
  else
    record "$name" FAIL "$(tail -c 1200 "$home/out.txt") $(grep -A3 '"failures"' "$report" 2>/dev/null)"
  fi
  cp -f "$report" "${RESULT%.json}-$name.json" 2>/dev/null || true
}

apt-get update -qq > /dev/null

if [ "$MODE" != "--gpu-e2e" ]; then
  # 1. AppImage on a bare image (only glibc): informational, shows which system
  #    libraries a desktop must provide (they are present on Ubuntu Desktop).
  out="$(cd /tmp && APPIMAGE_EXTRACT_AND_RUN=1 QT_QPA_PLATFORM=offscreen HOME=/tmp "$APPIMAGE" --version 2>&1)"
  record "appimage-bare-image-version" INFO "$out"
fi

# 2. .deb installation resolves its declared dependencies.
check "deb-install" apt-get install -y -qq --no-install-recommends "$DEB"
check "deb-metadata" dpkg -s ai-upscaler-studio
check "launcher-symlink" test -x /usr/bin/ai-upscaler-studio -a -L /usr/bin/ai-upscaler-studio
check "bundled-executables" test -x /opt/ai-upscaler-studio/ai-upscaler-studio -a \
  -x /opt/ai-upscaler-studio/_internal/ffmpeg/ffmpeg -a -x /opt/ai-upscaler-studio/_internal/ffmpeg/ffprobe
check "desktop-entry-present" test -f /usr/share/applications/ai-upscaler-studio.desktop
check "icon-present" test -f /usr/share/icons/hicolor/256x256/apps/ai-upscaler-studio.png
check "version" /usr/bin/ai-upscaler-studio --version

ww="$(find /opt/ai-upscaler-studio -perm -o+w ! -type l | head -n 5)"
nonroot="$(find /opt/ai-upscaler-studio ! -user root | head -n 5)"
if [ -z "$ww$nonroot" ]; then record "file-permissions" PASS "root-owned, no world-writable files"
else record "file-permissions" FAIL "world-writable: $ww non-root: $nonroot"; fi

# 3. Every bundled ELF resolves its shared libraries (driver libraries excepted).
missing="$(find /opt/ai-upscaler-studio -type f \( -name '*.so*' -o -name ai-upscaler-studio \) -print0 |
  xargs -0 -n 50 env LD_LIBRARY_PATH=/opt/ai-upscaler-studio/_internal ldd 2>/dev/null |
  grep 'not found' | grep -vE 'libcuda\.so|libnvidia-|libnvcuvid|libcuda' | sort -u)"
if [ -z "$missing" ]; then record "shared-libraries" PASS "all bundled ELF dependencies resolve"
else record "shared-libraries" FAIL "$missing"; fi

if [ "$MODE" = "--gpu-e2e" ]; then
  home="$(mktemp -d)"
  if HOME="$home" QT_QPA_PLATFORM=offscreen /usr/bin/ai-upscaler-studio --self-test --e2e --require-gpu \
      --report "$home/r.json" > "$home/o.txt" 2>&1; then
    record "deb-cuda-inference-e2e" PASS "$(grep -E '"(passed|codec|avg_frame_rate|sample_rate|fp16)"' "$home/r.json" | tr -s ' ')"
  else
    record "deb-cuda-inference-e2e" FAIL "$(tail -c 1500 "$home/o.txt")"
  fi
  cp -f "$home/r.json" "${RESULT%.json}-cuda-e2e.json" 2>/dev/null || true
else
  # 4. Packaged self-test (offscreen Qt) and the real xcb plugin under Xvfb.
  self_test "deb-self-test-offscreen" /usr/bin/ai-upscaler-studio env QT_QPA_PLATFORM=offscreen
  apt-get install -y -qq --no-install-recommends xvfb xauth desktop-file-utils > /dev/null
  check "desktop-file-validate" desktop-file-validate /usr/share/applications/ai-upscaler-studio.desktop
  self_test "deb-self-test-xcb" /usr/bin/ai-upscaler-studio xvfb-run -a env QT_QPA_PLATFORM=xcb
fi

# 5. Uninstall leaves no files behind in /opt.
check "deb-remove" apt-get remove -y -qq ai-upscaler-studio
check "deb-removed-cleanly" test ! -e /opt/ai-upscaler-studio -a ! -e /usr/bin/ai-upscaler-studio

# 6. AppImage (system libraries from the .deb dependencies remain, like a desktop).
if [ "$MODE" = "--gpu-e2e" ]; then
  home="$(mktemp -d)"
  if (cd /tmp && HOME="$home" APPIMAGE_EXTRACT_AND_RUN=1 QT_QPA_PLATFORM=offscreen "$APPIMAGE" --self-test \
      --e2e --require-gpu --report "$home/r.json" > "$home/o.txt" 2>&1); then
    record "appimage-cuda-inference-e2e" PASS "$(grep -E '"(passed|codec|avg_frame_rate)"' "$home/r.json" | tr -s ' ')"
  else
    record "appimage-cuda-inference-e2e" FAIL "$(tail -c 1500 "$home/o.txt")"
  fi
else
  check "appimage-executable-bit" test -x "$APPIMAGE"
  self_test "appimage-self-test-xcb" "$APPIMAGE" env APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a env QT_QPA_PLATFORM=xcb
fi

{
  printf '{"os":"%s","mode":"%s","failed":%s,"checks":[' "$PRETTY_NAME" "${MODE:-standard}" "$FAILED"
  paste -sd, "$CHECKS"
  printf ']}\n'
} > "$RESULT"
echo "Result: $([ "$FAILED" -eq 0 ] && echo PASS || echo FAIL) ($RESULT)"
exit "$FAILED"
