"""Verify an FFmpeg/FFprobe pair before it is bundled.

Checks that both executables run, that every encoder the application exposes is
compiled in, and classifies the licence from the build configuration. A build
configured with --enable-nonfree is not redistributable and fails the check.
NVENC being compiled in does not prove the end user's GPU can use it; the
application performs a runtime encode test for that.

Usage: python verify_ffmpeg.py <dir-with-ffmpeg-and-ffprobe> [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_ENCODERS = (
    "h264_nvenc", "hevc_nvenc", "av1_nvenc",  # NVIDIA hardware encoders offered in the UI
    "libx264", "libx265", "libvpx-vp9", "libaom-av1", "prores_ks",  # software video encoders
    "aac", "libopus",  # audio encoders used by the AI runner
)
REQUIRED_DECODERS = ("h264", "hevc", "vp9", "av1", "prores")
REQUIRED_FILTERS = ("scale", "fps", "pad", "aresample")


def run(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60, check=False, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise SystemExit(f"{args[0]} failed ({result.returncode}): {result.stderr[-2000:]}")
    return result.stdout


FLAG_TOKEN = re.compile(r"^[A-Z.|]{1,8}$")


def names(output: str) -> set[str]:
    """Names from -encoders/-decoders/-filters listings (flag column width varies by version)."""
    found = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and FLAG_TOKEN.match(parts[0]) and parts[1] != "=":
            found.add(parts[1])
    return found


def classify(configuration: str) -> str:
    flags = set(configuration.split())
    if "--enable-nonfree" in flags:
        return "NONFREE (not redistributable)"
    gpl = "--enable-gpl" in flags
    v3 = "--enable-version3" in flags
    if gpl:
        return "GPL-3.0-or-later" if v3 else "GPL-2.0-or-later"
    return "LGPL-3.0-or-later" if v3 else "LGPL-2.1-or-later"


def verify(directory: Path) -> dict[str, object]:
    suffix = ".exe" if os.name == "nt" else ""
    ffmpeg, ffprobe = directory / f"ffmpeg{suffix}", directory / f"ffprobe{suffix}"
    for tool in (ffmpeg, ffprobe):
        if not tool.is_file():
            raise SystemExit(f"Missing {tool}")
        if os.name != "nt" and not os.access(tool, os.X_OK):
            raise SystemExit(f"{tool} is not executable")
    version_text = run([str(ffmpeg), "-hide_banner", "-version"])
    probe_version = run([str(ffprobe), "-hide_banner", "-version"]).splitlines()[0]
    configuration = next((line.split(":", 1)[1].strip() for line in version_text.splitlines()
                          if line.startswith("configuration:")), "")
    encoders = names(run([str(ffmpeg), "-hide_banner", "-encoders"]))
    decoders = names(run([str(ffmpeg), "-hide_banner", "-decoders"]))
    filters = names(run([str(ffmpeg), "-hide_banner", "-filters"]))
    report = {
        "ffmpeg_version": version_text.splitlines()[0],
        "ffprobe_version": probe_version,
        "configuration": configuration,
        "license": classify(configuration),
        "enabled_libraries": sorted(f[len("--enable-"):] for f in configuration.split()
                                    if f.startswith("--enable-lib")),
        "missing_encoders": [e for e in REQUIRED_ENCODERS if e not in encoders],
        "missing_decoders": [d for d in REQUIRED_DECODERS if d not in decoders],
        "missing_filters": [f for f in REQUIRED_FILTERS if f not in filters],
        "nvenc_compiled": [e for e in ("h264_nvenc", "hevc_nvenc", "av1_nvenc") if e in encoders],
    }
    report["ok"] = (not report["missing_encoders"] and not report["missing_decoders"]
                    and not report["missing_filters"] and "NONFREE" not in str(report["license"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = verify(args.directory)
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    print(text)
    if not report["ok"]:
        sys.exit("FFmpeg verification FAILED")


if __name__ == "__main__":
    main()
