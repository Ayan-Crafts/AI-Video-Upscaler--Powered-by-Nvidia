"""Collect licence and notice files into the bundle's licenses/ directory.

Includes the application licence notice, vendored model-code licences, the
FFmpeg build's licence files and the licence metadata of every bundled Python
distribution (PyTorch, Qt for Python, NumPy, NVIDIA CUDA wheels, ...).

Usage: python stage_licenses.py <stage-licenses-dir> --ffmpeg-dir <extracted ffmpeg root>
       [--ffmpeg-report report.json]
Run with the build environment's Python so installed distributions are visible.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LICENSE_HINTS = ("license", "licence", "copying", "notice", "authors", "third_party", "thirdparty")


def copy_distribution_licenses(out: Path) -> list[str]:
    copied = []
    skip = {"pip", "pyinstaller", "pyinstaller-hooks-contrib", "pytest", "pytest-qt", "ruff",
            "pluggy", "iniconfig", "pygments", "colorama", "altgraph", "pefile", "pywin32-ctypes",
            "packaging"}
    for dist in md.distributions():
        name = dist.metadata["Name"] or "unknown"
        if name.lower() in skip:
            continue
        target = out / "python-packages" / f"{name}-{dist.version}"
        for file in dist.files or []:
            lower = str(file).lower()
            if ".dist-info/" in lower and any(hint in lower for hint in LICENSE_HINTS):
                source = Path(dist.locate_file(file))
                if source.is_file():
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target / source.name)
        license_expr = dist.metadata.get("License-Expression") or dist.metadata.get("License") or ""
        if license_expr:
            target.mkdir(parents=True, exist_ok=True)
            (target / "LICENSE-METADATA.txt").write_text(
                f"{name} {dist.version}\nLicense: {license_expr[:4000]}\n", encoding="utf-8")
        if target.exists():
            copied.append(f"{name} {dist.version}")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--ffmpeg-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg-report", type=Path)
    args = parser.parse_args()
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", out / "THIRD_PARTY_NOTICES.md")
    shutil.copy2(ROOT / "packaging" / "common" / "LICENSE-NOTICE.txt", out / "AI-UPSCALER-STUDIO-LICENSE.txt")
    shutil.copytree(ROOT / "app" / "engines" / "vendor_licenses", out / "model-code")

    ffmpeg_out = out / "ffmpeg"
    ffmpeg_out.mkdir()
    for candidate in args.ffmpeg_dir.rglob("*"):
        if candidate.is_file() and any(h in candidate.name.lower() for h in LICENSE_HINTS + ("readme",)):
            shutil.copy2(candidate, ffmpeg_out / candidate.name)
    if args.ffmpeg_report and args.ffmpeg_report.is_file():
        shutil.copy2(args.ffmpeg_report, ffmpeg_out / "BUILD_CONFIGURATION.json")
    shutil.copy2(ROOT / "packaging" / "common" / "FFMPEG-SOURCE-OFFER.txt", ffmpeg_out / "SOURCE-OFFER.txt")

    packages = copy_distribution_licenses(out)
    (out / "BUNDLED-PYTHON-PACKAGES.txt").write_text("\n".join(sorted(packages)) + "\n", encoding="utf-8")
    print(f"Staged licences for {len(packages)} Python distributions into {out}")


if __name__ == "__main__":
    main()
