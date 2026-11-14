"""Write BUILD_MANIFEST-<platform>.json, or merge platform manifests.

Run with the build environment's Python (it imports torch/PySide6 to record
their exact versions).

  python write_manifest.py --platform windows-x64 --artifacts DIR --ffmpeg-report F \
      --tool "PyInstaller=6.22.3" --tool "Inno Setup=6.x" --output DIR/BUILD_MANIFEST-windows-x64.json
  python write_manifest.py --merge DIR/BUILD_MANIFEST-*.json --output DIR/BUILD_MANIFEST.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ARTIFACT_SUFFIXES = (".exe", ".zip", ".AppImage", ".deb")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    names = ["torch", "numpy", "PySide6-Essentials", "shiboken6", "psutil", "nvidia-ml-py",
             "pyinstaller", "pyinstaller-hooks-contrib"]
    versions = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            continue
    for dist in md.distributions():
        name = dist.metadata["Name"] or ""
        if name.startswith("nvidia-") and name != "nvidia-ml-py":
            versions[name] = dist.version
    return dict(sorted(versions.items()))


def model_sources() -> list[dict[str, object]]:
    from app.engines.realesrgan import REALESRGAN_MODELS
    from app.engines.rife import RIFE_MODEL_URL

    models: list[dict[str, object]] = [{
        "name": "Practical-RIFE v4.25 flownet.pkl",
        "url": RIFE_MODEL_URL,
        "sha256": "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2",
        "size_bytes": 24_636_301,
        "checksum_source": "Hugging Face LFS object SHA-256 (X-Linked-ETag) of the pinned revision",
        "bundled": False,
    }]
    for info in REALESRGAN_MODELS.values():
        models.append({
            "name": info["name"], "url": info["url"], "sha256": info["sha256"],
            "size_bytes": info["min_bytes"],
            "checksum_source": "maintainer-pinned from the official GitHub release asset "
                               "(upstream publishes no checksum)",
            "bundled": False,
        })
    return models


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from PySide6 import __version__ as pyside_version

    from app.version import APP_CODENAME, APP_VERSION

    ffmpeg = json.loads(Path(args.ffmpeg_report).read_text(encoding="utf-8")) if args.ffmpeg_report else None
    # Container builds work on a copy without .git; the host passes these in.
    commit = os.environ.get("AUS_GIT_COMMIT") or git("rev-parse", "HEAD")
    branch = os.environ.get("AUS_GIT_BRANCH") or git("rev-parse", "--abbrev-ref", "HEAD")
    dirty_env = os.environ.get("AUS_GIT_DIRTY")
    dirty = dirty_env == "1" if dirty_env is not None else bool(git("status", "--porcelain"))
    artifacts = []
    if args.artifacts:
        for path in sorted(Path(args.artifacts).iterdir()):
            if path.is_file() and path.name.endswith(ARTIFACT_SUFFIXES):
                artifacts.append({"file": path.name, "size_bytes": path.stat().st_size,
                                  "sha256": sha256(path)})
    tools = dict(item.split("=", 1) for item in args.tool)
    return {
        "application": "AI Upscaler Studio",
        "version": APP_VERSION,
        "codename": APP_CODENAME,
        "git_commit": commit,
        "git_branch": branch,
        "git_worktree_dirty": dirty,
        "built_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "build_platform": {
            "target": args.platform,
            "system": platform.platform(),
            "machine": platform.machine(),
            "build_environment": args.environment,
        },
        "python_version": platform.python_version(),
        "pytorch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_arch_list": torch.cuda.get_arch_list() if torch.version.cuda else [],
        },
        "pyside6_version": pyside_version,
        "python_packages": package_versions(),
        "ffmpeg": ffmpeg,
        "models": model_sources(),
        "packaging_tools": tools,
        "code_signing": "unsigned",
        "artifacts": artifacts,
    }


def merge(paths: list[Path]) -> dict[str, object]:
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    first = manifests[0]
    return {
        "application": first["application"],
        "version": first["version"],
        "codename": first["codename"],
        "git_commit": first["git_commit"],
        "code_signing": "unsigned",
        "platforms": {m["build_platform"]["target"]: m for m in manifests},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform")
    parser.add_argument("--environment", default="")
    parser.add_argument("--artifacts")
    parser.add_argument("--ffmpeg-report")
    parser.add_argument("--tool", action="append", default=[])
    parser.add_argument("--merge", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = merge(args.merge) if args.merge else build_manifest(args)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
