"""Download a pinned third-party binary and verify its SHA-256.

Entries come from packaging/common/external_binaries.json. Files are cached in
build/downloads/ and re-verified on every use; a mismatching file is deleted
and the build fails. Nothing downloaded is executed by this script.

Usage: python packaging/scripts/fetch_binary.py <entry-key> [--dest DIR] [--extract DIR]
Prints the path of the verified archive (or the extracted directory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "packaging" / "common" / "external_binaries.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(key: str, dest: Path) -> Path:
    entry = json.loads(MANIFEST.read_text(encoding="utf-8"))[key]
    url, expected = entry["url"], entry["sha256"].lower()
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / url.rsplit("/", 1)[-1]
    if target.is_file() and sha256(target) == expected:
        return target
    target.unlink(missing_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {url}", file=sys.stderr)
    request = urllib.request.Request(url, headers={"User-Agent": "ai-upscaler-studio-build"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    actual = sha256(partial)
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 mismatch for {url}\n  expected {expected}\n  actual   {actual}")
    partial.replace(target)
    return target


def _safe_members(names: list[str], root: Path) -> None:
    for name in names:
        resolved = (root / name).resolve()
        if root.resolve() not in resolved.parents and resolved != root.resolve():
            raise SystemExit(f"Refusing to extract unsafe archive member: {name}")


def extract(archive: Path, out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            _safe_members(zf.namelist(), out_dir)
            zf.extractall(out_dir)
    else:
        with tarfile.open(archive) as tf:
            _safe_members(tf.getnames(), out_dir)
            tf.extractall(out_dir, filter="data")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    default_dest = Path(os.environ.get("AUS_DOWNLOAD_CACHE", ROOT / "build" / "downloads"))
    parser.add_argument("--dest", type=Path, default=default_dest)
    parser.add_argument("--extract", type=Path, default=None)
    args = parser.parse_args()
    archive = fetch(args.key, args.dest)
    print(extract(archive, args.extract) if args.extract else archive)


if __name__ == "__main__":
    main()
