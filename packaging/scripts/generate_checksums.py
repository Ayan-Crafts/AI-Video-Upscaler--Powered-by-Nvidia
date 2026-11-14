"""Write SHA256SUMS.txt for distribution artifacts (``sha256sum -c`` compatible).

Usage: python generate_checksums.py <artifact-dir> [--output SHA256SUMS.txt]
Existing SHA256SUMS*.txt and BUILD_MANIFEST*.json files are not hashed into
themselves; the manifest is listed so its integrity can also be checked.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ARTIFACT_SUFFIXES = (".exe", ".zip", ".AppImage", ".deb", ".json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", default="SHA256SUMS.txt")
    args = parser.parse_args()
    files = sorted(p for p in args.directory.iterdir()
                   if p.is_file() and p.name.endswith(ARTIFACT_SUFFIXES)
                   and not p.name.startswith("SHA256SUMS"))
    lines = [f"{sha256(p)}  {p.name}" for p in files]
    (args.directory / args.output).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
