"""Create a ZIP64 archive of a directory under a chosen top-level folder name.

PowerShell 5.1 Compress-Archive cannot handle files larger than 2 GB, which the
CUDA PyTorch libraries can approach, so the portable ZIP is built here.

Usage: python make_zip.py <source-dir> <output.zip> --root "AI Upscaler Studio"
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", required=True)
    parser.add_argument("--level", type=int, default=6)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".zip.part")
    count = 0
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=args.level, allowZip64=True) as archive:
        for path in sorted(args.source.rglob("*")):
            arcname = Path(args.root) / path.relative_to(args.source)
            if path.is_dir():
                continue
            info = zipfile.ZipInfo.from_file(path, arcname.as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            if os.access(path, os.X_OK) and os.name != "nt":
                info.external_attr = (0o755 << 16)
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    target.write(chunk)
            count += 1
    partial.replace(args.output)
    print(f"{args.output} ({count} files, {args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
