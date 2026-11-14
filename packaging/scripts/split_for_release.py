"""Prepare GitHub Release uploads: split files larger than GitHub's 2 GiB asset limit.

Files up to the limit are copied unchanged; larger files are split into
``<name>.part1``, ``<name>.part2``… of at most ``--chunk`` bytes. The output
SHA256SUMS.txt lists every uploaded file *and* every reassembled original, so
users can verify both the parts and the joined file.

Join on Windows (cmd):   copy /b "<name>.part1" + "<name>.part2" "<name>"
Join on Linux/macOS:     cat <name>.part1 <name>.part2 > <name>

Usage: python split_for_release.py <dist/release> <out-dir> [--chunk 1900000000]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

GITHUB_ASSET_LIMIT = 2 * 1024**3
ARTIFACT_SUFFIXES = (".exe", ".zip", ".AppImage", ".deb", ".json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split(path: Path, out_dir: Path, chunk: int) -> list[Path]:
    parts: list[Path] = []
    with path.open("rb") as source:
        index = 1
        while True:
            part = out_dir / f"{path.name}.part{index}"
            written = 0
            with part.open("wb") as target:
                while written < chunk:
                    data = source.read(min(1 << 20, chunk - written))
                    if not data:
                        break
                    target.write(data)
                    written += len(data)
            if written == 0:
                part.unlink()
                break
            parts.append(part)
            index += 1
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--chunk", type=int, default=1_900_000_000)
    args = parser.parse_args()
    if args.chunk >= GITHUB_ASSET_LIMIT:
        raise SystemExit("Chunk size must be below GitHub's 2 GiB asset limit.")
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    whole: list[str] = []
    uploads: list[Path] = []
    for path in sorted(args.source.iterdir()):
        if not path.is_file() or not path.name.endswith(ARTIFACT_SUFFIXES):
            continue
        whole.append(f"{sha256(path)}  {path.name}")
        if path.stat().st_size <= args.chunk:
            target = args.out / path.name
            shutil.copy2(path, target)
            uploads.append(target)
        else:
            uploads.extend(split(path, args.out, args.chunk))

    part_lines = [f"{sha256(p)}  {p.name}" for p in uploads if ".part" in p.name]
    text = "# Complete files (verify after joining split parts)\n" + "\n".join(whole) + "\n"
    if part_lines:
        text += "\n# Individual download parts\n" + "\n".join(part_lines) + "\n"
    (args.out / "SHA256SUMS.txt").write_text(text, encoding="utf-8", newline="\n")
    for p in uploads:
        print(f"{p.stat().st_size:>14,}  {p.name}")
    print(text)


if __name__ == "__main__":
    main()
