"""Generate the AI Upscaler Studio application icons (PNG set + Windows .ico).

The icon is a simple original mark: a small pixel grid growing into a larger
one with an upward arrow. It deliberately contains no NVIDIA logo or other
third-party trademark. Output is committed under packaging/common/icons so
builds do not need Pillow.

Usage: python packaging/scripts/make_icons.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "packaging" / "common" / "icons"
BACKGROUND = (16, 19, 21, 255)
BORDER = (42, 51, 56, 255)
GREEN = (118, 185, 0, 255)
LIGHT = (168, 227, 78, 255)
SIZES = (16, 24, 32, 48, 64, 128, 256, 512)


def draw_icon(size: int = 1024) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    s = size / 1024
    draw.rounded_rectangle((40 * s, 40 * s, 984 * s, 984 * s), radius=200 * s,
                           fill=BACKGROUND, outline=BORDER, width=max(1, round(16 * s)))
    # Small 2x2 source grid (bottom left).
    cell = 70 * s
    gap = 18 * s
    x0, y0 = 170 * s, 640 * s
    for row in range(2):
        for col in range(2):
            x = x0 + col * (cell + gap)
            y = y0 + row * (cell + gap)
            draw.rounded_rectangle((x, y, x + cell, y + cell), radius=12 * s, fill=GREEN)
    # Large 3x3 output grid (top right).
    cell = 120 * s
    gap = 22 * s
    x0, y0 = 430 * s, 170 * s
    for row in range(3):
        for col in range(3):
            x = x0 + col * (cell + gap)
            y = y0 + row * (cell + gap)
            draw.rounded_rectangle((x, y, x + cell, y + cell), radius=18 * s, fill=LIGHT)
    # Upward arrow from the small grid to the large one.
    width = round(46 * s)
    draw.line((360 * s, 660 * s, 560 * s, 470 * s), fill=GREEN, width=width)
    draw.polygon([(600 * s, 420 * s), (470 * s, 440 * s), (585 * s, 555 * s)], fill=GREEN)
    return image


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    master = draw_icon()
    for size in SIZES:
        master.resize((size, size), Image.Resampling.LANCZOS).save(OUT / f"ai-upscaler-studio-{size}.png")
    master.resize((256, 256), Image.Resampling.LANCZOS).save(OUT / "ai-upscaler-studio.png")
    master.save(OUT / "ai-upscaler-studio.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    print(f"Icons written to {OUT}")


if __name__ == "__main__":
    main()
