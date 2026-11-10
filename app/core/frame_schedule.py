from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from math import floor


def as_rate(value: str | float | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        rate = value
    elif isinstance(value, int):
        rate = Fraction(value, 1)
    elif isinstance(value, float):
        rate = Fraction(str(value)).limit_denominator(1_000_000)
    else:
        text = value.strip()
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            rate = Fraction(int(numerator), int(denominator))
        else:
            rate = Fraction(text).limit_denominator(1_000_000)
    if rate <= 0:
        raise ValueError("Frame rates must be positive.")
    return rate


def intermediate_timesteps(
    source_fps: str | float | Fraction,
    target_fps: str | float | Fraction,
    source_pair_index: int,
) -> Iterator[tuple[Fraction, float]]:
    """Yield (target presentation time, RIFE timestep) within one CFR source pair.

    The output timestamps are computed with Fractions to avoid cumulative FPS
    drift. A target timestamp exactly on a source frame is deliberately omitted;
    the video runner writes that decoded source frame unchanged.
    """
    if source_pair_index < 0:
        raise ValueError("source_pair_index must be non-negative.")
    src = as_rate(source_fps)
    dst = as_rate(target_fps)
    start = Fraction(source_pair_index, 1) / src
    end = Fraction(source_pair_index + 1, 1) / src
    first_target_index = floor(start * dst) + 1
    target_index = first_target_index

    while True:
        timestamp = Fraction(target_index, 1) / dst
        if timestamp >= end:
            break
        alpha = float((timestamp - start) / (end - start))
        if 0.0 < alpha < 1.0:
            yield timestamp, alpha
        target_index += 1


def nearest_source_frame_index(
    timestamp: Fraction,
    source_fps: str | float | Fraction,
    frame_count: int,
) -> int:
    """Return the nearest CFR source frame index for non-interpolated output."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive.")
    src = as_rate(source_fps)
    index = round(float(timestamp * src))
    return max(0, min(frame_count - 1, index))
