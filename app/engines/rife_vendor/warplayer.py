from __future__ import annotations

import torch
from torch.nn import functional as F

_GRID_CACHE: dict[tuple[str, str, int, int, int], torch.Tensor] = {}


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp an image using pixel-space optical flow (Practical-RIFE)."""
    batch, _, height, width = flow.shape
    key = (str(flow.device), str(flow.dtype), batch, height, width)
    grid = _GRID_CACHE.get(key)
    if grid is None:
        horizontal = torch.linspace(
            -1.0, 1.0, width, device=flow.device, dtype=flow.dtype
        ).view(1, 1, 1, width).expand(batch, -1, height, -1)
        vertical = torch.linspace(
            -1.0, 1.0, height, device=flow.device, dtype=flow.dtype
        ).view(1, 1, height, 1).expand(batch, -1, -1, width)
        grid = torch.cat((horizontal, vertical), dim=1)
        if len(_GRID_CACHE) >= 16:
            _GRID_CACHE.pop(next(iter(_GRID_CACHE)))
        _GRID_CACHE[key] = grid

    normalized_flow = torch.cat(
        (
            flow[:, 0:1] / max((image.shape[3] - 1.0) / 2.0, 1.0),
            flow[:, 1:2] / max((image.shape[2] - 1.0) / 2.0, 1.0),
        ),
        dim=1,
    )
    sample_grid = (grid + normalized_flow).permute(0, 2, 3, 1)
    return F.grid_sample(
        image,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
