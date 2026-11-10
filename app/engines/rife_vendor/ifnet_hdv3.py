from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from app.engines.rife_vendor.warplayer import warp


def _conv(in_planes: int, out_planes: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, 3, 1, dilation, dilation=dilation, groups=1
        )
        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes: int, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(in_planes, channels // 2, 2),
            _conv(channels // 2, channels, 2),
        )
        self.convblock = nn.Sequential(*(ResConv(channels) for _ in range(8)))
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(channels, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(
        self, x: torch.Tensor, flow: torch.Tensor | None = None, scale: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = F.interpolate(
                flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            ) / scale
            x = torch.cat((x, flow), dim=1)
        feat = self.convblock(self.conv0(x))
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    """Practical-RIFE HDv3/v4.25 IFNet inference network."""

    def __init__(self) -> None:
        super().__init__()
        self.block0 = IFBlock(7 + 8, 192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, 128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, 96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, 64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, 32)
        self.encode = Head()

    @torch.no_grad()
    def interpolate(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float,
        scale_list: tuple[float, float, float, float, float],
    ) -> torch.Tensor:
        if img0.shape != img1.shape or img0.ndim != 4 or img0.shape[1] != 3:
            raise ValueError("RIFE inputs must have identical B×3×H×W tensor shapes.")
        imgs = torch.cat((img0, img1), dim=1)
        timestep_tensor = (imgs[:, :1].clone() * 0 + 1) * float(timestep)
        feat0 = self.encode(img0)
        feat1 = self.encode(img1)
        warped0, warped1 = img0, img1
        flow = None
        mask = None
        feat = None
        blocks = (self.block0, self.block1, self.block2, self.block3, self.block4)

        for index, block in enumerate(blocks):
            if flow is None:
                flow, mask, feat = block(
                    torch.cat((img0, img1, feat0, feat1, timestep_tensor), dim=1),
                    None,
                    scale_list[index],
                )
            else:
                warped_feat0 = warp(feat0, flow[:, :2])
                warped_feat1 = warp(feat1, flow[:, 2:4])
                delta_flow, mask, feat = block(
                    torch.cat(
                        (warped0, warped1, warped_feat0, warped_feat1, timestep_tensor, mask, feat),
                        dim=1,
                    ),
                    flow,
                    scale_list[index],
                )
                flow = flow + delta_flow
            warped0 = warp(img0, flow[:, :2])
            warped1 = warp(img1, flow[:, 2:4])

        blend = torch.sigmoid(mask)
        return warped0 * blend + warped1 * (1.0 - blend)
