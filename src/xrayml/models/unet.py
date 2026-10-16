"""U-Net from scratch (revision plan 1.4): 4 levels down, bottleneck, 4 levels up with
concatenated skips. ``base`` sets the width (64 in the paper; 16 is plenty for CI)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from xrayml.models.layers import Normalize
from xrayml.registry import MODELS


def _norm(c: int, kind: str) -> nn.Module:
    return nn.GroupNorm(min(8, c), c) if kind == "group" else nn.BatchNorm2d(c)


def double_conv(c_in: int, c_out: int, norm: str = "batch") -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        _norm(c_out, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        _norm(c_out, norm),
        nn.ReLU(inplace=True),
    )


@MODELS.register("unet")
class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = 1, base: int = 32, norm: str = "batch") -> None:
        super().__init__()
        w = [base * 2**i for i in range(5)]
        self.norm_in = Normalize()
        self.downs = nn.ModuleList(
            [double_conv(in_ch, w[0], norm)] + [double_conv(w[i], w[i + 1], norm) for i in range(3)]
        )
        self.bottleneck = double_conv(w[3], w[4], norm)
        # bilinear upsampling + conv: avoids transposed-conv checkerboard artefacts
        self.ups = nn.ModuleList([nn.Conv2d(w[i + 1], w[i], 1) for i in reversed(range(4))])
        self.decs = nn.ModuleList([double_conv(w[i] * 2, w[i], norm) for i in reversed(range(4))])
        self.out = nn.Conv2d(w[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # H, W divisible by 16
        x = self.norm_in(x)
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips), strict=True):
            x = up(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
            x = dec(torch.cat([skip, x], dim=1))  # skip connection = concatenation
        return self.out(x)  # logits (B, n_classes, H, W)
