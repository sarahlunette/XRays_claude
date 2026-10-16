"""Building blocks from revision plan chapter 1, each one small and unit-tested.

* ``Normalize``: intensity normalisation inside the model -> exported into ONNX (no skew).
* ``bn_to_gn`` / ``freeze_bn``: the small-batch BatchNorm problem (2-4 X-rays per GPU).
* ``Bottleneck`` (ResNet), ``SqueezeExcite``, ``MBConv`` (EfficientNet), ``DenseLayer``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Normalize(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: float = 0.5, std: float = 0.25) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))  # buffer: in state_dict, not trained
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def bn_to_gn(module: nn.Module, groups: int = 32) -> nn.Module:
    """Replace every BatchNorm2d by GroupNorm (batch-size independent, same at train/test)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            g = groups
            while c % g:  # GroupNorm needs channels divisible by groups
                g //= 2
            setattr(module, name, nn.GroupNorm(max(1, g), c))
        else:
            bn_to_gn(child, groups)
    return module


def freeze_bn(module: nn.Module) -> nn.Module:
    """Use stored running stats and stop BN gradients. Re-apply after every ``model.train()``."""
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)
    return module


class Bottleneck(nn.Module):
    """ResNet-50 block: 1x1 reduce -> 3x3 -> 1x1 expand (x4), plus identity/projection skip."""

    expansion = 4

    def __init__(self, c_in: int, c_mid: int, stride: int = 1) -> None:
        super().__init__()
        c_out = c_mid * self.expansion
        self.conv1, self.bn1 = nn.Conv2d(c_in, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid)
        self.conv2, self.bn2 = nn.Conv2d(c_mid, c_mid, 3, stride, 1, bias=False), nn.BatchNorm2d(c_mid)
        self.conv3, self.bn3 = nn.Conv2d(c_mid, c_out, 1, bias=False), nn.BatchNorm2d(c_out)
        self.shortcut: nn.Module = (
            nn.Identity()
            if stride == 1 and c_in == c_out
            else nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))  # gradient has a direct path: d(x + F(x))/dx = 1 + F'(x)


class SqueezeExcite(nn.Module):
    def __init__(self, c: int, hidden: int) -> None:
        super().__init__()
        self.fc1, self.fc2 = nn.Conv2d(c, hidden, 1), nn.Conv2d(hidden, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean((2, 3), keepdim=True)  # squeeze: global average pool
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(w))))  # excite: per-channel weight


class MBConv(nn.Module):
    """Inverted residual: thin -> wide (expand) -> depthwise -> SE -> thin (linear projection)."""

    def __init__(
        self, c_in: int, c_out: int, k: int = 3, stride: int = 1, expand: int = 6, drop_path: float = 0.0
    ) -> None:
        super().__init__()
        c_mid = c_in * expand
        self.use_skip = stride == 1 and c_in == c_out
        self.drop_path = drop_path
        layers: list[nn.Module] = []
        if expand != 1:
            layers += [nn.Conv2d(c_in, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.SiLU()]
        layers += [
            nn.Conv2d(c_mid, c_mid, k, stride, k // 2, groups=c_mid, bias=False),  # depthwise
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),
            SqueezeExcite(c_mid, max(1, c_in // 4)),
            nn.Conv2d(c_mid, c_out, 1, bias=False),  # projection, no activation: linear bottleneck
            nn.BatchNorm2d(c_out),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if not self.use_skip:
            return out
        if self.training and self.drop_path > 0:  # stochastic depth, per sample
            keep = (torch.rand(x.size(0), 1, 1, 1, device=x.device) > self.drop_path).to(out.dtype)
            out = out * keep / (1 - self.drop_path)
        return out + x


class DenseLayer(nn.Module):
    """DenseNet: pre-activation bottleneck whose output is *concatenated* to its input."""

    def __init__(self, c_in: int, growth: int = 32, bn_size: int = 4) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, bn_size * growth, 1, bias=False),
            nn.BatchNorm2d(bn_size * growth),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth, growth, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.f(x)], dim=1)


def conv_macs(h_out: int, w_out: int, c_in: int, c_out: int, k: int, groups: int = 1) -> int:
    """Multiply-accumulates of a conv layer (plan 1.1 FLOPs): H·W·C_out·(C_in/groups)·k²."""
    return h_out * w_out * c_out * (c_in // groups) * k * k
