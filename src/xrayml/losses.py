"""Losses (revision plan 1.2, 1.4). All take *logits* — never apply sigmoid before them."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from xrayml.registry import LOSSES


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """FL(p_t) = -α_t (1 - p_t)^γ log(p_t). γ=0 gives BCE; at p_t=0.9, γ=2 cuts the loss 100x."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - Dice per image, averaged. ``eps`` keeps empty masks finite (Dice undefined at 0/0)."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    denom = p.sum((2, 3)) + target.sum((2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def tversky_loss(
    logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, eps: float = 1.0
) -> torch.Tensor:
    """β > α penalises false negatives more: favours recall (fewer missed fracture pixels)."""
    p = torch.sigmoid(logits)
    tp = (p * target).sum((2, 3))
    fp = (p * (1 - target)).sum((2, 3))
    fn = ((1 - p) * target).sum((2, 3))
    return 1 - ((tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


class _Fn(nn.Module):
    def __init__(self, fn, **kw) -> None:
        super().__init__()
        self.fn, self.kw = fn, kw

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fn(logits, target, **self.kw)


@LOSSES.register("bce")
def _bce(pos_weight: float | None = None) -> nn.Module:
    """Weighted BCE: pos_weight = n_neg / n_pos multiplies the positive term."""
    return nn.BCEWithLogitsLoss(pos_weight=None if pos_weight is None else torch.tensor([pos_weight]))


@LOSSES.register("focal")
def _focal(alpha: float = 0.25, gamma: float = 2.0) -> nn.Module:
    return _Fn(focal_loss, alpha=alpha, gamma=gamma)


@LOSSES.register("dice")
def _dice() -> nn.Module:
    return _Fn(soft_dice_loss)


@LOSSES.register("bce_dice")
def _bce_dice(dice_weight: float = 1.0) -> nn.Module:
    """The segmentation default: BCE for stable gradients + Dice for imbalance."""

    def fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target) + dice_weight * soft_dice_loss(
            logits, target
        )

    return _Fn(fn)


@LOSSES.register("tversky")
def _tversky(alpha: float = 0.3, beta: float = 0.7) -> nn.Module:
    return _Fn(tversky_loss, alpha=alpha, beta=beta)


@LOSSES.register("detector_builtin")
def _detector_builtin() -> nn.Module:
    """Torchvision detectors compute their own losses (RPN + RoI); this is a placeholder."""
    return nn.Identity()
