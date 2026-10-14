"""Grad-CAM (revision plan 1.2 explainability): is the model looking at the bone, or at the
"L"/"R" marker, a cast, a ruler? Shortcut learning shows up here before it shows up in a
site with different markers. Uses forward/backward hooks (plan 5.6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        self._h = [
            layer.register_forward_hook(self._save_act),
            layer.register_full_backward_hook(self._save_grad),
        ]

    def _save_act(self, _m: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        self.acts = out.detach()

    def _save_grad(self, _m: nn.Module, _gin: tuple, gout: tuple) -> None:
        self.grads = gout[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = 0) -> torch.Tensor:
        """x (B,1,H,W) -> heatmap (B,H,W) in [0,1]."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad(set_to_none=True)
        logits[:, class_idx].sum().backward()
        if self.acts is None or self.grads is None:
            raise RuntimeError("hooks did not fire: is the target layer part of the forward pass?")
        weights = self.grads.mean((2, 3), keepdim=True)  # channel importance = mean gradient
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin((1, 2), keepdim=True)
        return cam / cam.amax((1, 2), keepdim=True).clamp(min=1e-8)

    def remove(self) -> None:
        for h in self._h:
            h.remove()
