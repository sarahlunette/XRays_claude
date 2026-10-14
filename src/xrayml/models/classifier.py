"""Image-level fracture classifier: backbone -> GAP -> dropout -> linear -> logit.

Any timm backbone works (``resnet18``, ``resnet50``, ``densenet121``, ``convnext_tiny``,
``efficientnet_b0``...). ``in_chans=1`` makes timm sum the pretrained RGB stem filters into
a 1-channel stem, so grayscale X-rays keep the ImageNet initialisation.
"""

from __future__ import annotations

from typing import cast

import timm
import torch
from torch import nn

from xrayml.models.layers import Normalize, bn_to_gn
from xrayml.registry import MODELS


@MODELS.register("timm_classifier")
class FractureClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        num_classes: int = 1,
        drop: float = 0.2,
        group_norm: bool = False,
    ) -> None:
        super().__init__()
        self.norm = Normalize()
        self.body = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=1)
        if group_norm:  # for tiny per-GPU batches when training from scratch (plan 1.1, GN)
            bn_to_gn(self.body)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(cast(int, self.body.num_features), num_classes))
        self.backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.norm(x)))  # raw logits: the loss applies the sigmoid

    def gradcam_target_layer(self) -> nn.Module:
        """Last conv stage, for Grad-CAM (``xrayml.explain.gradcam``)."""
        for name in ("layer4", "stages", "features", "blocks"):
            if hasattr(self.body, name):
                mod = getattr(self.body, name)
                return mod[-1] if isinstance(mod, nn.Sequential | nn.ModuleList) else mod
        raise AttributeError(f"no known last stage for {self.backbone_name}")
