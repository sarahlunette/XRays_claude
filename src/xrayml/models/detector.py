"""Fracture detectors on torchvision (revision plan 1.3).

* ``fasterrcnn``: two-stage (backbone + FPN -> RPN -> RoIAlign -> RoI head). Strong on small
  objects; ``mobilenet_v3`` variant for CPU/CI, ``resnet50_fpn_v2`` for real training.
* ``retinanet``: one-stage + focal loss (the dense-anchor imbalance case).

Torchvision detectors take a *list* of 3-channel images in [0, 1] and do their own resize
(``min_size``/``max_size``) and normalisation. We feed letterboxed 1-channel images, repeat
them to 3 channels, and set ``min_size = max_size = image_size`` so the internal transform
is a no-op and box coordinates stay in our letterbox frame.

Licensing note for a commercial device: torchvision is BSD-3; Ultralytics YOLO is AGPL-3.0.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from xrayml.registry import MODELS

_FASTER = {"mobilenet_v3": fasterrcnn_mobilenet_v3_large_fpn, "resnet50": fasterrcnn_resnet50_fpn_v2}


class _GrayToRGB(nn.Module):
    """Wrap a torchvision detector so it accepts 1-channel images."""

    def __init__(self, det: nn.Module) -> None:
        super().__init__()
        self.det = det

    def forward(self, images: list[torch.Tensor], targets: list[dict] | None = None):
        images = [im.expand(3, -1, -1) if im.shape[0] == 1 else im for im in images]
        return self.det(images, targets)


@MODELS.register("fasterrcnn")
def build_fasterrcnn(
    backbone: str = "mobilenet_v3",
    pretrained: bool = False,
    num_classes: int = 2,  # background + fracture
    image_size: int = 512,
    small_anchors: bool = True,
    trainable_backbone_layers: int = 3,
) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    model = _FASTER[backbone](
        weights=weights,
        weights_backbone="DEFAULT" if pretrained else None,
        min_size=image_size,
        max_size=image_size,
        trainable_backbone_layers=trainable_backbone_layers if pretrained else None,
        box_detections_per_img=20,
        box_score_thresh=0.001,  # keep low-score boxes: the operating point decides, not a hidden constant
    )
    if small_anchors and backbone == "mobilenet_v3":
        # Hairline fractures are small and elongated: smaller anchors, extreme aspect ratios.
        n_levels = len(model.rpn.anchor_generator.sizes)
        sizes = ((16, 32), (48, 96), (128, 256), (256, 512))[:n_levels]
        ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * n_levels
        model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
        from torchvision.models.detection.rpn import RPNHead

        in_ch = model.backbone.out_channels
        model.rpn.head = RPNHead(in_ch, model.rpn.anchor_generator.num_anchors_per_location()[0])
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return _GrayToRGB(model)


@MODELS.register("retinanet")
def build_retinanet(pretrained: bool = False, num_classes: int = 2, image_size: int = 512) -> nn.Module:
    model = retinanet_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None,
        weights_backbone="DEFAULT" if pretrained else None,
        min_size=image_size,
        max_size=image_size,
    )
    if num_classes != 91:
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead

        head = model.head.classification_head
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=head.conv[0][0].in_channels, num_anchors=head.num_anchors, num_classes=num_classes
        )
    return _GrayToRGB(model)


def image_scores(detections: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    """Image-level suspicion score = max box score (0 if no box). Feeds ROC/AUC + operating point."""
    return torch.tensor([float(d["scores"].max()) if len(d["scores"]) else 0.0 for d in detections])
