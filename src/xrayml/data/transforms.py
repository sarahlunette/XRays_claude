"""Train-time augmentation for X-rays (revision plan 1.5).

Applied *after* the shared letterbox (``preprocess.letterbox``), so the validation path is
exactly the serving path. Only label-preserving, anatomically plausible ops: small affine,
exposure (brightness/contrast/gamma), noise. No vertical flips, no cutout (it can erase the
only fracture), horizontal flip off by default (laterality markers).
"""

from __future__ import annotations

import albumentations as A
import cv2

from xrayml.config import AugmentConfig, Task


def build_train_augment(cfg: AugmentConfig, task: Task) -> A.Compose:
    ops: list = []
    if cfg.hflip_p > 0:
        ops.append(A.HorizontalFlip(p=cfg.hflip_p))
    ops += [
        A.Affine(
            rotate=(-cfg.rotate_deg, cfg.rotate_deg),
            scale=cfg.scale,
            translate_percent=(-0.05, 0.05),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.7,
        ),
        A.RandomBrightnessContrast(cfg.brightness_contrast, cfg.brightness_contrast, p=0.5),
        A.RandomGamma(gamma_limit=cfg.gamma, p=0.3),
        A.GaussNoise(std_range=(0.005, 0.02), p=cfg.noise_p),
    ]
    bbox_params = None
    if task == "detection":
        # pascal_voc = xyxy in pixels. Boxes mostly rotated/cropped out are dropped.
        bbox_params = A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3)
    return A.Compose(ops, bbox_params=bbox_params)
