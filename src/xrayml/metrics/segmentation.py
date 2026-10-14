"""Segmentation metrics (revision plan 1.4). Empty masks are scored explicitly: both empty ->
1.0 (correct "nothing here"), one empty -> 0.0. Always state this convention in a report."""

from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    return float(2 * (pred & gt).sum() / s)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    return float((pred & gt).sum() / union)
