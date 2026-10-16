"""Task metrics from prediction tables — one implementation for training logs and reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from xrayml.config import Task
from xrayml.metrics.classification import expected_calibration_error, safe_auc
from xrayml.metrics.detection import froc

MONITOR = {"classification": "auc", "detection": "cpm", "segmentation": "dice_pos"}


def froc_from_boxes(boxes: list[dict], hit_iou: float = 0.2) -> dict:
    preds = [
        (b["image_id"], float(s), bx) for b in boxes for bx, s in zip(b["boxes"], b["scores"], strict=True)
    ]
    gts = {b["image_id"]: np.asarray(b["gt"]).reshape(-1, 4) for b in boxes}
    return froc(preds, gts, n_images=len(boxes), hit_iou=hit_iou)


def box_table(boxes: list[dict], meta: pd.DataFrame, hit_iou: float = 0.2) -> pd.DataFrame:
    """One row per predicted box with is_tp (greedy by score, one hit per lesion)."""
    from xrayml.metrics.detection import _iou_np

    part = dict(zip(meta.image_id, meta.body_part, strict=True))
    rows = []
    for b in boxes:
        gt = np.asarray(b["gt"]).reshape(-1, 4)
        found = np.zeros(len(gt), bool)
        for bx, s in sorted(zip(b["boxes"], b["scores"], strict=True), key=lambda t: -t[1]):
            tp = False
            if len(gt):
                ious = _iou_np(np.asarray(bx, float), gt)
                j = int(np.argmax(ious))
                if ious[j] >= hit_iou and not found[j]:
                    found[j] = tp = True
            rows.append(
                {"image_id": b["image_id"], "score": float(s), "is_tp": tp, "body_part": part[b["image_id"]]}
            )
    return pd.DataFrame(rows, columns=["image_id", "score", "is_tp", "body_part"])


def task_metrics(task: Task, preds: pd.DataFrame, boxes: list[dict] | None = None) -> dict[str, float | None]:
    y, p = preds.y.to_numpy(), preds.score.to_numpy()
    out: dict[str, float | None] = {
        "n": float(len(preds)),
        "prevalence": float(y.mean()) if len(y) else None,
        "auc": safe_auc(y, p),
        "ap": float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else None,
    }
    if task == "classification":
        out["ece"] = expected_calibration_error(y, p)
    if task == "detection" and boxes is not None:
        fr = froc_from_boxes(boxes)
        out["cpm"] = fr["cpm"]
        out |= {f"sens@{k:g}fp": v for k, v in fr["sens_at_fp"].items()}
    if task == "segmentation":
        pos = preds[preds.y == 1]
        out["dice_pos"] = float(pos.dice.mean()) if len(pos) else None  # Dice on positive images only
        out["dice_all"] = float(preds.dice.mean())  # empty-vs-empty counts as 1.0
    return out
