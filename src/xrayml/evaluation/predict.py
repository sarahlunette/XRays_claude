"""Run a model over a loader and return one row per image (+ boxes for detection).

Used by both the trainer (validation every epoch) and ``xrayml evaluate`` so the numbers
in the training log and in the validation report come from the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from xrayml.config import Task
from xrayml.metrics.segmentation import dice_score


@torch.no_grad()  # no autograd graph: saves memory; model.eval() separately switches BN/Dropout
def predict(
    model: nn.Module,
    loader: DataLoader,
    task: Task,
    device: torch.device | str = "cpu",
    seg_threshold: float = 0.5,
) -> tuple[pd.DataFrame, list[dict]]:
    model.eval()
    df = loader.dataset.df  # type: ignore[attr-defined]
    rows: list[dict] = []
    boxes_out: list[dict] = []
    for images, targets, idx in loader:
        if task == "detection":
            dets = model([im.to(device) for im in images])
            for im, d, t, i in zip(images, dets, targets, idx.tolist(), strict=True):
                scores = d["scores"].float().cpu().numpy()
                rows.append(
                    {
                        "_i": i,
                        "score": float(scores.max()) if len(scores) else 0.0,
                        "n_gt": len(t["boxes"]),
                        "mean_intensity": float(im.mean()),
                        "std_intensity": float(im.std()),
                    }
                )
                boxes_out.append(
                    {
                        "image_id": df.image_id.iloc[i],
                        "boxes": d["boxes"].float().cpu().numpy(),
                        "scores": scores,
                        "gt": t["boxes"].numpy(),
                    }
                )
            continue
        logits = model(images.to(device)).float().cpu()
        # image statistics of the model input: the drift-monitoring reference (same as serving logs)
        stats = [{"mean_intensity": float(im.mean()), "std_intensity": float(im.std())} for im in images]
        if task == "classification":
            for lg, i, st in zip(logits[:, 0].tolist(), idx.tolist(), stats, strict=True):
                rows.append({"_i": i, "logit": lg, "score": float(1 / (1 + np.exp(-lg))), **st})
        else:  # segmentation: image score = max pixel probability; dice vs ground-truth mask
            probs = torch.sigmoid(logits)
            for pm, gt, i, st in zip(probs, targets, idx.tolist(), stats, strict=True):
                pred = (pm[0] >= seg_threshold).numpy()
                rows.append(
                    {
                        "_i": i,
                        "score": float(pm.max()),
                        "dice": dice_score(pred, gt[0].numpy() > 0.5),
                        "pred_area": int(pred.sum()),
                        **st,
                    }
                )
    out = pd.DataFrame(rows).sort_values("_i")
    meta_cols = [
        c
        for c in ["image_id", "group_id", "body_part", "site", "view", "hardware", "fractured", "split"]
        if c in df.columns
    ]
    meta = df.iloc[out._i.to_numpy()][meta_cols].reset_index(drop=True)
    out = pd.concat([meta, out.drop(columns="_i").reset_index(drop=True)], axis=1).rename(
        columns={"fractured": "y"}
    )
    return out, boxes_out
