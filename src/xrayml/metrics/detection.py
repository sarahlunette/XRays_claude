"""Detection metrics from scratch (revision plan 1.3, 2.8): IoU, NMS, FROC, CPM.

FROC (lesion-level sensitivity vs mean false positives per image) is the metric a clinical
team reasons with — "how many fractures missed, how many false alarms per exam" — and the
detection operating point is chosen on that curve. The "hit" criterion is looser than COCO
(IoU >= 0.2 here) because a roughly placed box already directs the radiologist's attention.
"""

from __future__ import annotations

import numpy as np
import torch


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a (N,4), b (M,4) in xyxy -> (N, M)."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter).clamp(min=1e-9)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
    """Greedy NMS. Production code uses torchvision.ops.batched_nms; this one is for review/tests."""
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)


def _iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum((box[2] - box[0]) * (box[3] - box[1]) + area - inter, 1e-9)


def froc(
    preds: list[tuple[str, float, np.ndarray]],
    gts: dict[str, np.ndarray],
    n_images: int,
    hit_iou: float = 0.2,
    fp_rates: tuple[float, ...] = (0.125, 0.25, 0.5, 1, 2, 4, 8),
) -> dict:
    """preds: (image_id, score, box xyxy); gts: image_id -> (N,4). Returns curve + CPM.

    A second hit on an already-found lesion counts as neither TP nor FP.
    """
    n_lesions = sum(len(b) for b in gts.values())
    found = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tps = fps = 0
    curve_fp, curve_sens, curve_thr = [0.0], [0.0], [float("inf")]
    for img, score, box in sorted(preds, key=lambda t: -t[1]):  # sweep threshold high -> low
        g = gts.get(img, np.zeros((0, 4)))
        if len(g):
            ious = _iou_np(np.asarray(box, float), np.asarray(g, float))
            j = int(np.argmax(ious))
            if ious[j] >= hit_iou:
                if not found[img][j]:
                    found[img][j] = True
                    tps += 1
            else:
                fps += 1
        else:
            fps += 1
        curve_fp.append(fps / max(n_images, 1))
        curve_sens.append(tps / max(n_lesions, 1))
        curve_thr.append(float(score))
    fp_axis, sens = np.array(curve_fp), np.array(curve_sens)
    points = {float(r): float(sens[fp_axis <= r].max(initial=0.0)) for r in fp_rates}
    return {
        "fp_per_image": fp_axis.tolist(),
        "sensitivity": sens.tolist(),
        "thresholds": curve_thr,
        "sens_at_fp": points,
        "cpm": float(np.mean(list(points.values()))),  # LUNA16 Competition Performance Metric
        "n_lesions": n_lesions,
    }


def threshold_at_fp_rate(froc_out: dict, max_fp_per_image: float) -> tuple[float, float]:
    """Lowest score threshold whose FP/image stays within budget -> (threshold, sensitivity)."""
    fp = np.array(froc_out["fp_per_image"])
    ok = np.flatnonzero(fp <= max_fp_per_image)
    i = int(ok[-1])
    return float(froc_out["thresholds"][i]), float(froc_out["sensitivity"][i])
