"""Choosing operating points (revision plan 2.3, 2.8) — a core part of this job.

Rule: choose on the *tuning* split (val), confirm on the untouched *test* split, never tune
on test. The output is an ``OperatingPointSet`` artefact, versioned separately from weights.

Strategies:
* ``sensitivity``: highest threshold reaching a target sensitivity (fractures: a miss costs
  more than a false alarm), then report specificity / FP per image.
* ``youden``: maximise Se + Sp − 1.
Per body part thresholds are only fitted when the subgroup has enough positives; otherwise
the global ("all") threshold applies — small subgroups give unstable thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from xrayml.artifacts import OperatingPoint, OperatingPointSet
from xrayml.metrics.classification import confusion_at, wilson_ci


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.9) -> float:
    _, tpr, thr = roc_curve(y, p)  # thresholds in decreasing order
    i = int(np.argmax(tpr >= target))  # first = highest threshold reaching the target
    return float(min(thr[i], 1.0))


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(min(thr[int(np.argmax(tpr - fpr))], 1.0))


def _point(
    y: np.ndarray, p: np.ndarray, thr: float, target: float, body_part: str, fp_per_image: float | None
) -> OperatingPoint:
    c = confusion_at(y, p, thr)
    return OperatingPoint(
        body_part=body_part,
        threshold=thr,
        target_sensitivity=target,
        validated_sensitivity=c["sensitivity"],
        sensitivity_ci=wilson_ci(c["tp"], c["tp"] + c["fn"]),
        specificity=c["specificity"],
        fp_per_image=fp_per_image,
        n_positive=c["tp"] + c["fn"],
        n_negative=c["tn"] + c["fp"],
    )


def select_operating_points(
    preds: pd.DataFrame,
    target_sensitivity: float = 0.9,
    strategy: str = "sensitivity",
    per_body_part: bool = True,
    min_positives: int = 30,
    version: str = "op-local",
    model_version: str = "local",
    dataset_version: str = "untracked",
    tuning_split: str = "val",
    box_preds: pd.DataFrame | None = None,
) -> OperatingPointSet:
    """``preds`` columns: y, score, body_part (one row per image, tuning split only).

    ``box_preds`` (detection only): one row per predicted box with columns image_id, score,
    is_tp, body_part — used to report false positives per image at each threshold.
    """
    y, p = preds.y.to_numpy(), preds.score.to_numpy()

    def choose(yy: np.ndarray, pp: np.ndarray) -> float:
        return (
            threshold_for_sensitivity(yy, pp, target_sensitivity)
            if strategy == "sensitivity"
            else youden_threshold(yy, pp)
        )

    def fp_img(thr: float, part: str | None = None) -> float | None:
        if box_preds is None:
            return None
        b = box_preds if part is None else box_preds[box_preds.body_part == part]
        n_img = len(preds) if part is None else int((preds.body_part == part).sum())
        return float(((b.score >= thr) & ~b.is_tp.astype(bool)).sum() / max(n_img, 1))

    thr_all = choose(y, p)
    points = [_point(y, p, thr_all, target_sensitivity, "all", fp_img(thr_all))]
    if per_body_part:
        for part, g in preds.groupby("body_part"):
            if g.y.sum() >= min_positives and (g.y == 0).sum() >= min_positives:
                yy, pp = g.y.to_numpy(), g.score.to_numpy()
                t = choose(yy, pp)
                points.append(_point(yy, pp, t, target_sensitivity, str(part), fp_img(t, str(part))))
    return OperatingPointSet(
        version=version,
        model_version=model_version,
        dataset_version=dataset_version,
        tuning_split=tuning_split,
        points=points,
    )


def confirm_on_test(ops: OperatingPointSet, test_preds: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen thresholds to the test split; one row per operating point."""
    rows = []
    for op in ops.points:
        g = test_preds if op.body_part == "all" else test_preds[test_preds.body_part == op.body_part]
        if g.empty:
            continue
        c = confusion_at(g.y.to_numpy(), g.score.to_numpy(), op.threshold)
        rows.append(
            {
                "body_part": op.body_part,
                "threshold": op.threshold,
                "n": len(g),
                "tuned_sensitivity": op.validated_sensitivity,
                "test_sensitivity": c["sensitivity"],
                "test_sensitivity_ci": wilson_ci(c["tp"], c["tp"] + c["fn"]),
                "test_specificity": c["specificity"],
            }
        )
    return pd.DataFrame(rows)
