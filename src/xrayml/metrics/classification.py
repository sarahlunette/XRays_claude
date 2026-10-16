"""Diagnostic statistics (revision plan 2.2, 2.7, 2.8). numpy + sklearn only.

Always say at which *level* a number is computed (image, study, patient) and give an
interval: bootstrap over *groups* (patients), never over images — images of one patient are
correlated, so image-level resampling gives intervals that are too narrow.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import roc_auc_score


def confusion_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= thr
    tp = int((pred & (y == 1)).sum())
    fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def ppv_at_prevalence(sens: float, spec: float, prevalence: float) -> float:
    """Bayes: what a site with another prevalence will see. (0.9, 0.9, 5%) -> PPV ≈ 32%."""
    return sens * prevalence / (sens * prevalence + (1 - spec) * (1 - prevalence))


def wilson_ci(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: well-behaved near 0 and 1 and on small test sets."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """ROC-AUC, or None when only one class is present (instead of crashing)."""
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def group_bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float | None],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI, resampling *groups* (patients) with replacement."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        v = metric(y[idx], p[idx])
        if v is not None and np.isfinite(v):
            stats.append(v)
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE = Σ_b (n_b/N)·|accuracy_b − confidence_b| with equal-width bins on P(positive)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)
