"""Automated release gate (revision plan 2.7 non-inferiority, 8.1 step 3, 8.7 PCCP).

Two families of checks, both pre-agreed (that is what makes a PCCP-style fast path possible):

1. Absolute acceptance criteria on the frozen test split: lower bound of the sensitivity CI
   at the shipped threshold >= ``min_sensitivity_lcb``; specificity >= ``min_specificity``.
2. Non-inferiority vs the production model on the *same* images: lower bound of the paired
   bootstrap CI of (candidate − production) sensitivity >= −margin, and specificity drop
   within tolerance.

The gate returns every check with its value, so the list goes straight into the release
record. CI fails the job when ``passed`` is False.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from xrayml.artifacts import OperatingPointSet
from xrayml.metrics.classification import confusion_at, wilson_ci


@dataclass
class Check:
    name: str
    passed: bool
    value: float
    limit: float
    detail: str = ""


@dataclass
class GateResult:
    passed: bool
    checks: list[Check] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}


def _apply(preds: pd.DataFrame, ops: OperatingPointSet) -> np.ndarray:
    """Decision per image using its body part's threshold (falls back to 'all')."""
    thr = preds.body_part.map(lambda bp: ops.threshold_for(bp).threshold).to_numpy()
    return preds.score.to_numpy() >= thr


def paired_diff_ci(
    y: np.ndarray,
    dec_c: np.ndarray,
    dec_p: np.ndarray,
    groups: np.ndarray,
    positives: bool,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI of rate(candidate) − rate(production), resampling groups, same images."""
    mask = y == 1 if positives else y == 0
    hit_c = dec_c if positives else ~dec_c  # sensitivity: flagged positives; specificity: cleared negatives
    hit_p = dec_p if positives else ~dec_p
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    point = float(hit_c[mask].mean() - hit_p[mask].mean())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        m = mask[idx]
        if m.any():
            diffs.append(hit_c[idx][m].mean() - hit_p[idx][m].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def release_gate(
    cand_preds: pd.DataFrame,
    cand_ops: OperatingPointSet,
    prod_preds: pd.DataFrame | None = None,
    prod_ops: OperatingPointSet | None = None,
    min_sensitivity_lcb: float = 0.80,
    min_specificity: float = 0.50,
    margin: float = 0.02,
    spec_tolerance: float = 0.03,
    n_boot: int = 1000,
) -> GateResult:
    checks: list[Check] = []
    y = cand_preds.y.to_numpy().astype(int)
    dec_c = _apply(cand_preds, cand_ops)
    c = confusion_at(y, dec_c.astype(float), 0.5)
    lcb = wilson_ci(c["tp"], c["tp"] + c["fn"])[0]
    checks.append(
        Check(
            "sensitivity_lower_ci",
            lcb >= min_sensitivity_lcb,
            lcb,
            min_sensitivity_lcb,
            f"sens={c['sensitivity']:.3f} on {c['tp'] + c['fn']} positives",
        )
    )
    checks.append(
        Check(
            "specificity",
            c["specificity"] >= min_specificity,
            c["specificity"],
            min_specificity,
            f"on {c['tn'] + c['fp']} negatives",
        )
    )

    if prod_preds is not None and prod_ops is not None:
        merged = cand_preds.merge(prod_preds[["image_id", "score"]], on="image_id", suffixes=("", "_prod"))
        if len(merged) != len(cand_preds):
            raise ValueError("candidate and production must be evaluated on the same images")
        prod_view = merged.assign(score=merged.score_prod)
        dc, dp = _apply(merged, cand_ops), _apply(prod_view, prod_ops)
        yy, gg = merged.y.to_numpy().astype(int), merged.group_id.to_numpy()
        d, lo, _ = paired_diff_ci(yy, dc, dp, gg, positives=True, n_boot=n_boot)
        checks.append(
            Check(
                "sensitivity_non_inferiority",
                lo >= -margin,
                lo,
                -margin,
                f"Δsens={d:+.3f}, CI lower {lo:+.3f}",
            )
        )
        d, lo, _ = paired_diff_ci(yy, dc, dp, gg, positives=False, n_boot=n_boot)
        checks.append(
            Check(
                "specificity_drop",
                d >= -spec_tolerance,
                d,
                -spec_tolerance,
                f"Δspec={d:+.3f} (CI lower {lo:+.3f})",
            )
        )
    return GateResult(passed=all(ch.passed for ch in checks), checks=checks)
