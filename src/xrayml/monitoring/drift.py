"""Post-deployment monitoring (revision plan 8.6).

Ground truth arrives late and partially, so we watch what we can see immediately:
* input drift — PSI of image statistics (a new X-ray machine changes exposure/contrast);
* output drift — positive rate per site vs the reference (validation) rate.
PSI rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 investigate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(ref: np.ndarray, new: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    ref, new = np.asarray(ref, float), np.asarray(new, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))  # quantile bins of the reference
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref) + eps
    q = np.histogram(new, edges)[0] / len(new) + eps
    return float(np.sum((q - p) * np.log(q / p)))


def drift_report(
    reference: pd.DataFrame,
    live: pd.DataFrame,
    features: tuple[str, ...] = ("mean_intensity", "std_intensity", "score"),
    by: str = "site",
    psi_alert: float = 0.25,
    rate_rel_alert: float = 0.5,
    min_n: int = 30,
) -> pd.DataFrame:
    """One row per site: n, PSI per feature, positive rate vs reference, alert flag + reason."""
    ref_rate = float(reference.positive.mean())
    rows = []
    for site, g in live.groupby(by):
        row: dict = {by: site, "n": len(g)}
        reasons = []
        for f in features:
            if f in reference.columns and f in g.columns:
                v = psi(reference[f].to_numpy(), g[f].to_numpy())
                row[f"psi_{f}"] = round(v, 4)
                if v > psi_alert and len(g) >= min_n:
                    reasons.append(f"PSI({f})={v:.2f}")
        rate = float(g.positive.mean())
        row |= {"positive_rate": round(rate, 4), "reference_rate": round(ref_rate, 4)}
        if len(g) >= min_n and abs(rate - ref_rate) > rate_rel_alert * ref_rate:
            reasons.append(f"positive rate {rate:.1%} vs {ref_rate:.1%}")
        row["alert"] = bool(reasons)
        row["reason"] = "; ".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)
