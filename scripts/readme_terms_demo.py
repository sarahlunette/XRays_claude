"""Worked numbers for the terms explained in docs/readme_explained.md, computed with the repo's code.

    uv run --no-sync python scripts/readme_terms_demo.py      # < 1 min on CPU (mostly imports), no data

Each section prints a small, reproducible example so a term ("Wilson interval", "PSI",
"decision flip"...) comes with a number you can quote and the function that produced it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xrayml.calibration import apply_temperature, fit_temperature
from xrayml.data.manifest import assign_splits, fingerprint
from xrayml.data.quality import find_near_duplicates
from xrayml.evaluation.release_gate import release_gate
from xrayml.metrics.classification import (
    confusion_at,
    expected_calibration_error,
    group_bootstrap_ci,
    ppv_at_prevalence,
    safe_auc,
    wilson_ci,
)
from xrayml.monitoring.drift import psi
from xrayml.operating_points import confirm_on_test, select_operating_points


def header(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def toy_predictions(n_patients: int = 400, seed: int = 0) -> pd.DataFrame:
    """2 images per patient, ~30% fractured patients, scores higher for fractures (AUC ≈ 0.9)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_patients):
        y = int(rng.random() < 0.3)
        part = rng.choice(["hand", "leg"])
        for v in range(2):
            logit = rng.normal(1.8 if y else -1.0, 1.0)
            rows.append(
                {
                    "image_id": f"p{g}_v{v}",
                    "group_id": f"p{g}",
                    "y": y,
                    "body_part": part,
                    "logit": logit,
                    "score": 1 / (1 + np.exp(-logit)),
                }
            )
    return pd.DataFrame(rows)


def data_terms() -> None:
    header("Data: deterministic patient-level split + fingerprint")
    df = pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(1000)],
            "group_id": [f"patient{i // 2}" for i in range(1000)],  # 2 images per patient
            "fractured": [i % 3 == 0 for i in range(1000)],
            "site": ["site_c" if i >= 900 else "site_a" for i in range(1000)],
            "boxes": ["[]"] * 1000,
        }
    )
    df["fractured"] = df.fractured.astype(int)
    a = assign_splits(df, external={"site": ["site_c"]})
    b = assign_splits(df.sample(frac=1, random_state=1), external={"site": ["site_c"]})
    same = (a.set_index("image_id").split == b.set_index("image_id").split.reindex(a.image_id)).all()
    print(a.split.value_counts().to_string())
    print(f"same split after shuffling the rows: {bool(same)}  (hash of patient id decides)")
    print(f"patients in >1 split: {int((a.groupby('group_id').split.nunique() > 1).sum())}")
    print(f"fingerprint: {fingerprint(a)}  (changes if any id/label/box/split changes)")

    header("Data: pHash near-duplicates with band-LSH")
    rng = np.random.default_rng(0)
    base = {f"x{i}": int(rng.integers(0, 2**63)) for i in range(2000)}
    base["copy_of_x0"] = base["x0"] ^ 0b101  # re-export: 2 bits differ
    pairs = find_near_duplicates(base, max_distance=2)
    print(f"2,000 random 64-bit hashes + 1 near copy -> pairs found: {pairs}")
    print("only items sharing a band are compared; all-pairs would be ~2,000,000 comparisons")


def evaluation_terms(preds: pd.DataFrame) -> None:
    header("Evaluation: AUC, patient-level bootstrap, Wilson, ECE")
    y, p, g = preds.y.to_numpy(), preds.score.to_numpy(), preds.group_id.to_numpy()
    auc = safe_auc(y, p)
    lo, hi = group_bootstrap_ci(y, p, g, safe_auc, n_boot=300)
    print(f"AUC = {auc:.3f}, 95% CI resampling patients = [{lo:.3f}, {hi:.3f}]")
    c = confusion_at(y, p, 0.5)
    w = wilson_ci(c["tp"], c["tp"] + c["fn"])
    print(f"at threshold 0.5: sensitivity {c['sensitivity']:.3f}, Wilson 95% CI [{w[0]:.3f}, {w[1]:.3f}]")
    small = wilson_ci(19, 20)
    print(f"19/20 positives found -> 95% but Wilson CI [{small[0]:.2f}, {small[1]:.2f}]: small n is wide")

    # Calibration: draw labels *from* a known probability, so the true logit is calibrated;
    # multiplying it by 3 gives an overconfident model that temperature scaling should undo.
    rng = np.random.default_rng(0)
    true_logit = rng.normal(-0.8, 1.5, 4000)
    yc = (rng.random(4000) < 1 / (1 + np.exp(-true_logit))).astype(float)
    over_logit = 3 * true_logit
    t = fit_temperature(over_logit, yc)
    over, fixed = apply_temperature(over_logit, 1.0), apply_temperature(over_logit, t)
    print(
        f"ECE overconfident = {expected_calibration_error(yc, over):.3f} -> fitted T = {t:.2f} (true 3) "
        f"-> ECE {expected_calibration_error(yc, fixed):.3f}; AUC unchanged: "
        f"{safe_auc(yc, over):.4f} vs {safe_auc(yc, fixed):.4f}"
    )

    header("Evaluation: PPV depends on prevalence")
    for prev in (0.30, 0.10, 0.05):
        print(f"Se 0.90, Sp 0.90, prevalence {prev:>4.0%} -> PPV {ppv_at_prevalence(0.9, 0.9, prev):.1%}")


def release_terms(preds: pd.DataFrame) -> None:
    header("Release: operating point (tune on val, confirm on test)")
    groups = preds.group_id.unique()
    val_groups = set(groups[: len(groups) // 2])
    val, test = preds[preds.group_id.isin(val_groups)], preds[~preds.group_id.isin(val_groups)]
    ops = select_operating_points(val, target_sensitivity=0.9, min_positives=30, version="op-demo")
    for op in ops.points:
        print(
            f"{op.body_part:>4}: threshold {op.threshold:.3f}  val Se {op.validated_sensitivity:.3f}  "
            f"Sp {op.specificity:.3f}  (n+ = {op.n_positive})"
        )
    print(confirm_on_test(ops, test)[["body_part", "test_sensitivity", "test_specificity"]].to_string())

    header("Release: gate (acceptance + non-inferiority vs production)")
    prod = test.assign(score=np.clip(test.score + np.random.default_rng(1).normal(0, 0.05, len(test)), 0, 1))
    res = release_gate(test, ops, prod_preds=prod, prod_ops=ops, min_sensitivity_lcb=0.8, n_boot=300)
    for ch in res.checks:
        print(f"{'PASS' if ch.passed else 'FAIL'} {ch.name:<28} value {ch.value:+.3f} limit {ch.limit:+.3f}")
    print(f"gate passed: {res.passed}")
    n_pos = int(test.y.sum())
    print(f"(a FAIL is information: {n_pos} test positives cannot prove Se >= 0.80 with 95% confidence)")

    header("Release: why parity is checked on decisions, not only on scores")
    thr = ops.threshold_for("all").threshold
    many = np.random.default_rng(2).beta(2, 3, 20_000)  # 20k production-like scores
    for delta in (1e-5, 1e-3, 1e-2):
        flips = int(((many >= thr) != (many + delta >= thr)).sum())
        print(
            f"scores shifted by {delta:.0e} at threshold {thr:.3f} -> {flips:>3} flipped decisions / 20,000"
        )
    print("a tolerance like allclose(atol=1e-3) would pass, yet some patients get a different answer")


def monitoring_terms() -> None:
    header("Monitoring: PSI of an input statistic")
    rng = np.random.default_rng(0)
    ref = rng.normal(0.45, 0.08, 5000)  # mean intensity on the validation set
    for name, live in [
        ("same machine", rng.normal(0.45, 0.08, 800)),
        ("slightly brighter", rng.normal(0.48, 0.08, 800)),
        ("new X-ray device", rng.normal(0.55, 0.10, 800)),
    ]:
        v = psi(ref, live)
        verdict = "stable" if v < 0.1 else "moderate" if v < 0.25 else "investigate"
        print(f"{name:<18} PSI = {v:.3f}  -> {verdict}")


def main() -> None:
    preds = toy_predictions()
    data_terms()
    evaluation_terms(preds)
    release_terms(preds)
    monitoring_terms()


if __name__ == "__main__":
    main()
