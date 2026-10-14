import numpy as np
import pandas as pd
import pytest

from xrayml.artifacts import OperatingPoint, OperatingPointSet
from xrayml.evaluation.release_gate import release_gate
from xrayml.operating_points import confirm_on_test, select_operating_points, threshold_for_sensitivity


def _preds(n=600, sep=2.0, seed=0, parts=("hand", "leg")):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    score = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + sep * (y - 0.5))))
    return pd.DataFrame(
        {
            "image_id": [f"i{i}" for i in range(n)],
            "group_id": [f"g{i // 2}" for i in range(n)],
            "y": y,
            "score": score,
            "body_part": rng.choice(parts, n),
        }
    )


def test_threshold_reaches_target_sensitivity():
    df = _preds()
    t = threshold_for_sensitivity(df.y, df.score, 0.9)
    pos = df.score[df.y == 1]
    assert (pos >= t).mean() >= 0.9
    assert (pos >= np.nextafter(t, 1)).mean() < (pos >= t).mean()  # highest threshold reaching the target


def test_select_ops_per_body_part_needs_enough_positives():
    df = _preds()
    ops = select_operating_points(df, 0.9, min_positives=30)
    assert {p.body_part for p in ops.points} == {"all", "hand", "leg"}
    ops_small = select_operating_points(df, 0.9, min_positives=10_000)
    assert [p.body_part for p in ops_small.points] == ["all"]
    assert ops.threshold_for("hip").body_part == "all"  # fallback
    assert ops.threshold_for("leg").body_part == "leg"


def test_ops_roundtrip_and_confirm(tmp_path):
    ops = select_operating_points(_preds(), 0.9, version="op-x", model_version="m1")
    ops.save(tmp_path / "t.json")
    assert OperatingPointSet.load(tmp_path / "t.json") == ops
    table = confirm_on_test(ops, _preds(seed=1))
    assert set(table.body_part) == {"all", "hand", "leg"} and table.test_sensitivity.between(0.8, 1).all()


def test_fp_per_image_reported_with_box_table():
    df = _preds(n=100)
    boxes = pd.DataFrame(
        {"image_id": df.image_id, "score": df.score, "is_tp": df.y == 1, "body_part": df.body_part}
    )
    ops = select_operating_points(df, 0.9, per_body_part=False, box_preds=boxes)
    assert ops.points[0].fp_per_image is not None and ops.points[0].fp_per_image >= 0


def _ops(thr, version="m"):
    return OperatingPointSet(
        version="op",
        model_version=version,
        dataset_version="d",
        points=[
            OperatingPoint(
                threshold=thr,
                target_sensitivity=0.9,
                validated_sensitivity=0.9,
                sensitivity_ci=(0.8, 0.95),
                specificity=0.5,
                n_positive=10,
                n_negative=10,
            )
        ],
    )


def test_gate_passes_good_model_and_fails_bad_one():
    good = _preds(sep=4.0)
    ops = select_operating_points(good, 0.95, per_body_part=False)
    assert release_gate(good, ops, n_boot=200).passed
    bad = _preds(sep=0.0)
    res = release_gate(bad, _ops(0.6), n_boot=200)
    assert not res.passed and any(c.name == "sensitivity_lower_ci" and not c.passed for c in res.checks)


def test_gate_non_inferiority_against_production():
    prod = _preds(sep=3.0, seed=2)
    worse = prod.assign(score=np.where(prod.y == 1, prod.score * 0.5, prod.score))  # misses more fractures
    thr = threshold_for_sensitivity(prod.y, prod.score, 0.9)
    res = release_gate(
        worse, _ops(thr), prod, _ops(thr), min_sensitivity_lcb=0, min_specificity=0, n_boot=200
    )
    names = {c.name: c.passed for c in res.checks}
    assert names["sensitivity_non_inferiority"] is False
    same = release_gate(
        prod, _ops(thr), prod, _ops(thr), min_sensitivity_lcb=0, min_specificity=0, n_boot=200
    )
    assert same.passed


def test_gate_requires_same_images():
    a = _preds(n=50)
    with pytest.raises(ValueError, match="same images"):
        release_gate(a, _ops(0.5), a.iloc[:10], _ops(0.5), n_boot=10)
