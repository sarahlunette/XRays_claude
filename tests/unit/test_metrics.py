import numpy as np
import pytest
import torch
import torchvision
from hypothesis import given, settings
from hypothesis import strategies as st

from xrayml.metrics.classification import (
    confusion_at,
    expected_calibration_error,
    group_bootstrap_ci,
    ppv_at_prevalence,
    safe_auc,
    wilson_ci,
)
from xrayml.metrics.detection import box_iou, froc, nms, threshold_at_fp_rate
from xrayml.metrics.segmentation import dice_score, iou_score

box = st.tuples(st.floats(0, 500), st.floats(0, 500), st.floats(1, 200), st.floats(1, 200)).map(
    lambda t: torch.tensor([[t[0], t[1], t[0] + t[2], t[1] + t[3]]])
)


@settings(max_examples=60, deadline=None)
@given(box, box)
def test_iou_properties(a, b):
    iou = box_iou(a, b).item()
    assert 0.0 <= iou <= 1.0 + 1e-6
    assert box_iou(a, a).item() == pytest.approx(1.0)
    assert iou == pytest.approx(box_iou(b, a).item())
    assert iou == pytest.approx(torchvision.ops.box_iou(a, b).item(), abs=1e-5)


def test_nms_matches_torchvision():
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(50, 2, generator=g) * 100
    boxes = torch.cat([xy, xy + 10 + torch.rand(50, 2, generator=g) * 30], 1)
    scores = torch.rand(50, generator=g)
    assert sorted(nms(boxes, scores, 0.5).tolist()) == sorted(
        torchvision.ops.nms(boxes, scores, 0.5).tolist()
    )


def test_froc_counts_hits_fps_and_double_hits():
    gts = {"a": np.array([[0, 0, 10, 10]]), "b": np.zeros((0, 4))}
    preds = [
        ("a", 0.9, [0, 0, 10, 10]),
        ("a", 0.8, [1, 1, 10, 10]),
        ("b", 0.7, [5, 5, 9, 9]),
        ("a", 0.6, [50, 50, 60, 60]),
    ]
    out = froc(preds, gts, n_images=2)
    assert out["sensitivity"][-1] == 1.0
    assert out["fp_per_image"][-1] == 1.0  # 2 FPs (b, far box) / 2 images; double hit is not an FP
    thr, sens = threshold_at_fp_rate(out, 0.5)
    assert sens == 1.0 and thr <= 0.7


def test_confusion_and_bayes_ppv():
    c = confusion_at(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)
    assert ppv_at_prevalence(0.9, 0.9, 0.05) == pytest.approx(0.32, abs=0.005)
    assert ppv_at_prevalence(0.9, 0.9, 0.30) == pytest.approx(0.79, abs=0.005)


def test_wilson_interval():
    lo, hi = wilson_ci(9, 10)
    assert lo < 0.9 < hi and hi <= 1.0
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_auc_single_class_is_none_not_crash():
    assert safe_auc(np.zeros(5), np.random.rand(5)) is None


def test_group_bootstrap_ci_contains_point_and_widens_with_correlation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 400), 0, 1)
    lo, hi = group_bootstrap_ci(y, p, np.arange(400), safe_auc, n_boot=300)
    assert lo < safe_auc(y, p) < hi
    # 4 copies per patient: patient-level CI is wider than the naive image-level CI
    y4, p4 = np.repeat(y[:100], 4), np.repeat(p[:100], 4)
    naive = group_bootstrap_ci(y4, p4, np.arange(400), safe_auc, n_boot=300)
    grouped = group_bootstrap_ci(y4, p4, np.repeat(np.arange(100), 4), safe_auc, n_boot=300)
    assert grouped[1] - grouped[0] > naive[1] - naive[0]


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p * 0.5, 0, 1)) > 0.1


def test_dice_iou_relation_and_empty_convention():
    a = np.zeros((10, 10), bool)
    b = a.copy()
    a[2:6, 2:6] = True
    b[3:7, 3:7] = True
    d, j = dice_score(a, b), iou_score(a, b)
    assert j == pytest.approx(d / (2 - d))
    assert dice_score(np.zeros((3, 3)), np.zeros((3, 3))) == 1.0
