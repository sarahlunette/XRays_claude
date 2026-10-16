import numpy as np
import pytest
import torch
import torch.nn.functional as F

from xrayml.calibration import apply_temperature, fit_temperature
from xrayml.losses import LOSSES, focal_loss, soft_dice_loss, tversky_loss


def test_focal_with_gamma0_alpha_half_is_half_bce():
    z, y = torch.randn(64), (torch.rand(64) > 0.5).float()
    assert focal_loss(z, y, alpha=0.5, gamma=0) == pytest.approx(
        0.5 * F.binary_cross_entropy_with_logits(z, y).item(), rel=1e-5
    )


def test_focal_downweights_easy_examples_100x():
    y = torch.ones(1)
    z = torch.logit(torch.tensor([0.9]))  # p_t = 0.9
    ratio = focal_loss(z, y, alpha=1.0, gamma=2) / F.binary_cross_entropy_with_logits(z, y)
    assert ratio.item() == pytest.approx(0.01, rel=1e-4)


def test_dice_and_tversky_perfect_prediction():
    t = torch.zeros(1, 1, 8, 8)
    t[..., 2:5, 2:5] = 1
    logits = (t * 2 - 1) * 50
    assert soft_dice_loss(logits, t).item() < 1e-3
    assert tversky_loss(logits, t).item() < 1e-3
    assert soft_dice_loss(torch.full_like(t, -50), torch.zeros_like(t)).item() < 1e-3  # empty/empty finite


def test_all_registered_losses_build():
    for name in LOSSES.names():
        assert LOSSES.build(name) is not None


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 2, 5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-true_logit))).astype(float)
    t = fit_temperature(true_logit * 3.0, y)  # model is 3x over-confident
    assert t == pytest.approx(3.0, rel=0.15)
    p = apply_temperature(true_logit * 3.0, t)
    assert np.argsort(p).tolist() == np.argsort(true_logit).tolist()  # ranking (AUC) unchanged
