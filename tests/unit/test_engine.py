import math

import pytest
import torch

from xrayml.config import TrainConfig
from xrayml.engine.checkpoint import load_model_from_checkpoint
from xrayml.engine.trainer import Trainer, param_groups, warmup_cosine


def _cfg(tmp_path, manifest_path, **optim):
    return TrainConfig.model_validate(
        {
            "task": "classification",
            "output_dir": str(tmp_path / "run"),
            "seed": 0,
            "data": {"manifest": str(manifest_path), "image_size": 64, "batch_size": 8, "num_workers": 0},
            "model": {"name": "timm_classifier", "params": {"backbone": "resnet10t"}},
            "optim": {"epochs": 2, "lr": 1e-3, "warmup_steps": 2, **optim},
        }
    )


def test_warmup_then_cosine():
    assert warmup_cosine(0, 10, 100) == pytest.approx(0.1)
    assert warmup_cosine(9, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(100, 10, 100) == pytest.approx(0.01)
    assert warmup_cosine(55, 10, 100) == pytest.approx(0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * 0.5)))


def test_no_weight_decay_on_bias_and_norm():
    m = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3), torch.nn.BatchNorm2d(4))
    decay, no_decay = param_groups(m, 0.05)
    assert len(decay["params"]) == 1 and len(no_decay["params"]) == 3 and no_decay["weight_decay"] == 0


def test_overfit_one_batch(manifest):
    """The first debugging step (plan 5.10): if the loss cannot go to ~0 on 8 images, there is a bug."""
    from xrayml.data.datasets import XrayDataset, collate
    from xrayml.models import build_model

    torch.manual_seed(0)
    ds = XrayDataset(manifest, "classification", 64)
    x, y, _ = collate([ds[i] for i in range(8)], "classification")
    m = build_model("timm_classifier", backbone="resnet10t", drop=0.0)
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    first = None
    for _ in range(60):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.1 * first


@pytest.mark.slow
def test_resume_reproduces_uninterrupted_run(tmp_path, synth_dir, manifest):
    """Pre-empted at epoch 1 + resumed == trained straight through (checkpoint has RNG + scheduler)."""
    tr, va = manifest[manifest.split == "train"], manifest[manifest.split == "val"]
    a = Trainer(_cfg(tmp_path / "a", synth_dir / "manifest.parquet", ema_decay=0.9), tr, va, device="cpu")
    a.fit()
    b = Trainer(_cfg(tmp_path / "b", synth_dir / "manifest.parquet", ema_decay=0.9), tr, va, device="cpu")
    b.fit(stop_at_epoch=1)
    c = Trainer(_cfg(tmp_path / "b", synth_dir / "manifest.parquet", ema_decay=0.9), tr, va, device="cpu")
    c.resume(tmp_path / "b" / "run" / "last.ckpt")
    c.fit()
    for (k, pa), pc in zip(a.raw_model.state_dict().items(), c.raw_model.state_dict().values(), strict=True):
        assert torch.allclose(pa.float(), pc.float(), atol=1e-6), k
    model, ck = load_model_from_checkpoint(tmp_path / "a" / "run" / "last.ckpt")
    assert ck["lineage"]["dataset_fingerprint"] and not model.training


def test_divergence_is_caught(tmp_path, synth_dir, manifest):
    tr, va = manifest[manifest.split == "train"], manifest[manifest.split == "val"]
    t = Trainer(_cfg(tmp_path, synth_dir / "manifest.parquet"), tr, va, device="cpu")
    t.criterion = lambda logits, y: logits.sum() * float("nan")
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        t.train_one_epoch()
