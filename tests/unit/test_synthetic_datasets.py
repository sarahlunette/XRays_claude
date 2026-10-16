import json

import numpy as np
import pytest
import torch

from xrayml.config import AugmentConfig
from xrayml.data.datasets import XrayDataset, collate, make_loader
from xrayml.data.synthetic import render_radiograph


def test_positive_box_covers_the_dark_gap():
    rng = np.random.default_rng(0)
    for _ in range(20):
        img, boxes, polys, _ = render_radiograph(rng, 128, fractured=True)
        if not boxes:
            continue
        x0, y0, x1, y1 = (round(v) for v in boxes[0])
        assert (img[max(y0, 0) : y1, max(x0, 0) : x1] < 0.35).any()
        assert polys and len(polys[0]) >= 6


def test_negative_has_no_annotation():
    img, boxes, polys, _ = render_radiograph(np.random.default_rng(1), 64, fractured=False)
    assert boxes == [] and polys == [] and img.dtype == np.float32


@pytest.mark.parametrize("task", ["classification", "detection", "segmentation"])
def test_targets_per_task(manifest, task):
    ds = XrayDataset(manifest, task, 64, train=True, augment=AugmentConfig())
    pos = int(np.flatnonzero(manifest.fractured.to_numpy())[0])
    img, target, i = ds[pos]
    assert img.shape == (1, 64, 64) and img.dtype == torch.float32 and i == pos
    if task == "classification":
        assert target.tolist() == [1.0]
    elif task == "detection":
        assert target["boxes"].shape[1] == 4 and target["labels"].dtype == torch.int64
        assert (target["boxes"][:, 2:] > target["boxes"][:, :2]).all()
    else:
        assert target.shape == (1, 64, 64) and set(target.unique().tolist()) <= {0.0, 1.0}


def test_seg_mask_lies_inside_the_box(manifest):
    ds = XrayDataset(manifest, "segmentation", 64, train=False)
    pos = int(np.flatnonzero(manifest.fractured.to_numpy())[0])
    _, mask, _ = ds[pos]
    _, boxes, _, _ = ds.load(pos)
    ys, xs = np.nonzero(mask[0].numpy())
    x0, y0, x1, y1 = boxes[0]
    assert len(xs) and xs.min() >= x0 - 2 and xs.max() <= x1 + 2 and ys.min() >= y0 - 2 and ys.max() <= y1 + 2


def test_detection_collate_keeps_lists(manifest):
    ds = XrayDataset(manifest, "detection", 64)
    images, targets, idx = collate([ds[0], ds[1]], "detection")
    assert isinstance(images, list) and isinstance(targets, list) and idx.tolist() == [0, 1]


def test_oversampling_balances_classes(manifest):
    from xrayml.config import DataConfig

    cfg = DataConfig(
        manifest="x.parquet", image_size=64, batch_size=200, num_workers=0, oversample_positives=True
    )
    df = manifest.assign(
        fractured=(np.arange(len(manifest)) % 10 == 0).astype(int),
        boxes=[json.dumps([[1, 1, 5, 5]]) if i % 10 == 0 else "[]" for i in range(len(manifest))],
    )
    loader = make_loader(df, cfg, "classification", train=True)
    idx = torch.cat([b[2] for b, _ in zip(loader, range(3), strict=False)])
    assert 0.35 < df.fractured.to_numpy()[idx.numpy()].mean() < 0.65


def test_validation_is_deterministic(manifest):
    ds = XrayDataset(manifest, "classification", 64, train=False)
    assert torch.equal(ds[3][0], ds[3][0])
