import numpy as np
import pytest
import torch

from xrayml.data.datasets import XrayDataset
from xrayml.data.io import read_image
from xrayml.data.preprocess import boxes_to_original, letterbox, to_model_input


@pytest.mark.parametrize(("h", "w"), [(300, 200), (200, 300), (64, 64), (50, 70)])
def test_letterbox_keeps_aspect_and_pads_bottom_right(h, w):
    img = np.ones((h, w), np.float32)
    out, s = letterbox(img, 128)
    assert out.shape == (128, 128)
    assert s == pytest.approx(128 / max(h, w))
    nh, nw = round(h * s), round(w * s)
    assert out[:nh, :nw].min() == pytest.approx(1.0)
    assert out[nh:, :].sum() == 0 and out[:, nw:].sum() == 0


def test_mask_letterbox_stays_binary():
    m = np.zeros((101, 77), np.uint8)
    m[20:60, 10:30] = 1
    out, _ = letterbox(m, 64, is_mask=True)
    assert set(np.unique(out)) <= {0, 1}


def test_boxes_round_trip():
    box = np.array([[10.0, 20.0, 50.0, 80.0]])
    _, s = letterbox(np.zeros((400, 300), np.float32), 128)
    assert np.allclose(boxes_to_original(box * s, s), box)


def test_training_val_path_equals_serving_path(manifest):
    """Train/serve skew guard: the dataset (val) tensor and the API tensor are identical."""
    ds = XrayDataset(manifest, "classification", 64, train=False)
    img_t, _, _ = ds[0]
    served, _ = to_model_input(read_image(manifest.abs_path.iloc[0]), 64)
    assert torch.equal(img_t, torch.from_numpy(served[0]))


def test_to_model_input_rejects_color():
    with pytest.raises(ValueError, match="grayscale"):
        to_model_input(np.zeros((8, 8, 3), np.float32), 32)
