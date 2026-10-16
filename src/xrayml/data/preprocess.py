"""The ONE preprocessing implementation, shared by training (validation path) and serving.

Train/serve skew is the most frequent root cause of "the model is worse in production"
(revision plan 8.5: different resize implementation between training and inference).
``tests/unit/test_preprocess.py`` asserts that the dataset's validation path and the
serving path produce byte-identical tensors.

No torch import here: the serving container must be able to use it.
"""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(img: np.ndarray, size: int, *, is_mask: bool = False) -> tuple[np.ndarray, float]:
    """Resize the long side to ``size`` and pad bottom/right with zeros.

    Keeps the aspect ratio (a plain resize would distort bone shapes) and pads top-left
    anchored, so mapping a prediction back to original pixels is just ``coord / scale``.
    Masks use nearest-neighbour interpolation so they stay binary.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    # AREA avoids aliasing when shrinking; masks stay binary with NEAREST
    interp = cv2.INTER_NEAREST if is_mask else (cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((size, size, *img.shape[2:]), dtype=img.dtype)
    out[:nh, :nw] = resized
    return out, scale


def to_model_input(img01: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """float32 [0,1] HxW image -> (1, 1, size, size) float32 batch + the letterbox scale.

    Intensity normalisation (mean/std) is *inside the model* (``models.layers.Normalize``),
    so it is exported into the ONNX graph and cannot drift between training and serving.
    """
    if img01.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {img01.shape}")
    lb, scale = letterbox(img01.astype(np.float32, copy=False), size)
    return lb[None, None], scale


def boxes_to_original(boxes: np.ndarray, scale: float) -> np.ndarray:
    """Map xyxy boxes predicted on the letterboxed image back to original pixel coordinates."""
    return np.asarray(boxes, dtype=np.float32) / scale
