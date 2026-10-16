"""Export to ONNX + parity checks (revision plan 8.5).

Parity at *decision level* matters more than raw tolerance: a 1e-3 score change that moves
one image across the shipped threshold is a clinical regression. ``check_parity`` reports
both the max score difference and the number of flipped decisions.

Detection: torchvision Faster R-CNN exports poorly (data-dependent NMS, custom ops); the
classifier and the U-Net export cleanly. A production detector would either keep NMS
outside the graph or ship via TorchScript/TensorRT — say so instead of pretending.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn


def export_onnx(model: nn.Module, image_size: int, path: str | Path, opset: int = 18) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()  # BN in eval mode at export, or running stats are ignored
    dummy = torch.rand(2, 1, image_size, image_size)
    try:  # torch.export-based exporter (full graph capture)
        batch = torch.export.Dim("batch", min=1, max=256)
        torch.onnx.export(
            model,
            (dummy,),
            str(path),
            input_names=["image"],
            output_names=["logits"],
            dynamic_shapes=({0: batch},),
            dynamo=True,
            opset_version=opset,
        )
    except Exception as e:  # pragma: no cover - fallback for older torch / unsupported ops
        warnings.warn(f"dynamo export failed ({e!r}); falling back to the legacy exporter", stacklevel=2)
        torch.onnx.export(
            model,
            (dummy,),
            str(path),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
            opset_version=min(opset, 17),
        )
    import onnx

    onnx.checker.check_model(onnx.load(str(path)))
    return path


def _image_score(logits: np.ndarray) -> np.ndarray:
    """Classifier: sigmoid of the logit. Segmentation: max pixel probability."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.reshape(len(probs), -1).max(axis=1)


@torch.no_grad()
def check_parity(
    model: nn.Module,
    onnx_path: str | Path,
    loader,
    thresholds: list[float],
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = model.eval().cpu()
    max_diff, flips, n = 0.0, 0, 0
    for images, _, _ in loader:
        pt = model(images).float().numpy()
        ox = sess.run(None, {"image": images.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(pt - ox).max()))
        s_pt, s_ox = _image_score(pt), _image_score(ox)
        for t in thresholds:
            flips += int(((s_pt >= t) != (s_ox >= t)).sum())
        n += len(images)
    ok = max_diff <= atol + rtol * 1.0 and flips == 0
    return {"n_images": n, "max_abs_diff": max_diff, "decision_flips": flips, "passed": bool(ok)}
