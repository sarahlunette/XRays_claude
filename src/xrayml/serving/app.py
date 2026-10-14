"""Inference service (revision plan 8.5): FastAPI + ONNX Runtime, *no torch*.

A bundle directory holds the three release artefacts, each versioned independently:
    model.onnx        weights (from `xrayml export`)
    thresholds.json   OperatingPointSet (from `xrayml select-ops`)
    model_card.json   lineage: git SHA, dataset version + fingerprint, config, metrics

Every response carries model_version + thresholds_version (traceability of each decision).
Each prediction is appended to a JSONL log (score, decision, site, simple image statistics):
that log is the input of drift monitoring (`xrayml drift`).

Run:  XRAYML_BUNDLE=bundle uvicorn xrayml.serving.app:app --port 8000
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from xrayml import __version__
from xrayml.artifacts import ModelCard, OperatingPointSet
from xrayml.data.io import decode_bytes
from xrayml.data.preprocess import boxes_to_original, to_model_input


class Bundle:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        self.card = ModelCard.load(path / "model_card.json")
        self.ops = OperatingPointSet.load(path / "thresholds.json")
        if self.ops.model_version != self.card.model_version:
            # Guard against shipping thresholds tuned for another model.
            raise RuntimeError(
                f"thresholds tuned for model {self.ops.model_version}, bundle has {self.card.model_version}"
            )
        self.session = ort.InferenceSession(str(path / "model.onnx"), providers=ort.get_available_providers())


def _mask_boxes(prob: np.ndarray, thr: float, scale: float, min_area: int = 4) -> list[list[float]]:
    """Segmentation output -> boxes in original pixels via connected components."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((prob >= thr).astype(np.uint8))
    boxes = [
        [float(x), float(y), float(x + w), float(y + h)] for x, y, w, h, a in stats[1:n] if a >= min_area
    ]
    return boxes_to_original(np.asarray(boxes).reshape(-1, 4), scale).round(1).tolist()


def create_app(bundle_dir: str | Path | None = None) -> FastAPI:
    bundle_dir = bundle_dir or os.environ.get("XRAYML_BUNDLE", "bundle")
    log_path = os.environ.get("XRAYML_PREDICTION_LOG")
    app = FastAPI(title="xrayml fracture detection", version=__version__)
    bundle = Bundle(bundle_dir)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict[str, Any]:
        c = bundle.card
        return {
            "service": __version__,
            "model_version": c.model_version,
            "task": c.task,
            "thresholds_version": bundle.ops.version,
            "dataset_version": c.dataset_version,
            "git_sha": c.git_sha,
        }

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        body_part: str | None = Form(None),
        x_site_id: str | None = Header(None),
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        data = await file.read()
        try:
            img = decode_bytes(data, file.filename or "")
        except Exception as e:  # corrupted upload -> 422, never a 500
            raise HTTPException(status_code=422, detail=f"cannot decode image: {e}") from e
        if body_part is None and data[128:132] == b"DICM":
            import io

            import pydicom

            body_part = (
                str(
                    getattr(
                        pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True), "BodyPartExamined", ""
                    )
                    or ""
                ).lower()
                or None
            )
        x, scale = to_model_input(img, bundle.card.image_size)
        logits = bundle.session.run(None, {"image": x})[0]
        prob = 1.0 / (1.0 + np.exp(-logits[0]))
        score = float(prob.max())
        op = bundle.ops.threshold_for(body_part)
        result: dict[str, Any] = {
            "finding": op.finding,
            "score": round(score, 5),
            "positive": score >= op.threshold,
            "threshold": op.threshold,
            "operating_point_body_part": op.body_part,
            "model_version": bundle.card.model_version,
            "thresholds_version": bundle.ops.version,
        }
        if bundle.card.task == "segmentation":
            result["boxes"] = _mask_boxes(prob[0], op.threshold, scale)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if log_path:
            rec = {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "site": x_site_id or "unknown",
                "body_part": body_part,
                "score": score,
                "positive": result["positive"],
                # stats on the model input (letterboxed), same definition as evaluation's reference
                "mean_intensity": float(x.mean()),
                "std_intensity": float(x.std()),
                "height": img.shape[0],
                "width": img.shape[1],
                "model_version": bundle.card.model_version,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return result

    return app


# Module-level app for `uvicorn xrayml.serving.app:app`; built lazily so importing is cheap in tests.
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)
