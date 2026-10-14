"""Synthetic "radiographs" with hairline fractures (fissures).

Why: CI must exercise the whole pipeline (manifest -> train -> evaluate -> operating point ->
export -> serve) in minutes, without downloading 338 MB of medical images. The generator
produces the *same manifest schema* as the FracAtlas adapter, so every downstream line of
code is identical for real and synthetic data.

Each image: soft-tissue background, one or two bone shafts with a brighter cortex, an
optional laterality marker ("L"/"R" text — a classic shortcut), optional metal hardware,
and for positives a thin dark gap crossing a shaft (fracture line, ``gap_px`` wide). The
gap polygon is the segmentation target; its bounding box is the detection target.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from xrayml.data.io import write_png16

BODY_PARTS = ["hand", "leg", "hip", "shoulder"]
SITES = ["site_a", "site_b", "site_c"]


def _shaft_polygon(cx: float, cy: float, length: float, width: float, angle: float) -> np.ndarray:
    d = np.array([np.cos(angle), np.sin(angle)])
    n = np.array([-d[1], d[0]])
    c = np.array([cx, cy])
    hl, hw = length / 2, width / 2
    return np.stack([c - d * hl - n * hw, c + d * hl - n * hw, c + d * hl + n * hw, c - d * hl + n * hw])


def render_radiograph(
    rng: np.random.Generator,
    size: int = 256,
    fractured: bool = False,
    site: str = "site_a",
    gap_px: tuple[float, float] = (2.0, 5.0),
) -> tuple[np.ndarray, list[list[float]], list[list[float]], int]:
    """Return (image float32 [0,1], boxes xyxy, polygons flat-xy, hardware flag).

    ``gap_px`` is the fracture-line width in output pixels. (2, 5) is learnable by a small
    CNN on CPU in minutes (CI); (0.7, 1.5) is a true hairline at this resolution — try it
    to see why resolution is the most valuable knob for fissures (plan 1.1 FLOPs/scaling).
    """
    img: np.ndarray = np.zeros((size, size), np.float32)
    # Soft tissue: a large blurred ellipse.
    tissue = np.zeros_like(img)
    cv2.ellipse(
        tissue,
        (size // 2, size // 2),
        (int(size * rng.uniform(0.3, 0.45)), int(size * rng.uniform(0.35, 0.48))),
        float(rng.uniform(0, 180)),
        0,
        360,
        1.0,
        -1,
    )
    img += 0.25 * cv2.GaussianBlur(tissue, (0, 0), size / 20)

    shafts = []
    bone = np.zeros((size, size), np.uint8)
    for _ in range(int(rng.integers(1, 3))):
        poly = _shaft_polygon(
            cx=size * rng.uniform(0.35, 0.65),
            cy=size * rng.uniform(0.35, 0.65),
            length=size * rng.uniform(0.55, 0.8),
            width=size * rng.uniform(0.08, 0.14),
            angle=rng.uniform(0, np.pi),
        )
        shafts.append(poly)
        cv2.fillPoly(bone, [np.round(poly).astype(np.int32)], 1)
    cortex = bone.astype(np.float32) - cv2.erode(bone, np.ones((5, 5), np.uint8)).astype(np.float32)
    img = np.where(bone > 0, 0.62, img) + 0.18 * cortex

    boxes: list[list[float]] = []
    polygons: list[list[float]] = []
    if fractured:
        poly = shafts[int(rng.integers(len(shafts)))]
        a, b, _, d_ = poly  # corners: a->b along the axis, a->d across the width
        axis, across = b - a, d_ - a
        t = rng.uniform(0.25, 0.75)
        width = float(np.linalg.norm(across))
        center = a + axis * t + across / 2
        # Fracture direction: across the shaft, tilted up to ±35°.
        base = np.arctan2(across[1], across[0]) + rng.uniform(-0.6, 0.6)
        u = np.array([np.cos(base), np.sin(base)])
        v = np.array([-u[1], u[0]])
        half_len = width * 0.62
        thick = rng.uniform(*gap_px)
        gap = np.stack(
            [
                center - u * half_len - v * thick / 2,
                center + u * half_len - v * thick / 2,
                center + u * half_len + v * thick / 2,
                center - u * half_len + v * thick / 2,
            ]
        )
        gap_mask = np.zeros_like(bone)
        cv2.fillPoly(gap_mask, [np.round(gap).astype(np.int32)], 1)
        gap_mask &= bone
        if gap_mask.any():
            img = np.where(gap_mask > 0, 0.22, img)  # gap back to soft-tissue density
            ys, xs = np.nonzero(gap_mask)
            pad = 2.0
            boxes.append(
                [
                    float(xs.min() - pad),
                    float(ys.min() - pad),
                    float(xs.max() + 1 + pad),
                    float(ys.max() + 1 + pad),
                ]
            )
            # Segmentation target = the gap *inside* the bone, as a COCO-style polygon.
            contours, _ = cv2.findContours(gap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 3:
                    polygons.append([float(c) for c in cnt.reshape(-1)])

    hardware = int(rng.random() < 0.1)
    if hardware:  # metal plate: very bright, uncorrelated with the label
        x0, y0 = (int(v) for v in rng.integers(size // 5, size // 2, 2))
        cv2.rectangle(img, (x0, y0), (x0 + size // 12, y0 + size // 4), 0.98, -1)

    # Laterality marker, as on real radiographs.
    text = np.zeros((size, size), np.uint8)  # OpenCV draws text on uint8 only
    cv2.putText(
        text,
        "L" if rng.random() < 0.5 else "R",
        (size // 16, size // 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        size / 400,
        255,
        max(1, size // 128),
    )
    img = np.where(text > 0, 0.95, img)

    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    # Site-specific acquisition: different exposure (gamma) and noise level.
    gamma = {"site_a": 1.0, "site_b": 0.85, "site_c": 1.15}.get(site, 1.0)
    noise = {"site_a": 0.02, "site_b": 0.03, "site_c": 0.015}.get(site, 0.02)
    img = np.clip(img, 0, 1) ** gamma + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 1).astype(np.float32), boxes, polygons, hardware


def make_synthetic_dataset(
    out_dir: str | Path,
    n_images: int = 400,
    image_size: int = 256,
    positive_rate: float = 0.35,
    images_per_patient: int = 2,
    seed: int = 0,
    gap_px: tuple[float, float] = (2.0, 5.0),
):
    """Write 16-bit PNGs and return a manifest DataFrame (paths relative to ``out_dir``)."""
    import pandas as pd

    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_images):
        patient = f"synth_p{i // images_per_patient:05d}"
        fractured = bool(rng.random() < positive_rate)
        if i % images_per_patient == 0:  # site is a property of the patient's hospital
            site = SITES[int(rng.integers(len(SITES)))]
        img, boxes, polys, hardware = render_radiograph(rng, image_size, fractured, site, gap_px)
        fractured = bool(boxes)  # a gap can miss the bone in rare cases
        rel = f"images/synth_{i:05d}.png"
        write_png16(out_dir / rel, img)
        rows.append(
            {
                "image_id": f"synth_{i:05d}",
                "path": rel,
                "group_id": patient,
                "source": "synthetic",
                "site": site,
                "body_part": BODY_PARTS[int(rng.integers(len(BODY_PARTS)))],
                "view": "frontal",
                "hardware": hardware,
                "fractured": int(fractured),
                "fracture_count": len(boxes),
                "width": image_size,
                "height": image_size,
                "boxes": json.dumps(boxes),
                "polygons": json.dumps(polys),
            }
        )
    return pd.DataFrame(rows)
