"""Decode once to a fast training format (revision plan 3.5, 5.10).

Decoding a 2,880 px JPEG/DICOM every epoch makes the GPU wait on the CPU ("GPU utilisation
40%"). We resize once (long side = ``max_side``, aspect kept, no padding), store lossless
16-bit PNGs, and rescale boxes and polygons in the manifest accordingly.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

from xrayml.data.io import read_image, write_png16


def _resize_one(args: tuple[str, str, int]) -> tuple[float, int, int]:
    src, dst, max_side = args
    img = read_image(src)
    h, w = img.shape
    scale = min(1.0, max_side / max(h, w))  # never upsample
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    write_png16(dst, img)
    return scale, img.shape[1], img.shape[0]


def _scale_json(js: str, s: float) -> str:
    return json.dumps([[c * s for c in item] for item in json.loads(js)])


def prepare_cache(
    df: pd.DataFrame, out_dir: str | Path, max_side: int = 1024, workers: int = 2
) -> pd.DataFrame:
    """``df`` must have ``abs_path`` (from ``load_manifest``). Returns the cached manifest."""
    out_dir = Path(out_dir)
    jobs = [
        (src, str(out_dir / "images" / f"{iid}.png"), max_side)
        for src, iid in zip(df.abs_path, df.image_id, strict=True)
    ]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(_resize_one, jobs, chunksize=16))
    out = df.copy()
    out["orig_width"], out["orig_height"] = df.width, df.height
    out["cache_scale"] = [r[0] for r in results]
    out["width"] = [r[1] for r in results]
    out["height"] = [r[2] for r in results]
    out["boxes"] = [_scale_json(b, s) for b, s in zip(df.boxes, out.cache_scale, strict=True)]
    out["polygons"] = [_scale_json(p, s) for p, s in zip(df.polygons, out.cache_scale, strict=True)]
    out["path"] = [f"images/{iid}.png" for iid in df.image_id]
    return out.drop(columns=["abs_path"])
