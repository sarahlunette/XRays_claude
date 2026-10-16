"""FracAtlas adapter: download (checksum-verified) and convert to the common manifest schema.

FracAtlas — Abedeen et al., *Scientific Data* 10, 521 (2023), doi:10.6084/m9.figshare.22363012,
licence CC BY 4.0. 4,083 musculoskeletal radiographs (hand, leg, hip, shoulder), ~720 with
fractures, annotated with COCO polygons + boxes, so it supports the three CV tasks of the
role: classification, detection and segmentation.

Known limitations, worth saying in an interview:
* No patient identifier -> splits are by image, so multi-view images of one patient may
  land in different splits (optimistic estimates). ``multiscan`` flags those images.
* JPEGs, not DICOM: windowing and bit depth were already applied upstream.
* Single site, single label ("fractured"); no hairline/displaced sub-type.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

FRACATLAS_URL = "https://ndownloader.figshare.com/files/65518038"
FRACATLAS_MD5 = "fe9da2c7c285915ebee69dfdab8fd396"
FRACATLAS_DOI = "10.6084/m9.figshare.22363012"
BODY_PART_COLUMNS = ["hand", "leg", "hip", "shoulder"]
VIEW_COLUMNS = ["frontal", "lateral", "oblique"]


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_fracatlas(dest: str | Path) -> Path:
    """Download + verify + extract. Idempotent: re-running with the data present is a no-op."""
    dest = Path(dest)
    root = dest / "FracAtlas"
    if (root / "dataset.csv").exists():
        return root
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "FracAtlas.zip"
    if not archive.exists() or _md5(archive) != FRACATLAS_MD5:
        tmp = archive.with_suffix(".part")
        with urllib.request.urlopen(FRACATLAS_URL) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(archive)
    if (got := _md5(archive)) != FRACATLAS_MD5:
        raise RuntimeError(f"checksum mismatch for {archive}: {got} != {FRACATLAS_MD5}")
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():  # refuse path traversal in archive members
            if member.startswith("/") or ".." in Path(member).parts:
                raise RuntimeError(f"unsafe path in archive: {member}")
        z.extractall(dest)
    return root


def _body_part(row) -> str:
    parts = [c for c in BODY_PART_COLUMNS if int(row[c]) == 1]
    if int(row["mixed"]) == 1 or len(parts) != 1:
        return "mixed"
    return parts[0]


def _view(row) -> str:
    views = [c for c in VIEW_COLUMNS if int(row[c]) == 1]
    return "+".join(views) if views else "unknown"


def build_fracatlas_manifest(root: str | Path):
    """Return a manifest DataFrame with paths relative to ``root``."""
    import pandas as pd
    from PIL import Image

    root = Path(root)
    meta = pd.read_csv(root / "dataset.csv")
    coco = json.loads((root / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json").read_text())
    id_to_file = {im["id"]: im for im in coco["images"]}
    boxes: dict[str, list[list[float]]] = defaultdict(list)
    polygons: dict[str, list[list[float]]] = defaultdict(list)
    for ann in coco["annotations"]:
        fname = id_to_file[ann["image_id"]]["file_name"]
        x, y, w, h = ann["bbox"]  # COCO is xywh; we use xyxy everywhere (plan 1.3: format bugs)
        boxes[fname].append([float(x), float(y), float(x + w), float(y + h)])
        seg = ann.get("segmentation") or []
        polygons[fname].extend([[float(c) for c in poly] for poly in seg if len(poly) >= 6])

    rows = []
    for row in meta.itertuples(index=False):
        r = row._asdict()
        fname = r["image_id"]
        fractured = int(r["fractured"])
        rel = f"images/{'Fractured' if fractured else 'Non_fractured'}/{fname}"
        if fname in id_to_file:
            width, height = id_to_file[fname]["width"], id_to_file[fname]["height"]
        else:
            with Image.open(root / rel) as im:  # header read only
                width, height = im.size
        rows.append(
            {
                "image_id": Path(fname).stem,
                "path": rel,
                "group_id": Path(fname).stem,  # no patient ID in FracAtlas (see module docstring)
                "source": "fracatlas",
                "site": "fracatlas",
                "body_part": _body_part(r),
                "view": _view(r),
                "hardware": int(r["hardware"]),
                "multiscan": int(r["multiscan"]),
                "fractured": fractured,
                "fracture_count": int(r["fracture_count"]),
                "width": int(width),
                "height": int(height),
                "boxes": json.dumps(boxes.get(fname, [])),
                "polygons": json.dumps(polygons.get(fname, [])),
            }
        )
    return pd.DataFrame(rows)
