"""Image I/O: every reader returns float32 HxW in [0, 1] (no torch)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        from xrayml.data.dicom import load_dicom

        return load_dicom(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return to_float01(img)


def decode_bytes(data: bytes, filename: str = "") -> np.ndarray:
    """Decode an uploaded file (DICOM, PNG, JPEG) — used by the API."""
    if filename.lower().endswith((".dcm", ".dicom")) or data[128:132] == b"DICM":
        import io

        import pydicom

        from xrayml.data.dicom import dicom_to_array

        return dicom_to_array(pydicom.dcmread(io.BytesIO(data)))
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("unsupported or corrupted image")
    return to_float01(img)


def to_float01(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 65535.0).astype(np.float32)
    if img.dtype == np.uint8:
        return (img.astype(np.float32) / 255.0).astype(np.float32)
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def write_png16(path: str | Path, img01: np.ndarray) -> None:
    """Store as 16-bit PNG: lossless and keeps the dynamic range of the original radiograph."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(img01 * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), arr):
        raise OSError(f"failed to write {path}")
