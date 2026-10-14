"""DICOM handling for radiographs (revision plan 2.1, 2.6, 2.8).

Order matters:
1. Modality LUT (RescaleSlope / RescaleIntercept): stored value -> physical value.
2. VOI LUT / windowing (WindowCenter / WindowWidth): physical value -> display range.
3. MONOCHROME1 inversion (white = low value on those devices).
4. Normalise to float32 [0, 1]. Never cast to uint8 before windowing: X-rays carry 10-16 bits.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

DX_STORAGE_FOR_PRESENTATION = UID("1.2.840.10008.5.1.4.1.1.1.1")

# Tags we audit before training (see `xrayml dicom-audit`) and stratify results by.
AUDIT_TAGS = [
    "PatientID",
    "StudyInstanceUID",
    "Modality",
    "BodyPartExamined",
    "ViewPosition",
    "PhotometricInterpretation",
    "BitsStored",
    "Manufacturer",
    "PixelSpacing",
]

# Minimal PHI list for the demo. Real pipelines follow the DICOM PS3.15 confidentiality profile.
PHI_TAGS = [
    "PatientName",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "AccessionNumber",
]


def load_dicom(path: str | Path) -> np.ndarray:
    """Read a radiograph as float32 in [0, 1], display-oriented (bone = bright)."""
    ds = pydicom.dcmread(path)
    return dicom_to_array(ds)


def dicom_to_array(ds: Dataset) -> np.ndarray:
    arr = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
    has_voi = "WindowCenter" in ds or "VOILUTSequence" in ds
    if has_voi:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
    else:
        # No windowing in the header: robust percentiles so one hot pixel does not crush contrast.
        lo, hi = (float(v) for v in np.percentile(arr, [0.5, 99.5]))
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = 1.0 - arr
    return arr.astype(np.float32)


def header_record(path: str | Path) -> dict[str, Any]:
    """Header-only read (fast: no pixel decoding) used to audit a folder before training."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    rec: dict[str, Any] = {}
    for tag in AUDIT_TAGS:
        v = getattr(ds, tag, None)
        rec[tag] = [float(x) for x in v] if tag == "PixelSpacing" and v is not None else v
    rec["path"] = str(path)
    return rec


def pseudonymise_id(patient_id: str, salt: str) -> str:
    """Stable pseudonym: same patient -> same hash, so patient-level splits stay possible."""
    return "p_" + hashlib.sha256(f"{salt}:{patient_id}".encode()).hexdigest()[:16]


def deidentify(ds: Dataset, salt: str) -> Dataset:
    """Remove direct identifiers and private tags; keep what the model and the audit need.

    Burned-in text (BurnedInAnnotation = YES) is *not* handled here: it needs pixel-level
    OCR/masking and is also a shortcut-learning risk.
    """
    ds.remove_private_tags()
    for tag in PHI_TAGS:
        if tag in ds:
            delattr(ds, tag)
    if "PatientID" in ds:
        ds.PatientID = pseudonymise_id(str(ds.PatientID), salt)
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "xrayml basic profile (demo)"
    return ds


def write_synthetic_dicom(
    path: str | Path,
    pixels: np.ndarray,
    *,
    photometric: str = "MONOCHROME2",
    bits_stored: int = 12,
    window: tuple[float, float] | None = None,
    slope: float = 1.0,
    intercept: float = 0.0,
    patient_id: str = "PAT0001",
    body_part: str = "WRIST",
    view: str = "PA",
    manufacturer: str = "SynthCo",
    pixel_spacing: tuple[float, float] = (0.15, 0.15),
) -> Path:
    """Write a valid uncompressed DX DICOM from a uint16 array (for tests and demos).

    FracAtlas ships JPEGs, so the DICOM path of the pipeline is exercised on synthetic files.
    """
    if pixels.dtype != np.uint16:
        raise TypeError("pixels must be uint16 (stored values)")
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DX_STORAGE_FOR_PRESENTATION
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = DX_STORAGE_FOR_PRESENTATION
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "DX"
    ds.PatientID = patient_id
    ds.PatientName = "Doe^Jane"
    ds.PatientBirthDate = "19800101"
    ds.InstitutionName = "Synthetic Hospital"
    ds.BodyPartExamined = body_part
    ds.ViewPosition = view
    ds.Manufacturer = manufacturer
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [str(pixel_spacing[0]), str(pixel_spacing[1])]
    ds.RescaleSlope = str(slope)
    ds.RescaleIntercept = str(intercept)
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = str(window[0]), str(window[1])
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return Path(path)
