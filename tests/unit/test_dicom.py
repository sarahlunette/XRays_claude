import numpy as np
import pydicom
import pytest

from xrayml.data.dicom import deidentify, header_record, load_dicom, pseudonymise_id, write_synthetic_dicom
from xrayml.data.io import decode_bytes, read_image


def _ramp():
    return (np.tile(np.arange(64, dtype=np.uint16), (32, 1)) * 64).astype(np.uint16)  # 0..4032, 12-bit


def test_monochrome1_is_inverted(tmp_path):
    a = load_dicom(write_synthetic_dicom(tmp_path / "m2.dcm", _ramp(), photometric="MONOCHROME2"))
    b = load_dicom(write_synthetic_dicom(tmp_path / "m1.dcm", _ramp(), photometric="MONOCHROME1"))
    assert a.dtype == np.float32 and a.min() >= 0 and a.max() <= 1
    assert a[0, -1] > a[0, 0]  # bright on the right
    assert np.allclose(a, 1 - b, atol=1e-6)


def test_window_clips_to_range(tmp_path):
    arr = load_dicom(write_synthetic_dicom(tmp_path / "w.dcm", _ramp(), window=(2000, 1000)))
    # values below center - width/2 are black, above center + width/2 are white
    assert arr[0, :20].max() == pytest.approx(0.0, abs=1e-6)
    assert arr[0, -10:].min() == pytest.approx(1.0, abs=1e-6)


def test_modality_lut_applied_before_normalisation(tmp_path):
    a = load_dicom(write_synthetic_dicom(tmp_path / "a.dcm", _ramp(), slope=1.0, intercept=0))
    b = load_dicom(write_synthetic_dicom(tmp_path / "b.dcm", _ramp(), slope=2.0, intercept=-100))
    assert np.allclose(a, b, atol=1e-3)  # a linear rescale does not change the normalised image


def test_deidentify_removes_phi_and_keeps_stable_pseudonym(tmp_path):
    p = write_synthetic_dicom(tmp_path / "x.dcm", _ramp(), patient_id="12345")
    ds = deidentify(pydicom.dcmread(p), salt="s")
    assert "PatientName" not in ds and "PatientBirthDate" not in ds and "InstitutionName" not in ds
    assert ds.PatientID == pseudonymise_id("12345", "s") != pseudonymise_id("12345", "other-salt")
    assert ds.PatientIdentityRemoved == "YES"


def test_header_audit_and_readers(tmp_path):
    p = write_synthetic_dicom(tmp_path / "x.dcm", _ramp(), manufacturer="VendorX", body_part="WRIST")
    rec = header_record(p)
    assert (
        rec["Manufacturer"] == "VendorX" and rec["BitsStored"] == 12 and rec["PixelSpacing"] == [0.15, 0.15]
    )
    assert np.allclose(read_image(p), decode_bytes(p.read_bytes(), "upload.bin"))  # DICM magic detected


def test_write_requires_uint16(tmp_path):
    with pytest.raises(TypeError):
        write_synthetic_dicom(tmp_path / "x.dcm", np.zeros((4, 4), np.float32))
