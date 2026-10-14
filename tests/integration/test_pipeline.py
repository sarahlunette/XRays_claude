"""End-to-end on synthetic data, through the CLI: the same commands CI and DVC run.

synth -> train -> evaluate (val, test) -> select-ops -> export (+parity) -> gate -> report
-> serve (PNG + DICOM) -> prediction log -> drift.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from xrayml.cli import main

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    data, run, rep, bundle = root / "data", root / "run", root / "reports", root / "bundle"
    assert main(["synth", "--out", str(data), "--n", "160", "--size", "64", "--external-site", "site_c"]) == 0
    cfg = {
        "task": "classification",
        "output_dir": str(run),
        "seed": 0,
        "data": {
            "manifest": str(data / "manifest.parquet"),
            "image_size": 64,
            "batch_size": 16,
            "num_workers": 0,
            "dataset_version": "synthetic-e2e",
        },
        "model": {"name": "timm_classifier", "params": {"backbone": "resnet10t"}},
        "optim": {"epochs": 2, "lr": 1e-3, "warmup_steps": 2},
    }
    (root / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    assert main(["train", str(root / "cfg.yaml"), "--device", "cpu"]) == 0
    ck = str(run / "best.ckpt")
    m = str(data / "manifest.parquet")
    for split in ("val", "test", "external_test"):
        assert (
            main(
                [
                    "evaluate",
                    "--checkpoint",
                    ck,
                    "--manifest",
                    m,
                    "--split",
                    split,
                    "--out",
                    str(rep),
                    "--n-boot",
                    "50",
                ]
            )
            == 0
        )
    assert (
        main(
            [
                "select-ops",
                "--checkpoint",
                ck,
                "--preds",
                str(rep / "preds_val.parquet"),
                "--target-sensitivity",
                "0.9",
                "--global-only",
                "--version",
                "op-test",
                "--out",
                str(bundle / "thresholds.json"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "export",
                "--checkpoint",
                ck,
                "--manifest",
                m,
                "--thresholds",
                str(bundle / "thresholds.json"),
                "--out",
                str(bundle),
            ]
        )
        == 0
    )
    gate_rc = main(
        [
            "gate",
            "--preds",
            str(rep / "preds_test.parquet"),
            "--ops",
            str(bundle / "thresholds.json"),
            "--min-sensitivity-lcb",
            "0.0",
            "--min-specificity",
            "0.0",
            "--out",
            str(rep / "gate.json"),
        ]
    )
    assert (
        main(
            [
                "report",
                "--bundle",
                str(bundle),
                "--test-metrics",
                str(rep / "metrics_test.json"),
                "--test-preds",
                str(rep / "preds_test.parquet"),
                "--gate",
                str(rep / "gate.json"),
                "--external-metrics",
                str(rep / "metrics_external_test.json"),
                "--out",
                str(rep / "report.md"),
            ]
        )
        == 0
    )
    return {"root": root, "data": data, "rep": rep, "bundle": bundle, "gate_rc": gate_rc}


def test_artifacts_and_lineage(pipeline):
    b = pipeline["bundle"]
    card = json.loads((b / "model_card.json").read_text())
    ops = json.loads((b / "thresholds.json").read_text())
    assert card["model_version"] == ops["model_version"]  # thresholds belong to these weights
    assert card["dataset_version"] == "synthetic-e2e" and card["dataset_fingerprint"]
    assert json.loads((b / "parity.json").read_text())["decision_flips"] == 0
    assert pipeline["gate_rc"] == 0
    report = (pipeline["rep"] / "report.md").read_text()
    assert "Release gate" in report and "External test" in report and card["model_version"] in report


def test_api_png_dicom_and_prediction_log(pipeline, tmp_path, monkeypatch):
    import cv2
    from fastapi.testclient import TestClient

    from xrayml.data.dicom import write_synthetic_dicom

    log = tmp_path / "pred.jsonl"
    monkeypatch.setenv("XRAYML_PREDICTION_LOG", str(log))
    from xrayml.serving.app import create_app

    client = TestClient(create_app(pipeline["bundle"]))
    assert client.get("/health").json() == {"status": "ok"}
    v = client.get("/version").json()
    assert v["thresholds_version"] == "op-test"

    png = next((pipeline["data"] / "images").glob("*.png"))
    r = client.post(
        "/predict", files={"file": ("x.png", png.read_bytes(), "image/png")}, headers={"X-Site-Id": "site_a"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["score"] <= 1 and body["positive"] == (body["score"] >= body["threshold"])

    img16 = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    dcm = write_synthetic_dicom(tmp_path / "x.dcm", (img16 >> 4).astype(np.uint16), body_part="HAND")
    r = client.post("/predict", files={"file": ("x.dcm", dcm.read_bytes(), "application/dicom")})
    assert r.status_code == 200 and r.json()["operating_point_body_part"] == "all"

    assert (
        client.post("/predict", files={"file": ("bad.png", b"not an image", "image/png")}).status_code == 422
    )
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["site"] == "site_a"

    # drift: live log vs validation reference
    out = tmp_path / "drift.csv"
    assert (
        main(
            [
                "drift",
                "--reference",
                str(pipeline["rep"] / "preds_val.parquet"),
                "--ops",
                str(pipeline["bundle"] / "thresholds.json"),
                "--live",
                str(log),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert out.exists()


def test_thresholds_for_another_model_are_refused(pipeline, tmp_path):
    import shutil

    from xrayml.serving.app import Bundle

    b = tmp_path / "bundle"
    shutil.copytree(pipeline["bundle"], b)
    ops = json.loads((b / "thresholds.json").read_text())
    ops["model_version"] = "cla-0000000000"
    (b / "thresholds.json").write_text(json.dumps(ops))
    with pytest.raises(RuntimeError, match="thresholds tuned for model"):
        Bundle(b)
