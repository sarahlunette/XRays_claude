# Start here — a guided reading path (≈ 2 hours)

The repository is large; you do not need all of it. Read these files **in this order**. Each
step says what to look for and which revision-plan chapter it rehearses. Open the file, read
the module docstring first, then the code.

## Part 1 — follow one image from disk to a decision (45 min)

| # | File | Look for | Plan |
|---|---|---|---|
| 1 | [`src/xrayml/data/manifest.py`](../src/xrayml/data/manifest.py) | Why training reads a *manifest*, not a folder. `assign_splits`: a hash of the patient id decides the split (deterministic, no leakage). `validate_manifest` collects every problem before failing. | 6.1, 6.3 |
| 2 | [`src/xrayml/data/fracatlas.py`](../src/xrayml/data/fracatlas.py) | Download with checksum, COCO `xywh` → our `xyxy` (the #1 detection bug), and the honest list of dataset limitations in the docstring. | 1.3, 2.5 |
| 3 | [`src/xrayml/data/quality.py`](../src/xrayml/data/quality.py) | `find_near_duplicates` (band-LSH instead of O(n²)) and `regroup_near_duplicates` (union-find). This fixed **real leakage in FracAtlas**: 45 exact duplicates sat across train/test. | 6.3, 6.9 |
| 4 | [`src/xrayml/data/dicom.py`](../src/xrayml/data/dicom.py) | Modality LUT → VOI LUT → MONOCHROME1 inversion → [0,1]. De-identification with a stable pseudonym. | 2.1, 2.6 |
| 5 | [`src/xrayml/data/preprocess.py`](../src/xrayml/data/preprocess.py) | **One** letterbox implementation used by training *and* serving. Then read its guard: `tests/unit/test_preprocess.py::test_training_val_path_equals_serving_path`. | 8.5 |
| 6 | [`src/xrayml/data/datasets.py`](../src/xrayml/data/datasets.py) | Three target formats (cls / det / seg), boxes re-filtered *after* augmentation (this fixed a NaN), seeded workers, oversampling. | 5.2 |

## Part 2 — training you can defend line by line (30 min)

| # | File | Look for | Plan |
|---|---|---|---|
| 7 | [`src/xrayml/engine/trainer.py`](../src/xrayml/engine/trainer.py) | The docstring lists what interviewers check. Then: accumulation + `no_sync`, `unscale_` before clipping, scheduler per optimizer step, `loss.item()`, no weight decay on norms, EMA, NaN guard, per-epoch seeding. | 5.1, 5.3, 5.7 |
| 7b | [`src/xrayml/engine/gpu.py`](../src/xrayml/engine/gpu.py) + [`docs/gpu.md`](gpu.md) | The GPU version of the same loop: `amp: auto` (bf16 vs fp16 + scaler), TF32, cudnn.benchmark, channels_last, compile after DDP, fused AdamW, `barrier()` after rank-0 validation, effective batch across GPUs. | 5.9 |
| 8 | [`src/xrayml/engine/checkpoint.py`](../src/xrayml/engine/checkpoint.py) | Full state + RNG + lineage; atomic save. Proven by `tests/unit/test_engine.py::test_resume_reproduces_uninterrupted_run`. | 5.4 |
| 9 | [`src/xrayml/models/`](../src/xrayml/models/) | `layers.py` (ResNet bottleneck, MBConv, DenseNet layer, BN→GN), `detector.py` (Faster R-CNN on 1-channel images, small elongated anchors, `box_score_thresh` lowered so *the operating point* decides). | 1.1, 1.3 |

## Part 3 — from scores to a shipped decision (30 min) ← the core of the role

| # | File | Look for | Plan |
|---|---|---|---|
| 10 | [`src/xrayml/metrics/classification.py`](../src/xrayml/metrics/classification.py) | Patient-level bootstrap, Wilson interval, PPV at another prevalence. | 2.2, 2.7 |
| 11 | [`src/xrayml/metrics/detection.py`](../src/xrayml/metrics/detection.py) | FROC from scratch (double hit = neither TP nor FP), CPM. | 1.3, 2.8 |
| 12 | [`src/xrayml/operating_points.py`](../src/xrayml/operating_points.py) | Tune on val, confirm on test; per-body-part thresholds only with enough positives. | 2.3 |
| 13 | [`src/xrayml/artifacts.py`](../src/xrayml/artifacts.py) | Thresholds are a **separate, versioned artefact** tied to a `model_version`. | 8.1 |
| 14 | [`src/xrayml/evaluation/release_gate.py`](../src/xrayml/evaluation/release_gate.py) | Pre-agreed acceptance criteria + paired non-inferiority vs production. | 2.7, 8.7 |

## Part 4 — production (15 min)

| # | File | Look for | Plan |
|---|---|---|---|
| 15 | [`src/xrayml/export/onnx_export.py`](../src/xrayml/export/onnx_export.py) | Parity at *decision* level (flipped decisions), not only `allclose`. | 8.5 |
| 16 | [`src/xrayml/serving/app.py`](../src/xrayml/serving/app.py) | No torch; refuses thresholds tuned for another model; every response carries versions; prediction log → drift. | 8.5, 8.6 |
| 17 | [`src/xrayml/monitoring/drift.py`](../src/xrayml/monitoring/drift.py) | PSI on image statistics + positive-rate drift per site. | 8.6 |

## Part 5 — how the team works (read once)

* [`docs/development.md`](development.md) — branches, PRs, review checklist, test pyramid, adding a model.
* [`docs/deployment.md`](deployment.md) — dev → test → deploy, environments, threshold-only releases, rollback.
* [`.github/workflows/`](../.github/workflows/) — `ci.yml` (dev), `integration.yml` (test), `release.yml` (deploy).
* [`.github/PULL_REQUEST_TEMPLATE.md`](../.github/PULL_REQUEST_TEMPLATE.md) — the regulated-change checklist.

## Then run it

```bash
make setup          # uv venv from the lockfile (CPU torch)
make test           # lint + types + 111 tests (~40 s; CUDA tests skip without a GPU)
make synth-demo     # full release path on synthetic fissures (~5 min CPU)
cat reports/synth/validation_report.md
make serve          # then: python scripts/smoke_test.py
```

Every chapter of the revision plan is mapped to code in [`docs/interview_map.md`](interview_map.md).
