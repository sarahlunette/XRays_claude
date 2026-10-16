# xrayml — fracture & fissure detection on X-rays, engineered end to end

A complete, runnable reference for the ML **Engineer** side of an AI radiology product
(fracture / fissure detection on musculoskeletal radiographs): versioned datasets, three CV
tasks (classification, detection, segmentation), a defensible training loop, operating points
as release artefacts, automated validation and release gate, ONNX export with parity checks,
a torch-free inference API, drift monitoring, and a GitHub dev → test → deploy pipeline.

It is built as a companion to a 2-day interview revision plan: every chapter of the plan is
implemented, tested, and mapped to code in [`docs/interview_map.md`](docs/interview_map.md).

> **Not a medical device.** Research / educational code. No clinical use.

## 👉 Where to look first

1. **[`docs/START_HERE.md`](docs/START_HERE.md)** — a 2-hour reading path through 17 files, in order.
   New to a term in this README? **[`docs/readme_explained.md`](docs/readme_explained.md)** explains
   every one, section by section, with worked numbers from `scripts/readme_terms_demo.py`.
2. [`src/xrayml/engine/trainer.py`](src/xrayml/engine/trainer.py) — the training loop you should be able to explain line by line.
3. [`src/xrayml/operating_points.py`](src/xrayml/operating_points.py) + [`src/xrayml/evaluation/release_gate.py`](src/xrayml/evaluation/release_gate.py) — from scores to a shipped, gated decision.
4. [`src/xrayml/serving/app.py`](src/xrayml/serving/app.py) — what production looks like (no torch, versioned responses, prediction log).
5. [`.github/workflows/`](.github/workflows/) — `ci.yml` (dev) · `integration.yml` (test) · `release.yml` (deploy).
6. [`docs/development.md`](docs/development.md) and [`docs/deployment.md`](docs/deployment.md) — how work flows, and how a model/threshold reaches production.

## Quickstart

```bash
pip install uv
make setup            # venv from uv.lock (CPU torch) + pre-commit hooks
make test             # ruff + mypy + 95 tests (~40 s)
make synth-demo       # full release path on synthetic fissures (~5 min on CPU)
cat reports/synth/validation_report.md
make serve            # API on :8000, then: python scripts/smoke_test.py
```

Real data (FracAtlas, CC BY 4.0, 338 MB):

```bash
make fracatlas-data   # download (md5) -> manifest -> 1024 px cache -> dedup regroup -> quality report
make fracatlas-train  # GPU recommended; CPU recipe: configs/classification_fracatlas_cpu.yaml
```

## The pipeline, one command per step

```
xrayml synth | download-fracatlas → build-manifest → prepare → regroup-duplicates → quality
      → train → evaluate (val/test/external) → select-ops → export (+parity) → gate → report
      → serve → drift
```

Each step reads files and writes files, so the same commands run from a laptop, `dvc repro`
([`dvc.yaml`](dvc.yaml)), an Airflow task ([`pipelines/`](pipelines/)) or GitHub Actions.

## What is inside

| Area | Highlights |
|---|---|
| **Data** | Manifest + schema validation + fingerprint; deterministic patient-level splits; external-site hold-out; DICOM (modality/VOI LUT, MONOCHROME1, de-identification); decode-once cache; pHash band-LSH near-duplicates + union-find regrouping; datasheets |
| **Models** | timm classifier (1-channel stem), Faster R-CNN / RetinaNet (small elongated anchors), U-Net; ResNet bottleneck, MBConv+SE, DenseNet layer, BN→GN; registry pattern |
| **Training** | AMP (bf16 / fp16 + GradScaler), accumulation with DDP `no_sync`, clipping after `unscale_`, warmup+cosine per step, AdamW groups, EMA, NaN guard, exact resume (RNG + per-epoch seeding), torchrun-ready, optional MLflow |
| **Evaluation** | AUC/AP/ECE, patient-level bootstrap CIs, Wilson intervals, FROC + CPM, Dice, subgroup tables, temperature scaling, Grad-CAM |
| **Release** | Operating points as a versioned artefact (per body part), release gate (acceptance criteria + paired non-inferiority vs production), ONNX export with decision-level parity, model card with lineage, generated validation report |
| **Serving** | FastAPI on ONNX Runtime, PNG/JPEG/DICOM, body-part thresholds, refuses mismatched thresholds, JSONL prediction log |
| **Monitoring** | PSI on input statistics, positive-rate drift per site |
| **NLP** | FR/EN rule-based report labeller (negation, uncertainty, conclusion section) + κ evaluation |
| **Platform** | `uv.lock`, pre-commit = CI, Docker (serving without torch; CUDA training image), DVC pipeline, Airflow DAG, Spark job, SQL & Mongo references |

## Results

See [`docs/results.md`](docs/results.md) for the synthetic CI run and a CPU-only FracAtlas run
(numbers, how they were produced, and their limits).

## Data and credits

FracAtlas — Abedeen I. et al., *FracAtlas: A Dataset for Fracture Classification, Localization
and Segmentation of Musculoskeletal Radiographs*, Scientific Data 10, 521 (2023),
[doi:10.6084/m9.figshare.22363012](https://doi.org/10.6084/m9.figshare.22363012), CC BY 4.0.
Data is downloaded at run time and never committed. See [`docs/data.md`](docs/data.md).

## Layout

```
src/xrayml/   data/ models/ engine/ evaluation/ export/ serving/ monitoring/ nlp/ explain/ metrics/
configs/      YAML experiments (synthetic for CI, FracAtlas for real runs)
tests/        unit/ (fast) · integration/ (CLI → bundle → API)
pipelines/    airflow/ spark/ sql/ mongo/   (+ dvc.yaml, params.yaml at the root)
docker/       Dockerfile.serve (runtime | release) · Dockerfile.train
docs/         START_HERE · readme_explained · interview_map · development · deployment · data · results
.github/      workflows (ci, integration, release) · PR template · issue forms · CODEOWNERS · dependabot
```
