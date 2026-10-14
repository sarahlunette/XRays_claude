# Development guide

How code gets from an idea to `main`, and the conventions that keep a research codebase
fast to change *and* safe to ship. This is the "own the codebase, review PRs, spread good
practices" part of the role.

## 1. Setup (once)

```bash
pip install uv                      # or: curl -LsSf https://astral.sh/uv/install.sh | sh
make setup                          # uv sync --frozen (CPU torch from uv.lock) + pre-commit hooks
make test                           # lint + mypy + all tests, ~40 s on a laptop
```

`uv.lock` pins every transitive dependency: a fresh clone, CI and the Docker image resolve the
same versions. GPU machines install CUDA torch on top (see `docker/Dockerfile.train`).

## 2. Repository layout and dependency direction

```
src/xrayml/
  config.py registry.py artifacts.py        <- no torch: shared contracts
  data/      manifest, splits, quality, DICOM, preprocessing, datasets
  models/    layers, classifier, detector, unet      (registered by name)
  losses.py  calibration.py  operating_points.py
  metrics/   classification, detection (FROC), segmentation
  engine/    trainer, checkpoint, distributed, seed
  evaluation/ predict, summary, evaluate, release_gate, report
  export/    onnx_export (+ parity)
  serving/   FastAPI app on ONNX Runtime  (imports: artifacts, data.io, data.preprocess only)
  monitoring/ drift     nlp/ report_labeler     explain/ gradcam
  cli.py     every pipeline step as one command
configs/     YAML experiments (validated by pydantic)
tests/       unit/ (fast, run on every push) + integration/ (end-to-end)
pipelines/   Airflow, Spark, SQL, Mongo reference code;  dvc.yaml at the root
```

Rule: **research experiments change configs, not core code.** A new backbone is a config line;
a new loss or model is one registered function; a new dataset is one adapter that outputs the
manifest schema. Nothing under `engine/` or `evaluation/` should need to change.

## 3. Day-to-day workflow (trunk-based)

1. Open or pick an issue. Model / threshold / dataset changes use the **Model change request**
   issue form: it forces the acceptance criteria to be written *before* the experiment.
2. Branch from `main`: `feat/<short-name>`, `fix/…`, `exp/…` (experiments), `data/…`.
3. Commit small, with Conventional Commits: `feat(detector): small elongated anchors`,
   `fix(data): re-filter boxes after augmentation`, `test: …`, `docs: …`, `ci: …`.
   Pre-commit runs ruff, ruff-format, mypy and refuses DICOM files.
4. Push and open a PR early (draft). The template asks *what/why*, *how tested*, and for
   regulated changes a checklist (dataset fingerprint, ops tuned on val, gate passed,
   subgroups reviewed, parity, risk impact).
5. CI must be green (see §6). One approving review from a CODEOWNER.
6. **Squash-merge**; delete the branch. `main` is always releasable.

Keep PRs under ~400 changed lines; split refactors from behaviour changes so the reviewer can
check "no output changed" separately from "this output changed on purpose".

## 4. How to review a PR here (checklist)

In order of what costs most if missed:

1. **Intent** — is this the right change? Linked issue? For model changes: were acceptance
   criteria agreed before looking at test results?
2. **Correctness** — box formats (xyxy vs xywh), letterbox scale, empty boxes / masks,
   batch of 1, single-class splits (`safe_auc`), dtype/device, `model.eval()` + `no_grad`.
3. **Leakage** — anything that touches splits, dedup, `group_id`, or tunes on `test`.
4. **Train/serve skew** — does preprocessing change in one path only? (`data/preprocess.py`
   is the only place allowed.)
5. **Reproducibility** — seeds, config logged, no hard-coded paths, dataset version passed.
6. **Tests** — does a test fail without the change? For a bug fix: a regression test.
7. **Readability, performance** — Python loops over pixels, `.item()` in the hot loop.
8. **Style** — tools handle it; do not comment on it.

Prefix comments: `blocking:`, `suggestion:`, `question:`, `nit:`, and say what you liked.

## 5. Tests — what goes where

| Layer | Folder / marker | Runs | Examples |
|---|---|---|---|
| Unit (fast, pure) | `tests/unit`, not `slow` | every push (`ci.yml`) | IoU properties (Hypothesis), focal loss values, DICOM LUTs, split determinism, NLP cases |
| Slow unit | `@pytest.mark.slow` | `integration.yml` | resume after pre-emption reproduces the uninterrupted run |
| Integration | `tests/integration`, `@pytest.mark.integration` | `integration.yml` | CLI chain → bundle → API with PNG + DICOM → drift |
| Model validation | `make synth-demo` | `integration.yml` | the model must *learn* (val AUC > 0.9) and pass the release gate |
| Container | `scripts/smoke_test.py` | `integration.yml`, `release.yml` | the image starts and answers `/predict` |

ML-specific tests worth copying: *overfit one batch*, *metric equals a reference
implementation* (`nms` vs torchvision, LSH vs brute force), *property tests* on geometry,
*determinism / resume* tests, *train/serve parity* tests.

## 6. CI stages: dev → test → deploy

| Stage | Workflow | Trigger | Jobs | Time |
|---|---|---|---|---|
| **dev** | `.github/workflows/ci.yml` | every PR, push to main | ruff, mypy, unit tests on py3.11 + 3.12 with coverage, guard against medical images in git | ~3 min |
| **test** | `.github/workflows/integration.yml` | PRs touching code/config, push to main, manual | integration tests; model validation on synthetic data (report in the job summary, bundle as artifact); build the serving image and smoke-test it | ~12 min |
| **deploy** | `.github/workflows/release.yml` | tag `v*`, manual | rebuild bundle + gate → push image to GHCR → **staging** environment smoke test → **production** environment (manual approval) → GitHub Release = release record | ~15 min |

Recommended repository settings (Settings → Branches → `main`): require PR, require status
checks `lint (ruff)`, `typecheck (mypy)`, `unit tests (py3.12)`, `integration-tests`,
`model-validation`; require CODEOWNER review; no force-push. Settings → Environments:
`staging` (no rule), `production` (required reviewers, deployment branches = tags `v*`).

## 7. Extending the codebase

**A new backbone** — config only: `model.params.backbone: convnext_tiny` (any timm name).

**A new model**

```python
# src/xrayml/models/my_model.py
from xrayml.registry import MODELS


@MODELS.register("my_model")
class MyModel(nn.Module): ...
```

Import it in `models/__init__.py`, add a shape test in `tests/unit/test_models.py`, reference
it from a config. The trainer, evaluator, exporter and API do not change.

**A new loss** — `@LOSSES.register("name")` in `losses.py`, takes logits, returns a scalar;
add a "known value" test (e.g. focal with γ=0 equals BCE).

**A new dataset** (e.g. GRAZPEDWRI-DX, 20k paediatric wrists with real patient IDs) — write an
adapter like `data/fracatlas.py` that returns the manifest columns (`image_id, path,
group_id, source, site, body_part, fractured, width, height, boxes, polygons`). Use the real
patient id as `group_id`. Everything downstream works unchanged.

**A foundation-model fine-tune** — layer-wise LR decay belongs in
`engine/trainer.py::param_groups` (one group per block with `lr * decay**depth`); freeze /
unfreeze schedules as trainer callbacks.

## 8. Debugging training — what actually happened while building this repo

These three bugs were found by the guards in this codebase; they make good interview stories.

1. **Detector loss → NaN at step 16.** The `FloatingPointError` guard stopped the run. Tracing
   per-step losses showed `loss_classifier` climbing 0.7 → 650 → NaN: LR 1e-3 is too high for
   Faster R-CNN from scratch. Fix: LR 1e-4 + warmup (`configs/detection_*.yaml`). A second,
   independent fix: boxes shrunk below 2 px by augmentation are now dropped (`log(0)` in the
   box encoding), see `datasets._valid_boxes`.
2. **Patient leakage in the synthetic generator.** `assign_splits` failed its own
   `assert_no_group_leakage`: the site was drawn per image, so one patient spanned two sites
   and the external-site hold-out split the patient. Fixes: site per patient in the generator,
   external membership decided per group.
3. **Real leakage in FracAtlas.** `xrayml quality --dedupe` found 148 near-duplicate pairs
   across splits, 45 of them pixel-identical (Hamming 0). FracAtlas has no patient id, so
   views/re-exports of one patient were split apart → optimistic test metrics. Fix:
   `xrayml regroup-duplicates` (pHash band-LSH + union-find, Hamming ≤ 2): 4,083 → 3,950
   groups, 0 cross-split duplicates. Numbers in `docs/results.md` use the fixed split.
4. **Label-mining regex**: "Fracture cannot be excluded" was labelled *positive* and
   "ne peut être exclue" would have been *negative*. Fix: add the cue, and check uncertainty
   before post-negation. The test table in `test_nlp_drift.py` grew by two cases.

General recipe (plan 5.10): overfit one batch → check the loss on a batch by hand → log per
step → `torch.autograd.set_detect_anomaly(True)` briefly → profile the loader if GPU < 80 %.
