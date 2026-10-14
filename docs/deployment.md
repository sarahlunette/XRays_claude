# Deployment guide — models and operating points to production

Goal from the job description: *significantly reduce the time it takes to move models and
operating points into production*. In a regulated device, speed comes from making validation
**cheap, automatic and pre-agreed** — not from skipping it.

## 1. The release bundle

A release is three files, each versioned and each traceable to the others:

| File | Produced by | Contains |
|---|---|---|
| `model.onnx` | `xrayml export` | weights + normalisation (inside the graph) |
| `thresholds.json` | `xrayml select-ops` | `OperatingPointSet`: version, `model_version` it was tuned for, dataset version, per-body-part thresholds with validated sensitivity, CI, specificity |
| `model_card.json` | `xrayml export` | `model_version` (hash of the weights), git SHA, dataset version + fingerprint, full config, parity result |

Safety interlocks in code:
* the API refuses to start if `thresholds.model_version != model_card.model_version`
  (`serving/app.py::Bundle`), and so does `xrayml export`;
* every API response carries `model_version` and `thresholds_version`;
* the serving image bakes the bundle in (`docker/Dockerfile.serve --target release`), so the
  **image digest identifies weights + thresholds**.

## 2. The 8 gates, and where each lives

```
 train ──► register ──► validate ──► operating points ──► export+parity ──► integration ──► release ──► monitor
 config     model_card   evaluate     select-ops           export            smoke test      staging     drift
 +dataset   (+MLflow)    report+gate  thresholds.json      parity.json       container       production  PSI, rate
```

| Gate | Command | Fails the pipeline when… |
|---|---|---|
| 1 Train | `xrayml train cfg.yaml data.dataset_version=…` | config invalid, loss non-finite |
| 2 Register | lineage written into the checkpoint and `model_card.json` | — |
| 3 Validate | `xrayml evaluate --split val/test/external_test` | — (produces numbers + CIs + subgroups) |
| 4 Operating points | `xrayml select-ops --target-sensitivity 0.9` (on **val**) | — |
| 5 Export | `xrayml export` | any score difference > tolerance or **any decision flips** at the shipped thresholds |
| 6 Gate | `xrayml gate` (on **test**, optionally vs production predictions) | sensitivity lower CI bound < criterion, specificity < criterion, non-inferiority not shown, specificity drop > tolerance |
| 7 Release | `release.yml`: staging → production (approval) | smoke test fails; reviewer rejects |
| 8 Monitor | prediction log → `xrayml drift` | PSI > 0.25 on image statistics, positive rate ±50 % vs reference |

The validation report (`xrayml report`) is generated from these artefacts and attached to the
GitHub Release: the ML part of the technical documentation can never disagree with what shipped.

## 3. Environments

| Environment | What runs | How it is reached | Protection |
|---|---|---|---|
| local / dev | `make synth-demo`, `make serve` | developer laptop / codespace | pre-commit |
| CI (test) | `integration.yml`: model validation + container smoke test | every PR | required status checks |
| **staging** | `release.yml` job `deploy-staging`: image pushed to `ghcr.io/<repo>-serve:<tag>` and `:<model_version>`, started and smoke-tested | tag or manual run | none (automatic) |
| **production** | `release.yml` job `deploy-production`: GitHub Release with bundle + report | after staging | **required reviewers** on the environment |

The staging/production "deploy" steps are deliberately thin: this repo has no cluster. In a
real setup the staging step is `helm upgrade --install … --set image.tag=<model_version>` (or a
GitOps commit to ArgoCD/Flux), and production is a staged rollout per site (shadow mode first,
then canary sites, then all), driven by the same image tag.

## 4. Three kinds of release

**A. Threshold-only release** (fastest path — no retraining):

```bash
xrayml select-ops --checkpoint <same ckpt> --preds reports/preds_val.parquet \
    --target-sensitivity 0.92 --version op-2026.11.1 --out bundle/thresholds.json
xrayml gate --preds reports/preds_test.parquet --ops bundle/thresholds.json --prod-preds prod/preds_test.parquet --prod-ops prod/thresholds.json
git tag v0.2.1 && git push --tags      # release.yml rebuilds, gates, stages, asks for approval
```

**B. Retraining (same architecture, new data)** — new dataset version → train → full gate set.
Under a PCCP (FDA) / pre-agreed change protocol (MDR), this follows the documented protocol
with the acceptance criteria from the change-request issue.

**C. New architecture / body part / finding** — same pipeline, plus a reader study and a
regulatory assessment (significant change). The code path does not change; the paperwork does.

## 5. Rollback

The previous image tag and bundle stay in GHCR and in the previous GitHub Release. Rollback =
redeploy the previous tag (one command / one GitOps revert). Because thresholds are part of
the bundle, a model and its thresholds always roll back together.

## 6. Monitoring after release

* The API appends one JSON line per prediction when `XRAYML_PREDICTION_LOG` is set: site
  (`X-Site-Id` header), body part, score, decision, image statistics (computed on the model
  input — the same definition as the validation reference).
* `xrayml drift --reference reports/preds_val.parquet --ops bundle/thresholds.json --live preds.jsonl`
  gives one row per site: PSI per statistic, positive rate vs reference, alert + reason.
* Ground truth arrives late: sample positive and negative cases per site for radiologist
  audit; feed confirmed misses into the dataset (new version) — the loop closes in Chapter 6.

## 7. Mapping to the regulated lifecycle

| Requirement | Evidence produced by this repo |
|---|---|
| IEC 62304 configuration management | git SHA + `uv.lock` + image digest in the model card / release |
| IEC 62304 SOUP list | `uv.lock` (serving image has no torch: shorter list) |
| ISO 14971 risk controls | operating point (sensitivity target), refusal of mismatched thresholds, input validation (422 on corrupt files), drift alerts |
| ISO 13485 change control | PR template checklist, model-change issue form, required reviewers on `production` |
| Data governance (EU AI Act, GMLP) | dataset version + fingerprint, datasheet, leakage checks, external-site test |
| Verification & validation report | `validation_report.md` generated and attached to every release |

## 8. Running the serving image yourself

```bash
make synth-demo                                   # produces ./bundle
docker build -f docker/Dockerfile.serve --target release -t xrayml-serve:local .
docker run --rm -p 8000:8000 -e XRAYML_PREDICTION_LOG=/tmp/p.jsonl xrayml-serve:local
python scripts/smoke_test.py http://localhost:8000
curl -F "file=@some_xray.dcm" -H "X-Site-Id: site_a" localhost:8000/predict
```
