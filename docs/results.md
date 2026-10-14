# Results

Two runs, both produced with the commands in this repo on a **2-vCPU codespace without GPU**.
They prove the pipeline works and show what the guards catch. They are not benchmarks.

## 1. Synthetic fissures (what CI runs) — `make synth-demo`, 6 min

800 synthetic radiographs at 128 px, 2–5 px fracture lines, 3 sites; `site_c` held out as an
external test site. ResNet-10t from scratch, 8 epochs.

| | AUC (95 % CI, patient bootstrap) | notes |
|---|---|---|
| val | 0.994 | after epoch 7 (0.73 after epoch 0) |
| test | 0.986 [0.959, 1.000] | 82 images |
| external site | 0.956 [0.930, 0.978] | unseen exposure/noise profile → the expected drop |

* Operating point tuned on val for sensitivity ≥ 0.90 → threshold 0.307 → **test sensitivity
  0.92 [0.75, 0.98], specificity 0.965**.
* ECE 0.21: ranking is excellent but probabilities are poorly calibrated. That is fine for a
  thresholded decision, not fine if the product displays a probability → temperature scaling.
* ONNX parity: max |Δlogit| 2e-6, **0 decision flips**.
* Release gate **passed** (sensitivity lower CI bound 0.75 ≥ 0.70) — with only 25 test
  positives the bound is wide, which is the argument for larger frozen test sets.

Full report: [`reports/synthetic_validation_report.md`](reports/synthetic_validation_report.md).
The same report is regenerated in every `integration` workflow run (job summary).

## 2. FracAtlas, CPU-only — `configs/classification_fracatlas_cpu.yaml`, ~75 min

4,083 radiographs, 512 px cache, deduplicated split (3,105 train / 377 val / 601 test).
ImageNet-pretrained ResNet-18 at 256 px, 5 epochs, weighted BCE, EMA.

| | value |
|---|---|
| val AUC | 0.928 |
| **test AUC** | **0.899 [0.864, 0.933]** · AP 0.768 · ECE 0.047 |
| threshold (val, target sens 0.90) | 0.287 global, 0.307 for hand (enough positives) |
| test sensitivity at that threshold | **0.78 [0.70, 0.85]**, specificity 0.85 |
| ONNX parity (601 images) | max Δ 1.8e-5, 0 decision flips |
| release gate | **FAILED** — sensitivity lower CI bound 0.68 < 0.80 |

Full report: [`reports/fracatlas_cpu_validation_report.md`](reports/fracatlas_cpu_validation_report.md) ·
datasheet: [`reports/fracatlas_datasheet.md`](reports/fracatlas_datasheet.md).

### What this run teaches (interview material)

1. **The gate did its job.** A threshold tuned on 63 validation positives for 90 % sensitivity
   gave 78 % on test. Small tuning sets make operating points unstable; the fix is more
   tuning positives (or cross-validated threshold selection), not a lower bar.
2. **Shortcut learning, found by the subgroup table.** In FracAtlas, 97 of the 99 images with
   metal hardware are fractured. The model scores hardware images 0.89 on average vs 0.61 for
   fractures without hardware: it partly learned "metal ⇒ fracture". Mitigations: report
   performance on `hardware == 0` (AUC 0.887 here), balance or stratify hardware cases,
   check Grad-CAM (`xrayml.explain.gradcam`), and state it as a known issue in the datasheet.
3. **Leakage was real before the fix.** Before `regroup-duplicates`, 45 pixel-identical images
   sat across splits; numbers above use the fixed split.
4. **Weak subgroups.** Oblique views (AUC 0.74, 5 positives) and "mixed" body parts (0.81)
   are where to collect data next.
5. **Resolution matters.** 256 px is far below the native ~2,900 px; hairline fractures
   vanish. The GPU recipe (`configs/classification_fracatlas.yaml`: 384–1024 px, 12+ epochs,
   bf16) and detection (`configs/detection_fracatlas.yaml`) are the next runs.

### Reproduce

```bash
make fracatlas-data                      # or use --max-side 512 in the prepare step for CPU
uv run xrayml train configs/classification_fracatlas_cpu.yaml
for s in val test; do uv run xrayml evaluate --checkpoint runs/cls_fracatlas_cpu/best.ckpt \
  --manifest data/fracatlas_cache/manifest.parquet --split $s --out reports/fracatlas; done
uv run xrayml select-ops --checkpoint runs/cls_fracatlas_cpu/best.ckpt --preds reports/fracatlas/preds_val.parquet \
  --target-sensitivity 0.9 --version op-fracatlas-cpu-1 --out bundle_fracatlas/thresholds.json
uv run xrayml export --checkpoint runs/cls_fracatlas_cpu/best.ckpt --manifest data/fracatlas_cache/manifest.parquet \
  --thresholds bundle_fracatlas/thresholds.json --out bundle_fracatlas
uv run xrayml gate --preds reports/fracatlas/preds_test.parquet --ops bundle_fracatlas/thresholds.json --out reports/fracatlas/gate.json
```
