# The README, explained term by term

The [README](../README.md) is dense: almost every line packs in a technical term. This page
follows the README section by section and, for each term, says **what it is**, **why it matters
for a fracture-detection product**, and **where it lives in the code**.

Numbers marked 🔢 come from [`scripts/readme_terms_demo.py`](../scripts/readme_terms_demo.py),
which computes small worked examples with the repo's own functions:

```bash
uv run --no-sync python scripts/readme_terms_demo.py   # < 1 min on CPU, no data needed
```

---

## 1. Title and opening paragraph

| Term | Meaning |
|---|---|
| **Fracture / fissure** | A fissure is a hairline fracture: a thin crack with no displacement. Fissures are the hard case because they are small, thin and low-contrast, and they are the ones radiologists miss. |
| **Musculoskeletal radiograph** | A plain X-ray of bones and joints (wrist, hand, ankle, hip…). It is 2-D, grayscale, often 10–16 bits deep, and stored as DICOM. |
| **ML Engineer side** | The repo focuses on everything *around* the model: data versioning, reproducible training, choosing thresholds, validation, export, serving and monitoring. A researcher invents the model; the ML engineer makes it shippable and auditable. |
| **Versioned datasets** | Every training set is a frozen, named table (a *manifest*) with a content hash, so you can answer "which images trained model v3.2?". → [`data/manifest.py`](../src/xrayml/data/manifest.py) |
| **Three CV tasks** | **Classification**: is there a fracture in this image (one score)? **Detection**: where is it (boxes)? **Segmentation**: which pixels (mask)? The same pipeline handles all three. |
| **Defensible training loop** | A loop where every line has a reason you can explain in an interview, e.g. why `unscale_` comes before clipping. → [`engine/trainer.py`](../src/xrayml/engine/trainer.py) |
| **Operating point** | The score threshold that turns a probability into a yes/no answer, together with the sensitivity and specificity it gives. It is the actual "product decision". |
| **Release artefact** | A file that ships to production and is versioned on its own: weights (`model.onnx`), thresholds (`thresholds.json`) and lineage (`model_card.json`). → [`artifacts.py`](../src/xrayml/artifacts.py) |
| **Release gate** | An automatic pass/fail check against criteria agreed *before* looking at the results. If it fails, the release is blocked. |
| **ONNX export with parity checks** | Convert the PyTorch model to ONNX (a portable model format), then check that the ONNX model gives the same *decisions* as PyTorch. → [`export/onnx_export.py`](../src/xrayml/export/onnx_export.py) |
| **Torch-free inference API** | The production service runs on ONNX Runtime only. The image is smaller, has fewer vulnerabilities, and has fewer third-party components to declare (SOUP, see §7). |
| **Drift monitoring** | Watching live inputs and outputs for changes against what the model was validated on, e.g. a new X-ray machine at a hospital. |
| **dev → test → deploy pipeline** | The three GitHub Actions workflows: lint and unit tests on every PR, end-to-end tests, then a gated release. |
| **Not a medical device** | Software that makes a diagnostic claim falls under MDR in the EU (CE marking) and needs an FDA 510(k) clearance in the US. This repo makes no such claim. |

## 2. "Where to look first"

These are the files to read first and why each one matters. For the full 2-hour path, see [START_HERE](START_HERE.md).

- **`trainer.py`**: the training loop. Interviewers like asking "walk me through your training loop".
- **`operating_points.py` + `release_gate.py`**: how scores become a decision that ships. This is the core of the role.
- **`serving/app.py`**: what production looks like. Every response carries `model_version` and `thresholds_version` (**traceability**: for any past answer, you can say which model and which thresholds produced it).
- **`ci.yml` / `integration.yml` / `release.yml`**: the dev, test and deploy stages.

## 3. Quickstart

| Term | Meaning |
|---|---|
| **uv / `uv.lock`** | A fast Python package manager. The lock file pins the *exact* version of every dependency, so your laptop, CI and Docker install identical environments. |
| **CPU torch** | The lock file uses the CPU build of PyTorch (smaller download) because CI and this codespace have no GPU. |
| **pre-commit hooks** | Checks that run automatically on `git commit`: ruff, notebook output stripping (outputs can contain patient images), and a block on large files (weights belong in DVC). These are the same checks CI runs. → [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) |
| **ruff** | Linter + formatter: style issues and common bugs. |
| **mypy** | Static type checker: catches `None` passed where a `float` is expected, before anything runs. |
| **Synthetic fissures** | Procedurally drawn "bones" with thin dark lines. They let CI run the full pipeline in minutes without downloading real data. → [`data/synthetic.py`](../src/xrayml/data/synthetic.py) |
| **Validation report** | A generated Markdown document with metrics, confidence intervals, gate results and lineage. It is the kind of evidence a regulatory file needs. → [`evaluation/report.py`](../src/xrayml/evaluation/report.py) |
| **Smoke test** | A quick "is it alive and answering sensibly?" check against the running API: `/health`, `/version` and one `/predict`. → [`scripts/smoke_test.py`](../scripts/smoke_test.py) |
| **FracAtlas, CC BY 4.0** | A public dataset of 4,083 musculoskeletal X-rays with fracture labels, boxes and masks. CC BY 4.0 lets you use it, including commercially, as long as you credit the authors. |
| **md5** | A checksum of the downloaded archive: it proves you got the exact file the dataset authors published. |
| **1024 px cache** | The images are decoded and resized once, then stored as lossless 16-bit PNGs. Otherwise each epoch re-decodes 2,880 px JPEGs and the GPU sits waiting for the CPU. → [`data/prepare.py`](../src/xrayml/data/prepare.py) |
| **Dedup regroup** | Near-identical images are put into the same group so they cannot end up on both sides of the train/test split (see pHash in §5). |
| **Quality report** | Counts, prevalence per split and body part, and cross-split duplicates, written as a *datasheet*. |

## 4. The pipeline, one command per step

```
synth | download → build-manifest → prepare → regroup-duplicates → quality
      → train → evaluate → select-ops → export (+parity) → gate → report → serve → drift
```

| Step | What it does |
|---|---|
| `build-manifest` | Scans the raw dataset and writes the manifest (one row per image: path, label, boxes, patient/group id, body part, site), then validates it. |
| `prepare` | Builds the decode-once cache. |
| `regroup-duplicates` | Merges groups linked by near-duplicate images. |
| `quality` | Writes the datasheet and fails on cross-split leakage. |
| `train` | Runs the trainer from a YAML config; writes `best.ckpt`, `last.ckpt` and `metrics.jsonl`. |
| `evaluate` | Predicts on `val`, `test` or `external_test`; writes per-image predictions + metrics with confidence intervals. |
| `select-ops` | Chooses thresholds on **val** and writes `thresholds.json`. |
| `export` | Converts to ONNX, checks parity and writes the bundle (model + thresholds + model card). |
| `gate` | Runs the release gate on **test**. |
| `report` | Generates the validation report. |
| `serve` / `drift` | Runs the API / compares the prediction log against the reference. |

| Term | Meaning |
|---|---|
| **"Each step reads files and writes files"** | No hidden state between steps. That is what lets the same commands run unchanged from a laptop, DVC, Airflow or GitHub Actions. |
| **DVC / `dvc repro`** | Data Version Control: git-like versioning for large files plus a pipeline (`dvc.yaml`). `dvc repro` reruns only the steps whose inputs changed, and `dvc.lock` records the hash of every input and output, which serves as proof of lineage. |
| **Airflow task / DAG** | Airflow is a scheduler that runs a DAG (directed acyclic graph) of tasks, e.g. a weekly "ingest → de-identify → QC → manifest" job. → [`pipelines/airflow/`](../pipelines/airflow/dags/dataset_release.py) |
| **GitHub Actions** | GitHub's CI/CD runner. It executes the workflows in `.github/workflows/`. |

## 5. "What is inside", row by row

### Data

| Term | Meaning |
|---|---|
| **Manifest** | The single table training reads (Parquet). Training never globs folders, so the training set is exactly what the table says. |
| **Schema validation** | `validate_manifest` checks columns, 0/1 labels, known body parts, boxes inside the image, and that every positive has a box. It collects *every* problem before failing once. |
| **Fingerprint** | A SHA-256 hash of ids, labels, boxes and splits. If one label changes, the fingerprint changes. It is stored in the model card. 🔢 `fingerprint: 73424e438e653fec` |
| **Deterministic patient-level split** | `hash(patient_id) % 100` decides train/val/test. It gives the same result on every machine and on every re-run, and adding new patients never moves old ones. Because the unit is the *patient*, two X-rays of the same wrist cannot sit in both train and test (**leakage**, which inflates test scores). 🔢 *same split after shuffling the rows: True; patients in >1 split: 0* |
| **External-site hold-out** | A whole hospital kept out of training and tuning: the honest test of "does it generalise to a site it never saw?". Internal test performance is usually optimistic. |
| **DICOM** | The medical image format: pixels plus a header (patient, device, body part, pixel spacing…). |
| **Modality LUT** | `RescaleSlope/Intercept`: converts stored integers to physical values. Applied first. |
| **VOI LUT / windowing** | `WindowCenter/Width`: maps physical values to the display range the radiologist sees. Applied second. Never cast to 8 bits before this step, because X-rays carry 10–16 bits. |
| **MONOCHROME1** | A photometric interpretation where low values display as *white*. These images must be inverted, or the model sees negatives. → [`data/dicom.py`](../src/xrayml/data/dicom.py) |
| **De-identification** | Removes names, dates, institution and private tags, and replaces the patient ID with a **stable pseudonym** (salted hash): the same patient always gets the same pseudonym, so patient-level splits still work. GDPR-relevant. |
| **Decode-once cache** | See §3 (1024 px cache). |
| **pHash** | A perceptual hash: a 64-bit fingerprint of what an image *looks like*, based on the DCT of a 32×32 thumbnail. Re-compressed or slightly resized copies get hashes only a few bits apart (small **Hamming distance**). |
| **Band-LSH** | Locality-sensitive hashing. Split the 64 bits into *k + 1* bands; two hashes differing in ≤ *k* bits must match exactly on at least one band (pigeonhole principle). So you only compare images that share a band, which is near-linear instead of O(n²). 🔢 *2,000 hashes + 1 near copy → finds exactly `('copy_of_x0', 'x0', 2)`* |
| **Union-find regrouping** | A data structure that merges sets efficiently. If A≈B and B≈C, A, B and C all end up in one group, hence in one split. This fixed real leakage in FracAtlas, which has no patient IDs: 45 exact duplicates sat across train and test. |
| **Datasheet** | A short document per dataset version: counts, prevalence, known issues, fingerprint ("Datasheets for Datasets", Gebru et al.). |

### Models

| Term | Meaning |
|---|---|
| **timm** | PyTorch Image Models: a library of pretrained backbones (ResNet, EfficientNet, ConvNeXt…). |
| **1-channel stem** | X-rays are grayscale but ImageNet weights expect RGB. `in_chans=1` sums the three RGB filters of the first conv layer, so the pretrained initialisation is kept. → [`models/classifier.py`](../src/xrayml/models/classifier.py) |
| **Faster R-CNN** | A *two-stage* detector. A region proposal network (RPN) suggests boxes, then RoIAlign + a head classifies and refines them. Strong on small objects. |
| **RetinaNet** | A *one-stage* detector with **focal loss**, which down-weights the huge number of easy background anchors. |
| **Small elongated anchors** | Anchors are the template boxes a detector starts from. Fissures are thin and long, so this repo uses smaller sizes and extreme aspect ratios (1:4, 4:1). → [`models/detector.py`](../src/xrayml/models/detector.py) |
| **U-Net** | An encoder–decoder segmentation network with skip connections that pass fine detail to the decoder. → [`models/unet.py`](../src/xrayml/models/unet.py) |
| **ResNet bottleneck** | A 1×1 conv (reduce channels) → 3×3 conv → 1×1 conv (expand), plus a skip connection. The skip connection is what makes very deep networks trainable. |
| **MBConv + SE** | The EfficientNet block: expand → depthwise conv → squeeze-and-excitation (learns per-channel weights) → project. Cheap in FLOPs. |
| **DenseNet layer** | Each layer receives the concatenated features of all earlier layers (CheXNet used DenseNet-121). |
| **BN → GN** | BatchNorm statistics are noisy with 2–4 large X-rays per GPU. GroupNorm normalises within each image, so it does not depend on batch size. → [`models/layers.py`](../src/xrayml/models/layers.py) |
| **Registry pattern** | `@MODELS.register("unet")` maps a name to a class. Configs pick a model by name, so adding a model never edits the trainer (open/closed principle). → [`registry.py`](../src/xrayml/registry.py) |

### Training

| Term | Meaning |
|---|---|
| **AMP** | Automatic mixed precision: compute in 16-bit for speed and memory, keep master weights in 32-bit. |
| **bf16 / fp16 + GradScaler** | bf16 has fp32's exponent range, so no scaling is needed. fp16 can underflow small gradients to zero, so `GradScaler` multiplies the loss before `backward()` and skips steps that produce inf/NaN. |
| **Gradient accumulation** | Sum gradients over N small batches before one optimizer step, to emulate a larger batch on limited memory. |
| **DDP `no_sync`** | In DistributedDataParallel, GPUs average gradients at every `backward()`. During accumulation, `no_sync()` skips that communication on all but the last micro-batch. |
| **Clipping after `unscale_`** | With fp16, gradients are still multiplied by the scale factor. You must unscale them first, otherwise you clip the *scaled* values and the clip threshold is meaningless. |
| **Warmup + cosine per step** | The learning rate ramps up linearly at the start, which stabilises early training, then decays along a cosine curve. It is updated at every *optimizer step*, not once per epoch. |
| **AdamW groups** | AdamW decouples weight decay from the gradient update. "Groups": no weight decay on biases and norm layers, where decay only hurts. |
| **EMA** | Exponential moving average of the weights. The averaged copy is smoother and often validates better, and it is the copy that gets evaluated and exported. |
| **NaN guard** | Stop immediately with the epoch and step if the loss becomes non-finite, instead of silently training on garbage. |
| **Exact resume (RNG + per-epoch seeding)** | The checkpoint saves model, optimizer, scheduler, scaler, EMA *and* random generator states. Each epoch's shuffle and augmentation depend only on `(seed, epoch)`. A run interrupted and resumed then gives the *same* result as an uninterrupted one, and a test proves it. |
| **torchrun-ready** | Launch on N GPUs with `torchrun --nproc_per_node=N`. → [`engine/distributed.py`](../src/xrayml/engine/distributed.py) |
| **MLflow (optional)** | An experiment tracker: params, metrics per epoch, lineage tags and the checkpoint. |

### Evaluation

| Term | Meaning |
|---|---|
| **Sensitivity / specificity** | Sensitivity = share of fractures flagged (misses are the costly error). Specificity = share of normal images correctly cleared. |
| **AUC** | Area under the ROC curve: the probability that a random fractured image scores higher than a random normal one. Threshold-free, so it measures *ranking* only. 🔢 *AUC = 0.975* |
| **AP** | Average precision: area under the precision–recall curve. More informative than AUC when positives are rare. |
| **PPV** | Positive predictive value: of the flagged images, how many are really fractures. It depends on **prevalence**, so the same model looks worse at a site with fewer fractures. 🔢 *Se 0.90 / Sp 0.90: PPV 79% at 30% prevalence, 50% at 10%, 32% at 5%* |
| **ECE** | Expected calibration error: does "0.8" really mean 80%? Bin the predictions and average the gap between predicted probability and observed frequency. |
| **Patient-level bootstrap CI** | Resample *patients* (not images) with replacement 1,000 times and take the 2.5–97.5 percentiles. Images of one patient are correlated, so image-level resampling gives intervals that are too narrow. 🔢 *AUC 95% CI [0.966, 0.984]* |
| **Wilson interval** | A confidence interval for a proportion (e.g. sensitivity) that behaves well near 0/1 and on small samples, unlike the textbook `p ± 1.96·√(p(1−p)/n)`. 🔢 *19/20 = 95%, but CI [0.76, 0.99]: 20 positives prove very little* |
| **FROC** | Free-response ROC, for detection: lesion-level sensitivity vs **false positives per image**. A second box on an already-found fracture counts as neither TP nor FP. → [`metrics/detection.py`](../src/xrayml/metrics/detection.py) |
| **CPM** | Competition Performance Metric: average FROC sensitivity at 1/8, 1/4, 1/2, 1, 2, 4 and 8 FP per image (from the LUNA16 challenge). |
| **Dice** | Segmentation overlap: 2·\|A∩B\| / (\|A\|+\|B\|), from 0 to 1. |
| **Subgroup tables** | Metrics per body part, site and view. A good global number can hide a bad subgroup, e.g. hips. |
| **Temperature scaling** | Divide logits by a single learned T, fitted on val, to fix over- or under-confidence. Ranking, and therefore AUC, is unchanged. 🔢 *overconfident ECE 0.147 → T = 3.02 (true 3) → ECE 0.010; AUC identical* → [`calibration.py`](../src/xrayml/calibration.py) |
| **Grad-CAM** | A heat map of which image regions drove the score, from the gradients of the last conv layer. Useful to catch **shortcut learning**, e.g. a model that looks at the "L/R" marker or a cast instead of the bone. → [`explain/gradcam.py`](../src/xrayml/explain/gradcam.py) |

### Release

| Term | Meaning |
|---|---|
| **Tune on val, confirm on test** | Thresholds are chosen on the validation split, then frozen and measured on the untouched test split. Tuning on test would make the test result optimistic. 🔢 *`all` threshold 0.647: val Se 0.904 → test Se 0.877. The drop is normal; the `hand` subgroup (30 positives) drops further to 0.786, which is why small subgroups fall back to the global threshold.* |
| **Operating points as a versioned artefact (per body part)** | `thresholds.json` is versioned *separately from the weights* and records which `model_version` it was tuned for. Changing a threshold goes back through the gate without retraining. Per-body-part thresholds are fitted only when there are ≥ 30 positives *and* ≥ 30 negatives. |
| **Acceptance criteria** | Pre-agreed absolute bars, e.g. *lower* bound of the sensitivity CI ≥ 0.80 and specificity ≥ 0.50. Using the lower bound means "we are 95% confident it is at least 0.80". |
| **Paired non-inferiority vs production** | Candidate and current production model run on the *same* test images. Bootstrap the difference in sensitivity; pass if its CI lower bound ≥ −margin (2 points). "Not meaningfully worse", without having to prove "better". 🔢 the demo gate **fails** both sensitivity checks: with 130 test positives, the evidence is not strong enough. That is the gate doing its job. |
| **PCCP** | Predetermined Change Control Plan (FDA): if the change protocol and the gate are agreed in advance, re-trained models that pass the gate can ship without a new submission. |
| **Decision-level parity** | Compare PyTorch and ONNX outputs on real images and count **decisions that flip** at the shipped thresholds, not only the maximum score difference. 🔢 *a uniform 1e-3 shift flips 18 of 20,000 decisions; tolerance checks like `allclose` would pass it* |
| **Model card with lineage** | `model_card.json` records git SHA, dataset version + fingerprint, config, metrics and date. It answers "exactly what is this model and where did it come from?" during an audit. |

### Serving

| Term | Meaning |
|---|---|
| **FastAPI** | A Python web framework; `/predict` takes an uploaded image and returns JSON. |
| **ONNX Runtime** | Microsoft's inference engine for `.onnx` models, on CPU or GPU, without PyTorch. |
| **PNG/JPEG/DICOM** | All three are decoded by the *same* preprocessing function as training. Different preprocessing in training and serving (**train/serve skew**) is the classic "worse in production" bug, and a test asserts the two paths give byte-identical tensors. → [`data/preprocess.py`](../src/xrayml/data/preprocess.py) |
| **Letterbox** | Resize the long side and pad the rest, keeping the aspect ratio (a plain resize would distort bone shapes). |
| **Body-part thresholds** | The request (or the DICOM `BodyPartExamined` tag) selects the matching operating point; otherwise the `all` default applies. |
| **Refuses mismatched thresholds** | At start-up, if `thresholds.json` was tuned for another `model_version`, the service refuses to start, so thresholds for model A can never run on model B. |
| **JSONL prediction log** | One JSON line per prediction (time, site, score, decision, image stats). It is the input of drift monitoring and an audit trail. |

### Monitoring

| Term | Meaning |
|---|---|
| **PSI** | Population Stability Index: compares the distribution of a feature (e.g. mean image intensity) between reference and live data, using quantile bins. Rule of thumb: < 0.1 stable, 0.1–0.25 moderate, > 0.25 investigate. 🔢 *same machine 0.011 · slightly brighter 0.189 · new X-ray device 1.078* |
| **Positive-rate drift per site** | Ground truth arrives late, but the share of positive predictions is visible immediately. A site whose rate jumps by more than 50% from the reference gets flagged. → [`monitoring/drift.py`](../src/xrayml/monitoring/drift.py) |

### NLP

| Term | Meaning |
|---|---|
| **Rule-based report labeller** | Mines labels (positive / negative / uncertain / not mentioned) from free-text radiology reports in French and English. It is the baseline any BERT/LLM labeller must beat. → [`nlp/report_labeler.py`](../src/xrayml/nlp/report_labeler.py) |
| **Negation** | NegEx-style: a cue like "pas de", "absence de" or "no" *before* the finding makes it negative. |
| **Uncertainty** | "suspicion de", "à confirmer", "cannot be excluded" → uncertain. These are kept separate rather than forced to 0/1. |
| **Conclusion section** | The findings section lists what was *looked for* ("recherche de fracture"); the conclusion states what was *found*, so the labeller reads the conclusion. |
| **κ (Cohen's kappa)** | Agreement between the labeller and a radiologist gold set, corrected for chance agreement (0 = chance, 1 = perfect). These labels are **weak labels** for pretraining, never test-set ground truth. |

### Platform

| Term | Meaning |
|---|---|
| **pre-commit = CI** | The same tool versions (from `uv.lock`) run locally and in CI, so there is no "works on my machine". |
| **Docker (serving without torch; CUDA training image)** | Two images: a slim ONNX Runtime image for serving (non-root user, healthcheck), and a CUDA image for GPU training. The `release` target bakes the bundle in, so the **image digest** identifies weights + thresholds exactly. |
| **SOUP / IEC 62304** | IEC 62304 is the medical-device software lifecycle standard. SOUP ("software of unknown provenance") is every third-party library you must list and risk-assess. Dropping torch from serving shortens that list. |
| **Spark job** | Distributed dataframe processing to build the manifest at hospital scale, using *the same* split function as the Python code. |
| **SQL & Mongo references** | Example analytics queries (window functions, anti-joins) and an annotation document schema with indexes. They show how the CLI fits a typical data stack. → [`pipelines/`](../pipelines/README.md) |
| **Dependabot / CODEOWNERS / PR template** | Automatic dependency-update PRs; required reviewers per folder; a checklist on every PR (e.g. "ONNX parity: 0 decision flips"). |

## 6. Results, Data and credits, Layout

- **Results**: [`results.md`](results.md) gives numbers *with how they were produced and their limits*. A CPU-only FracAtlas run is not a product claim, and the page says so.
- **Data and credits**: FracAtlas is cited with its DOI, and the CC BY licence requires that credit. Data is **downloaded at run time and never committed**: git is for code, DVC/storage is for data.
- **Layout**: `src/` is the importable package; `configs/` holds YAML experiments; `tests/unit` are fast tests and `tests/integration` run the full CLI → bundle → API chain; `pipelines/` is orchestrator code; `docker/` the images; `.github/` the CI/CD and review process.

## One sentence to keep

> Training gives you *scores*; the product ships a **decision**. The ML engineer's job is to
> make that decision versioned (operating points), proven (CIs, gate, parity), reproducible
> (manifest, fingerprint, exact resume) and watched (drift), and to be able to show where
> each of those lives in the code.
