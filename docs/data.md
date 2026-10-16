# Data

## Datasets

| Dataset | Content | Labels | Licence | Used here |
|---|---|---|---|---|
| **FracAtlas** (Abedeen et al., *Sci Data* 2023, [doi:10.6084/m9.figshare.22363012](https://doi.org/10.6084/m9.figshare.22363012)) | 4,083 radiographs: hand, leg, hip, shoulder; 717–719 fractured; JPEG, ~2,300×2,900 px | image label, COCO **boxes + polygons**, body part, view, hardware | CC BY 4.0 | **yes** — `make fracatlas-data` (338 MB, md5-checked) |
| **GRAZPEDWRI-DX** (Nagy et al., *Sci Data* 2022, figshare 14825193) | 20,327 paediatric wrist radiographs, 6,091 patients, 16-bit PNG | boxes for fracture + 8 other classes (periosteal reaction, metal, cast…) | CC BY 4.0 | adapter not included (16 GB); has real patient ids — the next dataset to add |
| MURA (Stanford) | 40k upper-limb studies | study-level normal/abnormal | research agreement | no (registration) |
| Synthetic (`xrayml synth`) | bone shafts with a 2–5 px fracture gap, L/R marker, hardware, 3 sites with different exposure | image label, box, polygon | — | CI and demos |

FracAtlas covers the three CV tasks of the role with one dataset: classification (fractured
or not), detection (boxes) and segmentation (polygons).

## Manifest schema

One Parquet row per image; paths relative to the manifest file.

| Column | Meaning |
|---|---|
| `image_id` | unique id |
| `path` | image file (PNG16, JPEG or DICOM) |
| `group_id` | **patient** (or duplicate-cluster) id — splits never separate a group |
| `source`, `site` | dataset and acquisition site (subgroups, external test, drift) |
| `body_part`, `view`, `hardware`, `multiscan` | metadata for subgroup tables and operating points |
| `fractured`, `fracture_count` | image label |
| `width`, `height` | image size (after caching: cached size; `orig_*` keep the original) |
| `boxes` | JSON list of `[x0, y0, x1, y1]` in pixels (**xyxy**) |
| `polygons` | JSON list of flat `[x, y, x, y, …]` polygons (COCO style) |
| `split` | `train` / `val` / `test` / `external_test` |

## Splits

* `assign_splits`: `sha256(salt:group_id) % 100` → test < 15 ≤ val < 25 ≤ train. Deterministic
  across machines and re-runs; new images never move old ones; changing the salt is an
  explicit new split version.
* `external={"site": [...]}` holds out whole sites (decided per group).
* `group_kfold` for cross-validation (StratifiedGroupKFold).

## Leakage found in FracAtlas, and the fix

FracAtlas has no patient id, so a naive split is by image. `xrayml quality --dedupe` found
**148 near-duplicate pairs across splits, 45 of them identical** (pHash Hamming 0). Fix:

```bash
xrayml regroup-duplicates --manifest data/fracatlas_cache/manifest.parquet --max-hamming 2
```

Union-find over near-duplicate pairs merges them into one group (4,083 → 3,950 groups), then the
split is recomputed: 0 cross-split duplicates. Threshold 2 is conservative; Hamming-4 pairs
include different images of similar anatomy and need a visual check before merging.

## DICOM

FracAtlas ships JPEGs, so the DICOM path is exercised with synthetic DICOM files
(`data/dicom.py::write_synthetic_dicom`): 12-bit stored values, modality LUT, windowing,
MONOCHROME1. The API accepts DICOM uploads and reads `BodyPartExamined` to choose the
body-part operating point. `xrayml dicom-audit <folder>` reads headers only (fast) and
summarises vendors / photometric interpretation before any training.

## Patient data hygiene

* `/data/` is git-ignored; CI refuses `.dcm`/`.nii` files and files > 1 MB in git.
* `nbstripout` removes notebook outputs (which may embed images) on commit.
* Issue forms ask reporters to confirm no patient data was pasted.
