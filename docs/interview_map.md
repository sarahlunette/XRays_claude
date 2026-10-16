# Revision plan → code map

Every topic of the 2-day revision plan that can be shown in code, with the place to look and
the test that proves it works. Use it the night before: pick a line, open the file, explain
it out loud.

## Chapter 1 — Computer vision

| Topic | Code | Test |
|---|---|---|
| Backbone / neck / head, 1-channel stem from RGB weights | `models/classifier.py` (`timm ... in_chans=1`) | `test_models.py::test_classifier_outputs_logits` |
| BatchNorm vs GroupNorm vs frozen BN | `models/layers.py::bn_to_gn`, `freeze_bn`; `unet.py` `norm="group"` | `test_models.py::test_bn_to_gn_and_freeze_bn` |
| DDP + SyncBN | `engine/distributed.py::wrap_ddp`; trainer `no_sync()` + `barrier()`; [`docs/gpu.md`](gpu.md) | `tests/gpu/test_cuda.py` (needs GPUs; `make fracatlas-gpu NPROC=N`) |
| GPU runtime: AMP auto, TF32, cudnn.benchmark, channels_last, compile, fused AdamW | `engine/gpu.py`; `configs/classification_fracatlas_gpu.yaml`; `scripts/gpu_benchmark.py` | `test_gpu_runtime.py` (CPU) · `tests/gpu/test_cuda.py` (CUDA) |
| Backpropagation, gradient accumulation | `engine/trainer.py::train_one_epoch` | `test_engine.py::test_overfit_one_batch` |
| ResNet bottleneck + projection shortcut | `models/layers.py::Bottleneck` | `test_bottleneck_projection_shortcut` |
| Grouped / depthwise conv cost (ResNeXt) | — | `test_grouped_and_depthwise_parameter_counts` |
| FLOPs / MACs | `models/layers.py::conv_macs` | `test_conv_macs_matches_revision_example` (116M) |
| MBConv + SE + drop path | `models/layers.py::MBConv`, `SqueezeExcite` | `test_mbconv_skip_only_when_shapes_match` |
| DenseNet layer (concatenate) | `models/layers.py::DenseLayer` | `test_dense_layer_grows_channels_by_growth_rate` |
| Multi-label BCE, logits not probabilities | `losses.py` (all take logits) | `test_losses_calibration.py` |
| Focal loss (γ, α) | `losses.py::focal_loss` | `test_focal_downweights_easy_examples_100x` |
| Temperature scaling, ECE | `calibration.py`, `metrics/classification.py::expected_calibration_error` | `test_temperature_scaling_recovers_overconfidence`, `test_ece_zero_for_perfect_calibration` |
| Grad-CAM / shortcut learning | `explain/gradcam.py` (synthetic images carry an "L"/"R" marker on purpose) | `test_gradcam_heatmap_range` |
| Box formats (xyxy vs COCO xywh) | `data/fracatlas.py` | `test_seg_mask_lies_inside_the_box` |
| IoU, NMS from scratch | `metrics/detection.py` | `test_iou_properties` (Hypothesis), `test_nms_matches_torchvision` |
| Faster R-CNN, anchors for thin fractures, FPN | `models/detector.py` | `test_detector_train_and_eval_contract` |
| FROC, CPM, FP/image | `metrics/detection.py::froc`, `evaluation/summary.py::box_table` | `test_froc_counts_hits_fps_and_double_hits` |
| U-Net (bilinear up + concat skips) | `models/unet.py` | `test_unet_shape_and_groupnorm` |
| Dice / Tversky, Dice vs IoU, empty masks | `losses.py`, `metrics/segmentation.py` | `test_dice_and_tversky_perfect_prediction`, `test_dice_iou_relation_and_empty_convention` |
| X-ray augmentation, boxes kept in sync | `data/transforms.py`, `data/datasets.py` | `test_targets_per_task` |
| Letterbox, mapping back to original pixels | `data/preprocess.py` | `test_letterbox_keeps_aspect_and_pads_bottom_right`, `test_boxes_round_trip` |

## Chapter 2 — Medical imaging, metrics, operating points

| Topic | Code | Test |
|---|---|---|
| DICOM LUTs, MONOCHROME1, bit depth | `data/dicom.py::dicom_to_array` | `test_dicom.py` (4 tests) |
| De-identification, pseudonym | `data/dicom.py::deidentify` | `test_deidentify_removes_phi_and_keeps_stable_pseudonym` |
| Header audit before training | `xrayml dicom-audit`, `data/dicom.py::header_record` | `test_header_audit_and_readers` |
| Sensitivity, specificity, PPV vs prevalence | `metrics/classification.py` | `test_confusion_and_bayes_ppv` (32 % / 79 %) |
| Wilson CI, patient-level bootstrap | `metrics/classification.py` | `test_group_bootstrap_ci_contains_point_and_widens_with_correlation` |
| Subgroups / hidden stratification | `evaluation/evaluate.py::subgroup_table` | integration report |
| Operating point: target sensitivity, Youden, per body part | `operating_points.py` | `test_operating_points_gate.py` |
| Tune on val, confirm on test | `operating_points.py::confirm_on_test`, `xrayml report` | `test_ops_roundtrip_and_confirm` |
| Non-inferiority, McNemar-style paired comparison | `evaluation/release_gate.py::paired_diff_ci` | `test_gate_non_inferiority_against_production` |
| External validation (unseen site) | `data/manifest.py::assign_splits(external=...)` | `test_external_site_holdout_is_group_level` |

## Chapter 3 — Foundation models (what is shown here)

| Topic | Code |
|---|---|
| Linear probe / fine-tune a pretrained backbone | `timm_classifier` with `pretrained: true`; swap `backbone:` in the config |
| Layer-wise LR decay, foundation-model pipeline | not implemented — see `docs/development.md` "Extending" for where it plugs in (`engine/trainer.py::param_groups`) |

## Chapter 4 — NLP

| Topic | Code | Test |
|---|---|---|
| Label mining from FR/EN reports, NegEx-style negation, uncertainty, conclusion section | `nlp/report_labeler.py`, `xrayml label-reports` | `test_nlp_drift.py::test_label_report_fr_en` (13 cases) |
| Agreement with a gold set (Cohen's κ) | `nlp/report_labeler.py::evaluate_labeler` | `test_other_findings_and_agreement` |

## Chapter 5 — PyTorch at scale

| Topic | Code | Test |
|---|---|---|
| Reference loop: AMP (bf16/fp16 + GradScaler), clipping, accumulation, scheduler per step | `engine/trainer.py` | `test_engine.py` |
| Warmup + cosine | `engine/trainer.py::warmup_cosine` | `test_warmup_then_cosine` |
| AdamW parameter groups | `engine/trainer.py::param_groups` | `test_no_weight_decay_on_bias_and_norm` |
| EMA weights | trainer `ema_decay` (`AveragedModel`) | resume test uses EMA |
| Seeds, deterministic resume | `engine/seed.py`, `engine/checkpoint.py`, per-epoch seeding in trainer | `test_resume_reproduces_uninterrupted_run` |
| Divergence → NaN guard | trainer `FloatingPointError` | `test_divergence_is_caught` (and see `docs/development.md` for the real detector NaN) |
| Collate for detection, WeightedRandomSampler | `data/datasets.py` | `test_detection_collate_keeps_lists`, `test_oversampling_balances_classes` |
| GroupKFold by patient | `data/manifest.py::group_kfold` | `test_group_kfold_has_no_patient_overlap` |

## Chapter 6 — Data engineering

| Topic | Code |
|---|---|
| Manifest, schema validation, fingerprint | `data/manifest.py` (+ tests) |
| Decode once (cache) | `data/prepare.py`, `xrayml prepare` |
| Near-duplicates, LSH, union-find regrouping | `data/quality.py` (+ property test vs brute force) |
| Datasheet per version | `data/quality.py::datasheet` → `DATASHEET.md` |
| DVC pipeline + params | `dvc.yaml`, `params.yaml` |
| Airflow DAG | `pipelines/airflow/dags/dataset_release.py` |
| Spark (broadcast join, mapInPandas with the same split function) | `pipelines/spark/build_manifest.py` |
| SQL window functions, anti-joins | `pipelines/sql/analytics.sql` |
| MongoDB schema, aggregation, ESR index | `pipelines/mongo/annotations.js` |

## Chapter 7 — Software engineering

| Topic | Where |
|---|---|
| Registry + factory (open/closed) | `registry.py`, `@MODELS.register` |
| Validated config that fails fast | `config.py` (+ `test_invalid_config_fails_fast`) |
| Typed code, mypy in CI | `pyproject.toml [tool.mypy]`, `ci.yml` |
| Test pyramid, fixtures, parametrize, Hypothesis | `tests/` |
| pre-commit = CI | `.pre-commit-config.yaml` |
| PR template, CODEOWNERS, issue forms | `.github/` |
| The buggy `compute_auc` snippet, fixed | `evaluation/summary.py::task_metrics` + `metrics/classification.py::safe_auc` |

## Chapter 8 — MLOps

| Gate | Code | Workflow |
|---|---|---|
| 1 Train from config + dataset version | `xrayml train` | `integration.yml` |
| 2 Register with lineage | `ModelCard` in `artifacts.py`; MLflow optional (`tracking: mlflow`) | — |
| 3 Automated validation + report | `xrayml evaluate`, `xrayml report` | job summary |
| 4 Operating points as artefact | `xrayml select-ops` → `thresholds.json` | |
| 5 Export + parity | `xrayml export` | `integration.yml` |
| 6 Integration test | `tests/integration/`, `scripts/smoke_test.py` | `container-smoke` |
| 7 Release, staged rollout, rollback | `release.yml` (staging → production approval) | `release.yml` |
| 8 Monitor | `monitoring/drift.py`, `xrayml drift`, prediction log | — |
