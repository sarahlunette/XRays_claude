"""``xrayml evaluate`` — one command from a checkpoint to a standard set of numbers.

Replacing ad-hoc evaluation notebooks with this command is the first fix for slow
time-to-production (revision plan 8.2). Outputs, in ``out_dir``:
* ``preds_<split>.parquet``: one row per image (score, label, metadata) -> operating points,
  release gate, monitoring reference.
* ``boxes_<split>.parquet`` (detection): one row per predicted box with is_tp.
* ``metrics_<split>.json``: task metrics + group-bootstrap CIs + subgroup tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from xrayml.config import TrainConfig
from xrayml.data.datasets import make_loader
from xrayml.data.manifest import load_manifest
from xrayml.engine.checkpoint import load_model_from_checkpoint
from xrayml.evaluation.predict import predict
from xrayml.evaluation.summary import box_table, task_metrics
from xrayml.metrics.classification import group_bootstrap_ci, safe_auc

SUBGROUP_COLUMNS = ["body_part", "site", "view", "hardware"]


def subgroup_table(preds: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    """Hidden stratification check: good average, bad subgroup (plan 2.7)."""
    rows = []
    for value, g in preds.groupby(by):
        rows.append(
            {
                by: value,
                "n": len(g),
                "n_pos": int(g.y.sum()),
                "auc": safe_auc(g.y.to_numpy(), g.score.to_numpy()),
            }
        )
    return rows


def evaluate_checkpoint(
    checkpoint: str | Path,
    manifest: str | Path,
    split: str,
    out_dir: str | Path,
    device: str = "cpu",
    n_boot: int = 1000,
    batch_size: int | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ck = load_model_from_checkpoint(checkpoint)
    cfg = TrainConfig.model_validate(ck["config"])
    model.to(device)
    df = load_manifest(manifest)
    sub = df[df.split.isin(split.split(","))]
    if sub.empty:
        raise ValueError(f"no rows for split={split!r} in {manifest}")
    data_cfg = cfg.data.model_copy(update={"batch_size": batch_size or cfg.data.batch_size, "num_workers": 0})
    loader = make_loader(sub, data_cfg, cfg.task, train=False)
    with torch.inference_mode():
        preds, boxes = predict(model, loader, cfg.task, device)

    metrics: dict[str, Any] = {
        "split": split,
        "checkpoint": str(checkpoint),
        "task": cfg.task,
        "lineage": ck.get("lineage", {}),
        **task_metrics(cfg.task, preds, boxes),
    }
    y, p, g = preds.y.to_numpy(), preds.score.to_numpy(), preds.group_id.to_numpy()
    metrics["auc_ci"] = group_bootstrap_ci(y, p, g, safe_auc, n_boot=n_boot)
    metrics["subgroups"] = {c: subgroup_table(preds, c) for c in SUBGROUP_COLUMNS if c in preds.columns}

    preds.to_parquet(out_dir / f"preds_{split}.parquet", index=False)
    if cfg.task == "detection":
        box_table(boxes, preds).to_parquet(out_dir / f"boxes_{split}.parquet", index=False)
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics
