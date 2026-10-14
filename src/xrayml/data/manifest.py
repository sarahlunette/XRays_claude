"""Manifest = the single table training reads (revision plan 6.1, 6.3).

Training never globs raw folders. It reads a versioned manifest (Parquet) whose rows say
which image, which label, which split. That table is what DVC versions and what the model
card fingerprints, which makes "which data trained model v3.2?" answerable.

Splits are *deterministic*: a hash of the group (patient) id decides the bucket, so re-runs,
new images and other machines all agree, and a patient can never straddle two splits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "image_id": "object",
    "path": "object",
    "group_id": "object",
    "source": "object",
    "site": "object",
    "body_part": "object",
    "fractured": "int",
    "width": "int",
    "height": "int",
    "boxes": "object",
    "polygons": "object",
}
BODY_PARTS = {"hand", "leg", "hip", "shoulder", "mixed", "wrist", "elbow", "ankle", "foot", "knee"}
SPLITS = {"train", "val", "test", "external_test"}


class ManifestError(ValueError):
    pass


def validate_manifest(df: pd.DataFrame) -> None:
    """Collect *every* problem, then raise once (like pandera's lazy=True)."""
    problems: list[str] = []
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ManifestError(f"missing columns: {sorted(missing)}")
    if df.image_id.duplicated().any():
        problems.append(f"{int(df.image_id.duplicated().sum())} duplicated image_id")
    if not df.fractured.isin([0, 1]).all():
        problems.append("fractured must be 0/1")
    bad_parts = set(df.body_part) - BODY_PARTS
    if bad_parts:
        problems.append(f"unknown body_part values: {sorted(bad_parts)}")
    if "split" in df.columns and not set(df.split).issubset(SPLITS):
        problems.append(f"unknown split values: {sorted(set(df.split) - SPLITS)}")
    n_bad_box = n_pos_without_box = 0
    for r in df.itertuples(index=False):
        bx = json.loads(r.boxes)
        if r.fractured == 1 and not bx:
            n_pos_without_box += 1
        for x0, y0, x1, y1 in bx:
            if not (
                x1 > x0 and y1 > y0 and x0 >= -5 and y0 >= -5 and x1 <= r.width + 5 and y1 <= r.height + 5
            ):
                n_bad_box += 1
    if n_bad_box:
        problems.append(f"{n_bad_box} boxes are degenerate or outside the image")
    if n_pos_without_box:
        problems.append(f"{n_pos_without_box} positive images have no box")
    if problems:
        raise ManifestError("; ".join(problems))


def _bucket(group_id: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{group_id}".encode()).hexdigest(), 16) % 100


def assign_splits(
    df: pd.DataFrame,
    val_pct: int = 10,
    test_pct: int = 15,
    external: dict[str, list[Any]] | None = None,
    salt: str = "split-v1",
) -> pd.DataFrame:
    """Deterministic group-level split; optional external test set (e.g. unseen sites).

    ``external={"site": ["site_c"]}`` holds out a whole site: the honest test of
    generalisation to a hospital the model never saw (plan 2.7, internal vs external).
    """
    df = df.copy()
    b = df.group_id.map(lambda g: _bucket(str(g), salt))
    split = pd.Series("train", index=df.index)
    split[b < val_pct + test_pct] = "val"
    split[b < test_pct] = "test"
    for col, values in (external or {}).items():
        # decided per group: if any image of a patient is external, the whole patient is
        ext_groups = set(df.loc[df[col].isin(values), "group_id"])
        split[df.group_id.isin(ext_groups)] = "external_test"
    df["split"] = split
    assert_no_group_leakage(df)
    return df


def assert_no_group_leakage(df: pd.DataFrame) -> None:
    n = df.groupby("group_id").split.nunique()
    if (n > 1).any():
        raise ManifestError(f"{int((n > 1).sum())} groups (patients) appear in several splits")


def group_kfold(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> list[tuple[Any, Any]]:
    """Cross-validation folds stratified on the label and grouped by patient (plan 5.5)."""
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(df, y=df.fractured, groups=df.group_id))


def save_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    validate_manifest(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["abs_path"], errors="ignore").to_parquet(path, index=False)
    return path


def load_manifest(path: str | Path, root: str | Path | None = None) -> pd.DataFrame:
    """Load and resolve image paths (relative to ``root``, default: the manifest's folder)."""
    path = Path(path)
    df = pd.read_parquet(path)
    base = Path(root) if root is not None else path.parent
    df["abs_path"] = [str(p) if Path(p).is_absolute() else str((base / p).resolve()) for p in df.path]
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """Content hash of what matters for training (ids, labels, boxes, splits)."""
    cols = [c for c in ["image_id", "fractured", "boxes", "split"] if c in df.columns]
    payload = df[cols].sort_values("image_id").to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and prevalence per split x body part — goes into the datasheet."""
    return (
        df.groupby(["split", "body_part"])
        .agg(n_images=("image_id", "size"), n_positive=("fractured", "sum"), prevalence=("fractured", "mean"))
        .reset_index()
    )
