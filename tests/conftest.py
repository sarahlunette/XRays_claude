"""Shared fixtures: a tiny synthetic dataset (same manifest schema as FracAtlas)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    from xrayml.data.manifest import assign_splits, save_manifest
    from xrayml.data.synthetic import make_synthetic_dataset

    out = tmp_path_factory.mktemp("synth")
    df = make_synthetic_dataset(out, n_images=80, image_size=64, positive_rate=0.5, seed=1, gap_px=(2, 4))
    df = assign_splits(df, val_pct=20, test_pct=20)
    save_manifest(df, out / "manifest.parquet")
    return out


@pytest.fixture(scope="session")
def manifest(synth_dir):
    from xrayml.data.manifest import load_manifest

    return load_manifest(synth_dir / "manifest.parquet")
