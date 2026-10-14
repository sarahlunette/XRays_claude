import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from xrayml.data.manifest import (
    ManifestError,
    assign_splits,
    fingerprint,
    group_kfold,
    summary_table,
    validate_manifest,
)
from xrayml.data.quality import datasheet, find_near_duplicates


def _df(n=200, per_group=3):
    return pd.DataFrame(
        {
            "image_id": [f"i{i}" for i in range(n)],
            "path": [f"{i}.png" for i in range(n)],
            "group_id": [f"p{i // per_group}" for i in range(n)],
            "source": "t",
            "site": ["s1" if i % 2 else "s2" for i in range(n)],
            "body_part": "hand",
            "fractured": [i % 3 == 0 for i in range(n)],
            "width": 100,
            "height": 100,
            "boxes": [json.dumps([[10, 10, 20, 20]] if i % 3 == 0 else []) for i in range(n)],
            "polygons": "[]",
        }
    ).astype({"fractured": int})


def test_splits_are_deterministic_and_group_disjoint():
    a, b = assign_splits(_df()), assign_splits(_df().sample(frac=1, random_state=0))
    assert a.set_index("image_id").split.equals(b.set_index("image_id").split.loc[a.image_id])
    assert (a.groupby("group_id").split.nunique() == 1).all()
    assert set(a.split) == {"train", "val", "test"}


def test_external_site_holdout_is_group_level():
    df = _df()  # sites alternate inside a group -> whole group must go external
    out = assign_splits(df, external={"site": ["s1"]})
    ext_groups = set(out[out.split == "external_test"].group_id)
    assert ext_groups == set(df[df.site == "s1"].group_id)
    assert (out.groupby("group_id").split.nunique() == 1).all()


def test_validate_reports_every_problem():
    df = _df(10)
    df.loc[0, "image_id"] = "i1"  # duplicate
    df.loc[3, "boxes"] = json.dumps([[50, 50, 40, 60]])  # x1 < x0
    df.loc[6, "boxes"] = "[]"  # positive without box
    df.loc[2, "body_part"] = "tail"
    with pytest.raises(ManifestError) as e:
        validate_manifest(df)
    msg = str(e.value)
    assert all(s in msg for s in ["duplicated", "degenerate", "no box", "tail"])


def test_fingerprint_changes_with_labels_only():
    df = assign_splits(_df())
    f = fingerprint(df)
    assert fingerprint(df.sample(frac=1, random_state=1)) == f  # row order does not matter
    df2 = df.copy()
    df2.loc[0, "fractured"] = 1 - df2.loc[0, "fractured"]
    assert fingerprint(df2) != f


def test_group_kfold_has_no_patient_overlap():
    df = _df()
    for tr, va in group_kfold(df, 4):
        assert set(df.group_id.iloc[tr]).isdisjoint(df.group_id.iloc[va])


def test_datasheet_lists_counts():
    df = assign_splits(_df())
    md = datasheet(df, "v-test", ["a known issue"])
    assert "v-test" in md and "a known issue" in md and fingerprint(df) in md
    assert summary_table(df).n_images.sum() == len(df)


@settings(max_examples=30, deadline=None)
@given(st.lists(st.integers(0, 2**64 - 1), min_size=2, max_size=40), st.integers(0, 6))
def test_band_lsh_finds_exactly_the_brute_force_pairs(raw, k):
    hashes = {f"h{i}": v for i, v in enumerate(raw)}
    # add guaranteed near-duplicates of the first hash
    rng = np.random.default_rng(k)
    for j in range(3):
        flip = sum(1 << int(b) for b in rng.choice(64, size=k % 5, replace=False))
        hashes[f"d{j}"] = raw[0] ^ flip
    got = {(a, b) for a, b, _ in find_near_duplicates(hashes, max_distance=4)}
    keys = sorted(hashes)
    brute = {
        (a, b) for i, a in enumerate(keys) for b in keys[i + 1 :] if (hashes[a] ^ hashes[b]).bit_count() <= 4
    }
    assert got == brute


def test_regroup_merges_duplicate_chains_so_splits_cannot_separate_them():
    from xrayml.data.manifest import assert_no_group_leakage
    from xrayml.data.quality import regroup_near_duplicates

    df = _df(30, per_group=1)
    rng = np.random.default_rng(0)
    hashes = {iid: int(rng.integers(0, 2**63)) for iid in df.image_id}  # random: ~32 bits apart
    hashes["i1"] = hashes["i0"] ^ 1  # i0 ~ i1
    hashes["i2"] = hashes["i1"] ^ 2  # i1 ~ i2 (chain: i0 and i2 are 2 bits apart)
    out = regroup_near_duplicates(df, max_distance=1, hashes=hashes)
    assert out.group_id.iloc[0] == out.group_id.iloc[1] == out.group_id.iloc[2]
    assert out.group_id.nunique() == 28
    assert_no_group_leakage(assign_splits(out))
