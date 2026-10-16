"""Dataset quality checks (revision plan 6.3, 6.9).

* Near-duplicate detection with perceptual hashes, scaled with band-LSH instead of O(n²).
* Cross-split duplicates = leakage, even when patient ids differ (re-exports, copies).
* Datasheet per release: counts, prevalence, known issues, fingerprint.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from xrayml.data.io import read_image
from xrayml.data.manifest import fingerprint, summary_table


def phash(path: str | Path, hash_size: int = 8) -> int:
    """64-bit perceptual hash (DCT of a 32x32 thumbnail), robust to re-compression."""
    import imagehash
    from PIL import Image

    img = (read_image(path) * 255).astype(np.uint8)
    return int(str(imagehash.phash(Image.fromarray(img), hash_size=hash_size)), 16)


def find_near_duplicates(hashes: dict[str, int], max_distance: int = 4) -> list[tuple[str, str, int]]:
    """Pairs whose Hamming distance <= max_distance.

    Band LSH: split the 64 bits into ``max_distance + 1`` bands. Two hashes differing in at
    most ``max_distance`` bits must agree exactly on at least one band (pigeonhole), so we
    only compare items sharing a band — near-linear instead of quadratic.
    """
    n_bands = max_distance + 1
    width = 64 // n_bands
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for key, h in hashes.items():
        for band in range(n_bands):
            buckets[(band, (h >> (band * width)) & ((1 << width) - 1))].append(key)
    pairs: dict[tuple[str, str], int] = {}
    for members in buckets.values():
        for a, b in combinations(sorted(members), 2):
            if (a, b) not in pairs:
                d = (hashes[a] ^ hashes[b]).bit_count()
                if d <= max_distance:
                    pairs[(a, b)] = d
    return [(a, b, d) for (a, b), d in sorted(pairs.items())]


def cross_split_duplicates(df: pd.DataFrame, max_distance: int = 4) -> pd.DataFrame:
    """Near-duplicate pairs that sit in different splits (needs ``abs_path`` and ``split``)."""
    hashes = {iid: phash(p) for iid, p in zip(df.image_id, df.abs_path, strict=True)}
    split = dict(zip(df.image_id, df.split, strict=True))
    rows = [
        (a, b, d, split[a], split[b])
        for a, b, d in find_near_duplicates(hashes, max_distance)
        if split[a] != split[b]
    ]
    return pd.DataFrame(rows, columns=["image_a", "image_b", "hamming", "split_a", "split_b"])


def datasheet(df: pd.DataFrame, version: str, notes: list[str] | None = None) -> str:
    """Markdown datasheet committed next to every dataset version."""
    table = summary_table(df)
    lines = [
        f"# Datasheet — {version}",
        "",
        f"- fingerprint: `{fingerprint(df)}`",
        f"- images: {len(df)}  |  positives: {int(df.fractured.sum())}  |  prevalence: {df.fractured.mean():.1%}",
        f"- groups (patients): {df.group_id.nunique()}  |  sources: {', '.join(sorted(df.source.unique()))}",
        "",
        "## Counts per split and body part",
        "",
        "| split | body_part | images | positives | prevalence |",
        "|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r.split} | {r.body_part} | {r.n_images} | {int(r.n_positive)} | {r.prevalence:.1%} |"
        for r in table.itertuples(index=False)
    ]
    lines += ["", "## Known issues", ""] + [f"- {n}" for n in (notes or ["none recorded"])]
    return "\n".join(lines) + "\n"


def regroup_near_duplicates(
    df: pd.DataFrame, max_distance: int = 2, hashes: dict[str, int] | None = None
) -> pd.DataFrame:
    """Merge groups linked by near-duplicate images (union-find), so a split cannot separate them.

    Use when patient IDs are missing or unreliable (FracAtlas has none): exact re-exports and
    multi-view scans of one patient then end up in the same split. Returns a copy with the
    new ``group_id`` and the original kept in ``orig_group_id``.
    """
    if hashes is None:
        hashes = {iid: phash(p) for iid, p in zip(df.image_id, df.abs_path, strict=True)}
    parent: dict[str, str] = {g: g for g in df.group_id}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    group_of = dict(zip(df.image_id, df.group_id, strict=True))
    for a, b, _ in find_near_duplicates(hashes, max_distance):
        ra, rb = find(group_of[a]), find(group_of[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # deterministic root
    out = df.copy()
    out["orig_group_id"] = out.group_id
    out["group_id"] = [find(g) for g in out.group_id]
    return out
