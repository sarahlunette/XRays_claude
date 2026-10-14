"""PyTorch datasets and loaders for the three tasks (revision plan 5.2).

One dataset class, three target formats:
* classification: ``tensor([fractured])`` (image-level label, BCE)
* detection: ``{"boxes": (N,4) xyxy, "labels": (N,)}`` (torchvision convention, 1 = fracture)
* segmentation: ``(1, H, W)`` binary mask rasterised from the COCO polygons
"""

from __future__ import annotations

import json
import random
from functools import partial

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler, WeightedRandomSampler

from xrayml.config import AugmentConfig, DataConfig, Task
from xrayml.data.io import read_image
from xrayml.data.preprocess import letterbox


def _valid_boxes(boxes: np.ndarray, size: int, min_side: float = 2.0) -> np.ndarray:
    boxes = np.clip(boxes, 0, size)
    keep = (boxes[:, 2] - boxes[:, 0] >= min_side) & (boxes[:, 3] - boxes[:, 1] >= min_side)
    return boxes[keep]


class XrayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        task: Task,
        image_size: int,
        train: bool = False,
        augment: AugmentConfig | None = None,
    ) -> None:
        if "abs_path" not in df.columns:
            raise ValueError("use xrayml.data.manifest.load_manifest() so image paths are resolved")
        self.df = df.reset_index(drop=True)
        self.task = task
        self.size = image_size
        self.aug = None
        if train:
            from xrayml.data.transforms import build_train_augment

            self.aug = build_train_augment(augment or AugmentConfig(), task)

    def __len__(self) -> int:
        return len(self.df)

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
        """Letterboxed image, boxes and mask *before* augmentation (also used by evaluation)."""
        row = self.df.iloc[i]
        img, scale = letterbox(read_image(row.abs_path), self.size)
        boxes = np.asarray(json.loads(row.boxes), np.float32).reshape(-1, 4) * scale
        boxes = _valid_boxes(boxes, self.size)
        mask = None
        if self.task == "segmentation":
            mask = np.zeros((self.size, self.size), np.uint8)
            polys = [
                np.round(np.asarray(p, np.float32).reshape(-1, 2) * scale).astype(np.int32)
                for p in json.loads(row.polygons)
            ]
            if polys:
                cv2.fillPoly(mask, polys, 1)
        return img, boxes, mask, scale

    def __getitem__(self, i: int):
        img, boxes, mask, _ = self.load(i)
        if self.aug is not None:
            kwargs: dict = {"image": img}
            if self.task == "detection":
                kwargs |= {"bboxes": boxes, "labels": np.ones(len(boxes), np.int64)}
            if mask is not None:
                kwargs["mask"] = mask
            out = self.aug(**kwargs)
            img = out["image"]
            if self.task == "detection":
                # re-filter: rotation/scale can shrink a box below 1 px -> log(0) in box encoding -> NaN
                boxes = _valid_boxes(np.asarray(out["bboxes"], np.float32).reshape(-1, 4), self.size)
            if mask is not None:
                mask = out["mask"]
        image = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).unsqueeze(0)  # (1, H, W)
        if self.task == "classification":
            return image, torch.tensor([float(self.df.fractured.iloc[i])]), i
        if self.task == "detection":
            target = {
                "boxes": torch.from_numpy(boxes).float(),
                "labels": torch.ones(len(boxes), dtype=torch.int64),
            }
            return image, target, i
        assert mask is not None
        return image, torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0), i


def collate(batch, task: Task):
    images, targets, idx = zip(*batch, strict=True)
    if task == "detection":
        return list(images), list(targets), torch.tensor(idx)  # variable box counts -> lists
    return torch.stack(images), torch.stack(targets), torch.tensor(idx)


def _seed_worker(_: int) -> None:
    """Each worker gets a distinct, reproducible seed (derived from the loader's generator)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
    info = torch.utils.data.get_worker_info()
    aug = getattr(info.dataset, "aug", None) if info is not None else None
    if aug is not None:  # albumentations 2.x has its own RNG, seeded from OS entropy otherwise
        aug.set_random_seed(s)


def make_loader(
    df: pd.DataFrame,
    cfg: DataConfig,
    task: Task,
    train: bool,
    seed: int = 0,
    distributed: bool = False,
) -> DataLoader:
    ds = XrayDataset(df, task, cfg.image_size, train=train, augment=cfg.augment)
    sampler: Sampler | None = None
    shuffle = train
    if distributed:
        sampler = DistributedSampler(ds, shuffle=train, seed=seed)  # call set_epoch() every epoch
        shuffle = False
    elif train and cfg.oversample_positives:
        y = ds.df.fractured.to_numpy()
        w = np.where(y == 1, 1.0 / max(1, y.sum()), 1.0 / max(1, (1 - y).sum()))
        sampler = WeightedRandomSampler(w.tolist(), num_samples=len(w), replacement=True)
        shuffle = False  # sampler and shuffle are mutually exclusive
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        collate_fn=partial(collate, task=task),  # picklable (spawn start method)
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )
