"""Full checkpoints: resuming must reproduce the curve (revision plan 5.4).

Saved: model, optimizer, scheduler, scaler, EMA, epoch/step, best metric, every RNG state,
plus the resolved config and lineage (git SHA, dataset fingerprint) so a checkpoint alone
is enough to rebuild the model and trace where it came from.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def _unwrap(model: nn.Module) -> nn.Module:
    inner = getattr(model, "module", model)  # DDP / AveragedModel wrap the real model
    assert isinstance(inner, nn.Module)
    return inner


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    epoch: int = 0,
    step: int = 0,
    best: float | None = None,
    config: dict | None = None,
    lineage: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema.state_dict() if ema else None,
        "epoch": epoch,
        "step": step,
        "best": best,
        "config": config or {},
        "lineage": lineage or {},
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never leaves a corrupt "last.ckpt"
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    ck = torch.load(path, map_location="cpu", weights_only=False)  # our own file, contains numpy RNG state
    _unwrap(model).load_state_dict(ck["model"])
    for obj, key in ((optimizer, "optimizer"), (scheduler, "scheduler"), (scaler, "scaler"), (ema, "ema")):
        if obj is not None and ck.get(key) is not None:
            obj.load_state_dict(ck[key])
    if restore_rng:
        torch.set_rng_state(ck["rng"]["torch"])
        if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
    return ck


def load_model_from_checkpoint(path: str | Path, use_ema: bool = True) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model from the config stored in the checkpoint (no external config needed)."""
    from xrayml.config import TrainConfig
    from xrayml.engine.trainer import model_params
    from xrayml.models import build_model

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig.model_validate(ck["config"])
    params = model_params(cfg) | {"pretrained": False}
    model = build_model(cfg.model.name, **params)
    state = ck["model"]
    if use_ema and ck.get("ema"):
        # AveragedModel state: "module.<param>" + "n_averaged"
        state = {k.removeprefix("module."): v for k, v in ck["ema"].items() if k.startswith("module.")}
    model.load_state_dict(state)
    model.eval()
    return model, ck
