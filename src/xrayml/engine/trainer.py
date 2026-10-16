"""The training loop (revision plan 5.1, 5.3, 5.7, 5.8) — read this file line by line.

Things an interviewer checks, all present below:
* ``model.train()`` / ``model.eval()`` switch BN + Dropout; ``no_grad`` in validation.
* ``zero_grad(set_to_none=True)`` once per *optimizer* step (gradient accumulation).
* loss divided by ``accum_steps``; DDP ``no_sync()`` on intermediate micro-steps.
* AMP: bf16 needs no scaler; fp16 needs GradScaler, and ``unscale_`` before clipping.
* scheduler stepped per optimizer step (warmup + cosine is defined in steps).
* ``loss.item()`` accumulated as a float, never the tensor (keeps the graph alive -> leak).
* AdamW without weight decay on biases and norm parameters.
* EMA weights evaluated and checkpointed; checkpoints include RNG states.

GPU path (``runtime:`` in the config, helpers in ``engine/gpu.py``): TF32 + cudnn.benchmark set
before the model is built, channels_last weights, fused AdamW, optional ``torch.compile`` of the
*training* forward only, ``non_blocking`` copies from pinned memory, and per-epoch throughput +
peak memory in ``metrics.jsonl``. On CPU every one of these is a no-op.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from xrayml.config import TrainConfig
from xrayml.data.datasets import make_loader
from xrayml.data.manifest import fingerprint
from xrayml.engine.checkpoint import load_checkpoint, save_checkpoint
from xrayml.engine.distributed import barrier, is_main
from xrayml.engine.gpu import (
    EpochMeter,
    configure_runtime,
    hardware_tag,
    maybe_compile,
    resolve_amp,
    resolve_device,
)
from xrayml.engine.seed import git_sha, seed_everything
from xrayml.evaluation.predict import predict
from xrayml.evaluation.summary import MONITOR, task_metrics
from xrayml.losses import LOSSES
from xrayml.models import build_model


def model_params(cfg: TrainConfig) -> dict[str, Any]:
    params = dict(cfg.model.params)
    if cfg.task == "detection":
        params.setdefault("image_size", cfg.data.image_size)
    return params


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim == 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(step: int, warmup: int, total: int, min_ratio: float = 0.01) -> float:
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        device: str | None = None,
        ddp_local_rank: int | None = None,
    ) -> None:
        self.cfg = cfg
        rt = cfg.runtime
        seed_everything(cfg.seed, deterministic=rt.deterministic)
        self.device = resolve_device(device)
        configure_runtime(
            self.device, tf32=rt.tf32, cudnn_benchmark=rt.cudnn_benchmark, deterministic=rt.deterministic
        )
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        distributed = ddp_local_rank is not None

        self.train_loader = make_loader(
            train_df, cfg.data, cfg.task, train=True, seed=cfg.seed, distributed=distributed
        )
        self.val_loader = make_loader(val_df, cfg.data, cfg.task, train=False, seed=cfg.seed)

        model = build_model(cfg.model.name, **model_params(cfg)).to(self.device)
        if rt.channels_last and self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)  # NHWC: native layout for tensor cores
        self.raw_model = model  # checkpoints, EMA and validation use this one
        if distributed:
            from xrayml.engine.distributed import wrap_ddp

            model = wrap_ddp(model, ddp_local_rank or 0)
        # compile after DDP so DDP's bucketing sees the compiled graph (DDPOptimizer splits it)
        self.model = maybe_compile(model, rt.compile)
        self.criterion = LOSSES.build(cfg.loss.name, **cfg.loss.params).to(self.device)

        o = cfg.optim
        fused = rt.fused_optimizer and self.device.type == "cuda"  # one kernel for the whole update
        self.optimizer = torch.optim.AdamW(
            param_groups(self.raw_model, o.weight_decay), lr=o.lr, fused=fused or None
        )
        steps = cfg.max_steps_per_epoch or len(self.train_loader)
        self.total_steps = max(1, o.epochs * math.ceil(steps / o.accum_steps))
        warm = min(o.warmup_steps, self.total_steps // 5)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: warmup_cosine(s, warm, self.total_steps)
        )
        self.amp_dtype = resolve_amp(o.amp, self.device)
        use_scaler = self.amp_dtype == torch.float16 and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.ema = (
            AveragedModel(self.raw_model, multi_avg_fn=get_ema_multi_avg_fn(o.ema_decay), use_buffers=True)
            if o.ema_decay
            else None
        )
        self.epoch, self.step, self.best = 0, 0, -float("inf")
        self.lineage = {
            "git_sha": git_sha(),
            "dataset_version": cfg.data.dataset_version,
            "dataset_fingerprint": fingerprint(pd.concat([train_df, val_df])),
            "hardware": hardware_tag(self.device),
            "amp": str(self.amp_dtype).removeprefix("torch.") if self.amp_dtype else "fp32",
        }
        self.meter: dict[str, float] = {}

    # ------------------------------------------------------------------ training
    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _loss(self, images, targets) -> torch.Tensor:
        if self.cfg.task == "detection":
            images = [im.to(self.device, non_blocking=True) for im in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            return sum(self.model(images, targets).values())  # RPN + RoI losses
        logits = self.model(images.to(self.device, non_blocking=True))
        return self.criterion(logits.float(), targets.to(self.device, non_blocking=True))

    def train_one_epoch(self) -> float:
        self.model.train()
        set_epoch = getattr(self.train_loader.sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(self.epoch)  # DDP: different shuffle each epoch
        # Reproducible epochs, also after a resume: shuffle order and augmentations depend only
        # on (seed, epoch), not on how many epochs this process has already run.
        epoch_seed = self.cfg.seed * 1000 + self.epoch
        if self.train_loader.generator is not None:
            self.train_loader.generator.manual_seed(epoch_seed)
        aug = getattr(self.train_loader.dataset, "aug", None)
        if aug is not None:
            aug.set_random_seed(epoch_seed)  # num_workers=0 path; workers reseed in _seed_worker
        accum, total, n = self.cfg.optim.accum_steps, 0.0, 0
        max_steps = self.cfg.max_steps_per_epoch
        meter = EpochMeter(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        for i, (images, targets, _) in enumerate(self.train_loader):
            if max_steps is not None and i >= max_steps:
                break
            last_micro = (i + 1) % accum == 0
            no_sync = getattr(self.model, "no_sync", None)  # DDP only (also through torch.compile)
            sync = contextlib.nullcontext() if last_micro or no_sync is None else no_sync()
            with sync:
                with self._autocast():
                    loss = self._loss(images, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {self.epoch} step {i}")
                self.scaler.scale(loss / accum).backward()
            if last_micro:
                self._optimizer_step()
            total += loss.item()  # also a sync point: one per micro-step is fine at these batch sizes
            n += 1
            meter.add(len(images))
        if n % accum:  # flush a partial accumulation at the end of the epoch
            self._optimizer_step()
        self.meter = meter.result()
        return total / max(n, 1)

    def _optimizer_step(self) -> None:
        if self.cfg.optim.grad_clip:
            self.scaler.unscale_(self.optimizer)  # clip true gradients, not scaled ones
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.cfg.optim.grad_clip)
        self.scaler.step(self.optimizer)  # skipped automatically on inf/NaN in fp16
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update_parameters(self.raw_model)
        self.step += 1

    # ---------------------------------------------------------------- validation
    def eval_model(self) -> nn.Module:
        return self.ema.module if self.ema is not None else self.raw_model

    def validate(self) -> dict[str, float | None]:
        preds, boxes = predict(self.eval_model(), self.val_loader, self.cfg.task, self.device)
        return task_metrics(self.cfg.task, preds, boxes)

    # ----------------------------------------------------------------- main loop
    def _save(self, name: str) -> Path:
        return save_checkpoint(
            self.out / name,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema=self.ema,
            epoch=self.epoch,
            step=self.step,
            best=self.best,
            config=json.loads(self.cfg.model_dump_json()),
            lineage=self.lineage,
        )

    def resume(self, path: str | Path) -> None:
        ck = load_checkpoint(
            path,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema=self.ema,
        )
        self.epoch, self.step, self.best = ck["epoch"] + 1, ck["step"], ck["best"]

    def fit(self, stop_at_epoch: int | None = None) -> dict[str, Any]:
        """Train until ``optim.epochs``; ``stop_at_epoch`` simulates a pre-emption (resume tests)."""
        monitor = MONITOR[self.cfg.task]
        log_path = self.out / "metrics.jsonl"
        tracker = _Tracker(self.cfg) if is_main() else None
        last: dict[str, Any] = {}
        end = self.cfg.optim.epochs if stop_at_epoch is None else min(stop_at_epoch, self.cfg.optim.epochs)
        while self.epoch < end:
            t0 = time.perf_counter()
            train_loss = self.train_one_epoch()
            if is_main():  # validation on rank 0 only (simple; all-gather is the scalable option)
                metrics = self.validate()
                score = metrics.get(monitor)
                last = {
                    "epoch": self.epoch,
                    "step": self.step,
                    "train_loss": train_loss,
                    "lr": self.scheduler.get_last_lr()[0],
                    "seconds": round(time.perf_counter() - t0, 1),
                    **self.meter,  # per-rank throughput; x world_size for the job total
                    **{f"val_{k}": v for k, v in metrics.items()},
                }
                with log_path.open("a") as f:
                    f.write(json.dumps(last) + "\n")
                print(json.dumps(last))
                if tracker:
                    tracker.log(last, self.epoch)
                if score is not None and score > self.best:
                    self.best = score
                    self._save("best.ckpt")
                self._save("last.ckpt")
            barrier()  # other ranks wait here instead of inside the next all-reduce (NCCL timeout)
            self.epoch += 1
        if tracker:
            tracker.close(self.out)
        return {"best": self.best, "monitor": monitor, "last": last}


class _Tracker:
    """Optional MLflow logging: params, per-epoch metrics, lineage tags, checkpoint artefact."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.mlflow = None
        if cfg.tracking == "mlflow":
            import mlflow

            self.mlflow = mlflow
            mlflow.set_experiment(f"xrayml-{cfg.task}")
            mlflow.start_run()
            mlflow.log_params(
                {
                    "model": cfg.model.name,
                    **cfg.model.params,
                    "lr": cfg.optim.lr,
                    "image_size": cfg.data.image_size,
                    "dataset_version": cfg.data.dataset_version,
                }
            )
            mlflow.set_tag("git_sha", git_sha())

    def log(self, metrics: dict, step: int) -> None:
        if self.mlflow:
            self.mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, int | float)}, step=step
            )

    def close(self, out: Path) -> None:
        if self.mlflow:
            if (out / "best.ckpt").exists():
                self.mlflow.log_artifact(str(out / "best.ckpt"))
            self.mlflow.end_run()
