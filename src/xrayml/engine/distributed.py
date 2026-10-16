"""DDP helpers (revision plan 1.1 "What is DDP", 5.9).

Launch: ``torchrun --nproc_per_node=8 -m xrayml.cli train configs/detection.yaml``.
One process per GPU, each with a full model copy; a DistributedSampler shards the data;
gradients are all-reduced in buckets during backward. BatchNorm statistics are NOT shared
unless converted to SyncBatchNorm — with 2 X-rays per GPU that matters (see ``wrap_ddp``).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn


def init_distributed() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank). No-op (0, 1, 0) outside torchrun."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, 0
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    return rank, world, local_rank


def is_main() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def wrap_ddp(model: nn.Module, local_rank: int, sync_bn: bool = True) -> nn.Module:
    if sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # BN stats over all GPUs
    device_ids = [local_rank] if torch.cuda.is_available() else None
    return nn.parallel.DistributedDataParallel(model, device_ids=device_ids)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
