"""GPU runtime: the knobs that make the same training loop fast on CUDA (revision plan 5.9).

Everything here is a no-op on CPU, so CPU runs, CI and tests keep their behaviour. The
trainer calls these helpers; the configs choose them (``runtime:`` and ``optim.amp``).

What each knob buys, and what it costs:

* **AMP bf16** — matmuls/convs in bfloat16 on tensor cores (~2x faster, ~half the
  activation memory). Same exponent range as fp32, so no GradScaler. Needs Ampere+ (A100,
  A10, L4, RTX 30xx+). ``amp: auto`` picks bf16 there, fp16 + GradScaler on older GPUs
  (V100, T4), and off on CPU.
* **TF32** — fp32 matmuls use 10-bit mantissas on Ampere+. Free speed for the parts that
  stay fp32 (loss, optimizer); invisible at the decision level for classification.
* **cudnn.benchmark** — cuDNN times several conv algorithms on the first batch of each
  input shape and keeps the fastest. Good when shapes are fixed (we letterbox to a fixed
  square size); bad with variable shapes (re-benchmarks every new shape).
* **channels_last** — NHWC memory layout, the native layout for tensor-core convolutions.
* **torch.compile** — fuses ops into Triton kernels. The first steps are slow (compilation);
  worth it for long runs at fixed shapes. Not used for torchvision detection (data-dependent
  shapes in the RPN/RoI heads cause graph breaks and recompiles).
* **fused AdamW** — one CUDA kernel for the whole parameter update instead of a loop of
  small kernels.
* **deterministic** — bit-exact reruns at a speed cost; for golden tests and the final
  validation run, not for exploration.
"""

from __future__ import annotations

import platform
import time
from typing import Any

import torch
from torch import nn


def resolve_device(device: str | None = None) -> torch.device:
    """``None``/``"auto"`` -> cuda if available, else cpu."""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def bf16_native() -> bool:
    """bf16 in hardware (compute capability >= 8.0). The default ``is_bf16_supported()`` also
    says yes on a T4 / V100, where bf16 is *emulated* and slower than fp16."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


def resolve_amp(amp: str, device: torch.device) -> torch.dtype | None:
    """Config value -> autocast dtype (``None`` = full fp32)."""
    if amp == "off":
        return None
    if amp == "auto":
        if device.type != "cuda":
            return None  # CPU autocast exists but rarely speeds up convs; keep fp32
        return torch.bfloat16 if bf16_native() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]


def configure_runtime(
    device: torch.device,
    *,
    tf32: bool = True,
    cudnn_benchmark: bool = True,
    deterministic: bool = False,
) -> None:
    """Set global CUDA backend flags once per process (before building the model)."""
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    # deterministic wins over benchmark: autotuning may pick a non-deterministic algorithm
    torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic
    torch.backends.cudnn.deterministic = deterministic


def maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    """``torch.compile`` when asked and on CUDA. The returned wrapper shares parameters with
    ``model``, so checkpoints and EMA keep using the original module."""
    if not enabled or not torch.cuda.is_available():
        return model
    return torch.compile(model)  # type: ignore[return-value]


def device_info(device: torch.device | None = None) -> dict[str, Any]:
    """Hardware + software stack, recorded in the run lineage (which GPU produced the model)."""
    device = device or resolve_device()
    info: dict[str, Any] = {
        "device": device.type,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info |= {
            "gpu": props.name,
            "gpu_count": torch.cuda.device_count(),
            "compute_capability": f"{props.major}.{props.minor}",
            "memory_gb": round(props.total_memory / 1024**3, 1),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "bf16_native": bf16_native(),
        }
    return info


def hardware_tag(device: torch.device) -> str:
    """One-line summary for lineage, e.g. ``"NVIDIA A100-SXM4-40GB x1 | torch 2.14.1 | cuda 12.6"``."""
    i = device_info(device)
    if i["device"] != "cuda":
        return f"cpu | torch {i['torch']}"
    return f"{i['gpu']} x{i['gpu_count']} | torch {i['torch']} | cuda {i['cuda']}"


class EpochMeter:
    """Throughput and peak GPU memory for one epoch: the two numbers to watch when tuning
    batch size, resolution, AMP and data-loader workers."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.images = 0
        self.t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def add(self, n_images: int) -> None:
        self.images += n_images

    def result(self) -> dict[str, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)  # kernels are async: wait before reading the clock
        dt = max(time.perf_counter() - self.t0, 1e-9)
        out = {"images_per_s": round(self.images / dt, 1)}
        if self.device.type == "cuda":
            out["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated(self.device) / 1024**3, 2)
        return out
