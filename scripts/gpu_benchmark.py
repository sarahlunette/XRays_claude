"""Measure what each GPU knob buys on *this* machine, before launching a long run.

Times the classifier's training step (forward + backward + AdamW) on random tensors, adding
one knob per row: fp32 -> TF32 -> channels_last -> AMP -> fused AdamW -> torch.compile. Run
it once per new GPU type; the speed-up table is the evidence behind the config choices
(and a good interview answer to "how do you know AMP helped?").

    uv run python scripts/gpu_benchmark.py                          # resnet50 @ 768 px, batch 32
    uv run python scripts/gpu_benchmark.py --backbone resnet18 --image-size 512 --batch-size 64
    uv run python scripts/gpu_benchmark.py --max-batch                # largest batch that fits (bf16)

On CPU it runs a tiny version (used by the unit tests) so the script cannot rot.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch

from xrayml.engine.gpu import configure_runtime, device_info, resolve_amp, resolve_device
from xrayml.models import build_model


@dataclass
class Variant:
    name: str
    tf32: bool = False
    channels_last: bool = False
    amp: str = "off"
    fused: bool = False
    compile: bool = False


VARIANTS = [
    Variant("fp32"),
    Variant("+tf32", tf32=True),
    Variant("+channels_last", tf32=True, channels_last=True),
    Variant("+amp", tf32=True, channels_last=True, amp="auto"),
    Variant("+fused_adamw", tf32=True, channels_last=True, amp="auto", fused=True),
    Variant("+compile", tf32=True, channels_last=True, amp="auto", fused=True, compile=True),
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_variant(
    v: Variant, backbone: str, image_size: int, batch_size: int, steps: int, warmup: int, device: torch.device
) -> dict:
    configure_runtime(device, tf32=v.tf32, cudnn_benchmark=True)
    torch.manual_seed(0)
    model = build_model("timm_classifier", backbone=backbone, pretrained=False).to(device)
    fmt = torch.channels_last if v.channels_last and device.type == "cuda" else torch.contiguous_format
    model = model.to(memory_format=fmt)
    fused = v.fused and device.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=fused or None)
    step_model = torch.compile(model) if v.compile and device.type == "cuda" else model
    dtype = resolve_amp(v.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16 and device.type == "cuda")
    x = torch.rand(batch_size, 1, image_size, image_size, device=device).to(memory_format=fmt)
    y = torch.randint(0, 2, (batch_size, 1), device=device).float()

    def step() -> None:
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(step_model(x).float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    for _ in range(warmup):  # cudnn autotuning + compilation happen here, not in the timing
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    _sync(device)  # CUDA is asynchronous: without this we would time the kernel *launches*
    dt = time.perf_counter() - t0
    row = {
        **asdict(v),
        "images_per_s": round(steps * batch_size / dt, 1),
        "ms_per_step": round(1000 * dt / steps, 1),
    }
    if device.type == "cuda":
        row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
    return row


def max_batch(backbone: str, image_size: int, device: torch.device, start: int = 8, limit: int = 1024) -> int:
    """Double the batch until CUDA OOM (bf16/fp16 + channels_last); return the last that fitted."""
    best, bs = 0, start
    while bs <= limit:
        try:
            time_variant(VARIANTS[4], backbone, image_size, bs, steps=1, warmup=1, device=device)
            best, bs = bs, bs * 2
        except torch.OutOfMemoryError:
            break
        finally:
            torch.cuda.empty_cache()
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default=None, help="default: resnet50 on GPU, resnet10t on CPU")
    p.add_argument("--image-size", type=int, default=None, help="default: 768 on GPU, 64 on CPU")
    p.add_argument("--batch-size", type=int, default=None, help="default: 32 on GPU, 4 on CPU")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-compile", action="store_true", help="skip the torch.compile row (slow to build)")
    p.add_argument("--max-batch", action="store_true", help="find the largest batch that fits, then exit")
    p.add_argument("--json", help="also write the rows to this file")
    a = p.parse_args(argv)

    device = resolve_device(a.device)
    gpu = device.type == "cuda"
    backbone = a.backbone or ("resnet50" if gpu else "resnet10t")
    size = a.image_size or (768 if gpu else 64)
    bs = a.batch_size or (32 if gpu else 4)
    steps = a.steps or (20 if gpu else 2)
    warmup = a.warmup or (5 if gpu else 1)
    print(json.dumps(device_info(device)))

    if a.max_batch:
        if not gpu:
            p.error("--max-batch needs a GPU")
        print(
            json.dumps(
                {"backbone": backbone, "image_size": size, "max_batch": max_batch(backbone, size, device)}
            )
        )
        return 0

    variants = [v for v in VARIANTS if not (v.compile and (a.no_compile or not gpu))]
    rows, base = [], None
    print(f"\n{backbone} @ {size}px, batch {bs}, {steps} timed steps on {device}\n")
    print(f"{'variant':<16}{'img/s':>10}{'ms/step':>10}{'speed-up':>10}{'peak GB':>10}")
    for v in variants:
        r = time_variant(v, backbone, size, bs, steps, warmup, device)
        base = base or r["images_per_s"]
        r["speedup"] = round(r["images_per_s"] / base, 2)
        rows.append(r)
        print(
            f"{v.name:<16}{r['images_per_s']:>10}{r['ms_per_step']:>10}{r['speedup']:>9}x{r.get('peak_mem_gb', '-'):>10}"
        )
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"device": device_info(device), "rows": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
