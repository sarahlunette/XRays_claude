# GPU version — same code, a GPU recipe, and the knobs that make it fast

The training loop has one code path. On CPU every GPU feature is a no-op; on CUDA the
`runtime:` section of the config and `optim.amp` switch them on. The CPU FracAtlas run in
[`results.md`](results.md) was a pipeline proof at 256 px; this is the recipe it pointed to.

> **No GPU numbers in this repo yet.** The development machine has no GPU, so everything
> below was tested on CPU (the GPU paths degrade cleanly) and the CUDA-only tests are written
> but skip here. Timings below are estimates, not measurements. Measure on your card with
> `make gpu-check` and record the run in `results.md`.

## Run it

**Cloud VM with an NVIDIA GPU** (A10G / L4 / A100; driver ≥ 560 for `cu126`):

```bash
git clone … && cd XRays_claude && pip install uv
make setup-gpu                # lockfile env, then CUDA torch of the *same* version (CUDA=cu130 for newer drivers)
make gpu-check                # GPU visible? + what AMP / channels_last / fused AdamW buy on this card
make test-gpu                 # CUDA-only tests: AMP dtype, compile, GPU->CPU decision parity, resume
make fracatlas-data           # 338 MB download -> 1024 px cache -> dedup regroup -> datasheet
make fracatlas-gpu            # train -> evaluate -> operating points -> ONNX -> gate -> report
cat reports/fracatlas_gpu/validation_report.md
```

**Several GPUs on one machine** (the effective batch must stay 32, see the config header):

```bash
make fracatlas-gpu NPROC=2 GPU_OVERRIDES="optim.accum_steps=1"
make fracatlas-gpu NPROC=4 GPU_OVERRIDES="optim.accum_steps=1 data.batch_size=8"
```

**Docker** (NGC PyTorch base image, `torchrun --nproc_per_node=gpu` entrypoint):

```bash
make docker-train             # builds docker/Dockerfile.train, mounts ./data (ro) and ./runs, --shm-size=16g
```

**CI**: [`.github/workflows/gpu.yml`](../.github/workflows/gpu.yml) runs on demand on a
self-hosted runner labelled `gpu`: CUDA tests + benchmark in the job summary, and
optionally the whole FracAtlas release path with the validation report attached.

## What changes from the CPU run

| | CPU proof (`classification_fracatlas_cpu.yaml`) | GPU recipe (`classification_fracatlas_gpu.yaml`) | Why |
|---|---|---|---|
| resolution | 256 px (512 cache) | **768 px** (1024 cache) | hairline fractures are 1–3 px wide at native ~2,900 px; at 256 they vanish |
| backbone | ResNet-18 | **ResNet-50** | capacity is cheap on GPU |
| epochs | 5 | **20** | the CPU run was still improving |
| precision | fp32 | **`amp: auto`** (bf16 on Ampere+, fp16 + GradScaler on T4/V100) | ~2× throughput, ~½ activation memory |
| batch | 32 | 16 per GPU × 2 accumulation = **32 effective** | same optimisation as CPU, fits a 24 GB card at 768 px |
| loader | 2 workers | 8 workers, prefetch 4 | at 768 px decoding + augmentation, not the GPU, is usually the bottleneck |
| runtime | — | `compile`, `channels_last`, TF32, cudnn.benchmark, fused AdamW | see below |
| EMA | 0.99 | 0.998 | ~1,940 optimizer steps instead of ~490 |

The detection and segmentation FracAtlas configs were already GPU recipes (bf16, 768 / 512 px);
they now also carry a `runtime:` block (detection rejects `compile`, see below).

**Estimated cost** (not measured): one A10G-class GPU, ~3,100 training images at 768 px →
roughly 30–60 s per epoch if the loader keeps up, so the 20 epochs take tens of minutes, not the
~75 min the CPU needed for 5 epochs at 256 px. Check `images_per_s` in
`runs/cls_fracatlas_gpu/metrics.jsonl` against the `gpu_benchmark.py` number: if training is
much slower than the benchmark, the GPU is waiting for data.

## Where the GPU code lives

| File | What to read |
|---|---|
| [`src/xrayml/engine/gpu.py`](../src/xrayml/engine/gpu.py) | Every knob, what it buys and costs (docstring). `resolve_amp`, `bf16_native`, `configure_runtime`, `maybe_compile`, `EpochMeter`, `device_info` |
| [`src/xrayml/engine/trainer.py`](../src/xrayml/engine/trainer.py) | Where they are applied: flags before the model is built, channels_last, DDP wrap *then* compile, fused AdamW, GradScaler only for fp16, `barrier()` after rank-0 validation, throughput + peak memory per epoch, hardware in lineage |
| [`src/xrayml/engine/distributed.py`](../src/xrayml/engine/distributed.py) | `torchrun` env → process group (NCCL), SyncBatchNorm, DDP |
| [`src/xrayml/config.py`](../src/xrayml/config.py) | `RuntimeConfig`, `amp: auto`, `prefetch_factor`; compile + detection is a config error |
| [`configs/classification_fracatlas_gpu.yaml`](../configs/classification_fracatlas_gpu.yaml) | The recipe, with the batch arithmetic in the header |
| [`scripts/gpu_benchmark.py`](../scripts/gpu_benchmark.py) | One knob per row: fp32 → TF32 → channels_last → AMP → fused AdamW → compile; `--max-batch` |
| [`tests/gpu/test_cuda.py`](../tests/gpu/test_cuda.py) | What can only break on a GPU (marker `gpu`, skipped without CUDA) |
| [`tests/unit/test_gpu_runtime.py`](../tests/unit/test_gpu_runtime.py) | The same knobs degrade to plain fp32 on CPU; every shipped config validates |
| [`docker/Dockerfile.train`](../docker/Dockerfile.train) · [`Makefile`](../Makefile) | `setup-gpu`, `gpu-check`, `test-gpu`, `fracatlas-gpu`, `docker-train` |

## The knobs, and how to explain them

**Mixed precision (AMP).** Inside `torch.autocast`, convs and matmuls run in 16-bit on tensor
cores; reductions and the loss stay fp32 (the trainer calls `logits.float()` before the loss).
*bf16* has fp32's exponent range, so gradients do not underflow → no scaler. *fp16* has 5
exponent bits → small gradients flush to zero → `GradScaler` multiplies the loss, then
`unscale_` before clipping (otherwise you clip scaled gradients), and skips the step on inf.
`amp: auto` decides from the compute capability. It does not trust `is_bf16_supported()`,
which also says yes on a T4, where bf16 is emulated and slower than fp16.

**TF32.** On Ampere+, fp32 matmuls/convs use a 10-bit mantissa. It speeds up whatever autocast
leaves in fp32. It is off in the CPU↔GPU parity test, because it changes logits by ~1e-3.

**cudnn.benchmark.** Times the conv algorithms on the first batch of each shape and keeps the
fastest. Inputs are letterboxed to a fixed square, so it pays off. With variable shapes it
would re-benchmark every time. `deterministic: true` turns it off.

**channels_last.** NHWC is the layout tensor-core convolutions want. With NCHW, cuDNN adds a
transpose around every conv.

**torch.compile.** It traces the forward and fuses element-wise ops (BN + ReLU + add) into
Triton kernels. The first steps are slow (compiling); after that it is faster at fixed shapes.
Only the *training* model is compiled. Validation uses the uncompiled model (or its EMA copy), so eval
mode does not trigger a recompile. It is applied *after* DDP so DDP's gradient buckets line
up with the compiled graph. It is rejected for torchvision detectors: the number of proposals
per image is data-dependent → graph breaks and recompiles.

**Fused AdamW.** One kernel updates all parameters, instead of several small kernels per
tensor. ResNet-50 has ~160 parameter tensors, so this matters.

**Pinned memory + `non_blocking=True`.** Page-locked host buffers let the host→GPU copy run
asynchronously, overlapping with compute. The loader pins when CUDA is present.

## Several GPUs (DDP)

* `torchrun --standalone --nproc_per_node=N -m xrayml.cli train …` starts one process per
  GPU. `init_distributed` reads `RANK / WORLD_SIZE / LOCAL_RANK`, sets the device and joins
  an NCCL group. With `N=1` it is a no-op, so the Makefile always uses torchrun.
* `DistributedSampler` shards the data. The trainer calls `set_epoch` so every epoch
  shuffles differently.
* **Effective batch = per-GPU batch × accum_steps × N.** The recipe keeps it at 32 so lr and
  warmup stay valid. If you let it grow, scale the lr (roughly linearly, with warmup).
* `no_sync()` on the intermediate accumulation micro-steps: all-reduce only once per
  optimizer step.
* SyncBatchNorm: with 8–16 X-rays per GPU, per-GPU BN statistics are noisy (the alternative
  is GroupNorm, `group_norm: true`).
* Validation runs on rank 0 only, then `barrier()`. Without the barrier the other ranks wait
  inside the next all-reduce, and a long validation hits the NCCL timeout. The scalable
  alternative is sharded validation + `all_gather`.
* Logged `images_per_s` is per rank; multiply by N for the job.

## Memory: what to do on CUDA OOM

Memory is weights + gradients + AdamW state (≈ 4× the weights in fp32) + **activations**,
and activations dominate at 768 px. They scale with batch × H × W. In order:

1. Turn on AMP (≈ ½ the activations) and channels_last.
2. Halve `data.batch_size` and double `optim.accum_steps`. Same effective batch, same
   results; this is what accumulation is for.
3. Lower the resolution last: it is the quality knob for hairline fractures.
4. `scripts/gpu_benchmark.py --max-batch` finds the ceiling; `gpu_mem_peak_gb` per epoch in
   `metrics.jsonl` shows how close you are.

(Gradient checkpointing is the next tool, trading ~30 % compute for activation memory; not
needed at this size.)

## Is the GPU busy? (the data pipeline)

`nvidia-smi dmon` or `watch nvidia-smi`. Low utilisation with high CPU means the loader is
the bottleneck. Raise `num_workers` (≈ CPU cores / GPUs) and `prefetch_factor`, and keep the
decode-once cache (`xrayml prepare`): never decode DICOM/full-size PNG in the training loop.
In Docker, DataLoader workers share batches through `/dev/shm`. The default 64 MB crashes
with "bus error", hence `--shm-size=16g` (or `--ipc=host`).

## Reproducibility on GPU

* Atomic checkpoints include the CUDA RNG state (`torch.cuda.get_rng_state_all`) and load
  with `map_location="cpu"`, so a checkpoint trained on GPU loads anywhere.
* GPU kernels (atomics in conv backward, cudnn autotuning) are not bit-exact by default.
  `runtime.deterministic: true` → `use_deterministic_algorithms`, cudnn deterministic, no
  benchmark, `CUBLAS_WORKSPACE_CONFIG`. It is slower. Use it for the final validation run
  and the resume test, not for exploration.
* The lineage records `hardware` (e.g. `NVIDIA A10G x1 | torch 2.14.1 | cuda 12.6`) and the
  AMP dtype. The model card then says what produced the model.
* Training on GPU and serving on CPU (ONNX Runtime) is the normal case. The export parity check
  and `test_gpu_checkpoint_gives_same_decisions_on_cpu` measure it at the *decision* level.

## Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `gpu-info` says `cuda_available: false` after `setup-gpu` | driver older than the wheel's CUDA → `nvidia-smi` (top right), `make setup-gpu CUDA=cu126` or update the driver |
| a later `uv sync` / `uv run` (without `--no-sync`) is slow and CUDA is gone | uv restored the CPU wheels from the lockfile → rerun `make setup-gpu`; use `make` targets (they pass `--no-sync`) |
| loss NaN only in fp16 | overflow in a layer autocast left in fp16 → use bf16 (Ampere+) or check the scaler is enabled (`amp: fp16`) |
| first epoch much slower than the rest | `torch.compile` and cudnn autotuning, expected; compare from epoch 1 |
| `images_per_s` far below the benchmark | data loading → workers, prefetch, cache resolution; check CPU usage |
| DDP hangs at the end of epoch 0 | a rank skipped a collective (e.g. different number of batches) → `drop_last`, or NCCL timeout during long validation (barrier is there; raise `TORCH_NCCL_` timeouts if needed) |
| "bus error" in Docker | `/dev/shm` too small → `--shm-size=16g` |

## Interview one-liners

* *"Why bf16 rather than fp16?"* Same exponent range as fp32, so no loss scaling and no
  overflow NaNs. It needs Ampere or newer. On a T4 I fall back to fp16 + GradScaler.
* *"Your GPU utilisation is 40 %. What do you do?"* Profile the loader first (workers,
  prefetch, decode-once cache, pinned memory). Then the GPU side: AMP, channels_last,
  compile, a bigger batch.
* *"How do you go from 1 to 8 GPUs without changing results?"* Keep the effective batch
  (lower accumulation, then per-GPU batch), SyncBN or GroupNorm, `set_epoch`, rank-0
  checkpoints, barrier. Otherwise scale lr with warmup and re-validate.
* *"Trained on GPU, served on CPU. How do you know it is the same model?"* Decision-level
  parity on the validation set (flipped decisions = 0), plus the lineage in the model card.
