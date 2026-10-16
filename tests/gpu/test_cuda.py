"""CUDA-only checks: run with ``make test-gpu`` on a GPU box (skipped everywhere else).

What can go wrong only on a GPU, and is cheap to catch before an 8-hour run:
* AMP picks the right dtype for this card, the step stays finite, GradScaler is on for fp16.
* torch.compile + channels_last + fused AdamW train without error.
* A checkpoint trained on GPU gives the same *decisions* on CPU (where export/serving run).
* Resume on GPU restores the CUDA RNG.
"""

import json

import pytest
import torch

from xrayml.config import TrainConfig
from xrayml.engine.checkpoint import load_model_from_checkpoint
from xrayml.engine.gpu import bf16_native
from xrayml.engine.trainer import Trainer
from xrayml.evaluation.predict import predict

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def _cfg(tmp_path, manifest_path, amp="auto", **runtime):
    return TrainConfig.model_validate(
        {
            "task": "classification",
            "output_dir": str(tmp_path / "run"),
            "data": {"manifest": str(manifest_path), "image_size": 64, "batch_size": 8, "num_workers": 2},
            "model": {"name": "timm_classifier", "params": {"backbone": "resnet10t"}},
            "optim": {"epochs": 2, "lr": 1e-3, "warmup_steps": 2, "amp": amp, "ema_decay": 0.9},
            "runtime": runtime,
        }
    )


def _split(manifest):
    return manifest[manifest.split == "train"], manifest[manifest.split == "val"]


def test_auto_amp_matches_hardware(tmp_path, synth_dir, manifest):
    t = Trainer(_cfg(tmp_path, synth_dir / "manifest.parquet"), *_split(manifest), device="cuda")
    assert t.amp_dtype is (torch.bfloat16 if bf16_native() else torch.float16)
    assert t.scaler.is_enabled() == (t.amp_dtype is torch.float16)
    assert t.optimizer.defaults.get("fused") is True


@pytest.mark.parametrize("amp", ["bf16", "fp16"])
def test_amp_training_is_finite(tmp_path, synth_dir, manifest, amp):
    if amp == "bf16" and not bf16_native():
        pytest.skip("bf16 is emulated on this GPU")
    t = Trainer(_cfg(tmp_path, synth_dir / "manifest.parquet", amp=amp), *_split(manifest), device="cuda")
    result = t.fit()
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[-1])
    assert result["best"] > 0 and row["gpu_mem_peak_gb"] > 0 and row["images_per_s"] > 0


def test_compile_channels_last_train(tmp_path, synth_dir, manifest):
    t = Trainer(
        _cfg(tmp_path, synth_dir / "manifest.parquet", compile=True), *_split(manifest), device="cuda"
    )
    assert t.model is not t.raw_model  # compiled wrapper, shared parameters
    t.fit()
    assert (tmp_path / "run" / "best.ckpt").exists()


def test_gpu_checkpoint_gives_same_decisions_on_cpu(tmp_path, synth_dir, manifest):
    """Train on GPU, serve on CPU: the release path must not depend on where training ran."""
    t = Trainer(_cfg(tmp_path, synth_dir / "manifest.parquet"), *_split(manifest), device="cuda")
    t.fit()
    model, _ = load_model_from_checkpoint(tmp_path / "run" / "last.ckpt")  # map_location=cpu
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 both sides
    cpu, _ = predict(model, t.val_loader, "classification", "cpu")
    gpu, _ = predict(model.to("cuda"), t.val_loader, "classification", "cuda")
    assert (cpu.score - gpu.score).abs().max() < 1e-4
    assert ((cpu.score >= 0.5) == (gpu.score >= 0.5)).all()


def test_resume_on_gpu_matches_straight_run(tmp_path, synth_dir, manifest):
    path = synth_dir / "manifest.parquet"
    rt = {"deterministic": True, "tf32": False}
    a = Trainer(_cfg(tmp_path / "a", path, amp="off", **rt), *_split(manifest), device="cuda")
    a.fit()
    b = Trainer(_cfg(tmp_path / "b", path, amp="off", **rt), *_split(manifest), device="cuda")
    b.fit(stop_at_epoch=1)
    c = Trainer(_cfg(tmp_path / "b", path, amp="off", **rt), *_split(manifest), device="cuda")
    c.resume(tmp_path / "b" / "run" / "last.ckpt")
    c.fit()
    for (k, pa), pc in zip(a.raw_model.state_dict().items(), c.raw_model.state_dict().values(), strict=True):
        assert torch.allclose(pa.float(), pc.float(), atol=1e-5), k
