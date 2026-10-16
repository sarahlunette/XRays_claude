"""GPU runtime helpers and config rules — all checkable on a CPU-only machine."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from xrayml.config import TrainConfig, load_config
from xrayml.engine.gpu import (
    EpochMeter,
    configure_runtime,
    device_info,
    hardware_tag,
    maybe_compile,
    resolve_amp,
    resolve_device,
)
from xrayml.engine.trainer import Trainer

ROOT = Path(__file__).parents[2]


def _cfg(tmp_path, manifest_path, task="classification", **runtime):
    return TrainConfig.model_validate(
        {
            "task": task,
            "output_dir": str(tmp_path / "run"),
            "data": {"manifest": str(manifest_path), "image_size": 64, "batch_size": 8, "num_workers": 0},
            "model": {"name": "timm_classifier", "params": {"backbone": "resnet10t"}},
            "optim": {"epochs": 1, "lr": 1e-3, "warmup_steps": 1, "amp": "auto"},
            "runtime": runtime,
            "max_steps_per_epoch": 2,
        }
    )


def test_amp_resolution():
    cpu = torch.device("cpu")
    assert resolve_amp("off", cpu) is None
    assert resolve_amp("auto", cpu) is None  # auto never turns on CPU autocast
    assert resolve_amp("bf16", cpu) is torch.bfloat16  # explicit wins
    assert resolve_amp("fp16", torch.device("cuda")) is torch.float16


def test_device_resolution_and_info():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    info = device_info(torch.device("cpu"))
    assert info["device"] == "cpu" and info["torch"] == torch.__version__
    assert hardware_tag(torch.device("cpu")).startswith("cpu | torch")


def test_runtime_flags_are_noops_on_cpu():
    before = torch.backends.cudnn.benchmark
    configure_runtime(torch.device("cpu"), cudnn_benchmark=not before)
    assert torch.backends.cudnn.benchmark == before
    m = torch.nn.Linear(2, 2)
    if not torch.cuda.is_available():
        assert maybe_compile(m, True) is m  # compile is CUDA-only here


def test_epoch_meter_counts_images():
    meter = EpochMeter(torch.device("cpu"))
    meter.add(8)
    meter.add(8)
    out = meter.result()
    assert out["images_per_s"] > 0 and "gpu_mem_peak_gb" not in out


def test_compile_rejected_for_detection(tmp_path, synth_dir):
    with pytest.raises(ValidationError, match="not supported for detection"):
        _cfg(tmp_path, synth_dir / "manifest.parquet", task="detection", compile=True)
    with pytest.raises(ValidationError):
        _cfg(tmp_path, synth_dir / "manifest.parquet", unknown_knob=True)  # typos fail fast


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_validates(path):
    cfg = load_config(path)
    if path.stem.endswith("_gpu"):
        assert cfg.optim.amp == "auto" and cfg.runtime.compile


def test_gpu_config_runs_on_cpu(tmp_path, synth_dir, manifest):
    """The GPU recipe's knobs (amp auto, compile, fused AdamW, channels_last) degrade to plain
    fp32 on CPU, and the run records which hardware produced it."""
    cfg = _cfg(tmp_path, synth_dir / "manifest.parquet", compile=True, fused_optimizer=True)
    t = Trainer(cfg, manifest[manifest.split == "train"], manifest[manifest.split == "val"], device="cpu")
    assert t.amp_dtype is None and not t.scaler.is_enabled()
    assert t.model is t.raw_model  # not compiled on CPU
    t.fit()
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[-1])
    assert row["images_per_s"] > 0
    assert t.lineage["hardware"].startswith("cpu") and t.lineage["amp"] == "fp32"


def test_prefetch_factor_reaches_loader(synth_dir, manifest):
    from xrayml.config import DataConfig
    from xrayml.data.datasets import make_loader

    cfg = DataConfig(manifest=synth_dir / "manifest.parquet", image_size=64, num_workers=1, prefetch_factor=3)
    assert make_loader(manifest, cfg, "classification", train=False).prefetch_factor == 3
    cfg0 = cfg.model_copy(update={"num_workers": 0})
    assert make_loader(manifest, cfg0, "classification", train=False).num_workers == 0  # no ValueError


def test_gpu_benchmark_script_runs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("gpu_benchmark", ROOT / "scripts" / "gpu_benchmark.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "gpu_benchmark", mod)  # @dataclass resolves its module here
    spec.loader.exec_module(mod)
    assert (
        mod.main(["--device", "cpu", "--steps", "1", "--warmup", "1", "--json", str(tmp_path / "b.json")])
        == 0
    )
    rows = json.loads((tmp_path / "b.json").read_text())["rows"]
    assert [r["name"] for r in rows][:2] == ["fp32", "+tf32"] and rows[0]["speedup"] == 1.0
