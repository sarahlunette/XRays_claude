"""Validated configuration (revision plan 7.1, 7.11).

YAML -> pydantic, so a typo or an out-of-range value fails in second 1, not after hour 10
of training. Overrides use the Hydra-like ``key.sub=value`` syntax from the CLI:

    xrayml train configs/classification.yaml optim.lr=1e-3 data.image_size=384
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Task = Literal["classification", "detection", "segmentation"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown keys are errors, not silently ignored


class AugmentConfig(_Strict):
    rotate_deg: float = Field(10.0, ge=0, le=30)
    scale: tuple[float, float] = (0.9, 1.1)
    brightness_contrast: float = Field(0.15, ge=0, le=0.5)
    gamma: tuple[int, int] = (80, 120)
    noise_p: float = Field(0.2, ge=0, le=1)
    # Horizontal flip is off by default: laterality markers become mirrored text (plan 1.5).
    hflip_p: float = Field(0.0, ge=0, le=1)


class DataConfig(_Strict):
    manifest: Path
    dataset_version: str = "untracked"
    image_size: int = Field(512, ge=64, le=4096, multiple_of=32)
    batch_size: int = Field(8, ge=1)
    num_workers: int = Field(2, ge=0)
    oversample_positives: bool = False
    augment: AugmentConfig = AugmentConfig()

    @field_validator("manifest")
    @classmethod
    def _parquet(cls, v: Path) -> Path:
        if v.suffix != ".parquet":
            raise ValueError("manifest must be a .parquet file produced by `xrayml build-manifest`")
        return v


class ModelConfig(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class LossConfig(_Strict):
    name: str = "bce"
    params: dict[str, Any] = Field(default_factory=dict)


class OptimConfig(_Strict):
    lr: float = Field(3e-4, gt=0, lt=1)
    weight_decay: float = Field(0.05, ge=0)
    epochs: int = Field(10, ge=1)
    warmup_steps: int = Field(100, ge=0)
    grad_clip: float | None = Field(1.0, gt=0)
    accum_steps: int = Field(1, ge=1)
    amp: Literal["off", "bf16", "fp16"] = "off"
    ema_decay: float | None = Field(None, gt=0, lt=1)


class TrainConfig(_Strict):
    task: Task
    data: DataConfig
    model: ModelConfig
    loss: LossConfig = LossConfig()
    optim: OptimConfig = OptimConfig()
    seed: int = 0
    output_dir: Path = Path("runs/default")
    tracking: Literal["none", "mlflow"] = "none"
    max_steps_per_epoch: int | None = Field(None, ge=1, description="cap steps (smoke tests / CI)")


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key.sub=value")
        key, value = item.split("=", 1)
        _set_dotted(raw, key, yaml.safe_load(value))  # "1e-3" -> float, "true" -> bool
    return TrainConfig.model_validate(raw)
