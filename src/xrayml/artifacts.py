"""Release artefacts shared by training, evaluation and serving (revision plan 8.1).

Operating points live in their own versioned file, *separate from the weights*: changing a
threshold re-enters the release pipeline at the validation gate, without retraining. This
module is pydantic-only so the serving container can import it without torch.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class OperatingPoint(BaseModel):
    finding: str = "fracture"
    body_part: str = "all"  # "all" = default threshold; specific body parts override it
    threshold: float = Field(ge=0, le=1)
    target_sensitivity: float = Field(ge=0, le=1)
    validated_sensitivity: float = Field(ge=0, le=1)
    sensitivity_ci: tuple[float, float]
    specificity: float = Field(ge=0, le=1)
    fp_per_image: float | None = None
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)


class OperatingPointSet(BaseModel):
    version: str  # e.g. "op-2026.10.1"
    model_version: str  # the model these thresholds were tuned for
    dataset_version: str
    tuning_split: str = "val"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    points: list[OperatingPoint]

    def threshold_for(self, body_part: str | None, finding: str = "fracture") -> OperatingPoint:
        """Body-part-specific point if one exists, else the 'all' default."""
        candidates = [p for p in self.points if p.finding == finding]
        for p in candidates:
            if body_part is not None and p.body_part == body_part:
                return p
        for p in candidates:
            if p.body_part == "all":
                return p
        raise KeyError(f"no operating point for finding={finding!r} body_part={body_part!r}")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> OperatingPointSet:
        return cls.model_validate_json(Path(path).read_text())


class ModelCard(BaseModel):
    """Lineage record written next to every exported model (traceability for audits)."""

    model_version: str
    task: str
    architecture: str
    image_size: int
    git_sha: str
    dataset_version: str
    dataset_fingerprint: str
    config: dict
    metrics: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> ModelCard:
        return cls.model_validate_json(Path(path).read_text())
