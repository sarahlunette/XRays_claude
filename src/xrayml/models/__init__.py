"""Models: building blocks, classifier, detector, U-Net — registered by name on import.

from xrayml.models import build_model
model = build_model("timm_classifier", backbone="resnet18")
"""

from typing import Any

from xrayml.models import classifier, detector, unet  # noqa: F401  (registration side effect)
from xrayml.registry import MODELS


def build_model(name: str, **params: Any):
    return MODELS.build(name, **params)
