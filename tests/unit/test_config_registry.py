import pytest
from pydantic import ValidationError

from xrayml.config import load_config
from xrayml.registry import Registry


def _write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


BASE = """
task: classification
data: {manifest: data/m.parquet, image_size: 128}
model: {name: timm_classifier}
"""


def test_overrides_are_typed(tmp_path):
    cfg = load_config(
        _write(tmp_path, BASE), ["optim.lr=1e-3", "data.image_size=256", "optim.ema_decay=0.99"]
    )
    assert cfg.optim.lr == pytest.approx(1e-3)
    assert cfg.data.image_size == 256
    assert cfg.optim.ema_decay == pytest.approx(0.99)


@pytest.mark.parametrize(
    "override",
    [
        "data.image_size=100",  # not a multiple of 32
        "optim.lr=5",  # out of range
        "optim.typo=1",  # unknown key: fails fast instead of being silently ignored
        "task=regression",
    ],
)
def test_invalid_config_fails_fast(tmp_path, override):
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, BASE), [override])


def test_manifest_must_be_parquet(tmp_path):
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, BASE.replace("m.parquet", "m.csv")))


def test_registry_register_build_and_errors():
    reg: Registry = Registry("thing")

    @reg.register("double")
    def double(x=1):
        return 2 * x

    assert reg.build("double", x=3) == 6
    assert "double" in reg
    with pytest.raises(KeyError, match="already registered"):
        reg.register("double")(double)
    with pytest.raises(KeyError, match="unknown thing"):
        reg.build("nope")
