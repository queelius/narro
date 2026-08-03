"""muse.core.image_preprocessing: shared image-processor fallback ladder.

Tests cover four-tier dispatch (override-first, AutoImageProcessor,
encoder-hints-derived, all-tiers-exhausted), the DerivedImageProcessor
class, and structured error reporting.
"""
import json
from unittest.mock import MagicMock

import pytest


def test_read_encoder_hints_finds_grayscale(tmp_path):
    """Vision-encoder-decoder configs nest hyperparams under encoder."""
    from muse.core.image_preprocessing import read_encoder_hints
    cfg = {
        "model_type": "vision-encoder-decoder",
        "encoder": {
            "model_type": "vit",
            "num_channels": 1,
            "image_size": 448,
            "patch_size": 16,
        },
        "decoder": {"model_type": "roberta"},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    out = read_encoder_hints(str(tmp_path))
    assert out["num_channels"] == 1
    assert out["image_size"] == 448


def test_read_encoder_hints_top_level_fallback(tmp_path):
    """Older repos put encoder hyperparams at top level."""
    from muse.core.image_preprocessing import read_encoder_hints
    cfg = {"model_type": "vit", "num_channels": 3, "image_size": 224}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    out = read_encoder_hints(str(tmp_path))
    assert out["num_channels"] == 3
    assert out["image_size"] == 224


def test_read_encoder_hints_missing_config_returns_empty(tmp_path):
    from muse.core.image_preprocessing import read_encoder_hints
    out = read_encoder_hints(str(tmp_path))
    assert out == {}


def test_read_encoder_hints_malformed_json_returns_empty(tmp_path):
    from muse.core.image_preprocessing import read_encoder_hints
    (tmp_path / "config.json").write_text("{broken json")
    out = read_encoder_hints(str(tmp_path))
    assert out == {}


@pytest.mark.parametrize(
    "num_channels,size,fill,expected_shape,expected_min,expected_max",
    [
        (1, 64, "white", (1, 1, 64, 64), 0.99, 1.01),
        (3, 32, "black", (1, 3, 32, 32), -1.01, -0.99),
    ],
)
def test_derived_image_processor_shape_and_normalization(
    num_channels, size, fill, expected_shape, expected_min, expected_max,
):
    """Output tensors have the right shape and symmetric [-1, 1] normalization."""
    from muse.core.image_preprocessing import DerivedImageProcessor
    from PIL import Image
    proc = DerivedImageProcessor(num_channels=num_channels, image_size=size)
    img = Image.new("RGB", (128, 128), fill)
    pv = proc(img, return_tensors="pt")["pixel_values"]
    assert pv.shape == expected_shape
    assert expected_min <= pv.min().item() <= expected_max
    assert expected_min <= pv.max().item() <= expected_max


def test_derived_image_processor_accepts_image_size_tuple():
    """image_size can be a (height, width) pair for non-square inputs."""
    from muse.core.image_preprocessing import DerivedImageProcessor
    from PIL import Image
    proc = DerivedImageProcessor(num_channels=1, image_size=(48, 96))
    img = Image.new("RGB", (100, 100), "white")
    out = proc(img, return_tensors="pt")
    assert out["pixel_values"].shape == (1, 1, 48, 96)


def test_derived_image_processor_to_device_chain():
    """Returned BatchFeature supports .to(device) for the runtime's chain."""
    from muse.core.image_preprocessing import DerivedImageProcessor
    from PIL import Image
    proc = DerivedImageProcessor(num_channels=1, image_size=32)
    out = proc(Image.new("RGB", (64, 64), "gray"), return_tensors="pt")
    moved = out.to("cpu")
    assert hasattr(moved, "__getitem__")
    assert moved["pixel_values"].device.type == "cpu"


def test_derived_image_processor_validates_image_mean_length():
    """image_mean length must match num_channels."""
    from muse.core.image_preprocessing import DerivedImageProcessor
    with pytest.raises(ValueError, match="image_mean has 3 values"):
        DerivedImageProcessor(
            num_channels=1, image_size=32, image_mean=[0.5, 0.5, 0.5],
        )


def test_derived_image_processor_validates_image_std_length():
    """image_std length must match num_channels."""
    from muse.core.image_preprocessing import DerivedImageProcessor
    with pytest.raises(ValueError, match="image_std has 1 values"):
        DerivedImageProcessor(
            num_channels=3, image_size=32, image_std=[0.5],
        )


def test_derived_image_processor_rejects_zero_std():
    """A zero std channel would divide by zero during normalization."""
    from muse.core.image_preprocessing import DerivedImageProcessor
    with pytest.raises(ValueError, match="must all be positive"):
        DerivedImageProcessor(
            num_channels=3, image_size=32, image_std=[0.5, 0.0, 0.5],
        )


def test_derived_image_processor_rejects_negative_std():
    """A negative std is physically meaningless and flips normalization."""
    from muse.core.image_preprocessing import DerivedImageProcessor
    with pytest.raises(ValueError, match="must all be positive"):
        DerivedImageProcessor(
            num_channels=1, image_size=32, image_std=[-0.5],
        )


def test_build_image_processor_overrides_skip_auto(tmp_path, monkeypatch):
    """Tier 1: when overrides is set, AutoImageProcessor is NOT called."""
    from muse.core import image_preprocessing as mod
    auto_factory = MagicMock()
    auto_factory.from_pretrained = MagicMock(side_effect=AssertionError(
        "AutoImageProcessor must not be called when overrides is set"
    ))
    monkeypatch.setattr(mod, "_load_auto_image_processor", lambda src: auto_factory.from_pretrained(src))

    proc = mod.build_image_processor(
        str(tmp_path),
        overrides={"num_channels": 1, "image_size": 448},
        model_id="texteller-test",
    )
    assert isinstance(proc, mod.DerivedImageProcessor)
    assert proc.num_channels == 1
    assert proc.height == 448
    auto_factory.from_pretrained.assert_not_called()


def test_build_image_processor_auto_succeeds_when_no_overrides(tmp_path, monkeypatch):
    """Tier 2: AutoImageProcessor returns a processor; tier 3 doesn't fire."""
    from muse.core import image_preprocessing as mod
    sentinel = MagicMock(name="auto-image-processor")
    monkeypatch.setattr(
        mod, "_load_auto_image_processor", lambda src: sentinel,
    )
    out = mod.build_image_processor(
        str(tmp_path), overrides=None, model_id="m",
    )
    assert out is sentinel


def test_build_image_processor_falls_back_to_encoder_hints(tmp_path, monkeypatch):
    """Tier 3: AutoImageProcessor raises; encoder hints in config.json --
    INCLUDING explicit image_mean/image_std -- drive DerivedImageProcessor.
    Hints that provide ground-truth normalization stats are trustworthy;
    hints that don't (see test below) must not be silently guessed."""
    from muse.core import image_preprocessing as mod
    (tmp_path / "config.json").write_text(json.dumps({
        "encoder": {
            "num_channels": 1, "image_size": 448,
            "image_mean": [0.5], "image_std": [0.5],
        },
    }))

    def _failing(src):
        raise RuntimeError("AutoImageProcessor failed")
    monkeypatch.setattr(mod, "_load_auto_image_processor", _failing)

    proc = mod.build_image_processor(
        str(tmp_path), overrides=None, model_id="texteller",
    )
    assert isinstance(proc, mod.DerivedImageProcessor)
    assert proc.num_channels == 1
    assert proc.height == 448


def test_build_image_processor_hints_without_mean_std_raise(tmp_path, monkeypatch):
    """Tier 3 must NOT silently default missing image_mean/image_std to
    0.5: that reintroduces the exact silently-wrong-normalization bug
    the old ViT-defaults tier was removed for in v0.42.1. When
    config.json exposes only num_channels/image_size (no mean/std), the
    ladder must fall through to the override-hatch error instead of
    guessing normalization stats."""
    from muse.core import image_preprocessing as mod
    (tmp_path / "config.json").write_text(json.dumps({
        "encoder": {"num_channels": 1, "image_size": 448},
    }))

    def _failing(src):
        raise RuntimeError("AutoImageProcessor failed")
    monkeypatch.setattr(mod, "_load_auto_image_processor", _failing)

    with pytest.raises(mod.ImageProcessorError) as exc_info:
        mod.build_image_processor(
            str(tmp_path), overrides=None, model_id="partial-hints",
        )
    assert "image_processor_overrides" in str(exc_info.value)


def test_build_image_processor_all_tiers_exhausted_raises(tmp_path, monkeypatch):
    """Tier 4: AutoImageProcessor fails AND no usable hints. Raise
    ImageProcessorError with override-hatch hint in the message."""
    from muse.core import image_preprocessing as mod

    underlying = ImportError("requires torchvision")

    def _failing(src):
        raise underlying
    monkeypatch.setattr(mod, "_load_auto_image_processor", _failing)

    with pytest.raises(mod.ImageProcessorError) as exc_info:
        mod.build_image_processor(
            str(tmp_path), overrides=None, model_id="m",
        )
    msg = str(exc_info.value)
    assert "ImportError: requires torchvision" in msg
    assert "image_processor_overrides" in msg
    assert "image_mean" in msg
    assert "image_std" in msg
    assert exc_info.value.__cause__ is underlying


def test_build_image_processor_empty_overrides_treated_as_none(tmp_path, monkeypatch):
    """An empty dict for overrides should be treated like None: try
    AutoImageProcessor first, not skip to DerivedImageProcessor."""
    from muse.core import image_preprocessing as mod
    sentinel = MagicMock(name="auto-image-processor")
    monkeypatch.setattr(
        mod, "_load_auto_image_processor", lambda src: sentinel,
    )
    out = mod.build_image_processor(
        str(tmp_path), overrides={}, model_id="m",
    )
    assert out is sentinel  # used auto, not derived
