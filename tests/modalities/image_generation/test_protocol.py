"""Tests for ImageModel protocol."""
import numpy as np
import pytest

from muse.images.generations.protocol import ImageModel, ImageResult


def _fake_img():
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_image_result_stores_image_and_metadata():
    img = _fake_img()
    res = ImageResult(image=img, width=2, height=2, seed=42, metadata={"prompt": "hello"})
    assert res.image is img
    assert res.width == 2
    assert res.height == 2
    assert res.seed == 42
    assert res.metadata["prompt"] == "hello"


def test_image_result_metadata_defaults_empty():
    res = ImageResult(image=_fake_img(), width=2, height=2, seed=1)
    assert res.metadata == {}


def test_image_model_protocol_accepts_structural_impl():
    class MyModel:
        model_id = "fake-sd"
        default_size = (512, 512)
        def generate(self, prompt, **kwargs): ...

    assert isinstance(MyModel(), ImageModel)


def test_image_model_protocol_rejects_incomplete():
    class Missing:
        pass

    assert not isinstance(Missing(), ImageModel)


def test_image_result_metadata_is_independent_per_instance():
    """Regression: shared mutable default would leak state across instances."""
    a = ImageResult(image=_fake_img(), width=1, height=1, seed=0)
    b = ImageResult(image=_fake_img(), width=1, height=1, seed=1)
    a.metadata["x"] = 1
    assert "x" not in b.metadata
