"""Tests for ModalityRegistry: {modality: {model_id: Model}}."""
from dataclasses import dataclass
from typing import Any

import pytest

from muse.core.registry import ModalityRegistry, ModelInfo


@dataclass
class FakeAudioModel:
    model_id: str = "fake-tts"
    sample_rate: int = 16000
    def synthesize(self, text: str) -> Any: ...  # noqa


@dataclass
class FakeImageModel:
    model_id: str = "fake-diffusion"
    default_size: tuple[int, int] = (512, 512)
    def generate(self, prompt: str) -> Any: ...  # noqa


@pytest.fixture
def reg():
    return ModalityRegistry()


def test_register_and_get_by_modality(reg):
    m = FakeAudioModel()
    reg.register("audio/speech", m)
    assert reg.get("audio/speech", "fake-tts") is m


def test_first_registered_becomes_default_per_modality(reg):
    a1 = FakeAudioModel(model_id="tts-1")
    a2 = FakeAudioModel(model_id="tts-2")
    reg.register("audio/speech", a1)
    reg.register("audio/speech", a2)
    assert reg.get("audio/speech") is a1  # default


def test_modalities_are_isolated(reg):
    a = FakeAudioModel()
    i = FakeImageModel()
    reg.register("audio/speech", a)
    reg.register("image/generation", i)
    assert reg.get("audio/speech") is a
    assert reg.get("image/generation") is i
    with pytest.raises(KeyError):
        reg.get("audio/speech", "fake-diffusion")


def test_set_default_overrides_first_registered(reg):
    a1 = FakeAudioModel(model_id="tts-1")
    a2 = FakeAudioModel(model_id="tts-2")
    reg.register("audio/speech", a1)
    reg.register("audio/speech", a2)
    reg.set_default("audio/speech", "tts-2")
    assert reg.get("audio/speech").model_id == "tts-2"


def test_list_models_returns_modelinfo_per_modality(reg):
    reg.register("audio/speech", FakeAudioModel())
    reg.register("image/generation", FakeImageModel())
    audio = reg.list_models("audio/speech")
    assert len(audio) == 1
    assert isinstance(audio[0], ModelInfo)
    assert audio[0].model_id == "fake-tts"
    assert audio[0].modality == "audio/speech"


def test_list_all_spans_modalities(reg):
    reg.register("audio/speech", FakeAudioModel())
    reg.register("image/generation", FakeImageModel())
    all_models = reg.list_all()
    modalities = {m.modality for m in all_models}
    assert modalities == {"audio/speech", "image/generation"}


def test_modalities_lists_registered_keys(reg):
    reg.register("audio/speech", FakeAudioModel())
    assert reg.modalities() == ["audio/speech"]
    reg.register("image/generation", FakeImageModel())
    assert set(reg.modalities()) == {"audio/speech", "image/generation"}


def test_missing_modality_raises(reg):
    with pytest.raises(KeyError, match="no models registered"):
        reg.get("audio/speech")


def test_duplicate_registration_overwrites(reg):
    a1 = FakeAudioModel(model_id="tts")
    a2 = FakeAudioModel(model_id="tts")
    reg.register("audio/speech", a1)
    reg.register("audio/speech", a2)
    assert reg.get("audio/speech", "tts") is a2
    # Default lookup must also reflect the overwrite
    assert reg.get("audio/speech") is a2


def test_get_error_messages_include_available_options(reg):
    """Errors should tell the operator what's actually available."""
    reg.register("audio/speech", FakeAudioModel(model_id="tts-1"))

    # Missing modality → lists known modalities
    with pytest.raises(KeyError, match="known modalities"):
        reg.get("video.clips")

    # Wrong model_id in a valid modality → lists available in that modality
    with pytest.raises(KeyError, match="available.*tts-1"):
        reg.get("audio/speech", "tts-99")
