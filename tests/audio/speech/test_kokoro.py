"""Tests for muse.audio.speech.backends.kokoro: Kokoro TTS adapter."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from muse.audio.speech.protocol import AudioChunk, AudioResult, TTSModel


class TestKokoroModel:
    def _make_adapter(self):
        from muse.audio.speech.backends.kokoro import KokoroModel

        mock_pipeline = MagicMock()
        result = MagicMock()
        result.audio = torch.randn(24000)
        mock_pipeline.return_value = [result]

        adapter = object.__new__(KokoroModel)
        adapter._pipeline = mock_pipeline
        adapter._device = "cpu"
        return adapter

    def test_protocol_conformance(self):
        assert isinstance(self._make_adapter(), TTSModel)

    def test_model_id(self):
        assert self._make_adapter().model_id == "kokoro-82m"

    def test_sample_rate(self):
        assert self._make_adapter().sample_rate == 24000

    def test_synthesize_returns_audio_result(self):
        result = self._make_adapter().synthesize("Hello")
        assert isinstance(result, AudioResult)
        assert result.sample_rate == 24000
        assert len(result.audio) == 24000

    def test_synthesize_passes_voice(self):
        adapter = self._make_adapter()
        adapter.synthesize("Hello", voice="am_adam", speed=1.2)
        adapter._pipeline.assert_called_once_with("Hello", voice="am_adam", speed=1.2)

    def test_stream_yields_chunks(self):
        adapter = self._make_adapter()
        chunks = list(adapter.synthesize_stream("Hello"))
        assert len(chunks) == 1
        assert isinstance(chunks[0], AudioChunk)

    def test_voices_list(self):
        from muse.audio.speech.backends.kokoro import KOKORO_VOICES
        assert "af_heart" in KOKORO_VOICES
        assert "am_adam" in KOKORO_VOICES
        assert len(KOKORO_VOICES) > 50


def test_kokoro_has_lowercase_voices_property():
    """routes.py + registry look for `voices` (lowercase); KokoroModel must satisfy."""
    from muse.audio.speech.backends.kokoro import KokoroModel

    assert "voices" in dir(KokoroModel), "KokoroModel must expose a `voices` attribute/property"

    # Verify via an instance (bypassing __init__) that it returns the VOICES list
    adapter = object.__new__(KokoroModel)
    assert hasattr(adapter, "voices")
    assert isinstance(adapter.voices, list)
    assert len(adapter.voices) > 0
    assert adapter.voices is KokoroModel.VOICES
