"""Tests for narro.protocol — TTSModel protocol and audio types."""

import numpy as np
import pytest

from narro.protocol import AudioChunk, AudioResult, TTSModel


class TestAudioResult:
    def test_duration(self):
        audio = np.zeros(32000, dtype=np.float32)
        result = AudioResult(audio=audio, sample_rate=32000)
        assert result.duration == pytest.approx(1.0)

    def test_duration_half_second(self):
        audio = np.zeros(16000, dtype=np.float32)
        result = AudioResult(audio=audio, sample_rate=32000)
        assert result.duration == pytest.approx(0.5)

    def test_metadata_defaults_empty(self):
        result = AudioResult(audio=np.zeros(1), sample_rate=16000)
        assert result.metadata == {}

    def test_metadata_round_trips(self):
        result = AudioResult(
            audio=np.zeros(1),
            sample_rate=16000,
            metadata={"alignment": [{"start": 0.0}]},
        )
        assert result.metadata["alignment"][0]["start"] == 0.0


class TestAudioChunk:
    def test_fields(self):
        audio = np.ones(1024, dtype=np.float32)
        chunk = AudioChunk(audio=audio, sample_rate=32000)
        assert chunk.sample_rate == 32000
        assert len(chunk.audio) == 1024


class TestTTSModelProtocol:
    """Verify that a plain class satisfying the shape is a TTSModel."""

    def test_conforming_class_passes_isinstance(self):
        class GoodModel:
            @property
            def model_id(self) -> str:
                return "test"

            @property
            def sample_rate(self) -> int:
                return 16000

            def synthesize(self, text, **kwargs):
                return AudioResult(audio=np.zeros(100), sample_rate=16000)

            def synthesize_stream(self, text, **kwargs):
                yield AudioChunk(audio=np.zeros(100), sample_rate=16000)

        assert isinstance(GoodModel(), TTSModel)

    def test_missing_method_fails_isinstance(self):
        class BadModel:
            @property
            def model_id(self):
                return "bad"

            @property
            def sample_rate(self):
                return 16000

            def synthesize(self, text, **kwargs):
                pass

            # missing synthesize_stream

        assert not isinstance(BadModel(), TTSModel)
