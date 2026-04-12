"""Tests for narro.models.bark: Bark TTS adapter."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from narro.protocol import AudioChunk, AudioResult, TTSModel


@pytest.fixture(autouse=True)
def _isolate_voices(tmp_path, monkeypatch):
    """Point voice storage to a temp directory."""
    monkeypatch.setenv("NARRO_HOME", str(tmp_path / "narro"))


class TestBarkModel:
    """Test the Bark adapter with mocked transformers."""

    def _make_adapter(self):
        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.randn(1, 24000)
        mock_model.to.return_value = mock_model

        from narro.models.bark import BarkModel
        adapter = object.__new__(BarkModel)
        adapter._processor = mock_processor
        adapter._model = mock_model
        adapter._device = "cpu"
        return adapter

    def test_protocol_conformance(self):
        adapter = self._make_adapter()
        assert isinstance(adapter, TTSModel)

    def test_model_id(self):
        adapter = self._make_adapter()
        assert adapter.model_id == "bark"

    def test_sample_rate(self):
        adapter = self._make_adapter()
        assert adapter.sample_rate == 24000

    def test_synthesize_returns_audio_result(self):
        adapter = self._make_adapter()
        result = adapter.synthesize("Hello world")
        assert isinstance(result, AudioResult)
        assert result.sample_rate == 24000
        assert len(result.audio) > 0

    def test_synthesize_with_voice_preset(self):
        adapter = self._make_adapter()
        result = adapter.synthesize("Hello", voice="v2/en_speaker_3")
        adapter._processor.assert_called_once()
        call_kwargs = adapter._processor.call_args
        assert call_kwargs[1]["voice_preset"] == "v2/en_speaker_3"

    def test_synthesize_with_temps(self):
        adapter = self._make_adapter()
        adapter.synthesize("Hello", text_temp=0.5, waveform_temp=0.9)
        gen_kwargs = adapter._model.generate.call_args[1]
        assert gen_kwargs["semantic_temperature"] == 0.5
        assert gen_kwargs["coarse_temperature"] == 0.9

    def test_synthesize_stream_yields_one_chunk(self):
        adapter = self._make_adapter()
        chunks = list(adapter.synthesize_stream("Hello"))
        assert len(chunks) == 1
        assert isinstance(chunks[0], AudioChunk)
        assert chunks[0].sample_rate == 24000

    def test_unknown_kwargs_ignored(self):
        adapter = self._make_adapter()
        result = adapter.synthesize("Hi", bogus_param=42)
        assert isinstance(result, AudioResult)

    def test_list_voices_includes_presets(self):
        adapter = self._make_adapter()
        voices = adapter.list_voices()
        assert "v2/en_speaker_0" in voices
        assert "v2/fr_speaker_5" in voices
        assert len(voices) >= 130  # 13 languages * 10 speakers

    def test_unknown_voice_raises(self):
        adapter = self._make_adapter()
        with pytest.raises(ValueError, match="Unknown voice"):
            adapter.synthesize("Hello", voice="nonexistent_voice")


class TestBarkVoiceCloning:
    """Test voice import and resolution."""

    def _make_adapter(self):
        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.randn(1, 24000)
        mock_model.to.return_value = mock_model

        from narro.models.bark import BarkModel
        adapter = object.__new__(BarkModel)
        adapter._processor = mock_processor
        adapter._model = mock_model
        adapter._device = "cpu"
        return adapter

    def test_create_voice(self, tmp_path):
        adapter = self._make_adapter()

        npz_path = tmp_path / "reference.npz"
        np.savez(
            str(npz_path),
            semantic_prompt=np.zeros(100),
            coarse_prompt=np.zeros((2, 100)),
            fine_prompt=np.zeros((8, 100)),
        )

        adapter.create_voice("alex", str(npz_path))
        assert "alex" in adapter.list_voices()

    def test_create_voice_missing_keys_raises(self, tmp_path):
        adapter = self._make_adapter()

        npz_path = tmp_path / "bad.npz"
        np.savez(str(npz_path), wrong_key=np.zeros(10))

        with pytest.raises(ValueError, match="must contain"):
            adapter.create_voice("bad", str(npz_path))

    def test_create_voice_missing_file_raises(self):
        adapter = self._make_adapter()
        with pytest.raises(FileNotFoundError):
            adapter.create_voice("ghost", "/nonexistent/path.npz")

    def test_custom_voice_resolves(self, tmp_path):
        adapter = self._make_adapter()

        npz_path = tmp_path / "reference.npz"
        np.savez(
            str(npz_path),
            semantic_prompt=np.zeros(100),
            coarse_prompt=np.zeros((2, 100)),
            fine_prompt=np.zeros((8, 100)),
        )
        adapter.create_voice("alex", str(npz_path))

        result = adapter.synthesize("Hello", voice="alex")
        assert isinstance(result, AudioResult)

        call_kwargs = adapter._processor.call_args[1]
        assert "alex.npz" in call_kwargs["voice_preset"]
