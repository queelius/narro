"""Tests for narro.models — model registry and Soprano adapter."""

import numpy as np
import pytest

from narro.models import ModelRegistry
from narro.protocol import AudioChunk, AudioResult, TTSModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubModel:
    """Minimal TTSModel for registry tests."""

    def __init__(self, model_id="stub", sample_rate=16000):
        self._model_id = model_id
        self._sample_rate = sample_rate

    @property
    def model_id(self):
        return self._model_id

    @property
    def sample_rate(self):
        return self._sample_rate

    def synthesize(self, text, **kwargs):
        samples = len(text) * 100
        return AudioResult(audio=np.zeros(samples, dtype=np.float32),
                           sample_rate=self._sample_rate)

    def synthesize_stream(self, text, **kwargs):
        yield AudioChunk(audio=np.zeros(100, dtype=np.float32),
                         sample_rate=self._sample_rate)


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        model = StubModel("alpha")
        reg.register(model)
        assert reg.get("alpha") is model

    def test_first_registered_is_default(self):
        reg = ModelRegistry()
        reg.register(StubModel("first"))
        reg.register(StubModel("second"))
        assert reg.default_model_id == "first"

    def test_get_none_returns_default(self):
        reg = ModelRegistry()
        m = StubModel("only")
        reg.register(m)
        assert reg.get(None) is m
        assert reg.get() is m

    def test_get_unknown_raises(self):
        reg = ModelRegistry()
        reg.register(StubModel("known"))
        with pytest.raises(KeyError, match="nope"):
            reg.get("nope")

    def test_get_empty_registry_raises(self):
        reg = ModelRegistry()
        with pytest.raises(RuntimeError, match="No models"):
            reg.get()

    def test_list_models(self):
        reg = ModelRegistry()
        reg.register(StubModel("a", sample_rate=22050))
        reg.register(StubModel("b", sample_rate=32000))
        infos = reg.list_models()
        assert len(infos) == 2
        ids = {i.id for i in infos}
        assert ids == {"a", "b"}

    def test_set_default(self):
        reg = ModelRegistry()
        reg.register(StubModel("a"))
        reg.register(StubModel("b"))
        reg.set_default("b")
        assert reg.default_model_id == "b"

    def test_set_default_unregistered_raises(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.set_default("ghost")

    def test_clear(self):
        reg = ModelRegistry()
        reg.register(StubModel("x"))
        assert len(reg) == 1
        reg.clear()
        assert len(reg) == 0
        assert reg.default_model_id is None

    def test_rejects_non_protocol(self):
        reg = ModelRegistry()
        with pytest.raises(TypeError, match="does not implement"):
            reg.register("not a model")


# ---------------------------------------------------------------------------
# SopranoModel adapter (unit-level, mocked tts.Narro)
# ---------------------------------------------------------------------------

class TestSopranoModel:
    """Test the adapter without loading the real model."""

    def _make_adapter(self, mock_narro_cls):
        """Instantiate SopranoModel with a mocked Narro class.

        Patches narro.tts.Narro (the source) so the lazy import inside
        SopranoModel.__init__ picks up the mock.
        """
        from unittest.mock import patch
        with patch("narro.tts.Narro", mock_narro_cls), \
             patch("narro.tts.SAMPLE_RATE", 32000):
            from narro.models.soprano import SopranoModel
            return SopranoModel(model_path=None, compile=False)

    def test_protocol_conformance(self):
        from unittest.mock import MagicMock
        import torch

        mock_cls = MagicMock()
        mock_tts = MagicMock()
        mock_tts.model_id = "soprano-80m"
        mock_tts.infer.return_value = torch.randn(3200)
        mock_cls.return_value = mock_tts

        adapter = self._make_adapter(mock_cls)
        assert isinstance(adapter, TTSModel)

    def test_synthesize_returns_audio_result(self):
        from unittest.mock import MagicMock
        import torch

        mock_cls = MagicMock()
        mock_tts = MagicMock()
        mock_tts.model_id = "soprano-80m"
        mock_tts.infer.return_value = torch.randn(3200)
        mock_cls.return_value = mock_tts

        adapter = self._make_adapter(mock_cls)
        result = adapter.synthesize("Hello world")

        assert isinstance(result, AudioResult)
        assert result.sample_rate == 32000
        assert len(result.audio) == 3200
        assert result.metadata == {}

    def test_synthesize_with_align(self):
        from unittest.mock import MagicMock, patch
        import torch

        mock_cls = MagicMock()
        mock_tts = MagicMock()
        mock_tts.model_id = "soprano-80m"
        mock_tts.encode_batch.return_value = MagicMock()
        mock_tts.decode.return_value = [torch.randn(3200)]
        mock_cls.return_value = mock_tts

        adapter = self._make_adapter(mock_cls)
        with patch("narro.alignment.extract_paragraph_alignment",
                   return_value=[{"paragraph": 0}]):
            result = adapter.synthesize("Hello\n\nWorld", align=True)

        assert "alignment" in result.metadata

    def test_synthesize_stream_yields_chunks(self):
        from unittest.mock import MagicMock
        import torch

        mock_cls = MagicMock()
        mock_tts = MagicMock()
        mock_tts.model_id = "soprano-80m"
        mock_tts.infer_stream.return_value = [torch.randn(1024), torch.randn(1024)]
        mock_cls.return_value = mock_tts

        adapter = self._make_adapter(mock_cls)
        chunks = list(adapter.synthesize_stream("Hello"))

        assert len(chunks) == 2
        assert all(isinstance(c, AudioChunk) for c in chunks)

    def test_unknown_kwargs_ignored(self):
        """Extra kwargs should not crash — _pick filters them out."""
        from unittest.mock import MagicMock
        import torch

        mock_cls = MagicMock()
        mock_tts = MagicMock()
        mock_tts.model_id = "soprano-80m"
        mock_tts.infer.return_value = torch.randn(3200)
        mock_cls.return_value = mock_tts

        adapter = self._make_adapter(mock_cls)
        result = adapter.synthesize("Hi", bogus_param=42)
        assert isinstance(result, AudioResult)
