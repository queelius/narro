"""Tests for narro.catalog: model catalog, pull state, and loading."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from narro.catalog import (
    KNOWN_MODELS,
    ModelEntry,
    is_pulled,
    load_backend,
    pull,
    pulled_models,
    remove,
)


@pytest.fixture(autouse=True)
def _isolate_catalog(tmp_path, monkeypatch):
    """Point NARRO_HOME to a temp directory for every test."""
    monkeypatch.setenv("NARRO_HOME", str(tmp_path / "narro"))


class TestKnownModels:
    def test_soprano_in_known_models(self):
        assert "soprano-80m" in KNOWN_MODELS

    def test_entry_has_required_fields(self):
        for mid, entry in KNOWN_MODELS.items():
            assert entry.id == mid
            assert entry.hf_repo
            assert entry.backend
            assert entry.sample_rate > 0


class TestPullState:
    def test_empty_catalog(self):
        assert pulled_models() == {}
        assert not is_pulled("soprano-80m")

    def test_pull_records_state(self):
        with patch("huggingface_hub.snapshot_download"):
            pull("soprano-80m")

        assert is_pulled("soprano-80m")
        state = pulled_models()
        assert "soprano-80m" in state
        assert "pulled_at" in state["soprano-80m"]
        assert "hf_repo" in state["soprano-80m"]

    def test_pull_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            pull("nonexistent")

    def test_remove_unpulled_raises(self):
        with pytest.raises(KeyError, match="not pulled"):
            remove("soprano-80m")

    def test_pull_then_remove(self):
        with patch("huggingface_hub.snapshot_download"):
            pull("soprano-80m")
        assert is_pulled("soprano-80m")
        remove("soprano-80m")
        assert not is_pulled("soprano-80m")


class TestLoadBackend:
    def test_load_unpulled_raises(self):
        with pytest.raises(RuntimeError, match="not pulled"):
            load_backend("soprano-80m")

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            load_backend("nonexistent")

    def test_load_pulled_model(self):
        with patch("huggingface_hub.snapshot_download"):
            pull("soprano-80m")

        mock_cls = MagicMock()
        with patch("narro.tts.Narro", mock_cls), \
             patch("narro.tts.SAMPLE_RATE", 32000):
            backend = load_backend("soprano-80m", device="cpu", compile=False)

        from narro.protocol import TTSModel
        assert isinstance(backend, TTSModel)
