"""Tests for narro.server — FastAPI TTS server."""
import json
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked Narro."""
    with patch('narro.server._get_tts') as mock_get:
        import torch
        mock_tts = MagicMock()
        mock_tts.device = 'cpu'
        mock_tts.model_id = 'test/model'
        mock_tts.infer.return_value = torch.randn(3200)
        mock_get.return_value = mock_tts

        from narro.server import app
        yield TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "device" in data


class TestSpeech:
    def test_speech_returns_audio(self, client):
        resp = client.post("/v1/audio/speech", json={"input": "Hello world"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert len(resp.content) > 44

    def test_speech_rejects_empty_input(self, client):
        resp = client.post("/v1/audio/speech", json={"input": ""})
        assert resp.status_code == 400

    def test_speech_rejects_missing_input(self, client):
        resp = client.post("/v1/audio/speech", json={})
        assert resp.status_code in (400, 422)
