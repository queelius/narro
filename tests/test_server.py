"""Tests for narro.server — FastAPI TTS server with model registry."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from narro.protocol import AudioChunk, AudioResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeModel:
    """Minimal TTSModel for server tests."""

    def __init__(self, model_id="test-model", sample_rate=32000):
        self._model_id = model_id
        self._sample_rate = sample_rate

    @property
    def model_id(self):
        return self._model_id

    @property
    def sample_rate(self):
        return self._sample_rate

    def synthesize(self, text, **kwargs):
        samples = max(3200, len(text) * 100)
        return AudioResult(
            audio=np.random.randn(samples).astype(np.float32) * 0.1,
            sample_rate=self._sample_rate,
            metadata={"alignment": [{"p": 0}]} if kwargs.get("align") else {},
        )

    def synthesize_stream(self, text, **kwargs):
        for _ in range(3):
            yield AudioChunk(
                audio=np.random.randn(1024).astype(np.float32) * 0.1,
                sample_rate=self._sample_rate,
            )


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure a clean registry for every test."""
    from narro.models import registry
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
def client():
    """Create a test client with a FakeModel registered."""
    from narro.models import registry
    from narro.server import app

    registry.register(FakeModel())
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "test-model" in data["models"]

    def test_health_no_models_503(self):
        from narro.server import app
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get("/health")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["sample_rate"] == 32000

    def test_list_multiple_models(self):
        from narro.models import registry
        from narro.server import app

        registry.register(FakeModel("model-a", 22050))
        registry.register(FakeModel("model-b", 32000))
        c = TestClient(app)
        resp = c.get("/v1/models")
        ids = {m["id"] for m in resp.json()["data"]}
        assert ids == {"model-a", "model-b"}


# ---------------------------------------------------------------------------
# /v1/audio/speech
# ---------------------------------------------------------------------------

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

    def test_speech_with_model_field(self, client):
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "model": "test-model"},
        )
        assert resp.status_code == 200

    def test_speech_unknown_model_400(self, client):
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "model": "nonexistent"},
        )
        assert resp.status_code == 400

    def test_speech_align_header(self, client):
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "align": True},
        )
        assert resp.status_code == 200
        assert "x-alignment" in resp.headers

    def test_speech_unsupported_format(self, client):
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "response_format": "mp3"},
        )
        assert resp.status_code == 422

    def test_speech_input_too_long(self, client):
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "x" * 60_000},
        )
        assert resp.status_code == 400

    def test_stream_rejects_opus(self, client):
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "stream": True, "response_format": "opus"},
        )
        assert resp.status_code == 422

    def test_stream_rejects_align(self, client):
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "stream": True, "align": True},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestStreaming:
    def test_stream_returns_sse_events(self, client):
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "stream": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.strip().split("\n")
            if line.startswith("data: ")
        ]
        assert len(events) >= 3  # start + N deltas + done

        assert events[0]["type"] == "speech.audio.start"
        assert events[0]["sample_rate"] == 32000
        assert events[0]["format"] == "pcm_s16le"

        assert events[-1]["type"] == "speech.audio.done"

        deltas = [e for e in events if e["type"] == "speech.audio.delta"]
        assert len(deltas) == 3  # FakeModel yields 3 chunks
        for d in deltas:
            import base64
            pcm = base64.b64decode(d["audio"])
            assert len(pcm) > 0
