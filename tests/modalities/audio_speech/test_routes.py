"""Tests for /v1/audio/speech FastAPI router."""
import asyncio
import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.modalities.audio_speech import routes as routes_mod
from muse.modalities.audio_speech.protocol import AudioChunk, AudioResult
from muse.modalities.audio_speech.routes import build_router
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app


class FakeTTS:
    model_id = "fake-tts"
    sample_rate = 16000
    voices = ["default", "alt"]

    def synthesize(self, text, **kwargs):
        n = max(1000, len(text) * 100)
        return AudioResult(
            audio=np.zeros(n, dtype=np.float32),
            sample_rate=self.sample_rate,
            metadata={"duration": n / self.sample_rate},
        )

    def synthesize_stream(self, text, **kwargs):
        for _ in range(3):
            yield AudioChunk(audio=np.zeros(500, dtype=np.float32), sample_rate=self.sample_rate)


@pytest.fixture
def client():
    reg = ModalityRegistry()
    reg.register("audio/speech", FakeTTS())
    app = create_app(registry=reg, routers={"audio/speech": build_router(reg)})
    return TestClient(app)


def test_list_voices_endpoint(client):
    r = client.get("/v1/audio/speech/voices")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake-tts"
    assert "default" in body["voices"]
    assert "alt" in body["voices"]


def test_speech_wav_response(client):
    r = client.post("/v1/audio/speech", json={
        "input": "hello world",
        "model": "fake-tts",
        "response_format": "wav",
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content.startswith(b"RIFF")


def test_speech_default_model_when_unspecified(client):
    r = client.post("/v1/audio/speech", json={"input": "hello"})
    assert r.status_code == 200
    assert r.content.startswith(b"RIFF")


def test_unknown_model_returns_404(client):
    r = client.post("/v1/audio/speech", json={
        "input": "hello",
        "model": "does-not-exist",
    })
    assert r.status_code == 404


def test_empty_input_returns_400(client):
    r = client.post("/v1/audio/speech", json={"input": ""})
    # Pydantic v2 validation yields 422 by default; either is acceptable
    assert r.status_code in (400, 422)


def test_oversize_input_returns_400(client):
    r = client.post("/v1/audio/speech", json={"input": "x" * 60_000})
    assert r.status_code in (400, 422)


def test_streaming_response(client):
    r = client.post("/v1/audio/speech", json={
        "input": "hello world",
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]


def test_404_uses_openai_error_envelope(client):
    """Unknown model must return {error:{...}} not {detail:...}."""
    r = client.post("/v1/audio/speech", json={
        "input": "hi", "model": "no-such",
    })
    assert r.status_code == 404
    body = r.json()
    # OpenAI envelope: top-level "error" with code/message/type
    assert "error" in body
    assert "detail" not in body
    err = body["error"]
    assert err["code"] == "model_not_found"
    assert err["type"] == "invalid_request_error"
    assert "no-such" in err["message"]


def test_voices_404_uses_openai_error_envelope(client):
    r = client.get("/v1/audio/speech/voices?model=no-such")
    assert r.status_code == 404
    assert "error" in r.json()


def test_streaming_yields_multiple_events_progressively(client):
    """With the producer-queue pattern, each chunk is a distinct SSE event."""
    r = client.post("/v1/audio/speech", json={
        "input": "hello world",
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    text = r.text
    # FakeTTS yields 3 chunks then "done". Expect at least 3 data events + done.
    data_event_count = text.count("data: ")  # each SSE event starts with "data: "
    assert data_event_count >= 3
    assert "event: done" in text


def test_encoding_failure_returns_openai_error_envelope(monkeypatch):
    """When wav encoding fails, the response uses the {error:{...}}
    envelope, not FastAPI's {detail:...}.

    The earlier code raised HTTPException(detail=str(e)) which yielded
    {"detail": "..."}, violating CLAUDE.md's stated convention.
    """
    from muse.modalities.audio_speech import routes as routes_mod
    from muse.modalities.audio_speech.codec import AudioFormatError

    def _boom(*a, **kw):
        raise AudioFormatError("PCM out of range")

    monkeypatch.setattr(routes_mod, "audio_to_wav_bytes", _boom)

    reg = ModalityRegistry()
    reg.register("audio/speech", FakeTTS())
    app = create_app(registry=reg, routers={"audio/speech": routes_mod.build_router(reg)})
    client_local = TestClient(app)

    r = client_local.post("/v1/audio/speech", json={
        "input": "hi", "model": "fake-tts",
    })
    assert r.status_code == 500
    body = r.json()
    assert "error" in body
    assert "detail" not in body
    err = body["error"]
    assert err["code"] == "encoding_failed"
    # Finding 1 (v0.58.1 review): the backend exception text must NOT
    # reach the client body; only a generic message does.
    assert "PCM out of range" not in err["message"]
    # 5xx statuses carry type "server_error" (L10: error_type is derived
    # from the status code, not hardcoded to invalid_request_error).
    assert err["type"] == "server_error"


class _FastStreamingTTS:
    model_id = "fast-streaming-tts"
    sample_rate = 16_000

    def __init__(self) -> None:
        self._inference_lock = threading.Lock()
        self.produced = 0
        self.closed = threading.Event()

    def synthesize_stream(self, text, **kwargs):
        try:
            for _ in range(1_000):
                self.produced += 1
                yield AudioChunk(
                    audio=np.zeros(8, dtype=np.float32),
                    sample_rate=self.sample_rate,
                )
        finally:
            self.closed.set()


@pytest.mark.asyncio
async def test_stream_backpressure_is_bounded_and_early_close_releases_model():
    model = _FastStreamingTTS()
    response = await routes_mod._stream(
        model,
        routes_mod.SpeechRequest(input="hello", stream=True),
    )
    events = response.body_iterator

    first = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert "data" in first

    # Stop consuming while the native producer runs. It may have yielded
    # the consumed item, one item waiting for a slot, and at most one item
    # per bounded queue slot; it must not run through all 1,000 chunks.
    await asyncio.sleep(0.15)
    assert model.produced <= routes_mod._STREAM_QUEUE_DEPTH + 2

    await events.aclose()
    assert model.closed.is_set()
    assert model._inference_lock.acquire(blocking=False)
    model._inference_lock.release()


class _LockWaitingTTS:
    model_id = "lock-waiting-tts"

    def __init__(self) -> None:
        self._inference_lock = threading.Lock()
        self.called = threading.Event()

    def synthesize_stream(self, text, **kwargs):
        self.called.set()
        yield AudioChunk(audio=np.zeros(8, dtype=np.float32), sample_rate=16_000)


@pytest.mark.asyncio
async def test_cancelled_stream_stops_waiting_for_inference_lock():
    model = _LockWaitingTTS()
    model._inference_lock.acquire()
    try:
        response = await routes_mod._stream(
            model,
            routes_mod.SpeechRequest(input="hello", stream=True),
        )
        events = response.body_iterator
        pending = asyncio.create_task(events.__anext__())
        await asyncio.sleep(0.1)

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        assert not model.called.is_set()
        assert model._inference_lock.locked()
    finally:
        model._inference_lock.release()


class _FailingStreamingTTS:
    model_id = "failing-streaming-tts"

    def __init__(self) -> None:
        self._inference_lock = threading.Lock()

    def synthesize_stream(self, text, **kwargs):
        raise RuntimeError("secret backend path: /srv/private/model.bin")
        yield  # pragma: no cover - makes this an iterator


@pytest.mark.asyncio
async def test_stream_backend_error_is_structured_and_sanitized():
    response = await routes_mod._stream(
        _FailingStreamingTTS(),
        routes_mod.SpeechRequest(input="hello", stream=True),
    )
    events = [event async for event in response.body_iterator]

    assert [event.get("event") for event in events] == ["error", "done"]
    payload = json.loads(events[0]["data"])
    assert payload["error"]["code"] == "streaming_failed"
    assert payload["error"]["type"] == "server_error"
    assert "secret" not in events[0]["data"]
    assert "/srv/private" not in events[0]["data"]


@pytest.mark.asyncio
async def test_stream_encoding_error_is_structured_and_sanitized(monkeypatch):
    def _fail_encoding(audio):
        raise ValueError("secret encoder state: /tmp/audio.raw")

    monkeypatch.setattr(routes_mod, "float_to_pcm16", _fail_encoding)
    model = _FastStreamingTTS()
    response = await routes_mod._stream(
        model,
        routes_mod.SpeechRequest(input="hello", stream=True),
    )
    events = [event async for event in response.body_iterator]

    assert [event.get("event") for event in events] == ["error", "done"]
    payload = json.loads(events[0]["data"])
    assert payload["error"]["message"] == (
        "audio streaming backend failed; see server logs"
    )
    assert "secret encoder state" not in events[0]["data"]
    assert model.closed.is_set()
