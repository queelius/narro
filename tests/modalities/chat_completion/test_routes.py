"""Tests for /v1/chat/completions (streaming + non-streaming).

Uses a FakeModel that implements ChatModel Protocol. No llama-cpp deps.
"""
import json
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.core.sse import iter_sse_events
from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatResult,
)
from muse.modalities.chat_completion.routes import build_router


class _FakeChatModel:
    model_id = "fake-chat"

    def chat(self, messages, **kwargs):
        return ChatResult(
            id="chatcmpl-fake-1",
            model_id=self.model_id,
            created=1_700_000_000,
            choices=[ChatChoice(
                index=0,
                message={"role": "assistant", "content": "hi"},
                finish_reason="stop",
            )],
            usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        )

    def chat_stream(self, messages, **kwargs):
        yield ChatChunk(
            id="chatcmpl-fake-2", model_id=self.model_id, created=0,
            choice_index=0, delta={"role": "assistant"}, finish_reason=None,
        )
        yield ChatChunk(
            id="chatcmpl-fake-2", model_id=self.model_id, created=0,
            choice_index=0, delta={"content": "hi"}, finish_reason=None,
        )
        yield ChatChunk(
            id="chatcmpl-fake-2", model_id=self.model_id, created=0,
            choice_index=0, delta={}, finish_reason="stop",
        )


@pytest.fixture
def client():
    reg = ModalityRegistry()
    reg.register("chat/completion", _FakeChatModel())
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    return TestClient(app)


def test_non_streaming_returns_openai_shape(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-chat",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "fake-chat"
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["usage"]["total_tokens"] == 5


def test_streaming_returns_sse(client):
    with client.stream(
        "POST", "/v1/chat/completions",
        json={
            "model": "fake-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = r.read().decode()
    payloads = [
        data for event_type, data in iter_sse_events(body.splitlines())
        if event_type is None
    ]
    assert payloads[-1] == "[DONE]"
    parsed = [json.loads(p) for p in payloads[:-1]]
    assert len(parsed) == 3
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert parsed[1]["choices"][0]["delta"]["content"] == "hi"
    assert parsed[2]["choices"][0]["finish_reason"] == "stop"


def test_streaming_backend_error_emits_sse_error_event():
    """When chat_stream raises mid-stream, the route emits an SSE
    `event: error` carrying the OpenAI error envelope, followed by
    [DONE]. The earlier behavior was to swallow the exception and emit
    only [DONE], indistinguishable from a clean empty completion.
    """
    reg = ModalityRegistry()

    class _BoomModel:
        model_id = "boom"

        def chat(self, messages, **kwargs):
            raise NotImplementedError

        def chat_stream(self, messages, **kwargs):
            yield ChatChunk(
                id="x", model_id=self.model_id, created=0,
                choice_index=0, delta={"role": "assistant"}, finish_reason=None,
            )
            raise RuntimeError("backend exploded")

    reg.register("chat/completion", _BoomModel())
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    client = TestClient(app)

    with client.stream(
        "POST", "/v1/chat/completions",
        json={
            "model": "boom",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        body = r.read().decode()

    events = list(iter_sse_events(body.splitlines()))

    # Expected sequence: at least one normal data chunk, then an error
    # event, then [DONE].
    assert events[-1] == (None, "[DONE]"), f"final event was {events[-1]!r}"
    error_events = [e for e in events if e[0] == "error"]
    assert len(error_events) == 1, f"expected one error event, got {events}"
    payload = json.loads(error_events[0][1])
    assert payload["error"]["code"] == "internal_error"
    assert "backend exploded" not in payload["error"]["message"]
    assert "see server logs" in payload["error"]["message"]
    assert payload["error"]["type"] == "server_error"


def test_unknown_model_returns_404_with_openai_envelope(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "nonexistent", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "model_not_found"


def test_empty_messages_returns_422(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "fake-chat", "messages": []},
    )
    assert r.status_code == 422


@pytest.mark.parametrize("field,value", [
    ("max_tokens", 0),
    ("max_tokens", 32_769),
    ("temperature", 2.1),
    ("top_p", 1.1),
    ("top_logprobs", 21),
])
def test_generation_controls_are_bounded(client, field, value):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-chat",
            "messages": [{"role": "user", "content": "hi"}],
            field: value,
        },
    )
    assert r.status_code == 422


def test_extra_body_cannot_bypass_max_tokens(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"max_new_tokens": 1_000_000},
        },
    )
    assert r.status_code == 422


def test_default_model_used_when_model_omitted(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json()["model"] == "fake-chat"


def test_tools_passthrough_to_backend():
    """Backend's chat() gets tools kwarg when client sends one."""
    reg = ModalityRegistry()
    seen = {}

    class _Capturing:
        model_id = "cap"

        def chat(self, messages, **kwargs):
            seen.update(kwargs)
            return ChatResult(
                id="x", model_id=self.model_id, created=0,
                choices=[ChatChoice(index=0, message={"role": "assistant", "content": ""}, finish_reason="stop")],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        def chat_stream(self, messages, **kwargs):
            return iter([])

    reg.register("chat/completion", _Capturing())
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    client = TestClient(app)
    tools = [{"type": "function", "function": {
        "name": "get_weather", "parameters": {"type": "object"},
    }}]
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "cap",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        },
    )
    assert r.status_code == 200
    assert seen["tools"] == tools
    assert seen["tool_choice"] == "auto"
    assert seen["temperature"] == 0.2


# --- v0.11.5: tools-with-unknown-support warning ---------------------------


def _model_with_supports_tools(value):
    """A tiny FakeChatModel whose supports_tools attribute is settable."""
    class _M:
        model_id = "tools-test"
        supports_tools = value
        def chat(self, messages, **kwargs):
            return ChatResult(
                id="x", model_id=self.model_id, created=0,
                choices=[ChatChoice(index=0, message={"role":"assistant","content":""}, finish_reason="stop")],
                usage={"prompt_tokens":0,"completion_tokens":0,"total_tokens":0},
            )
        def chat_stream(self, messages, **kwargs):
            return iter([])
    return _M()


def _client_for(model):
    reg = ModalityRegistry()
    reg.register("chat/completion", model)
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    return TestClient(app)


def test_route_warns_when_tools_requested_and_supports_tools_is_None(caplog):
    """capabilities.supports_tools == None means 'unknown'; warn at chat time."""
    import logging
    caplog.set_level(logging.WARNING, logger="muse.modalities.chat_completion.routes")
    client = _client_for(_model_with_supports_tools(None))
    r = client.post("/v1/chat/completions", json={
        "model": "tools-test",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
    })
    assert r.status_code == 200
    assert "unknown" in caplog.text.lower()
    assert "tools-test" in caplog.text


def test_route_warns_when_tools_requested_and_supports_tools_is_False(caplog):
    """capabilities.supports_tools == False means 'known not to support'; warn."""
    import logging
    caplog.set_level(logging.WARNING, logger="muse.modalities.chat_completion.routes")
    client = _client_for(_model_with_supports_tools(False))
    r = client.post("/v1/chat/completions", json={
        "model": "tools-test",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
    })
    assert r.status_code == 200
    assert "not known to support" in caplog.text.lower()


def test_route_silent_when_tools_requested_and_supports_tools_is_True(caplog):
    """capabilities.supports_tools == True: silent (we believe tools work)."""
    import logging
    caplog.set_level(logging.WARNING, logger="muse.modalities.chat_completion.routes")
    client = _client_for(_model_with_supports_tools(True))
    r = client.post("/v1/chat/completions", json={
        "model": "tools-test",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
    })
    assert r.status_code == 200
    # No tool-related warnings
    assert "tool" not in caplog.text.lower() or "warning" not in caplog.text.lower()


def test_route_silent_when_no_tools_requested(caplog):
    """No tools in request: never warn, even if supports_tools is None."""
    import logging
    caplog.set_level(logging.WARNING, logger="muse.modalities.chat_completion.routes")
    client = _client_for(_model_with_supports_tools(None))
    r = client.post("/v1/chat/completions", json={
        "model": "tools-test",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert caplog.text == ""


# H2 regression guards: _inference_lock must be held around backend calls.

def test_non_streaming_chat_called_under_inference_lock():
    """Non-streaming model.chat() must execute while _inference_lock is held.

    A backend that asserts self._inference_lock.locked() during chat() lets us
    verify the route holds the lock across the blocking call (H2 fix).
    """
    import threading

    class _LockAssertingModel:
        model_id = "lock-chat"

        def __init__(self):
            self._inference_lock = threading.Lock()
            self.lock_was_held = False

        def chat(self, messages, **kwargs):
            self.lock_was_held = self._inference_lock.locked()
            return ChatResult(
                id="x", model_id=self.model_id, created=0,
                choices=[ChatChoice(
                    index=0,
                    message={"role": "assistant", "content": "ok"},
                    finish_reason="stop",
                )],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

        def chat_stream(self, messages, **kwargs):
            return iter([])

    model = _LockAssertingModel()
    reg = ModalityRegistry()
    reg.register("chat/completion", model)
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "lock-chat",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    assert model.lock_was_held, (
        "model.chat() was called without holding _inference_lock"
    )


def test_streaming_chat_lock_released_after_stream_completes():
    """_inference_lock must be released after the stream completes.

    The producer thread acquires the lock for the whole stream; the event
    generator drains the queue. After the HTTP response is consumed, the
    lock must be free again (H2 fix for streaming path).
    """
    import threading
    import time

    class _StreamingLockModel:
        model_id = "lock-stream"

        def __init__(self):
            self._inference_lock = threading.Lock()

        def chat(self, messages, **kwargs):
            raise NotImplementedError

        def chat_stream(self, messages, **kwargs):
            # Verify the lock is held while streaming.
            assert self._inference_lock.locked(), (
                "lock should be held during chat_stream iteration"
            )
            yield ChatChunk(
                id="s", model_id=self.model_id, created=0,
                choice_index=0, delta={"content": "hi"}, finish_reason=None,
            )
            yield ChatChunk(
                id="s", model_id=self.model_id, created=0,
                choice_index=0, delta={}, finish_reason="stop",
            )

    model = _StreamingLockModel()
    reg = ModalityRegistry()
    reg.register("chat/completion", model)
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    client = TestClient(app)

    with client.stream(
        "POST", "/v1/chat/completions",
        json={
            "model": "lock-stream",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        r.read()

    # Give the producer thread a moment to finish and release the lock.
    deadline = time.time() + 2.0
    while model._inference_lock.locked() and time.time() < deadline:
        time.sleep(0.01)

    assert not model._inference_lock.locked(), (
        "_inference_lock was not released after streaming completed"
    )


def test_streaming_lock_released_after_chat_stream_raises():
    """H2+H6 composition: when chat_stream raises mid-stream, the
    _inference_lock must still be released (H2) AND the error must
    surface as an SSE error event (H6 interaction with the route layer).
    """
    import threading
    import time

    class _ExplodingStreamModel:
        model_id = "exploding"

        def __init__(self):
            self._inference_lock = threading.Lock()

        def chat(self, messages, **kwargs):
            raise NotImplementedError

        def chat_stream(self, messages, **kwargs):
            yield ChatChunk(
                id="e", model_id=self.model_id, created=0,
                choice_index=0, delta={"content": "partial"}, finish_reason=None,
            )
            raise RuntimeError("mid-stream kaboom")

    model = _ExplodingStreamModel()
    reg = ModalityRegistry()
    reg.register("chat/completion", model)
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    client = TestClient(app)

    with client.stream(
        "POST", "/v1/chat/completions",
        json={
            "model": "exploding",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        body = r.read().decode()

    # Error must surface as SSE error event (not silent truncation).
    assert "mid-stream kaboom" in body or "internal" in body

    # Lock must be released after the stream (H2 must compose with H6).
    deadline = time.time() + 2.0
    while model._inference_lock.locked() and time.time() < deadline:
        time.sleep(0.01)
    assert not model._inference_lock.locked(), (
        "_inference_lock was not released after chat_stream raised"
    )


@pytest.mark.asyncio
async def test_cancelled_stream_stops_waiting_for_inference_lock():
    """A disconnected queued stream must not become a ghost generation."""
    import asyncio
    import threading
    from muse.modalities.chat_completion.routes import ChatCompletionRequest

    class _LockWaitingModel:
        model_id = "lock-waiting-chat"

        def __init__(self):
            self._inference_lock = threading.Lock()
            self.called = threading.Event()

        def chat(self, messages, **kwargs):
            raise NotImplementedError

        def chat_stream(self, messages, **kwargs):
            self.called.set()
            yield ChatChunk(
                id="never", model_id=self.model_id, created=0,
                choice_index=0, delta={"content": "late"},
                finish_reason=None,
            )

    model = _LockWaitingModel()
    reg = ModalityRegistry()
    reg.register("chat/completion", model)
    endpoint = next(
        route.endpoint
        for route in build_router(reg).routes
        if route.path == "/v1/chat/completions"
    )

    model._inference_lock.acquire()
    try:
        response = await endpoint(ChatCompletionRequest(
            model=model.model_id,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        ))
        pending = asyncio.create_task(response.body_iterator.__anext__())
        await asyncio.sleep(0.15)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.sleep(0.15)

        assert not model.called.is_set()
        assert model._inference_lock.locked()
    finally:
        model._inference_lock.release()
