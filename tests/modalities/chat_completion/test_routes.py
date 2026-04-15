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
    lines = [line for line in body.splitlines() if line.startswith("data: ")]
    payloads = [line[len("data: "):] for line in lines]
    assert payloads[-1] == "[DONE]"
    parsed = [json.loads(p) for p in payloads[:-1]]
    assert len(parsed) == 3
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert parsed[1]["choices"][0]["delta"]["content"] == "hi"
    assert parsed[2]["choices"][0]["finish_reason"] == "stop"


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
