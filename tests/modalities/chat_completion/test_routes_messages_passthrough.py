"""Tests that EVERY OpenAI message field reaches the backend untouched.

This is the fast unit-level guard against muse silently dropping fields
in tool-use round-trips. The integration tests against a real model
(tests/integration/) prove tools work end-to-end; THESE tests prove
that whatever the SDK sends, the backend sees verbatim.

Specifically guards against the failure mode discovered live: tool
results not influencing the model's response. The first thing to
verify is that the tool result message is reaching the backend at all,
with all its OpenAI-shape fields (tool_call_id, name, content) intact.
"""
from typing import Any

import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatResult,
)
from muse.modalities.chat_completion.routes import build_router


class _CapturingModel:
    """Backend that records exactly what messages and kwargs reach it."""
    model_id = "captor"
    supports_tools = True

    def __init__(self):
        self.last_messages: list[dict] | None = None
        self.last_kwargs: dict | None = None

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return ChatResult(
            id="x", model_id=self.model_id, created=0,
            choices=[ChatChoice(
                index=0,
                message={"role": "assistant", "content": "ok"},
                finish_reason="stop",
            )],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def chat_stream(self, messages, **kwargs):
        return iter([])


@pytest.fixture
def captor_and_client():
    model = _CapturingModel()
    reg = ModalityRegistry()
    reg.register("chat/completion", model)
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    return model, TestClient(app)


def test_assistant_message_with_tool_calls_passes_through(captor_and_client):
    """An assistant message carrying tool_calls must reach the backend with
    tool_calls field intact (not flattened or dropped)."""
    model, client = captor_and_client
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc",
            "name": "get_weather",
            "content": '{"temp_c": 18}',
        },
    ]
    r = client.post("/v1/chat/completions", json={
        "model": "captor",
        "messages": messages,
    })
    assert r.status_code == 200, r.text
    assert model.last_messages == messages, (
        "Backend did not receive the messages verbatim"
    )


def test_tool_message_fields_preserved_exactly(captor_and_client):
    """The tool message's three required fields (role, tool_call_id, content)
    plus the optional name field must all reach the backend.

    Regression guard: muse's Pydantic model declares messages as list[dict]
    so it should pass through verbatim. This test makes that contract
    explicit so a future schema tightening cannot silently drop fields."""
    model, client = captor_and_client
    tool_msg = {
        "role": "tool",
        "tool_call_id": "call_xyz",
        "name": "lookup_database",
        "content": '{"rows": [{"id": 1, "value": "the moon is made of cheese"}]}',
    }
    r = client.post("/v1/chat/completions", json={
        "model": "captor",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{
                "id": "call_xyz", "type": "function",
                "function": {"name": "lookup_database", "arguments": "{}"},
            }]},
            tool_msg,
        ],
    })
    assert r.status_code == 200, r.text
    received_tool_msg = model.last_messages[2]
    assert received_tool_msg == tool_msg, (
        f"Tool message changed in transit:\n  sent:     {tool_msg}\n  received: {received_tool_msg}"
    )


def test_tools_field_passes_through_to_backend_chat(captor_and_client):
    """The tools list (function definitions) must reach backend.chat()
    via kwargs so the chat handler can format them for the model."""
    model, client = captor_and_client
    tools = [{
        "type": "function",
        "function": {
            "name": "calculate_tip",
            "description": "Calculate tip on a bill",
            "parameters": {
                "type": "object",
                "properties": {
                    "bill": {"type": "number"},
                    "percent": {"type": "number"},
                },
                "required": ["bill", "percent"],
            },
        },
    }]
    r = client.post("/v1/chat/completions", json={
        "model": "captor",
        "messages": [{"role": "user", "content": "tip on $42 at 18%?"}],
        "tools": tools,
        "tool_choice": "auto",
    })
    assert r.status_code == 200, r.text
    assert model.last_kwargs["tools"] == tools
    assert model.last_kwargs["tool_choice"] == "auto"


def test_response_format_passes_through(captor_and_client):
    """response_format={"type": "json_object"} must reach backend."""
    model, client = captor_and_client
    r = client.post("/v1/chat/completions", json={
        "model": "captor",
        "messages": [{"role": "user", "content": "json please"}],
        "response_format": {"type": "json_object"},
    })
    assert r.status_code == 200, r.text
    assert model.last_kwargs["response_format"] == {"type": "json_object"}


def test_full_multi_turn_tool_loop_messages_arrive_in_order(captor_and_client):
    """Realistic agent loop: system, user, assistant w/ tool_calls, tool result,
    next assistant turn. All five messages must arrive at the backend in
    order with all fields preserved. This is the structural test for the
    failure mode 'tool result not influencing the next turn'."""
    model, client = captor_and_client
    messages = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_w1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_w1",
            "name": "get_weather",
            "content": '{"city": "Paris", "temp_c": 47, "weird_marker": "FROGS_SKY_DIVING"}',
        },
    ]
    r = client.post("/v1/chat/completions", json={
        "model": "captor",
        "messages": messages,
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }],
    })
    assert r.status_code == 200, r.text
    # Same length, same order, same content per role
    assert len(model.last_messages) == 4
    assert [m["role"] for m in model.last_messages] == [
        "system", "user", "assistant", "tool",
    ]
    # The weird marker must survive; it's the test's smoking-gun signal
    # that an integration test could check for in the model's response.
    assert "FROGS_SKY_DIVING" in model.last_messages[3]["content"]


def test_extra_body_fields_pass_through_via_extra_body(captor_and_client):
    """Backend-specific fields (e.g. llama.cpp's `repeat_penalty`) reach
    the backend via extra_body."""
    model, client = captor_and_client
    r = client.post("/v1/chat/completions", json={
        "model": "captor",
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"repeat_penalty": 1.15, "weird_future_kwarg": "ok"},
    })
    assert r.status_code == 200, r.text
    assert model.last_kwargs["repeat_penalty"] == 1.15
    assert model.last_kwargs["weird_future_kwarg"] == "ok"
