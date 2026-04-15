"""Tests for ChatClient (HTTP client for /v1/chat/completions)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.chat_completion.client import ChatClient


def test_client_non_streaming_returns_dict():
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "id": "chatcmpl-1", "object": "chat.completion",
        "created": 0, "model": "fake",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    fake_response.raise_for_status = MagicMock()
    with patch("muse.modalities.chat_completion.client.httpx.post", return_value=fake_response) as mock_post:
        c = ChatClient(base_url="http://localhost:8000")
        result = c.chat(model="fake", messages=[{"role": "user", "content": "hi"}])
        assert result["choices"][0]["message"]["content"] == "hi"
        url = mock_post.call_args.args[0]
        assert url == "http://localhost:8000/v1/chat/completions"
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "fake"
        assert body["stream"] is False


def test_client_streaming_yields_chunks():
    """stream=True: client opens a stream and yields parsed chunk dicts."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.iter_lines.return_value = [
        "data: " + '{"choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}',
        "",
        "data: " + '{"choices":[{"delta":{"content":"hi"},"index":0,"finish_reason":null}]}',
        "",
        "data: [DONE]",
        "",
    ]
    fake_response.raise_for_status = MagicMock()

    fake_stream_cm = MagicMock()
    fake_stream_cm.__enter__ = lambda s: fake_response
    fake_stream_cm.__exit__ = lambda s, a, b, c: None

    with patch("muse.modalities.chat_completion.client.httpx.stream", return_value=fake_stream_cm):
        c = ChatClient(base_url="http://x")
        out = list(c.chat_stream(model="fake", messages=[{"role": "user", "content": "hi"}]))
        assert len(out) == 2
        assert out[0]["choices"][0]["delta"]["role"] == "assistant"
        assert out[1]["choices"][0]["delta"]["content"] == "hi"


def test_client_uses_muse_server_env_var(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://example.test:9000")
    c = ChatClient()
    assert c.base_url == "http://example.test:9000"


def test_client_strips_trailing_slash_in_base_url():
    c = ChatClient(base_url="http://example.test/")
    assert c.base_url == "http://example.test"


def test_client_forwards_tools_kwarg():
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"choices": [], "usage": {}}
    fake_response.raise_for_status = MagicMock()
    with patch("muse.modalities.chat_completion.client.httpx.post", return_value=fake_response) as mock_post:
        c = ChatClient(base_url="http://x")
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        c.chat(model="m", messages=[{"role": "user", "content": "x"}], tools=tools, tool_choice="auto")
        body = mock_post.call_args.kwargs["json"]
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"
