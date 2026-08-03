"""Tests for ChatClient (HTTP client for /v1/chat/completions)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.chat_completion.client import ChatClient, ChatStreamError


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
    fake_response.headers = {"content-type": "text/event-stream; charset=utf-8"}
    fake_response.iter_lines.return_value = [
        "data:" + '{"choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}',
        "",
        "data: " + '{"choices":[{"delta":{"content":"hi"},"index":0,"finish_reason":null}]}',
        "",
        "data:[DONE]",
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


def test_client_streaming_raises_on_sse_error_frame():
    """L6: a mid-stream `event: error` frame must raise, not be yielded as a
    normal chunk (callers iterating chunk["choices"] would KeyError)."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"content-type": "text/event-stream"}
    fake_response.iter_lines.return_value = [
        "data: " + '{"choices":[{"delta":{"content":"partial"},"index":0}]}',
        "",
        "event:error",
        "data:" + '{"error":{"code":"internal","message":"backend blew up","type":"server_error"}}',
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
        gen = c.chat_stream(model="fake", messages=[{"role": "user", "content": "hi"}])
        first = next(gen)  # the partial content chunk arrives normally
        assert first["choices"][0]["delta"]["content"] == "partial"
        with pytest.raises(ChatStreamError) as exc:
            next(gen)
        assert "backend blew up" in str(exc.value)
        assert exc.value.error["code"] == "internal"


def test_client_streaming_rejects_non_sse_response():
    fake_response = MagicMock()
    fake_response.headers = {"content-type": "application/json"}
    fake_response.iter_lines.return_value = []

    fake_stream_cm = MagicMock()
    fake_stream_cm.__enter__ = lambda s: fake_response
    fake_stream_cm.__exit__ = lambda s, a, b, c: None

    with patch(
        "muse.modalities.chat_completion.client.httpx.stream",
        return_value=fake_stream_cm,
    ):
        client = ChatClient(base_url="http://x")
        with pytest.raises(ChatStreamError) as exc:
            list(client.chat_stream(messages=[{"role": "user", "content": "hi"}]))
    assert exc.value.error["code"] == "invalid_stream_response"


def test_client_streaming_rejects_eof_before_done():
    fake_response = MagicMock()
    fake_response.headers = {"content-type": "text/event-stream"}
    fake_response.iter_lines.return_value = [
        'data: {"choices":[{"delta":{"content":"partial"}}]}',
        "",
    ]

    fake_stream_cm = MagicMock()
    fake_stream_cm.__enter__ = lambda s: fake_response
    fake_stream_cm.__exit__ = lambda s, a, b, c: None

    with patch(
        "muse.modalities.chat_completion.client.httpx.stream",
        return_value=fake_stream_cm,
    ):
        client = ChatClient(base_url="http://x")
        with pytest.raises(ChatStreamError) as exc:
            list(client.chat_stream(messages=[{"role": "user", "content": "hi"}]))
    assert exc.value.error["code"] == "incomplete_stream"


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
