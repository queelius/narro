"""Contract tests for the Muse streaming benchmark client."""

from unittest.mock import MagicMock, patch

import pytest

from scripts.bench.bench_llm import MuseClient


def _stream_response(lines: list[str], content_type: str = "text/event-stream"):
    response = MagicMock()
    response.headers = {"content-type": content_type}
    response.iter_lines.return_value = lines
    stream = MagicMock()
    stream.__enter__ = lambda _self: response
    stream.__exit__ = lambda _self, _type, _value, _traceback: None
    return stream


def test_muse_benchmark_stream_requires_sse_content_type():
    stream = _stream_response([], content_type="application/json")
    with patch("scripts.bench.bench_llm.httpx.stream", return_value=stream):
        client = MuseClient("http://example.test", "model")
        with pytest.raises(RuntimeError, match="not text/event-stream"):
            client.stream_ttft([{"role": "user", "content": "hi"}], 8)


def test_muse_benchmark_stream_requires_done_sentinel():
    stream = _stream_response([
        'data:{"choices":[{"delta":{"content":"partial"}}]}',
        "",
    ])
    with patch("scripts.bench.bench_llm.httpx.stream", return_value=stream):
        client = MuseClient("http://example.test", "model")
        with pytest.raises(RuntimeError, match="before the \\[DONE\\] sentinel"):
            client.stream_ttft([{"role": "user", "content": "hi"}], 8)


def test_muse_benchmark_stream_accepts_compact_complete_events():
    stream = _stream_response([
        'data:{"choices":[{"delta":{"content":"hello"}}]}',
        "",
        "data:[DONE]",
        "",
    ])
    with patch("scripts.bench.bench_llm.httpx.stream", return_value=stream):
        client = MuseClient("http://example.test", "model")
        result = client.stream_ttft([{"role": "user", "content": "hi"}], 8)

    assert result["tokens"] == 1
    assert result["ttft"] is not None
