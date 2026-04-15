"""Tests for chat/completion wire codec (SSE + OpenAI dict shapes)."""
import json

from muse.modalities.chat_completion.codec import (
    chunk_to_sse_data,
    chunk_to_openai_dict,
    result_to_openai_dict,
    DONE_SENTINEL,
)
from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatResult,
)


def test_chunk_to_openai_dict_maps_fields():
    c = ChatChunk(
        id="chatcmpl-1",
        model_id="qwen3-8b",
        created=1_700_000_000,
        choice_index=0,
        delta={"role": "assistant", "content": "hi"},
        finish_reason=None,
    )
    d = chunk_to_openai_dict(c)
    assert d["id"] == "chatcmpl-1"
    assert d["object"] == "chat.completion.chunk"
    assert d["created"] == 1_700_000_000
    assert d["model"] == "qwen3-8b"
    assert d["choices"] == [{
        "index": 0,
        "delta": {"role": "assistant", "content": "hi"},
        "finish_reason": None,
    }]


def test_chunk_with_finish_reason():
    c = ChatChunk(
        id="x", model_id="y", created=0, choice_index=0,
        delta={}, finish_reason="stop",
    )
    d = chunk_to_openai_dict(c)
    assert d["choices"][0]["finish_reason"] == "stop"


def test_chunk_to_sse_data_json_encodes():
    c = ChatChunk(
        id="a", model_id="b", created=1, choice_index=0,
        delta={"content": "hello"}, finish_reason=None,
    )
    sse = chunk_to_sse_data(c)
    decoded = json.loads(sse)
    assert decoded["choices"][0]["delta"]["content"] == "hello"


def test_done_sentinel():
    assert DONE_SENTINEL == "[DONE]"


def test_result_to_openai_dict_shape():
    r = ChatResult(
        id="chatcmpl-xyz",
        model_id="qwen3-8b",
        created=1_700_000_001,
        choices=[ChatChoice(
            index=0,
            message={"role": "assistant", "content": "world"},
            finish_reason="stop",
        )],
        usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    )
    d = result_to_openai_dict(r)
    assert d["object"] == "chat.completion"
    assert d["id"] == "chatcmpl-xyz"
    assert d["choices"][0]["message"]["content"] == "world"
    assert d["choices"][0]["finish_reason"] == "stop"
    assert d["usage"]["total_tokens"] == 4
