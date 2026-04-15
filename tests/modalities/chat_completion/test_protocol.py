"""Tests for chat/completion protocol types and ChatModel Protocol."""
from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatMessage,
    ChatResult,
    ChatModel,
)


def test_chat_message_roles():
    for role in ("system", "user", "assistant", "tool"):
        msg = ChatMessage(role=role, content="hi")
        assert msg.role == role


def test_chat_result_shape():
    r = ChatResult(
        id="chatcmpl-abc",
        model_id="qwen3-8b-gguf-q4_k_m",
        created=1_700_000_000,
        choices=[ChatChoice(
            index=0,
            message={"role": "assistant", "content": "hello"},
            finish_reason="stop",
        )],
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )
    assert r.choices[0].message["content"] == "hello"
    assert r.usage["total_tokens"] == 12


def test_chat_chunk_shape():
    c = ChatChunk(
        id="chatcmpl-abc",
        model_id="qwen3-8b",
        created=1_700_000_000,
        choice_index=0,
        delta={"role": "assistant", "content": "he"},
        finish_reason=None,
    )
    assert c.delta["content"] == "he"
    assert c.finish_reason is None


def test_chat_model_protocol_is_runtime_checkable():
    """A plain class with the right attrs/methods should satisfy ChatModel."""
    class Fake:
        model_id = "x"

        def chat(self, messages, **kwargs):
            return None

        def chat_stream(self, messages, **kwargs):
            return iter([])

    assert isinstance(Fake(), ChatModel)


def test_chat_model_protocol_rejects_missing_method():
    class NoStream:
        model_id = "x"

        def chat(self, messages, **kwargs):
            return None

    assert not isinstance(NoStream(), ChatModel)
