"""Chat completion modality: OpenAI-shape chat + tool calling.

Types mirror OpenAI's ChatCompletion / ChatCompletionChunk structure
exactly at the wire level (dict passthrough for messages, tool_calls,
tool_choice, response_format) but the ChatModel Protocol is
backend-agnostic. Backends translate their native format into
ChatResult / ChatChunk and return them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Protocol, runtime_checkable


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatMessage:
    """Input message. Primarily for type-checking tests; routes
    accept raw dicts and pass them through so backends can see
    fields we haven't modeled yet (logprobs, name, etc.)."""
    role: Role
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ChatChoice:
    """One choice in a non-streaming response.

    `message` is a raw dict (OpenAI shape) so backends can include
    tool_calls, refusal, or future fields without us modeling them.
    """
    index: int
    message: dict
    finish_reason: str | None


@dataclass
class ChatResult:
    """Non-streaming chat response. Mirrors OpenAI ChatCompletion."""
    id: str
    model_id: str
    created: int
    choices: list[ChatChoice]
    usage: dict
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatChunk:
    """Streaming delta. Mirrors OpenAI ChatCompletionChunk.

    `delta` is a partial message dict: {role?, content?, tool_calls?}.
    Clients accumulate across chunks by choice_index.
    """
    id: str
    model_id: str
    created: int
    choice_index: int
    delta: dict
    finish_reason: str | None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ChatModel(Protocol):
    """Backend protocol. Implementations satisfy this structurally;
    no base class needed."""

    model_id: str

    def chat(
        self,
        messages: list[dict],
        **kwargs: Any,
    ) -> ChatResult:
        ...

    def chat_stream(
        self,
        messages: list[dict],
        **kwargs: Any,
    ) -> Iterator[ChatChunk]:
        ...
