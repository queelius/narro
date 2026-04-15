"""/v1/chat/completions router.

Two call shapes:
  - stream=False (default): non-streaming. Calls ChatModel.chat() once,
    returns OpenAI ChatCompletion JSON.
  - stream=True: SSE. Producer thread calls ChatModel.chat_stream() and
    pushes ChatChunk items into an asyncio.Queue; the response iterator
    reads from the queue and serializes to SSE `data:` lines plus a
    final `data: [DONE]` sentinel.

Thread + queue pattern matches the audio.speech streaming code so we
do not buffer tokens on the server. Every token dispatches as produced.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from muse.core.errors import ModelNotFoundError
from muse.core.registry import ModalityRegistry
from muse.modalities.chat_completion.codec import (
    DONE_SENTINEL,
    chunk_to_sse_data,
    result_to_openai_dict,
)


logger = logging.getLogger(__name__)

MODALITY = "chat/completion"


class ChatCompletionRequest(BaseModel):
    """OpenAI-shape request. Most fields are passthrough to the backend."""
    model: str | None = None
    messages: list[dict] = Field(..., min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    response_format: dict | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    extra_body: dict | None = None

    @field_validator("messages")
    @classmethod
    def _non_empty_messages(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError("messages must be non-empty")
        return v

    def backend_kwargs(self) -> dict:
        """Dict of kwargs to forward to ChatModel.chat()/chat_stream().

        Omits `model` (routing metadata) and `stream` (handled by the
        route, not the backend). extra_body spreads in raw.
        """
        out: dict[str, Any] = {}
        for key in (
            "temperature", "top_p", "max_tokens", "stop", "seed",
            "tools", "tool_choice", "response_format", "logprobs",
            "top_logprobs",
        ):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        if self.extra_body:
            out.update(self.extra_body)
        return out


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["chat/completion"])

    def _get_model(model_id: str | None):
        try:
            return registry.get(MODALITY, model_id)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model_id or "(default)",
                modality=MODALITY,
            )

    @router.post("/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        model = _get_model(req.model)
        kwargs = req.backend_kwargs()

        if not req.stream:
            result = await asyncio.to_thread(model.chat, req.messages, **kwargs)
            return result_to_openai_dict(result)

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer():
            try:
                for chunk in model.chat_stream(req.messages, **kwargs):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(e), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        threading.Thread(target=_producer, daemon=True).start()

        async def _events():
            while True:
                item = await queue.get()
                if item is None:
                    yield {"data": DONE_SENTINEL}
                    return
                if isinstance(item, Exception):
                    logger.error("chat_stream backend error: %s", item)
                    yield {"data": DONE_SENTINEL}
                    return
                yield {"data": chunk_to_sse_data(item)}

        return EventSourceResponse(_events())

    return router
