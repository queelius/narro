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
import json
import logging
import threading
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._native_offload import run_native_offload
from muse.modalities.chat_completion.codec import (
    DONE_SENTINEL,
    chunk_to_sse_data,
    result_to_openai_dict,
)
from muse.modalities.image_generation.image_input import (
    close_decoded_images,
    decode_image_input,
    validate_total_image_pixels,
)


logger = logging.getLogger(__name__)

MODALITY = "chat/completion"

_MAX_MESSAGES = 256
_MAX_TOOLS = 128
_MAX_IMAGES = 16
_MAX_OUTPUT_TOKENS = 32_768
_STREAM_QUEUE_DEPTH = 64
_STREAM_POLL_SECONDS = 0.1
_STANDARD_BACKEND_KEYS = frozenset({
    "temperature", "top_p", "max_tokens", "stop", "seed", "tools",
    "tool_choice", "response_format", "logprobs", "top_logprobs",
})


class ChatCompletionRequest(BaseModel):
    """OpenAI-shape request. Most fields are passthrough to the backend."""
    model: str | None = None
    messages: list[dict] = Field(..., min_length=1, max_length=_MAX_MESSAGES)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=_MAX_OUTPUT_TOKENS)
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[dict] | None = Field(default=None, max_length=_MAX_TOOLS)
    tool_choice: str | dict | None = None
    response_format: dict | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    extra_body: dict | None = None

    @field_validator("messages")
    @classmethod
    def _non_empty_messages(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError("messages must be non-empty")
        return v

    @field_validator("extra_body")
    @classmethod
    def _no_standard_key_override(cls, value: dict | None) -> dict | None:
        if not value:
            return value
        overlap = sorted(_STANDARD_BACKEND_KEYS.intersection(value))
        if overlap:
            raise ValueError(
                "extra_body cannot override validated top-level fields: "
                + ", ".join(overlap)
            )
        # Common aliases can otherwise bypass max_tokens and hand a backend
        # an effectively unlimited generation request.
        token_aliases = sorted({"max_new_tokens", "max_length"}.intersection(value))
        if token_aliases:
            raise ValueError(
                "use the bounded top-level max_tokens field instead of: "
                + ", ".join(token_aliases)
            )
        return value

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


class _ImageDecodeError(Exception):
    """Raised by _decode_image_parts with a structured error code + message.

    The route handler catches this and translates to a 400 with the
    OpenAI-shape error envelope.
    """
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


async def _decode_image_parts(
    messages: list[dict],
    supports_vision: bool,
    supports_multi_image: bool,
    model_id: str,
) -> list[dict]:
    """Walk messages; validate + decode any image_url parts; return rewritten list.

    Pre-dispatch step that runs before ChatModel.chat()/chat_stream() to:
      - Detect image_url parts in any message.content list
      - Reject capability mismatches (vision_not_supported, too_many_images)
      - Validate part shape (invalid_content_part, unsupported_content_type)
      - Decode each url via decode_image_input (data: or http(s)://)
      - Rewrite the part as {type: image, image: <PIL.Image>} so the
        backend consumes a uniform muse-internal shape

    Capability flags come from the registry manifest (not instance attrs)
    so resolver-pulled VLMs whose capabilities live in the synthesized
    manifest are correctly gated.

    Check order per spec:
      1. unsupported_content_type  (early reject, no further work)
      2. supports_vision            (early reject before any fetch)
      3. invalid_content_part       (url field present)
      4. decode                     (only after all gates pass)

    The too_many_images count check runs in a first pass over the whole
    conversation (total image_url parts across all messages, not per
    message) before the decode loop, so a single-image model is gated on
    the conversation total and we don't waste fetches on a request that is
    about to 400.

    Raises _ImageDecodeError with a structured error code + message.

    Returns the original `messages` list unchanged when no image_url parts
    are found, preserving byte-identical behaviour for text-only requests.
    """
    # First pass over the WHOLE conversation: gate too_many_images on the
    # total image_url count, not per-message. A model with
    # supports_multi_image=False handles exactly one image; N messages each
    # carrying a single image still overflows it, yet per-message counting
    # (image_count <= 1 in every message) would wrongly admit them. Counting
    # here, before the decode loop, also avoids wasted HTTP fetches.
    decoded_images: list[Any] = []

    def _abort(code: str, message: str) -> None:
        close_decoded_images(decoded_images)
        raise _ImageDecodeError(code, message)

    total_image_count = sum(
        1
        for msg in messages
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    )
    if total_image_count > 1 and not supports_multi_image:
        _abort(
            "too_many_images",
            f"model {model_id!r} accepts only 1 image per conversation; "
            f"got {total_image_count}",
        )
    if total_image_count > _MAX_IMAGES:
        _abort(
            "too_many_images",
            f"a conversation may contain at most {_MAX_IMAGES} images; "
            f"got {total_image_count}",
        )

    has_any_image = False
    new_messages: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue

        new_content: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                new_content.append(part)
                continue
            ptype = part.get("type")
            if ptype == "text":
                new_content.append(part)
            elif ptype == "image_url":
                has_any_image = True
                # 1. Unsupported content-type check is for non-image_url
                #    types (handled in the else branch below).
                # 2. supports_vision check before url validation or decode.
                if not supports_vision:
                    _abort(
                        "vision_not_supported",
                        f"model {model_id!r} does not support vision input; "
                        f"pick a model with supports_vision=true",
                    )
                # 3. url field present.
                image_url = part.get("image_url")
                if not isinstance(image_url, dict):
                    _abort(
                        "invalid_content_part",
                        "image_url part must contain an object with a url field",
                    )
                url = image_url.get("url")
                if not isinstance(url, str) or not url:
                    _abort(
                        "invalid_content_part",
                        "image_url part missing required string url field",
                    )
                # 4. decode.
                try:
                    img = await decode_image_input(url)
                except ValueError as ve:
                    _abort(
                        "invalid_image",
                        f"could not decode image: {ve}",
                    )
                except BaseException:
                    close_decoded_images(decoded_images)
                    raise
                decoded_images.append(img)
                try:
                    validate_total_image_pixels(decoded_images)
                except ValueError as ve:
                    _abort("image_budget_exceeded", str(ve))
                new_content.append({"type": "image", "image": img})
            elif ptype is None:
                new_content.append(part)
            else:
                # 1. unsupported content type: always reject first.
                _abort(
                    "unsupported_content_type",
                    f"content type {ptype!r} not supported; "
                    f"allowed: text, image_url",
                )
        new_messages.append({**msg, "content": new_content})
    return new_messages if has_any_image else messages


def _decoded_message_images(messages: list[dict]) -> list[Any]:
    """Collect request-owned images produced by `_decode_image_parts`."""
    return [
        part["image"]
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict)
        and part.get("type") == "image"
        and "image" in part
    ]


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["chat/completion"])

    def _get_model(model_id: str | None):
        try:
            return registry.get(MODALITY, model_id)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model_id or "<default>",
                modality=MODALITY,
            )

    @router.post("/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        model = _get_model(req.model)

        # Derive effective_id the same way image_generation does: prefer
        # the model's own model_id attr; fall back to the request field or
        # the literal default marker.
        effective_id = getattr(model, "model_id", None) or (req.model or "<default>")

        # Read capability flags from the registry manifest (NOT instance
        # attrs) so resolver-pulled VLMs whose capabilities live in the
        # synthesized manifest are correctly gated.
        manifest = registry.manifest(MODALITY, effective_id) or {}
        caps = manifest.get("capabilities", {})
        supports_vision = caps.get("supports_vision", False)
        supports_multi_image = caps.get("supports_multi_image", False)

        # Pre-dispatch: decode image_url content parts and validate
        # capability flags BEFORE branching on stream/non-stream so that
        # errors always surface as 400 (never as mid-stream SSE errors).
        try:
            messages = await _decode_image_parts(
                req.messages,
                supports_vision=supports_vision,
                supports_multi_image=supports_multi_image,
                model_id=effective_id,
            )
        except _ImageDecodeError as e:
            return error_response(400, e.code, e.message)

        decoded_images = _decoded_message_images(messages)

        kwargs = req.backend_kwargs()

        # If the request asks for tool calling, warn when the loaded
        # model isn't known to support it. Tool-call quality is a
        # property of the model + chat_format combination; muse doesn't
        # block the request, but a warning lets the user know structured
        # tool_calls may not appear and the model may emit raw text in
        # `content` instead.
        if req.tools is not None:
            # Read supports_tools from the manifest capabilities first (same
            # source as the vision flags above) so resolver-pulled GGUFs,
            # whose tool support lives in the synthesized manifest rather
            # than on the instance, are gated correctly; fall back to the
            # instance attribute for bundled scripts that set it directly.
            supports = caps.get("supports_tools", getattr(model, "supports_tools", None))
            if supports is False:
                logger.warning(
                    "model %s is not known to support tool calling; "
                    "structured tool_calls may not appear in the response",
                    getattr(model, "model_id", "?"),
                )
            elif supports is None:
                logger.warning(
                    "tool support for model %s is unknown; "
                    "structured tool_calls may not appear in the response. "
                    "If you know this model works, set "
                    "capabilities.chat_format in its manifest "
                    "(see docs/CHAT_COMPLETION.md)",
                    getattr(model, "model_id", "?"),
                )

        if not req.stream:
            def _call_chat():
                with model._inference_lock:
                    return model.chat(messages, **kwargs)
            abandoned = False
            try:
                result = await run_native_offload(
                    _call_chat,
                    cleanup_abandoned=(
                        lambda _result: close_decoded_images(decoded_images)
                    ),
                )
            except asyncio.CancelledError:
                abandoned = True
                raise
            finally:
                if not abandoned:
                    close_decoded_images(decoded_images)
            return result_to_openai_dict(result)

        queue: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_DEPTH)
        loop = asyncio.get_running_loop()
        queue_slots = threading.Semaphore(_STREAM_QUEUE_DEPTH)
        cancelled = threading.Event()

        def _enqueue(item: object) -> bool:
            # The producer runs in a native thread and cannot await an
            # asyncio.Queue. A matching semaphore bounds scheduled+queued
            # items; the async consumer releases one slot per get().
            while not cancelled.is_set():
                if not queue_slots.acquire(timeout=0.1):
                    continue
                if cancelled.is_set():
                    queue_slots.release()
                    return False
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except RuntimeError:
                    queue_slots.release()
                    return False
                return True
            return False

        def _producer():
            # Acquire the per-model inference lock for the ENTIRE stream.
            # A single GPU model cannot run two generations concurrently;
            # holding the lock across the full stream serializes requests
            # to one model without blocking siblings on the same worker.
            # Lock is released in the finally block, which fires on normal
            # completion, client disconnect, or exception (H2 fix).
            lock_acquired = False
            try:
                while not cancelled.is_set():
                    if model._inference_lock.acquire(
                        timeout=_STREAM_POLL_SECONDS,
                    ):
                        lock_acquired = True
                        break
                if not lock_acquired or cancelled.is_set():
                    return
                for chunk in model.chat_stream(messages, **kwargs):
                    if not _enqueue(chunk):
                        break
            except Exception as e:
                logger.exception("chat_stream backend failed")
                _enqueue(e)
            finally:
                if lock_acquired:
                    model._inference_lock.release()
                close_decoded_images(decoded_images)
                _enqueue(None)

        producer = threading.Thread(target=_producer, daemon=True)
        try:
            producer.start()
        except BaseException:
            close_decoded_images(decoded_images)
            raise

        async def _events():
            try:
                while True:
                    item = await queue.get()
                    queue_slots.release()
                    if item is None:
                        yield {"data": DONE_SENTINEL}
                        return
                    if isinstance(item, Exception):
                        # Preserve a structured mid-stream failure without
                        # exposing backend paths, driver text, or secrets.
                        err_payload = {"error": {
                            "code": "internal_error",
                            "message": "chat streaming backend failed; see server logs",
                            "type": "server_error",
                        }}
                        yield {"event": "error", "data": json.dumps(err_payload)}
                        yield {"data": DONE_SENTINEL}
                        return
                    yield {"data": chunk_to_sse_data(item)}
            finally:
                # EventSourceResponse closes this generator on disconnect.
                # Wake a producer blocked on the bounded queue; the backend
                # generator is allowed to observe cancellation at its next
                # yielded chunk and release the inference lock in `finally`.
                cancelled.set()

        return EventSourceResponse(_events())

    return router
