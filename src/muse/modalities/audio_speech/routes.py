"""FastAPI router for /v1/audio/speech.

Ports narro/server.py's TTS handlers to muse's per-modality router
pattern. The router is built with a registry reference so handlers
look up backends by name.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading


import numpy as np
from fastapi import APIRouter, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from muse.modalities.audio_speech.codec import (
    AudioFormatError,
    audio_to_wav_bytes,
    float_to_pcm16,
    wav_bytes_to_opus,
)
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry

logger = logging.getLogger(__name__)

MODALITY = "audio/speech"
MAX_INPUT_LENGTH = 50_000
_STREAM_QUEUE_DEPTH = 8
_STREAM_POLL_SECONDS = 0.05
_STREAM_CLEANUP_GRACE_SECONDS = 0.5


def _stream_error_event() -> dict[str, str]:
    """Return a stable mid-stream failure without backend details."""
    payload = {
        "error": {
            "code": "streaming_failed",
            "message": "audio streaming backend failed; see server logs",
            "type": "server_error",
        },
    }
    return {"event": "error", "data": json.dumps(payload)}



class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)
    model: str | None = None
    voice: str | None = None
    response_format: str = Field(default="wav", pattern="^(wav|opus)$")
    stream: bool = False
    speed: float = 1.0
    align: bool = False


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/audio", tags=["audio/speech"])

    @router.get("/speech/voices")
    def list_voices(model: str | None = None):
        try:
            m = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(model_id=model or "<default>", modality=MODALITY)
        voices = getattr(m, "voices", [])
        return {"model": m.model_id, "voices": voices}

    @router.post("/speech")
    async def speech(req: SpeechRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(model_id=req.model or "<default>", modality=MODALITY)

        if req.stream:
            return await _stream(model, req)
        return await _non_stream(model, req)

    return router


async def _non_stream(model, req: SpeechRequest) -> Response:
    def _synth():
        with model._inference_lock:
            return model.synthesize(
                req.input,
                voice=req.voice,
                speed=req.speed,
                align=req.align,
            )

    result = await asyncio.to_thread(_synth)

    try:
        wav = audio_to_wav_bytes(result.audio, result.sample_rate)
    except AudioFormatError:
        # Log the real exception server-side but never leak it to the
        # client: str(e) can carry internal filesystem paths, CUDA
        # driver text, or other backend-implementation detail.
        logger.exception("wav encoding failed")
        return error_response(
            500, "encoding_failed",
            "audio encoding backend failed; see server logs",
        )

    if req.response_format == "opus":
        try:
            body = wav_bytes_to_opus(wav)
            media = "audio/ogg"
        except AudioFormatError:
            logger.warning("opus encoding unavailable; falling back to wav")
            body = wav
            media = "audio/wav"
    else:
        body = wav
        media = "audio/wav"

    headers: dict[str, str] = {}
    if req.align and result.metadata and "alignment" in result.metadata:
        headers["X-Alignment"] = json.dumps(result.metadata["alignment"])

    return Response(content=body, media_type=media, headers=headers)


async def _stream(model, req: SpeechRequest) -> EventSourceResponse:
    async def event_gen():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_DEPTH)
        queue_slots = threading.Semaphore(_STREAM_QUEUE_DEPTH)
        cancelled = threading.Event()
        producer_done = threading.Event()
        done_sentinel = object()
        error_sentinel = object()

        def _enqueue(item: object) -> bool:
            """Reserve bounded queue capacity from the producer thread."""
            while not cancelled.is_set():
                if not queue_slots.acquire(timeout=_STREAM_POLL_SECONDS):
                    continue
                if cancelled.is_set():
                    queue_slots.release()
                    return False
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except RuntimeError:
                    # The event loop can close during process shutdown.
                    queue_slots.release()
                    return False
                return True
            return False

        def _acquire_inference_lock() -> bool:
            """Wait for the model without stranding disconnected requests."""
            while not cancelled.is_set():
                if model._inference_lock.acquire(timeout=_STREAM_POLL_SECONDS):
                    return True
            return False

        def _produce():
            stream = None
            lock_acquired = False
            failure_reported = False
            try:
                lock_acquired = _acquire_inference_lock()
                if not lock_acquired or cancelled.is_set():
                    return

                stream = iter(model.synthesize_stream(
                    req.input, voice=req.voice, speed=req.speed,
                ))
                while not cancelled.is_set():
                    try:
                        chunk = next(stream)
                    except StopIteration:
                        break
                    if not _enqueue(chunk):
                        break
            except Exception:
                logger.exception("audio speech stream backend failed")
                failure_reported = _enqueue(error_sentinel)
            finally:
                if stream is not None:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            logger.exception("audio speech stream cleanup failed")
                            if not failure_reported:
                                failure_reported = _enqueue(error_sentinel)
                if lock_acquired:
                    model._inference_lock.release()
                if not cancelled.is_set():
                    _enqueue(done_sentinel)
                producer_done.set()

        # threading.Thread (not loop.run_in_executor) so the producer's
        # lifecycle is explicit and decoupled from the asyncio default
        # executor's bounded thread pool. Matches chat_completion. The
        # earlier run_in_executor pattern discarded the returned Future,
        # so executor saturation could silently hang the consumer on
        # an empty queue.
        producer = threading.Thread(
            target=_produce,
            name="muse-audio-stream",
            daemon=True,
        )
        try:
            producer.start()
        except Exception:
            logger.exception("audio speech stream producer failed to start")
            yield _stream_error_event()
            yield {"event": "done", "data": ""}
            return

        try:
            while True:
                item = await queue.get()
                queue_slots.release()
                if item is done_sentinel:
                    yield {"event": "done", "data": ""}
                    return
                if item is error_sentinel:
                    yield _stream_error_event()
                    yield {"event": "done", "data": ""}
                    return
                try:
                    pcm = float_to_pcm16(item.audio)
                except Exception:
                    logger.exception("audio speech stream encoding failed")
                    yield _stream_error_event()
                    yield {"event": "done", "data": ""}
                    return
                yield {"data": base64.b64encode(pcm.tobytes()).decode()}
        finally:
            # EventSourceResponse closes the generator when the client
            # disconnects. Wake a producer waiting for queue capacity or
            # model ownership, then briefly give it a chance to close its
            # backend iterator and release the inference lock. A backend
            # blocked inside next() cannot be force-killed safely; keep the
            # wait bounded and report that condition server-side.
            cancelled.set()
            deadline = loop.time() + _STREAM_CLEANUP_GRACE_SECONDS
            while not producer_done.is_set() and loop.time() < deadline:
                await asyncio.sleep(_STREAM_POLL_SECONDS)
            if not producer_done.is_set():
                logger.warning(
                    "audio speech stream producer did not stop within %.2fs",
                    _STREAM_CLEANUP_GRACE_SECONDS,
                )

    return EventSourceResponse(event_gen())
