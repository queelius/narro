"""FastAPI TTS server for Narro.

Model-agnostic: any backend implementing :class:`~narro.protocol.TTSModel`
can be registered and served.  The ``model`` request field selects the
backend; omit it to use the default (first registered).

Endpoints:

  GET  /health              -- liveness probe
  GET  /v1/models           -- list available models
  POST /v1/audio/speech     -- synthesize speech

Usage (programmatic):
    from narro.server import configure_app
    from narro.models.soprano import SopranoModel
    configure_app(models=[SopranoModel(device='cuda')])

Usage (CLI):
    narro serve --device cuda
"""

import asyncio
import base64
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .models import registry
from .protocol import TTSModel

logger = logging.getLogger(__name__)

INT16_MAX = 32767

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

_MAX_INPUT_LENGTH = 50_000  # characters


class SpeechRequest(BaseModel):
    input: str
    model: str | None = None
    voice: str | None = None
    response_format: str = "wav"
    stream: bool = False
    align: bool = False


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a float32 numpy waveform to raw WAV bytes (in-memory)."""
    from scipy.io import wavfile

    pcm = (np.clip(audio, -1.0, 1.0) * INT16_MAX).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, pcm)
    return buf.getvalue()


def _wav_bytes_to_opus(wav_bytes: bytes) -> bytes:
    """Transcode WAV bytes to Opus/OGG via ffmpeg subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in.write(wav_bytes)
        in_path = tmp_in.name
    out_path = in_path.replace(".wav", ".opus")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-c:a", "libopus", out_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def configure_app(models: list[TTSModel] | None = None) -> FastAPI:
    """Register models with the module-level FastAPI app and return it.

    If *models* is ``None``, no models are registered -- call
    :func:`serve` or register them manually via ``registry.register()``.
    """
    if models:
        for m in models:
            registry.register(m)
    return app


# ---------------------------------------------------------------------------
# FastAPI app (module-level, importable for testing)
# ---------------------------------------------------------------------------

app = FastAPI(title="Narro TTS Server", version="2.0.0")
_inference_lock = asyncio.Lock()


@app.get("/health")
async def health():
    """Liveness probe -- returns model info if loaded."""
    models = registry.list_models()
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")
    default = registry.default_model_id
    return {
        "status": "ok",
        "default_model": default,
        "models": [m.id for m in models],
    }


@app.get("/v1/models")
async def list_models():
    """List available TTS models."""
    return {
        "object": "list",
        "data": [asdict(m) for m in registry.list_models()],
    }


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    """Synthesize speech from text.

    Supports:
    - Non-streaming WAV or Opus response
    - SSE streaming of raw PCM chunks (base64-encoded)
    - Model-specific features via the model's metadata (e.g. alignment)
    """
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' must be non-empty text.")

    if len(req.input) > _MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"'input' exceeds maximum length of {_MAX_INPUT_LENGTH} characters.",
        )

    if req.response_format not in ("wav", "opus"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported response_format '{req.response_format}'. Use 'wav' or 'opus'.",
        )

    if req.response_format == "opus" and shutil.which("ffmpeg") is None:
        raise HTTPException(
            status_code=422,
            detail="Opus output requires ffmpeg, which was not found on PATH.",
        )

    try:
        model = registry.get(req.model)
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if req.stream:
        if req.response_format != "wav":
            raise HTTPException(
                status_code=422,
                detail="Streaming only supports response_format='wav' (raw PCM).",
            )
        if req.align:
            raise HTTPException(
                status_code=422,
                detail="Alignment is not supported with streaming.",
            )
        return _streaming_response(model, req)

    return await _non_streaming_response(model, req)


async def _non_streaming_response(model: TTSModel, req: SpeechRequest) -> Response:
    """Synthesize the full input and return a WAV or Opus response."""
    loop = asyncio.get_event_loop()
    kwargs = {"align": req.align}
    if req.voice:
        kwargs["voice"] = req.voice
    async with _inference_lock:
        result = await loop.run_in_executor(
            None, lambda: model.synthesize(req.input, **kwargs)
        )

    wav_bytes = _audio_to_wav_bytes(result.audio, result.sample_rate)

    if req.response_format == "opus":
        content, media_type = _wav_bytes_to_opus(wav_bytes), "audio/ogg; codecs=opus"
    else:
        content, media_type = wav_bytes, "audio/wav"

    headers = {}
    alignment = result.metadata.get("alignment")
    if alignment is not None:
        headers["X-Alignment"] = json.dumps(alignment)

    return Response(content=content, media_type=media_type, headers=headers)


def _streaming_response(model: TTSModel, req: SpeechRequest) -> StreamingResponse:
    """Return a Server-Sent Events stream of base64-encoded PCM chunks."""

    async def event_generator():
        meta = json.dumps({
            "type": "speech.audio.start",
            "format": "pcm_s16le",
            "sample_rate": model.sample_rate,
        })
        yield f"data: {meta}\n\n"

        # Run the synchronous generator in a thread so we don't block
        # the event loop during inference.  We collect one chunk at a
        # time via asyncio.to_thread to keep memory bounded.
        async with _inference_lock:
            stream_kwargs = {}
            if req.voice:
                stream_kwargs["voice"] = req.voice
            stream_iter = iter(model.synthesize_stream(req.input, **stream_kwargs))
            while True:
                chunk = await asyncio.to_thread(next, stream_iter, None)
                if chunk is None:
                    break
                pcm = (np.clip(chunk.audio, -1.0, 1.0) * INT16_MAX).astype(np.int16)
                encoded = base64.b64encode(pcm.tobytes()).decode("ascii")
                payload = json.dumps({"type": "speech.audio.delta", "audio": encoded})
                yield f"data: {payload}\n\n"

        done = json.dumps({"type": "speech.audio.done"})
        yield f"data: {done}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    models: list[str] | None = None,
    device: str = "auto",
    compile: bool = True,
    quantize: bool = False,
    log_level: str = "info",
) -> None:
    """Start the Narro TTS server.

    Loads all pulled models by default, or a specific subset if
    *models* is provided.

    Args:
        host: Network interface to bind.
        port: TCP port to listen on.
        models: Model IDs to load (None = all pulled).
        device: Compute device.
        compile: Enable torch.compile.
        quantize: Enable INT8 quantization.
        log_level: Uvicorn log level string.
    """
    import uvicorn
    from .catalog import load_backend, pulled_models

    model_ids = models or list(pulled_models().keys())
    if not model_ids:
        raise RuntimeError(
            "No models pulled. Pull a model first:\n"
            "  narro models pull soprano-80m"
        )

    backends = []
    for mid in model_ids:
        logger.info("Loading model: %s", mid)
        backends.append(load_backend(mid, device=device, compile=compile, quantize=quantize))

    configure_app(models=backends)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
