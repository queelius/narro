"""FastAPI TTS server for Narro.

Exposes two endpoints:

  GET  /health              — liveness probe
  POST /v1/audio/speech     — synthesize speech (non-streaming or SSE streaming)

Usage (programmatic):
    from narro.server import create_app
    app = create_app(device='cpu')

Usage (CLI via uvicorn):
    from narro.server import serve
    serve(host='0.0.0.0', port=8000)
"""

import asyncio
import base64
import io
import json
import logging
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy.io import wavfile

from .tts import INT16_MAX, SAMPLE_RATE, Narro

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global TTS instance — lazy, created once at first request (or by create_app)
# ---------------------------------------------------------------------------

_tts: Optional[Narro] = None
_tts_lock = asyncio.Lock()


def _get_tts() -> Narro:
    """Return the global Narro TTS instance (must be initialised first)."""
    if _tts is None:
        raise RuntimeError(
            "TTS model not initialised. Call create_app() with model parameters "
            "or use the serve() convenience function."
        )
    return _tts


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    input: str
    response_format: str = "wav"
    stream: bool = False
    align: bool = False


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _tensor_to_wav_bytes(audio: torch.Tensor) -> bytes:
    """Convert a float32 audio tensor to raw WAV bytes (in-memory)."""
    pcm = (np.clip(audio.numpy(), -1.0, 1.0) * INT16_MAX).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, pcm)
    return buf.getvalue()


def _wav_bytes_to_opus(wav_bytes: bytes) -> bytes:
    """Transcode WAV bytes to Opus/OGG via ffmpeg subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as tmp_out:
        in_path = tmp_in.name
        out_path = tmp_out.name

    try:
        with open(in_path, "wb") as f:
            f.write(wav_bytes)

        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-c:a", "libopus", out_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        import os
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    device: str = "auto",
    model_path: Optional[str] = None,
    compile: bool = True,
    quantize: bool = False,
) -> FastAPI:
    """Create and return a FastAPI application with a loaded Narro model.

    Args:
        device: Compute device ('auto', 'cpu', 'cuda', 'mps').
        model_path: Path to local model directory (None = HuggingFace).
        compile: Enable torch.compile for faster inference.
        quantize: Enable INT8 quantization.

    Returns:
        Configured FastAPI application instance.
    """
    global _tts
    _tts = Narro(model_path=model_path, compile=compile, quantize=quantize, device=device)
    logger.info("Narro model loaded on device=%s", _tts.device)
    return app


# ---------------------------------------------------------------------------
# Request handler for asyncio.Lock concurrency control
# ---------------------------------------------------------------------------

_inference_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# FastAPI app (module-level, importable for testing)
# ---------------------------------------------------------------------------

app = FastAPI(title="Narro TTS Server", version="1.0.0")


@app.get("/health")
async def health():
    """Liveness probe — returns model info if loaded."""
    tts = _get_tts()
    return {
        "status": "ok",
        "device": str(tts.device),
        "model": tts.model_id,
    }


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    """Synthesize speech from text.

    Supports:
    - Non-streaming WAV or Opus response
    - SSE streaming of raw PCM chunks (base64-encoded)
    - Optional paragraph-level alignment in X-Alignment response header

    Returns HTTP 400 for empty input, HTTP 422 if Opus requested without ffmpeg.
    """
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' must be non-empty text.")

    if req.response_format == "opus" and shutil.which("ffmpeg") is None:
        raise HTTPException(
            status_code=422,
            detail="Opus output requires ffmpeg, which was not found on PATH.",
        )

    tts = _get_tts()

    if req.stream:
        return _streaming_response(tts, req)

    return await _non_streaming_response(tts, req)


async def _non_streaming_response(tts: Narro, req: SpeechRequest) -> Response:
    """Encode + decode the full input and return a WAV or Opus response."""
    async with _inference_lock:
        if req.align:
            from .alignment import extract_paragraph_alignment

            paragraphs = [p.strip() for p in req.input.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [req.input]

            encoded = tts.encode_batch(paragraphs)
            audio_list = tts.decode(encoded)
            audio = torch.cat(audio_list) if audio_list else torch.zeros(0)
            alignment = extract_paragraph_alignment(encoded)
        else:
            audio = tts.infer(req.input)
            alignment = None

    wav_bytes = _tensor_to_wav_bytes(audio)

    if req.response_format == "opus":
        content = _wav_bytes_to_opus(wav_bytes)
        media_type = "audio/ogg; codecs=opus"
    else:
        content = wav_bytes
        media_type = "audio/wav"

    headers = {}
    if alignment is not None:
        headers["X-Alignment"] = json.dumps(alignment)

    return Response(content=content, media_type=media_type, headers=headers)


def _streaming_response(tts: Narro, req: SpeechRequest) -> StreamingResponse:
    """Return a Server-Sent Events stream of base64-encoded PCM chunks."""

    async def event_generator():
        # Send format metadata as first event
        meta = json.dumps({
            "type": "speech.audio.start",
            "format": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
        })
        yield f"data: {meta}\n\n"

        # Stream inference chunks — held under lock to protect Torch GIL
        async with _inference_lock:
            for chunk in tts.infer_stream(req.input):
                pcm = (np.clip(chunk.numpy(), -1.0, 1.0) * INT16_MAX).astype(np.int16)
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
    device: str = "auto",
    model_path: Optional[str] = None,
    compile: bool = True,
    quantize: bool = False,
    log_level: str = "info",
) -> None:
    """Start the Narro TTS server.

    Args:
        host: Network interface to bind (default '0.0.0.0').
        port: TCP port to listen on (default 8000).
        device: Compute device ('auto', 'cpu', 'cuda', 'mps').
        model_path: Path to local model directory (None = HuggingFace).
        compile: Enable torch.compile.
        quantize: Enable INT8 quantization.
        log_level: Uvicorn log level string.
    """
    import uvicorn

    create_app(device=device, model_path=model_path, compile=compile, quantize=quantize)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
