"""FastAPI routes for /v1/audio/embeddings.

Wire shape mirrors /v1/audio/transcriptions on the request side
(multipart/form-data with `file` part(s)) and /v1/embeddings on the
response side (OpenAI shape with `data: [{embedding, index, object}],
model, usage`).

Each request carries one or more `file` parts whose bytes decode to
audio (wav/mp3/flac/ogg/...) via librosa (in the runtime). The route
layer:
  - Validates the `file` field (non-empty bytes per file)
  - Enforces a size cap via MUSE_AUDIO_EMBEDDINGS_MAX_BYTES
  - Resolves the registered backend
  - Hands the raw bytes list to backend.embed(...)
  - Encodes the response per `encoding_format` (float | base64)

Multipart batching: repeating the `file` field name produces a list
of UploadFile objects. Single-file uploads remain the common path.

Error envelopes follow muse's OpenAI-compat convention. 404 raises
ModelNotFoundError; 400/413/415 use error_response() so the bare
{"error": ...} envelope reaches the client without FastAPI's
{"detail": ...} wrap.
"""
from __future__ import annotations

import asyncio
import logging


from fastapi import APIRouter, File, Form, UploadFile

from muse.core import config
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.audio_embedding.codec import embedding_to_base64


logger = logging.getLogger(__name__)


MODALITY = "audio/embedding"

_MAX_FILES = 32
_MAX_TOTAL_UPLOAD_BYTES = 64 * 1024 * 1024



def _max_upload_bytes() -> int:
    """Per-file byte cap via muse.core.config (env:
    MUSE_AUDIO_EMBEDDINGS_MAX_BYTES). Default 50MB; tunable via env."""
    return config.get("limits.audio_embeddings_max_bytes")


_VALID_FORMATS = {"float", "base64"}


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/embeddings")
    async def audio_embeddings(
        file: list[UploadFile] = File(...),
        model: str | None = Form(None),
        encoding_format: str = Form("float"),
        user: str | None = Form(None),  # OpenAI compat; ignored
    ):
        if encoding_format not in _VALID_FORMATS:
            return error_response(
                400, "invalid_parameter",
                f"encoding_format must be one of {sorted(_VALID_FORMATS)}; "
                f"got {encoding_format!r}",
            )

        if not file:
            return error_response(
                400, "invalid_parameter",
                "at least one `file` part is required",
            )
        if len(file) > _MAX_FILES:
            return error_response(
                413, "payload_too_large",
                f"at most {_MAX_FILES} audio files are allowed per request",
            )

        max_bytes = _max_upload_bytes()
        audio_bytes_list: list[bytes] = []
        total_bytes = 0
        for idx, upload in enumerate(file):
            try:
                data = await upload.read(max_bytes + 1)
            finally:
                try:
                    await upload.close()
                except Exception:  # noqa: BLE001
                    logger.debug("failed to close audio upload", exc_info=True)
            if len(data) > max_bytes:
                return error_response(
                    413, "payload_too_large",
                    f"file[{idx}] exceeds MUSE_AUDIO_EMBEDDINGS_MAX_BYTES="
                    f"{max_bytes}",
                )
            if not data:
                return error_response(
                    400, "invalid_parameter",
                    f"file[{idx}] is empty",
                )
            total_bytes += len(data)
            if total_bytes > _MAX_TOTAL_UPLOAD_BYTES:
                return error_response(
                    413, "payload_too_large",
                    "aggregate audio upload exceeds "
                    f"{_MAX_TOTAL_UPLOAD_BYTES} bytes",
                )
            audio_bytes_list.append(data)

        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model or "<default>", modality=MODALITY,
            )

        # backend.embed is sync (transformers forward + librosa decode);
        # offload so a slow inference doesn't block sibling /health,
        # /v1/models, or other in-flight requests on the same worker.
        def _call():
            with backend._inference_lock:
                return backend.embed(audio_bytes_list)

        try:
            result = await asyncio.to_thread(_call)
        except Exception as e:  # noqa: BLE001
            # librosa decode failures land here. Use the same heuristic
            # gate as audio_transcription/routes: treat decoder-shaped
            # error messages as 415; re-raise everything else so a real
            # bug surfaces instead of being masked.
            msg = str(e).lower()
            if (
                "audioread" in msg
                or "decoder" in msg
                or "decode" in msg
                or "ffmpeg" in msg
                or "soundfile" in msg
                or "format not recognised" in msg
                or "format not recognized" in msg
                or "no backend available" in msg
            ):
                logger.warning(
                    "backend.embed raised a decoder-looking exception; "
                    "returning 415. Error: %s", e,
                )
                return error_response(
                    415, "unsupported_media_type",
                    f"audio decode failed: {e}",
                )
            # Deliberately re-raise everything else so a real bug
            # surfaces instead of being masked as a 415. This used to
            # escape as FastAPI's bare `{"detail": "Internal Server
            # Error"}`; create_app's global Exception handler (added
            # v0.58.1) now catches it and returns the OpenAI-shape 500
            # envelope, logging the real exception server-side.
            raise

        data = []
        for i, vec in enumerate(result.embeddings):
            embedding_field = (
                embedding_to_base64(vec)
                if encoding_format == "base64"
                else vec
            )
            data.append({
                "object": "embedding",
                "embedding": embedding_field,
                "index": i,
            })

        return {
            "object": "list",
            "data": data,
            "model": result.model_id,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

    return router
