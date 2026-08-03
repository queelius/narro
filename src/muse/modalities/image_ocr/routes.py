"""FastAPI routes for /v1/images/ocr.

Multipart in (image file + optional model/prompt/max_new_tokens),
JSON out (id, model, text, usage). Mirrors /v1/audio/transcriptions.

Reuses decode_image_file from image_generation/image_input.py for
the multipart upload; MUSE_IMAGE_INPUT_MAX_BYTES env cap (v0.34.0)
applies. Per-backend _inference_lock from v0.34.0 serializes calls
into one model.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._native_offload import run_native_offload
from muse.modalities.image_generation.image_input import (
    close_decoded_images,
    decode_image_file,
)
from muse.modalities.image_ocr.codec import encode_ocr


MODALITY = "image/ocr"


logger = logging.getLogger(__name__)


# Hard ceiling on per-request decoder budget. Far above any sensible
# OCR (Nougat full-page is ~2-3K tokens). Beyond this is either a
# misuse or a budget-burner; reject with 400.
_HARD_MAX_NEW_TOKENS = 4096


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/images/ocr")
    async def images_ocr(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        prompt: str | None = Form(None),
        max_new_tokens: int | None = Form(None),
    ):
        # Cap validation: positive integer, bounded.
        if max_new_tokens is not None:
            if max_new_tokens < 1 or max_new_tokens > _HARD_MAX_NEW_TOKENS:
                return error_response(
                    400, "invalid_parameter",
                    f"max_new_tokens must be in [1, {_HARD_MAX_NEW_TOKENS}]; "
                    f"got {max_new_tokens}",
                )

        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model or "<default>", modality=MODALITY,
            )

        effective_id = backend.model_id

        # Decode the multipart upload (size-capped via decode_image_file;
        # MUSE_IMAGE_INPUT_MAX_BYTES env cap applies).
        try:
            pil_image = await decode_image_file(image)
        except ValueError as e:
            return error_response(400, "invalid_parameter", str(e))

        def _call():
            with backend._inference_lock:
                kwargs = {}
                if prompt is not None:
                    kwargs["prompt"] = prompt
                if max_new_tokens is not None:
                    kwargs["max_new_tokens"] = max_new_tokens
                return backend.ocr(pil_image, **kwargs)

        abandoned = False
        try:
            result = await run_native_offload(
                _call,
                cleanup_abandoned=(
                    lambda _result: close_decoded_images([pil_image])
                ),
            )
        except asyncio.CancelledError:
            abandoned = True
            raise
        except Exception:  # noqa: BLE001
            # Log the real exception server-side but never leak it to the
            # client: str(e) can carry internal filesystem paths, CUDA
            # driver text, or other backend-implementation detail.
            logger.exception("ocr call failed")
            return error_response(
                500, "internal_error",
                "ocr backend failed; see server logs",
            )
        finally:
            if not abandoned:
                close_decoded_images([pil_image])

        body = encode_ocr(result)
        # Override result.model_id with the effective registry id in
        # case the runtime's model_id drifted (paranoia guard, mirrors
        # the moderation route's pattern).
        body["model"] = effective_id
        return JSONResponse(content=body)

    return router
