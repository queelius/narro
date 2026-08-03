"""FastAPI route for raster-to-SVG vectorization."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from muse.core import config
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._native_offload import run_native_offload
from muse.modalities.image_generation.image_input import (
    close_decoded_images,
    decode_image_file,
)
from muse.modalities.image_vectorization.codec import encode_vectorization
from muse.modalities.image_vectorization.protocol import (
    VectorizationOutputError,
)
from muse.modalities.image_vectorization.svg import validate_static_svg


logger = logging.getLogger(__name__)
MODALITY = "image/vectorization"
_MAX_NEW_TOKENS = 7680
_MAX_SEED = 2**63 - 1


def _max_input_side() -> int:
    value = config.get("limits.vectorization_max_input_side")
    if value is None or value <= 0:
        return config.SETTINGS_BY_KEY[
            "limits.vectorization_max_input_side"
        ].default
    return value


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=[MODALITY])

    @router.post("/vectorize")
    async def vectorize(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        max_new_tokens: int = Form(4096),
        temperature: float = Form(1.0),
        top_p: float = Form(0.9),
        num_beams: int = Form(2),
        seed: int | None = Form(None),
        response_format: str = Form("json"),
    ):
        if not 1 <= max_new_tokens <= _MAX_NEW_TOKENS:
            return error_response(
                400, "invalid_parameter",
                f"max_new_tokens must be in [1, {_MAX_NEW_TOKENS}]",
            )
        if not 0.0 <= temperature <= 2.0:
            return error_response(
                400, "invalid_parameter",
                "temperature must be in [0.0, 2.0]",
            )
        if not 0.0 < top_p <= 1.0:
            return error_response(
                400, "invalid_parameter", "top_p must be in (0.0, 1.0]",
            )
        if not 1 <= num_beams <= 8:
            return error_response(
                400, "invalid_parameter", "num_beams must be in [1, 8]",
            )
        if seed is not None and not 0 <= seed <= _MAX_SEED:
            return error_response(
                400, "invalid_parameter",
                f"seed must be in [0, {_MAX_SEED}]",
            )
        if response_format not in ("json", "svg"):
            return error_response(
                400, "invalid_parameter",
                "response_format must be 'json' or 'svg'",
            )

        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model or "<default>", modality=MODALITY,
            )

        max_side = _max_input_side()
        try:
            source = await decode_image_file(image, max_side=max_side)
        except ValueError as exc:
            message = str(exc)
            if "exceeds max input side" in message:
                message += (
                    " (set MUSE_VECTORIZATION_MAX_INPUT_SIDE to raise)"
                )
            return error_response(400, "invalid_parameter", message)

        width, height = source.size

        def _call():
            with backend._inference_lock:
                result = backend.vectorize(
                    source,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    seed=seed,
                )
            # The HTTP boundary is authoritative even for third-party
            # backends. Never rely on a runtime to sanitize this active
            # document format before returning it to a browser.
            info = validate_static_svg(result.svg)
            result.svg = info.svg
            result.width = info.width
            result.height = info.height
            result.view_box = info.view_box
            result.source_width = width
            result.source_height = height
            return result

        abandoned = False
        try:
            result = await run_native_offload(
                _call,
                cleanup_abandoned=(
                    lambda _result: close_decoded_images([source])
                ),
            )
        except asyncio.CancelledError:
            abandoned = True
            raise
        except VectorizationOutputError as exc:
            logger.warning("vectorization produced invalid SVG: %s", exc)
            return error_response(
                502, "invalid_model_output",
                "vectorization backend produced invalid or unsafe SVG",
            )
        except Exception:  # noqa: BLE001
            logger.exception("vectorization backend failed")
            return error_response(
                500, "internal_error",
                "vectorization backend failed; see server logs",
            )
        finally:
            if not abandoned:
                close_decoded_images([source])

        # Registry identity is authoritative if a backend accidentally
        # reports an upstream repo id instead of Muse's catalog id.
        result.model_id = backend.model_id
        if response_format == "svg":
            return Response(
                content=result.svg,
                media_type="image/svg+xml",
                headers={
                    "X-Muse-Model": backend.model_id,
                    "X-Muse-Seed": str(result.seed),
                    "X-Content-Type-Options": "nosniff",
                    "Content-Security-Policy": (
                        "default-src 'none'; style-src 'unsafe-inline'"
                    ),
                },
            )
        return JSONResponse(content=encode_vectorization(result))

    return router
