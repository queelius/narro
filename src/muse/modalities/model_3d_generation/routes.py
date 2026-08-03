"""FastAPI routes for /v1/3d/generations and /v1/3d/from-image.

Two routes share this MIME tag (`3d/generation`):

  POST /v1/3d/generations
    Text-to-3d. JSON body, validated via Pydantic. Capability flag
    `supports_text_to_3d` on the manifest gates the route.

  POST /v1/3d/from-image
    Image-to-3d. multipart/form-data. Capability flag
    `supports_image_to_3d` on the manifest gates the route. Per-request
    size cap via `MUSE_3D_INPUT_MAX_BYTES` (default 20 MiB; matches the
    pattern of `MUSE_IMAGE_INPUT_MAX_BYTES` and `MUSE_AUDIO_CLS_MAX_BYTES`).

Both routes use the per-backend `_inference_lock` (auto-attached at
registration time since v0.34.0) to serialize calls into one model
without blocking siblings on the same worker. Backend exceptions
surface as 500 `internal_error` with the original exception text;
tracebacks are logged via `logger.exception` and never returned to
the client.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core import config
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._native_offload import run_native_offload
from muse.modalities.image_generation.image_input import (
    close_decoded_images,
    decode_image_file,
)
from muse.modalities.model_3d_generation.codec import encode_3d_response


try:
    from PIL import Image as _PIL_Image
except Exception:  # noqa: BLE001
    _PIL_Image = None  # type: ignore[assignment]


MODALITY = "3d/generation"


logger = logging.getLogger(__name__)


def _max_bytes() -> int:
    """Read the image-to-3d upload byte cap via muse.core.config (env:
    MUSE_3D_INPUT_MAX_BYTES) per call, so operators can change the cap
    without a server restart. Garbled values fall back to the registry
    default (20MB) with a logged warning. Matches the v0.34.0
    MUSE_IMAGE_INPUT_MAX_BYTES + MUSE_AUDIO_CLS_MAX_BYTES pattern. A
    resolved value that is None (empty env) or non-positive (parseable
    but nonsensical as a byte cap) also falls back to the registry
    default, mirroring image_generation/image_input.py's
    _default_max_bytes guard."""
    n = config.get("limits.model_3d_input_max_bytes")
    if n is None or n <= 0:
        return config.SETTINGS_BY_KEY["limits.model_3d_input_max_bytes"].default
    return n


class _Generation3DRequest(BaseModel):
    """JSON body for POST /v1/3d/generations.

    n is capped at 2 because each 3D asset is heavy (multi-GB VRAM,
    seconds-to-minutes per call). response_format is validated by the
    handler (codec also re-validates) so the 400 envelope reaches the
    client before we touch the backend.
    """
    prompt: str = Field(..., min_length=1)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=2)
    seed: int | None = None
    response_format: str = "b64_json"


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    # ---------------- /v1/3d/generations (text-to-3d, JSON) ----------------

    @router.post("/v1/3d/generations")
    async def text_to_3d_route(req: _Generation3DRequest):
        # response_format gate: 400 before invoking the backend.
        if req.response_format not in ("b64_json", "url"):
            return error_response(
                400, "invalid_parameter",
                f"response_format must be 'b64_json' or 'url'; "
                f"got {req.response_format!r}",
            )

        # Resolve backend or 404.
        try:
            backend = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        # Resolve effective_id from the backend so error messages name
        # the real model (the registry default when req.model is None).
        effective_id = (
            getattr(backend, "model_id", None)
            or req.model
            or "<default>"
        )

        # Capability gate: supports_text_to_3d=True required. Mismatch
        # returns 400 BEFORE invoking the backend, naming the alternate
        # route in the error message.
        manifest = registry.manifest(MODALITY, effective_id) or {}
        capabilities = manifest.get("capabilities") or {}
        if not capabilities.get("supports_text_to_3d"):
            return error_response(
                400, "unsupported_route",
                f"model {effective_id!r} does not support text-to-3d "
                f"generation; use POST /v1/3d/from-image with an "
                f"input image instead",
            )

        # Forward kwargs. Always include n; seed only when explicitly
        # provided (so backends that branch on `seed is None` for
        # randomness keep that branch).
        kwargs: dict = {"n": req.n}
        if req.seed is not None:
            kwargs["seed"] = req.seed

        def _call():
            with backend._inference_lock:
                return backend.text_to_3d(req.prompt, **kwargs)

        try:
            results = await run_native_offload(_call)
        except Exception:  # noqa: BLE001
            # Log the real exception server-side but never leak it to the
            # client: str(e) can carry internal filesystem paths, CUDA
            # driver text, or other backend-implementation detail.
            logger.exception("text_to_3d failed")
            return error_response(
                500, "internal_error",
                "text-to-3d backend failed; see server logs",
            )

        body = encode_3d_response(
            results,
            model_id=effective_id,
            response_format=req.response_format,
        )
        return JSONResponse(content=body)

    # ---------------- /v1/3d/from-image (image-to-3d, multipart) -----------

    @router.post("/v1/3d/from-image")
    async def image_to_3d_route(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        n: int = Form(1),
        seed: int | None = Form(None),
        response_format: str = Form("b64_json"),
    ):
        cap = _max_bytes()

        # n + response_format gates: 400 before invoking the backend.
        if not (1 <= n <= 2):
            return error_response(
                400, "invalid_parameter", "n must be in [1, 2]",
            )
        if response_format not in ("b64_json", "url"):
            return error_response(
                400, "invalid_parameter",
                f"response_format must be 'b64_json' or 'url'; "
                f"got {response_format!r}",
            )

        # Resolve backend or 404.
        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model or "<default>", modality=MODALITY,
            )

        effective_id = (
            getattr(backend, "model_id", None)
            or model
            or "<default>"
        )

        # Capability gate: supports_image_to_3d=True required.
        manifest = registry.manifest(MODALITY, effective_id) or {}
        capabilities = manifest.get("capabilities") or {}
        if not capabilities.get("supports_image_to_3d"):
            return error_response(
                400, "unsupported_route",
                f"model {effective_id!r} does not support image-to-3d "
                f"generation; use POST /v1/3d/generations with a text "
                f"prompt instead",
            )

        # Decode the bounded upload to a PIL Image once, in the async layer,
        # before entering the worker thread. All runtimes (TripoSR,
        # TRELLIS, Hunyuan3D) accept PIL.Image.Image; the unified contract
        # avoids runtime-specific path vs. PIL dispatching.
        if _PIL_Image is None:
            return error_response(
                500, "internal_error",
                "Pillow is not installed in this worker; cannot decode "
                "the uploaded image. Run `muse models refresh <id>`.",
            )
        try:
            pil_image = await decode_image_file(image, max_bytes=cap)
            if pil_image.mode != "RGB":
                original = pil_image
                try:
                    pil_image = original.convert("RGB")
                except BaseException:
                    close_decoded_images([original])
                    raise
                else:
                    close_decoded_images([original])
        except ValueError as e:
            message = str(e)
            if "image bytes exceeds max" in message:
                message = (
                    "uploaded image exceeds "
                    f"MUSE_3D_INPUT_MAX_BYTES={cap} bytes"
                )
            return error_response(
                400, "invalid_image",
                f"uploaded file is not a valid image: {message}",
            )

        kwargs: dict = {"n": n}
        if seed is not None:
            kwargs["seed"] = seed

        def _call():
            with backend._inference_lock:
                return backend.image_to_3d(pil_image, **kwargs)

        abandoned = False
        try:
            results = await run_native_offload(
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
            logger.exception("image_to_3d failed")
            return error_response(
                500, "internal_error",
                "image-to-3d backend failed; see server logs",
            )
        finally:
            if not abandoned:
                close_decoded_images([pil_image])

        body = encode_3d_response(
            results,
            model_id=effective_id,
            response_format=response_format,
        )
        return JSONResponse(content=body)

    return router
