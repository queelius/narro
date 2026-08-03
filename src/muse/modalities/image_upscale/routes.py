"""FastAPI router for /v1/images/upscale (multipart/form-data).

Mirrors the wire shape of /v1/images/edits and /v1/images/variations:
the source image arrives as a multipart file upload, all other fields
arrive as Form parameters. Output envelope mirrors /v1/images/generations.

Capability gating: the model's `supported_scales` capability must
include the requested `scale`; otherwise the route returns 400. An
env-tunable input-side cap (MUSE_UPSCALE_MAX_INPUT_SIDE, default 1024)
rejects oversized inputs before they reach the pipeline.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time


from fastapi import APIRouter, File, Form, UploadFile

from muse.core import config
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._native_offload import run_native_offload
from muse.modalities.image_generation.image_input import (
    close_decoded_images,
    decode_image_file,
)
from muse.modalities.image_upscale.codec import to_bytes, to_data_url

logger = logging.getLogger(__name__)

MODALITY = "image/upscale"


def _max_input_side() -> int:
    """Read the per-request input-side cap via muse.core.config.

    Default 1024. Tunable via MUSE_UPSCALE_MAX_INPUT_SIDE so users with
    more VRAM can lift the cap without a code change. A resolved value
    that is non-positive (parseable but nonsensical as a side cap)
    falls back to the registry default, mirroring
    image_generation/image_input.py's _default_max_bytes guard.
    """
    n = config.get("limits.upscale_max_input_side")
    if n is None or n <= 0:
        return config.SETTINGS_BY_KEY["limits.upscale_max_input_side"].default
    return n


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/upscale"])

    @router.post("/upscale")
    async def upscale(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        scale: int = Form(4),
        prompt: str = Form(""),
        negative_prompt: str | None = Form(None),
        steps: int | None = Form(None),
        guidance: float | None = Form(None),
        seed: int | None = Form(None),
        n: int = Form(1),
        response_format: str = Form("b64_json"),
    ):
        # Manual Form validation (Pydantic doesn't validate multipart).
        if not (1 <= n <= 4):
            return error_response(
                400, "invalid_parameter", "n must be in [1, 4]",
            )
        if response_format not in ("b64_json", "url"):
            return error_response(
                400, "invalid_parameter",
                "response_format must be 'b64_json' or 'url'",
            )
        if len(prompt) > 4000:
            return error_response(
                400, "invalid_parameter",
                "prompt must be 0 to 4000 characters",
            )
        if negative_prompt is not None and len(negative_prompt) > 4000:
            return error_response(
                400, "invalid_parameter",
                "negative_prompt must be 0 to 4000 characters",
            )
        if steps is not None and not (1 <= steps <= 100):
            return error_response(
                400, "invalid_parameter",
                "steps must be in [1, 100]",
            )
        if guidance is not None and not (0.0 <= guidance <= 20.0):
            return error_response(
                400, "invalid_parameter",
                "guidance must be in [0.0, 20.0]",
            )

        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(model_id=model or "<default>", modality=MODALITY)

        effective_id = getattr(backend, "model_id", None) or (model or "<default>")
        manifest = registry.manifest(MODALITY, effective_id) or {}
        capabilities = manifest.get("capabilities", {}) or {}
        supported = list(capabilities.get("supported_scales") or [4])
        if scale not in supported:
            return error_response(
                400, "invalid_parameter",
                f"model {effective_id!r} only supports scales: {supported}",
            )

        max_side = _max_input_side()
        try:
            init_image = await decode_image_file(image, max_side=max_side)
        except ValueError as e:
            message = str(e)
            if "exceeds max input side" in message:
                message += " (set MUSE_UPSCALE_MAX_INPUT_SIDE to raise)"
            return error_response(
                400, "invalid_parameter", f"image decode failed: {message}",
            )

        def _call_one(seed_offset: int):
            kwargs: dict = {
                "scale": scale,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance": guidance,
            }
            if seed is not None:
                kwargs["seed"] = seed + seed_offset
            with backend._inference_lock:
                return backend.upscale(init_image, **kwargs)

        results = []

        def _cleanup_abandoned(current_result) -> None:
            close_decoded_images(
                [init_image]
                + [getattr(result, "image", None) for result in results]
                + [getattr(current_result, "image", None)]
            )

        try:
            for i in range(n):
                result = await run_native_offload(
                    lambda i=i: _call_one(i),
                    cleanup_abandoned=_cleanup_abandoned,
                )
                results.append(result)
        except asyncio.CancelledError:
            raise
        except BaseException:
            close_decoded_images(
                [init_image]
                + [getattr(result, "image", None) for result in results]
            )
            raise

        data = []
        try:
            for r in results:
                entry: dict = {"revised_prompt": prompt or None}
                if response_format == "url":
                    entry["url"] = to_data_url(r.image, fmt="png")
                else:
                    entry["b64_json"] = base64.b64encode(
                        to_bytes(r.image, fmt="png"),
                    ).decode()
                data.append(entry)
        finally:
            close_decoded_images(
                [init_image]
                + [getattr(r, "image", None) for r in results]
            )

        return {"created": int(time.time()), "data": data}

    return router
