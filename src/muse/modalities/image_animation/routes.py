"""POST /v1/images/animations.

Wire contract documented in docs/superpowers/specs/2026-04-27-image-animation-modality-design.md.
"""
from __future__ import annotations

import asyncio
import base64
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._native_offload import run_native_offload
from muse.modalities.image_animation.codec import (
    encode_webp, encode_gif, encode_mp4, encode_frames_b64,
    UnsupportedFormatError,
)
from muse.modalities.image_generation.image_input import (
    close_decoded_images,
    decode_image_input,
)


MODALITY = "image/animation"

logger = logging.getLogger(__name__)

_MAX_SIDE = 2048
_MAX_PIXELS_PER_FRAME = 2_097_152
_MAX_REQUEST_PIXEL_FRAMES = 128 * 1024 * 1024
_DEFAULT_FRAMES = 16


def _animation_result_images(results: list[object]) -> list[object]:
    return [
        frame
        for result in results
        for frame in (getattr(result, "frames", ()) or ())
    ]


def _close_animation_results(results: list[object]) -> None:
    close_decoded_images(_animation_result_images(results))


class AnimationsRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    frames: int = Field(default=_DEFAULT_FRAMES, ge=4, le=64)
    fps: int | None = Field(default=None, ge=1, le=30)
    loop: bool = True
    negative_prompt: str | None = None
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = None
    image: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    response_format: str = Field(default="webp", pattern="^(webp|gif|mp4|frames_b64)$")
    size: str | None = Field(default=None, pattern=r"^\d+x\d+$")

    @field_validator("size")
    @classmethod
    def _bounded_size(cls, value: str | None) -> str | None:
        if value is None:
            return value
        width, height = map(int, value.split("x"))
        if width < 1 or height < 1:
            raise ValueError("size dimensions must be positive")
        if width > _MAX_SIDE or height > _MAX_SIDE:
            raise ValueError(f"size dimensions must not exceed {_MAX_SIDE}")
        if width * height > _MAX_PIXELS_PER_FRAME:
            raise ValueError(
                f"size must not exceed {_MAX_PIXELS_PER_FRAME} pixels per frame"
            )
        return value

    @model_validator(mode="after")
    def _bounded_workload(self):
        if self.size is None:
            return self
        width, height = map(int, self.size.split("x"))
        # ``frames`` has an API-level default, so the admitted count is the
        # exact value forwarded to every backend rather than an estimate of a
        # custom backend's potentially larger private default.
        pixel_frames = width * height * self.frames * self.n
        if pixel_frames > _MAX_REQUEST_PIXEL_FRAMES:
            raise ValueError(
                "requested n × frames × width × height exceeds the animation "
                f"budget of {_MAX_REQUEST_PIXEL_FRAMES} pixel-frames"
            )
        return self


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/animation"])

    @router.post("/animations")
    async def animations(req: AnimationsRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )
        manifest = registry.manifest(MODALITY, model.model_id) or {}
        capabilities = manifest.get("capabilities") or {}

        # Image-to-animation gate
        init_image = None
        if req.image is not None:
            if not capabilities.get("supports_image_to_animation"):
                return error_response(
                    400, "invalid_parameter",
                    f"model {model.model_id!r} does not support image-to-animation; "
                    f"use a model with supports_image_to_animation=True",
                )
            try:
                init_image = await decode_image_input(req.image)
            except ValueError as e:
                return error_response(
                    400, "invalid_parameter", f"image decode failed: {e}",
                )

        width = height = None
        if req.size is not None:
            width, height = map(int, req.size.split("x"))

        def _call_one(seed_offset: int):
            with model._inference_lock:
                kwargs = {
                    "negative_prompt": req.negative_prompt,
                    "frames": req.frames,
                    "fps": req.fps,
                    "width": width, "height": height,
                    "steps": req.steps, "guidance": req.guidance,
                    "init_image": init_image,
                    "strength": req.strength,
                }
                if req.seed is not None:
                    kwargs["seed"] = req.seed + seed_offset
                return model.generate(req.prompt, **kwargs)

        input_images = [init_image] if init_image is not None else []
        results = []

        def _cleanup_abandoned(current_result) -> None:
            close_decoded_images(
                input_images
                + _animation_result_images(results)
                + _animation_result_images([current_result])
            )

        try:
            for i in range(req.n):
                result = await run_native_offload(
                    lambda i=i: _call_one(i),
                    cleanup_abandoned=_cleanup_abandoned,
                )
                results.append(result)
        except asyncio.CancelledError:
            raise
        except BaseException:
            close_decoded_images(
                input_images + _animation_result_images(results)
            )
            raise

        try:
            # Encode each result according to response_format.
            data = []
            for r in results:
                try:
                    encoded = _encode(req.response_format, r, loop=req.loop)
                except UnsupportedFormatError as e:
                    return error_response(400, "invalid_parameter", str(e))
                if req.response_format == "frames_b64":
                    # encoded is list[str]; expand into per-frame data entries
                    for s in encoded:
                        data.append({"b64_json": s})
                else:
                    data.append({
                        "b64_json": base64.b64encode(encoded).decode("ascii"),
                    })

            # Use the first result for top-level metadata (n=1 is common).
            head = results[0]
            body = {
                "data": data,
                "model": model.model_id,
                "metadata": {
                    "frames": len(head.frames),
                    "fps": head.fps,
                    "duration_seconds": round(
                        len(head.frames) / max(head.fps, 1), 3,
                    ),
                    "format": req.response_format,
                    "size": [head.width, head.height],
                },
            }
            return JSONResponse(content=body)
        finally:
            close_decoded_images(input_images)
            _close_animation_results(results)

    return router


def _encode(fmt, result, *, loop):
    if fmt == "webp":
        return encode_webp(result.frames, result.fps, loop=loop)
    if fmt == "gif":
        return encode_gif(result.frames, result.fps, loop=loop)
    if fmt == "mp4":
        return encode_mp4(result.frames, result.fps)
    if fmt == "frames_b64":
        return encode_frames_b64(result.frames)
    raise ValueError(f"unknown format: {fmt}")
