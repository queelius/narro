"""POST /v1/video/generations.

Wire contract documented in
docs/superpowers/specs/2026-04-28-video-generation-modality-design.md.

video/generation is muse's narrative-clip sibling to image/animation:
longer durations, single play (no `loop` field), mp4 default,
transformer-based backbones (Wan, CogVideoX). The two modalities
deliberately don't overlap: short looping animations go to
/v1/images/animations; multi-second narrative clips go here.
"""
from __future__ import annotations

import asyncio
import base64
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from muse.core import config
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.video_generation.codec import (
    UnsupportedFormatError,
    encode_frames_b64,
    encode_mp4,
    encode_webm,
)


MODALITY = "video/generation"

logger = logging.getLogger(__name__)

_MAX_SIDE = 2048
_MAX_PIXELS_PER_FRAME = 2_097_152
_MAX_REQUEST_PIXEL_FRAMES = 256 * 1024 * 1024

# frames_b64 inlines every frame as a base64 PNG in one JSON body. At the
# request caps (duration<=30s, fps<=60) that can reach ~1800 frames, i.e. a
# multi-hundred-MB response that the worker must hold in memory and serialize
# at once. mp4/webm are container-compressed and have no such ceiling, so the
# cap applies only to frames_b64. Tunable for power users; clips above it
# should use response_format=mp4/webm.
#
# Read per-request via muse.core.config so an operator's env change takes
# effect on the next request, not at server restart. Matches the
# MUSE_IMAGE_INPUT_MAX_BYTES / MUSE_MODERATIONS_MAX_BATCH pattern.
def _max_frames_b64() -> int:
    return config.get("limits.video_max_frames_b64")


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.5, le=30.0)
    fps: int | None = Field(default=None, ge=1, le=60)
    size: str | None = Field(default=None, pattern=r"^\d+x\d+$")
    seed: int | None = None
    negative_prompt: str | None = Field(default=None, max_length=4000)
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    response_format: str = Field(
        default="mp4", pattern="^(mp4|webm|frames_b64)$",
    )
    n: int = Field(default=1, ge=1, le=2)

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
        duration = self.duration_seconds if self.duration_seconds is not None else 5.0
        fps = self.fps if self.fps is not None else 8
        pixel_frames = width * height * duration * fps * self.n
        if pixel_frames > _MAX_REQUEST_PIXEL_FRAMES:
            raise ValueError(
                "requested n × duration × fps × width × height exceeds the video "
                f"budget of {_MAX_REQUEST_PIXEL_FRAMES} pixel-frames"
            )
        return self


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/video", tags=["video/generation"])

    @router.post("/generations")
    async def generations(req: VideoGenerationRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        width = height = None
        if req.size is not None:
            width, height = map(int, req.size.split("x"))

        def _call_one(seed_offset: int):
            with model._inference_lock:
                kwargs = {
                    "negative_prompt": req.negative_prompt,
                    "duration_seconds": req.duration_seconds,
                    "fps": req.fps,
                    "width": width,
                    "height": height,
                    "steps": req.steps,
                    "guidance": req.guidance,
                }
                if req.seed is not None:
                    kwargs["seed"] = req.seed + seed_offset
                return model.generate(req.prompt, **kwargs)

        try:
            results = []
            for i in range(req.n):
                r = await asyncio.to_thread(_call_one, i)
                results.append(r)
        except Exception:  # noqa: BLE001
            # Log the real exception server-side but never leak it to the
            # client: str(e) can carry internal filesystem paths, CUDA
            # driver text, or other backend-implementation detail.
            logger.exception("video generation failed")
            return error_response(
                500, "internal_error",
                "video generation backend failed; see server logs",
            )

        # frames_b64 inlines every frame; guard the response payload size.
        # Checked post-generation because the frame count is only known once
        # the model has run (duration_seconds/fps may be model defaults).
        if req.response_format == "frames_b64":
            max_frames = _max_frames_b64()
            total_frames = sum(len(r.frames) for r in results)
            if total_frames > max_frames:
                return error_response(
                    400, "invalid_parameter",
                    f"frames_b64 would emit {total_frames} frames, exceeding "
                    f"MUSE_VIDEO_MAX_FRAMES_B64={max_frames}; request a "
                    f"shorter clip or use response_format=mp4 or webm "
                    f"(no frame cap).",
                )

        data = []
        for r in results:
            try:
                encoded = _encode(req.response_format, r)
            except UnsupportedFormatError as e:
                return error_response(400, "invalid_parameter", str(e))
            except ValueError as e:
                # mp4/webm encoders raise a plain ValueError when the
                # backend hands back zero frames ("frames list is
                # empty"). That's a backend/model misbehavior, not a
                # bad request, so surface it as the OpenAI-shape 500
                # envelope instead of letting it escape to FastAPI's
                # bare default handler. Log the real exception server-side
                # but never leak it to the client: str(e) can carry
                # internal filesystem paths or other implementation detail.
                logger.exception("video encode failed")
                return error_response(
                    500, "internal_error",
                    "video encoding backend failed; see server logs",
                )
            if req.response_format == "frames_b64":
                # encoded is list[str]; expand into per-frame data entries.
                # For n>1 the per-result frame lists are appended in order;
                # a future revision may add explicit per-result grouping.
                for s in encoded:
                    data.append({"b64_json": s})
            else:
                data.append({
                    "b64_json": base64.b64encode(encoded).decode("ascii"),
                })

        # Use the first result for top-level metadata (n=1 is common)
        head = results[0]
        body = {
            "data": data,
            "model": model.model_id,
            "metadata": {
                "frames": len(head.frames),
                "fps": head.fps,
                "duration_seconds": head.duration_seconds,
                "format": req.response_format,
                "size": [head.width, head.height],
            },
        }
        return JSONResponse(content=body)

    return router


def _encode(fmt, result):
    if fmt == "mp4":
        return encode_mp4(result.frames, result.fps)
    if fmt == "webm":
        return encode_webm(result.frames, result.fps)
    if fmt == "frames_b64":
        return encode_frames_b64(result.frames)
    raise ValueError(f"unknown format: {fmt}")
