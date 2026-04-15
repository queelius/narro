"""FastAPI router for /v1/images/generations.

Follows OpenAI's /v1/images/generations contract:
  - `prompt` (required, 1-4000 chars)
  - `n` number of images (1-10)
  - `size` "WIDTHxHEIGHT" (64-2048 per side)
  - `response_format` "b64_json" (default) | "url" (data URL)
  - muse extensions: `model`, `seed`, `steps`, `guidance`, `negative_prompt`
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from threading import Lock

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from muse.core.errors import ModelNotFoundError
from muse.core.registry import ModalityRegistry
from muse.images.generations.codec import to_bytes, to_data_url

logger = logging.getLogger(__name__)

MODALITY = "images.generations"
_inference_lock = Lock()


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=10)
    size: str = Field(default="512x512", pattern=r"^\d+x\d+$")
    response_format: str = Field(default="b64_json", pattern="^(b64_json|url)$")
    negative_prompt: str | None = None
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = None

    @field_validator("size")
    @classmethod
    def _validate_size(cls, v: str) -> str:
        w, h = map(int, v.split("x"))
        if w < 64 or h < 64 or w > 2048 or h > 2048:
            raise ValueError(f"size {v} out of supported range (64-2048 per side)")
        return v


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["images.generations"])

    @router.post("/generations")
    async def generations(req: GenerationRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(model_id=req.model or "<default>", modality=MODALITY)

        width, height = map(int, req.size.split("x"))

        def _call_one(seed_offset: int):
            kwargs: dict = {
                "width": width,
                "height": height,
                "negative_prompt": req.negative_prompt,
                "steps": req.steps,
                "guidance": req.guidance,
            }
            if req.seed is not None:
                kwargs["seed"] = req.seed + seed_offset
            with _inference_lock:
                return model.generate(req.prompt, **kwargs)

        results = []
        for i in range(req.n):
            result = await asyncio.to_thread(_call_one, i)
            results.append(result)

        data = []
        for r in results:
            entry = {"revised_prompt": r.metadata.get("prompt", req.prompt)}
            if req.response_format == "url":
                entry["url"] = to_data_url(r.image, fmt="png")
            else:
                entry["b64_json"] = base64.b64encode(to_bytes(r.image, fmt="png")).decode()
            data.append(entry)

        return {"created": int(time.time()), "data": data}

    return router
