"""FastAPI router for /v1/images/generations.

Follows OpenAI's /v1/images/generations contract:
  - `prompt` (required, 1-4000 chars)
  - `n` number of images (1-10)
  - `size` "WIDTHxHEIGHT" (64-2048 per side)
  - `response_format` "b64_json" (default) | "url" (data URL)
  - muse extensions: `model`, `seed`, `steps`, `guidance`, `negative_prompt`
  - img2img extensions (since v0.17.0): `image` (data URL or http(s) URL),
    `strength` (0.0 to 1.0). When `image` is set, the runtime routes through
    AutoPipelineForImage2Image. Requires the selected model to advertise
    `capabilities.supports_img2img: True` in its manifest.
  - LoRA extension: `lora_scale` (0.0 to 2.0). Requires the selected model
    to advertise `capabilities.lora_adapter: True` in its manifest.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field, field_validator

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._native_offload import run_native_offload
from muse.modalities.image_generation.codec import to_bytes, to_data_url
from muse.modalities.image_generation.image_input import (
    close_decoded_images,
    decode_image_file,
    decode_image_input,
    validate_total_image_pixels,
)

logger = logging.getLogger(__name__)

MODALITY = "image/generation"


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
    image: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    lora_scale: float | None = Field(default=None, ge=0.0, le=2.0)

    @field_validator("size")
    @classmethod
    def _validate_size(cls, v: str) -> str:
        w, h = map(int, v.split("x"))
        if w < 64 or h < 64 or w > 2048 or h > 2048:
            raise ValueError(f"size {v} out of supported range (64-2048 per side)")
        return v


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/generation"])

    @router.post("/generations")
    async def generations(req: GenerationRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(model_id=req.model or "<default>", modality=MODALITY)

        # Resolve the effective model id (the registry default when req.model is None)
        # so capability-gate errors and decode-failure errors name a real model.
        effective_id = getattr(model, "model_id", None) or (req.model or "<default>")

        # lora_scale only applies to LoRA adapter models; reject early
        # for others (mirrors the supports_img2img capability gate).
        if req.lora_scale is not None:
            manifest = registry.manifest(MODALITY, effective_id) or {}
            if not manifest.get("capabilities", {}).get("lora_adapter"):
                return error_response(
                    400,
                    "lora_not_supported",
                    f"model {effective_id!r} is not a LoRA adapter model; "
                    f"lora_scale only applies to LoRA models",
                )

        # img2img gating: if `image` is supplied, the selected model must
        # declare `supports_img2img: True` in its manifest, and the input
        # must decode to a PIL.Image.
        init_image = None
        if req.image is not None:
            manifest = registry.manifest(MODALITY, effective_id) or {}
            if not manifest.get("capabilities", {}).get("supports_img2img"):
                return error_response(
                    400,
                    "invalid_parameter",
                    f"model {effective_id!r} does not support img2img; "
                    f"use one of the diffusers-resolved models or sd-turbo",
                )
            try:
                init_image = await decode_image_input(req.image)
            except ValueError as e:
                return error_response(
                    400,
                    "invalid_parameter",
                    f"image decode failed: {e}",
                )

        width, height = map(int, req.size.split("x"))

        def _call_one(seed_offset: int):
            kwargs: dict = {
                "width": width,
                "height": height,
                "negative_prompt": req.negative_prompt,
                "steps": req.steps,
                "guidance": req.guidance,
                "init_image": init_image,
                "strength": req.strength,
                "lora_scale": req.lora_scale,
            }
            if req.seed is not None:
                kwargs["seed"] = req.seed + seed_offset
            with model._inference_lock:
                return model.generate(req.prompt, **kwargs)

        input_images = [init_image] if init_image is not None else []
        results = []

        def _cleanup_abandoned(current_result) -> None:
            close_decoded_images(
                input_images
                + [getattr(result, "image", None) for result in results]
                + [getattr(current_result, "image", None)]
            )

        try:
            for i in range(req.n):
                result = await run_native_offload(
                    lambda i=i: _call_one(i),
                    cleanup_abandoned=_cleanup_abandoned,
                )
                results.append(result)
        except asyncio.CancelledError:
            # The native ownership helper releases request inputs and any
            # eventual result only after the running backend call settles.
            raise
        except BaseException:
            close_decoded_images(
                input_images
                + [getattr(result, "image", None) for result in results]
            )
            raise

        data = []
        try:
            for r in results:
                entry = {"revised_prompt": r.metadata.get("prompt", req.prompt)}
                if req.response_format == "url":
                    entry["url"] = to_data_url(r.image, fmt="png")
                else:
                    entry["b64_json"] = base64.b64encode(
                        to_bytes(r.image, fmt="png"),
                    ).decode()
                data.append(entry)
        finally:
            close_decoded_images(
                input_images
                + [getattr(r, "image", None) for r in results]
            )

        return {"created": int(time.time()), "data": data}

    # ---------------- /v1/images/edits (inpainting, multipart) ----------------

    @router.post("/edits")
    async def edits(
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        prompt: str = Form(...),
        model: str | None = Form(None),
        n: int = Form(1),
        size: str = Form("512x512"),
        response_format: str = Form("b64_json"),
    ):
        # Validate Form fields by hand. Pydantic's BaseModel can't validate
        # a multipart body directly; we mirror the GenerationRequest range
        # checks here so /edits returns the same 400 envelopes.
        if not (1 <= len(prompt) <= 4000):
            return error_response(
                400, "invalid_parameter",
                "prompt must be 1 to 4000 characters",
            )
        if not (1 <= n <= 10):
            return error_response(
                400, "invalid_parameter", "n must be in [1, 10]",
            )
        if response_format not in ("b64_json", "url"):
            return error_response(
                400, "invalid_parameter",
                "response_format must be 'b64_json' or 'url'",
            )
        try:
            w, h = map(int, size.split("x"))
        except Exception:  # noqa: BLE001
            return error_response(
                400, "invalid_parameter",
                f"size must be 'WIDTHxHEIGHT'; got {size!r}",
            )
        if not (64 <= w <= 2048 and 64 <= h <= 2048):
            return error_response(
                400, "invalid_parameter",
                f"size {size} out of supported range (64-2048 per side)",
            )

        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(model_id=model or "<default>", modality=MODALITY)

        effective_id = getattr(backend, "model_id", None) or (model or "<default>")
        manifest = registry.manifest(MODALITY, effective_id) or {}
        if not manifest.get("capabilities", {}).get("supports_inpainting"):
            return error_response(
                400, "invalid_parameter",
                f"model {effective_id!r} does not support inpainting",
            )

        decoded_images = []
        try:
            init_image = await decode_image_file(image)
            decoded_images.append(init_image)
            mask_image = await decode_image_file(mask)
            decoded_images.append(mask_image)
            validate_total_image_pixels(decoded_images)
        except ValueError as e:
            close_decoded_images(decoded_images)
            return error_response(
                400, "invalid_parameter", f"image decode failed: {e}",
            )
        except BaseException:
            close_decoded_images(decoded_images)
            raise

        def _call_one():
            kwargs: dict = {
                "width": w,
                "height": h,
                "init_image": init_image,
                "mask_image": mask_image,
            }
            with backend._inference_lock:
                return backend.inpaint(prompt, **kwargs)

        results = []

        def _cleanup_abandoned(current_result) -> None:
            close_decoded_images(
                decoded_images
                + [getattr(result, "image", None) for result in results]
                + [getattr(current_result, "image", None)]
            )

        try:
            for _ in range(n):
                result = await run_native_offload(
                    _call_one,
                    cleanup_abandoned=_cleanup_abandoned,
                )
                results.append(result)
        except asyncio.CancelledError:
            raise
        except BaseException:
            close_decoded_images(
                decoded_images
                + [getattr(result, "image", None) for result in results]
            )
            raise

        data = []
        try:
            for r in results:
                entry = {"revised_prompt": r.metadata.get("prompt", prompt)}
                if response_format == "url":
                    entry["url"] = to_data_url(r.image, fmt="png")
                else:
                    entry["b64_json"] = base64.b64encode(
                        to_bytes(r.image, fmt="png"),
                    ).decode()
                data.append(entry)
        finally:
            close_decoded_images(
                decoded_images
                + [getattr(r, "image", None) for r in results]
            )

        return {"created": int(time.time()), "data": data}

    # ---------------- /v1/images/variations (multipart) ----------------

    @router.post("/variations")
    async def variations(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        n: int = Form(1),
        size: str = Form("512x512"),
        response_format: str = Form("b64_json"),
    ):
        if not (1 <= n <= 10):
            return error_response(
                400, "invalid_parameter", "n must be in [1, 10]",
            )
        if response_format not in ("b64_json", "url"):
            return error_response(
                400, "invalid_parameter",
                "response_format must be 'b64_json' or 'url'",
            )
        try:
            w, h = map(int, size.split("x"))
        except Exception:  # noqa: BLE001
            return error_response(
                400, "invalid_parameter",
                f"size must be 'WIDTHxHEIGHT'; got {size!r}",
            )
        if not (64 <= w <= 2048 and 64 <= h <= 2048):
            return error_response(
                400, "invalid_parameter",
                f"size {size} out of supported range (64-2048 per side)",
            )

        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(model_id=model or "<default>", modality=MODALITY)

        effective_id = getattr(backend, "model_id", None) or (model or "<default>")
        manifest = registry.manifest(MODALITY, effective_id) or {}
        if not manifest.get("capabilities", {}).get("supports_variations"):
            return error_response(
                400, "invalid_parameter",
                f"model {effective_id!r} does not support variations",
            )

        try:
            init_image = await decode_image_file(image)
        except ValueError as e:
            return error_response(
                400, "invalid_parameter", f"image decode failed: {e}",
            )

        def _call_one():
            kwargs: dict = {
                "width": w,
                "height": h,
                "init_image": init_image,
            }
            with backend._inference_lock:
                return backend.vary(**kwargs)

        results = []

        def _cleanup_abandoned(current_result) -> None:
            close_decoded_images(
                [init_image]
                + [getattr(result, "image", None) for result in results]
                + [getattr(current_result, "image", None)]
            )

        try:
            for _ in range(n):
                result = await run_native_offload(
                    _call_one,
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

        # Variations envelope: no revised_prompt (no prompt).
        data = []
        try:
            for r in results:
                entry: dict = {}
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
