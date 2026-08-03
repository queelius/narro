"""FastAPI routes for image/cv: depth, keypoints, object detection.

Three routes share the modality tag (image/cv) and registry namespace.
Capability flags on each model's manifest gate which route accepts it:

  POST /v1/images/depth        requires supports_depth=True
  POST /v1/images/keypoints    requires supports_keypoints=True
  POST /v1/images/detect       requires supports_detection=True

Mismatch returns 400 with code `wrong_primitive`.

All three routes:
  - Multipart in (image upload via decode_image_file from
    image_generation; MUSE_IMAGE_INPUT_MAX_BYTES env cap from v0.34.0).
  - Per-backend inference lock (v0.34.0 Task A) serializes calls.
  - JSON out via the codec layer's encode_*_envelope helpers.
  - OpenAI-shape error envelope on 400/404/500.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._native_offload import run_native_offload
from muse.modalities.image_cv.codec import (
    encode_depth_envelope,
    encode_detections_envelope,
    encode_keypoints_envelope,
)
from muse.modalities.image_generation.image_input import (
    close_decoded_images,
    decode_image_file,
)


MODALITY = "image/cv"


logger = logging.getLogger(__name__)


def _resolve_backend(registry: ModalityRegistry, model: str | None):
    """Look up the backend; raise ModelNotFoundError on miss.

    Mirrors the pattern from sibling modalities. Returns the registered
    backend object so the route can read .model_id and the manifest.
    """
    try:
        return registry.get(MODALITY, model)
    except KeyError:
        raise ModelNotFoundError(
            model_id=model or "<default>", modality=MODALITY,
        )


def _capability_gate(
    registry: ModalityRegistry, backend, *, primitive: str,
) -> JSONResponse | None:
    """Return None if the backend supports the primitive; else a 400.

    primitive is one of "depth", "keypoints", "detection". The model's
    manifest must set the matching `supports_*` capability flag.
    """
    flag_name = {
        "depth": "supports_depth",
        "keypoints": "supports_keypoints",
        "detection": "supports_detection",
    }[primitive]
    manifest = registry.manifest(MODALITY, backend.model_id) or {}
    capabilities = manifest.get("capabilities") or {}
    if not capabilities.get(flag_name):
        return error_response(
            400, "wrong_primitive",
            f"model {backend.model_id!r} does not support the {primitive} "
            f"primitive (no {flag_name}=True in manifest); use a model "
            f"declaring that capability",
        )
    return None


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    # ---------- POST /v1/images/depth ----------

    @router.post("/v1/images/depth")
    async def images_depth(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        response_format: str = Form("png16"),
    ):
        if response_format not in ("png16", "float32"):
            return error_response(
                400, "invalid_parameter",
                f"response_format must be 'png16' or 'float32'; "
                f"got {response_format!r}",
            )

        backend = _resolve_backend(registry, model)
        gate = _capability_gate(registry, backend, primitive="depth")
        if gate is not None:
            return gate

        try:
            pil_image = await decode_image_file(image)
        except ValueError as e:
            return error_response(400, "invalid_parameter", str(e))

        def _call():
            with backend._inference_lock:
                return backend.estimate_depth(pil_image)

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
            logger.exception("estimate_depth failed")
            return error_response(
                500, "internal_error",
                "depth estimation backend failed; see server logs",
            )
        finally:
            if not abandoned:
                close_decoded_images([pil_image])

        try:
            body = encode_depth_envelope(
                result, response_format=response_format,
            )
        except ValueError as e:
            return error_response(400, "invalid_parameter", str(e))
        body["model"] = backend.model_id
        return JSONResponse(content=body)

    # ---------- POST /v1/images/keypoints ----------

    @router.post("/v1/images/keypoints")
    async def images_keypoints(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        threshold: float | None = Form(None),
    ):
        if threshold is not None and not (0.0 <= threshold <= 1.0):
            return error_response(
                400, "invalid_parameter",
                f"threshold must be in [0, 1]; got {threshold}",
            )

        backend = _resolve_backend(registry, model)
        gate = _capability_gate(registry, backend, primitive="keypoints")
        if gate is not None:
            return gate

        try:
            pil_image = await decode_image_file(image)
        except ValueError as e:
            return error_response(400, "invalid_parameter", str(e))

        kwargs: dict = {}
        if threshold is not None:
            kwargs["threshold"] = threshold

        def _call():
            with backend._inference_lock:
                return backend.detect_keypoints(pil_image, **kwargs)

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
            logger.exception("detect_keypoints failed")
            return error_response(
                500, "internal_error",
                "keypoint detection backend failed; see server logs",
            )
        finally:
            if not abandoned:
                close_decoded_images([pil_image])

        body = encode_keypoints_envelope(result)
        body["model"] = backend.model_id
        return JSONResponse(content=body)

    # ---------- POST /v1/images/detect ----------

    @router.post("/v1/images/detect")
    async def images_detect(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        threshold: float | None = Form(None),
        max_detections: int | None = Form(None),
    ):
        if threshold is not None and not (0.0 <= threshold <= 1.0):
            return error_response(
                400, "invalid_parameter",
                f"threshold must be in [0, 1]; got {threshold}",
            )
        if max_detections is not None and (max_detections < 1 or max_detections > 10000):
            return error_response(
                400, "invalid_parameter",
                f"max_detections must be in [1, 10000]; got {max_detections}",
            )

        backend = _resolve_backend(registry, model)
        gate = _capability_gate(registry, backend, primitive="detection")
        if gate is not None:
            return gate

        try:
            pil_image = await decode_image_file(image)
        except ValueError as e:
            return error_response(400, "invalid_parameter", str(e))

        kwargs: dict = {}
        if threshold is not None:
            kwargs["threshold"] = threshold
        if max_detections is not None:
            kwargs["max_detections"] = max_detections

        def _call():
            with backend._inference_lock:
                return backend.detect_objects(pil_image, **kwargs)

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
            logger.exception("detect_objects failed")
            return error_response(
                500, "internal_error",
                "object detection backend failed; see server logs",
            )
        finally:
            if not abandoned:
                close_decoded_images([pil_image])

        body = encode_detections_envelope(result)
        body["model"] = backend.model_id
        return JSONResponse(content=body)

    return router
