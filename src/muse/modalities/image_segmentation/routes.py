"""FastAPI router for /v1/images/segment (multipart/form-data).

Wire shape: multipart upload of an image plus form fields for the
prompt mode dispatch. Modes: auto, points, boxes, text. Each mode
maps to a capability flag on the model; mismatched modes return 400
before the runtime is invoked.

Capability gating table:
    auto   -> supports_automatic
    points -> supports_point_prompts
    boxes  -> supports_box_prompts
    text   -> supports_text_prompts

points and boxes arrive as JSON-encoded strings; the route parses
them with json.loads and validates the shape (list of pairs / list
of quads of integers). Bad JSON or shape mismatches yield 400.

The output envelope (see codec.encode_segmentation) carries an id,
the model id, the dispatch mode, the input image_size in PIL
convention (W, H), and a list of masks with index, score, bbox, area,
plus the encoded mask (PNG b64 or COCO RLE).
"""
from __future__ import annotations

import asyncio
import json
import logging


from fastapi import APIRouter, File, Form, UploadFile

from muse.core import config
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._native_offload import run_native_offload
from muse.modalities.image_generation.image_input import (
    close_decoded_images,
    decode_image_file,
)
from muse.modalities.image_segmentation.codec import encode_segmentation
from muse.modalities.image_segmentation.protocol import CapabilityError


logger = logging.getLogger(__name__)

MODALITY = "image/segmentation"



_VALID_MODES = ("auto", "points", "boxes", "text")
_VALID_MASK_FORMATS = ("png_b64", "rle")

_MODE_CAPABILITY = {
    "auto": "supports_automatic",
    "points": "supports_point_prompts",
    "boxes": "supports_box_prompts",
    "text": "supports_text_prompts",
}

_MODE_HUMAN = {
    "auto": "automatic",
    "points": "point-prompted",
    "boxes": "box-prompted",
    "text": "text-prompted",
}


def _max_input_side() -> int:
    """Read the per-request input-side cap via muse.core.config (env:
    MUSE_SEGMENTATION_MAX_INPUT_SIDE) per call, so operators can change
    the cap without a server restart. A resolved value that is
    non-positive (parseable but nonsensical as a side cap) falls back
    to the registry default, mirroring image_generation/image_input.py's
    _default_max_bytes guard."""
    n = config.get("limits.segmentation_max_input_side")
    if n is None or n <= 0:
        return config.SETTINGS_BY_KEY["limits.segmentation_max_input_side"].default
    return n


def _parse_points_json(raw: str) -> list[list[int]] | None:
    """Parse a JSON-encoded list of [x, y] integer pairs.

    Returns None on bad shape or bad JSON; the caller decides the 400
    message.
    """
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    out: list[list[int]] = []
    for entry in parsed:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return None
        try:
            out.append([int(entry[0]), int(entry[1])])
        except (TypeError, ValueError):
            return None
    return out


def _parse_boxes_json(raw: str) -> list[list[int]] | None:
    """Parse a JSON-encoded list of [x1, y1, x2, y2] integer quads."""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    out: list[list[int]] = []
    for entry in parsed:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            return None
        try:
            out.append([int(entry[0]), int(entry[1]),
                        int(entry[2]), int(entry[3])])
        except (TypeError, ValueError):
            return None
    return out


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/segmentation"])

    @router.post("/segment")
    async def segment(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        mode: str = Form("auto"),
        prompt: str | None = Form(None),
        points: str | None = Form(None),
        boxes: str | None = Form(None),
        mask_format: str = Form("png_b64"),
        max_masks: int = Form(16),
    ):
        # Manual Form validation (Pydantic doesn't validate multipart).
        if mode not in _VALID_MODES:
            return error_response(
                400, "invalid_parameter",
                f"mode must be one of {list(_VALID_MODES)}",
            )
        if mask_format not in _VALID_MASK_FORMATS:
            return error_response(
                400, "invalid_parameter",
                f"mask_format must be one of {list(_VALID_MASK_FORMATS)}",
            )
        if not (1 <= max_masks <= 256):
            return error_response(
                400, "invalid_parameter",
                "max_masks must be in [1, 256]",
            )
        if mode == "text" and (prompt is None or not prompt.strip()):
            return error_response(
                400, "invalid_parameter",
                "mode='text' requires a non-empty prompt",
            )
        if prompt is not None and len(prompt) > 4000:
            return error_response(
                400, "invalid_parameter",
                "prompt must be 0 to 4000 characters",
            )

        parsed_points: list[list[int]] | None = None
        parsed_boxes: list[list[int]] | None = None
        if mode == "points":
            if points is None:
                return error_response(
                    400, "invalid_parameter",
                    "mode='points' requires points (JSON list of [x, y] pairs)",
                )
            parsed_points = _parse_points_json(points)
            if parsed_points is None:
                return error_response(
                    400, "invalid_parameter",
                    "points must be a JSON list of [x, y] integer pairs",
                )
        if mode == "boxes":
            if boxes is None:
                return error_response(
                    400, "invalid_parameter",
                    "mode='boxes' requires boxes (JSON list of [x1, y1, x2, y2] quads)",
                )
            parsed_boxes = _parse_boxes_json(boxes)
            if parsed_boxes is None:
                return error_response(
                    400, "invalid_parameter",
                    "boxes must be a JSON list of [x1, y1, x2, y2] integer quads",
                )

        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(model_id=model or "<default>", modality=MODALITY)

        effective_id = getattr(backend, "model_id", None) or (model or "<default>")
        manifest = registry.manifest(MODALITY, effective_id) or {}
        capabilities = manifest.get("capabilities", {}) or {}

        # Capability gate (defense-in-depth before runtime invocation).
        cap_key = _MODE_CAPABILITY[mode]
        if not capabilities.get(cap_key, True):
            human = _MODE_HUMAN[mode]
            return error_response(
                400, "invalid_parameter",
                f"model {effective_id!r} does not support {human} segmentation",
            )

        max_side = _max_input_side()
        try:
            pil_image = await decode_image_file(image, max_side=max_side)
        except ValueError as e:
            message = str(e)
            if "exceeds max input side" in message:
                message += (
                    " (set MUSE_SEGMENTATION_MAX_INPUT_SIDE to raise)"
                )
            return error_response(
                400, "invalid_parameter", f"image decode failed: {message}",
            )

        def _call() -> object:
            with backend._inference_lock:
                return backend.segment(
                    pil_image,
                    mode=mode,
                    prompt=prompt,
                    points=parsed_points,
                    boxes=parsed_boxes,
                    max_masks=max_masks,
                )

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
        except (CapabilityError, ValueError) as e:
            # Client faults: an unsupported mode (CapabilityError, the
            # runtime's defense-in-depth for a mismatch the gate above
            # should already have caught) or a malformed prompt/points/boxes
            # (ValueError). Both are 400 invalid_parameter.
            return error_response(
                400, "invalid_parameter", str(e),
            )
        except Exception:  # noqa: BLE001
            # Server faults: a bare RuntimeError (e.g. CUDA OOM) or any other
            # unexpected error is NOT the client's fault. Surface a 500 in
            # the OpenAI envelope so clients retry rather than treating a
            # valid request as malformed, and so nothing escapes as a bare
            # Starlette 500. Log the real exception server-side but never
            # leak it to the client: str(e) can carry internal filesystem
            # paths, CUDA driver text, or other backend-implementation
            # detail.
            logger.exception("segmentation inference failed")
            return error_response(
                500, "internal_error",
                "segmentation backend failed; see server logs",
            )
        finally:
            if not abandoned:
                close_decoded_images([pil_image])

        return encode_segmentation(
            result, model_id=effective_id, mask_format=mask_format,
        )

    return router
