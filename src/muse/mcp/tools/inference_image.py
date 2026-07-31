"""Inference image tools.

Eight tools wrapping the image modalities:
  - muse_generate_image      /v1/images/generations
  - muse_edit_image          /v1/images/edits           (image + mask)
  - muse_vary_image          /v1/images/variations      (image)
  - muse_upscale_image       /v1/images/upscale         (image)
  - muse_segment_image       /v1/images/segment         (image)
  - muse_vectorize_image     /v1/images/vectorize       (image)
  - muse_generate_animation  /v1/images/animations
  - muse_embed_image         /v1/images/embeddings

Image-output handlers return one ImageContent block per generated image
plus a TextContent summary. Mask outputs from segmentation come back as
an envelope (PNG b64 strings or COCO RLE dicts depending on
``mask_format``).
"""
from __future__ import annotations

import base64
import json
from typing import Any

from mcp.types import Tool

from muse.mcp.binary_io import (
    binary_input_schema,
    pack_image_output,
    resolve_binary_input,
)
from muse.mcp.client import MuseClient
from muse.mcp.tools import INFERENCE_TOOLS, ToolEntry


def _json_block(payload: Any) -> dict:
    return {"type": "text", "text": json.dumps(payload, indent=2)}


def _filter_keys(args: dict, prefixes: tuple[str, ...]) -> dict:
    """Return ``args`` minus any key starting with one of ``prefixes``,
    and minus None values. Used to peel off binary input fields before
    forwarding the rest to muse as request body."""
    out = {}
    for k, v in args.items():
        if any(k.startswith(p) for p in prefixes):
            continue
        if v is None:
            continue
        out[k] = v
    return out


# ---- handlers ----

def _images_envelope_to_blocks(out: dict, *, summary_extras: dict) -> list[dict]:
    blocks: list[dict] = []
    for entry in out.get("data", []):
        b64 = entry.get("b64_json")
        if b64:
            blocks.append(pack_image_output(
                base64.b64decode(b64), mime="image/png",
            ))
    summary = {
        "model": out.get("model"),
        "n": len(out.get("data", [])),
        **summary_extras,
    }
    blocks.append(_json_block(summary))
    return blocks


def _handle_generate_image(client: MuseClient, args: dict) -> list[dict]:
    body = _filter_keys(args, prefixes=())
    body.setdefault("response_format", "b64_json")
    out = client.generate_image(**body)
    return _images_envelope_to_blocks(out, summary_extras={
        "size": args.get("size", "1024x1024"),
    })


def _handle_edit_image(client: MuseClient, args: dict) -> list[dict]:
    image = resolve_binary_input(
        b64=args.get("image_b64"),
        url=args.get("image_url"),
        path=args.get("image_path"),
        field_name="image",
    )
    mask = resolve_binary_input(
        b64=args.get("mask_b64"),
        url=args.get("mask_url"),
        path=args.get("mask_path"),
        field_name="mask",
    )
    body = _filter_keys(args, prefixes=("image_", "mask_"))
    out = client.edit_image(image=image, mask=mask, **body)
    return _images_envelope_to_blocks(out, summary_extras={})


def _handle_vary_image(client: MuseClient, args: dict) -> list[dict]:
    image = resolve_binary_input(
        b64=args.get("image_b64"),
        url=args.get("image_url"),
        path=args.get("image_path"),
        field_name="image",
    )
    body = _filter_keys(args, prefixes=("image_",))
    out = client.vary_image(image=image, **body)
    return _images_envelope_to_blocks(out, summary_extras={})


def _handle_upscale_image(client: MuseClient, args: dict) -> list[dict]:
    image = resolve_binary_input(
        b64=args.get("image_b64"),
        url=args.get("image_url"),
        path=args.get("image_path"),
        field_name="image",
    )
    body = _filter_keys(args, prefixes=("image_",))
    out = client.upscale_image(image=image, **body)
    return _images_envelope_to_blocks(out, summary_extras={
        "scale": args.get("scale", 4),
    })


def _handle_segment_image(client: MuseClient, args: dict) -> list[dict]:
    image = resolve_binary_input(
        b64=args.get("image_b64"),
        url=args.get("image_url"),
        path=args.get("image_path"),
        field_name="image",
    )
    body = _filter_keys(args, prefixes=("image_",))
    out = client.segment_image(image=image, **body)
    # Segmentation envelopes carry masks per the codec convention; we
    # forward the full envelope as a TextContent block.
    return [_json_block(out)]


def _handle_vectorize_image(client: MuseClient, args: dict) -> list[dict]:
    image = resolve_binary_input(
        b64=args.get("image_b64"),
        url=args.get("image_url"),
        path=args.get("image_path"),
        field_name="image",
    )
    body = _filter_keys(args, prefixes=("image_",))
    body["response_format"] = "json"
    out = client.vectorize_image(image=image, **body)
    return [_json_block(out)]


def _handle_generate_animation(client: MuseClient, args: dict) -> list[dict]:
    body = _filter_keys(args, prefixes=())
    out = client.generate_animation(**body)
    return [_json_block(out)]


def _handle_embed_image(client: MuseClient, args: dict) -> list[dict]:
    body = _filter_keys(args, prefixes=())
    out = client.embed_image(**body)
    return [_json_block(out)]


# ---- tool definitions ----

INFERENCE_TOOLS.extend([
    ToolEntry(
        tool=Tool(
            name="muse_generate_image",
            description=(
                "Generate one or more images from a text prompt using "
                "a diffusion model. Use when the user asks for a new "
                "image to be created from scratch. Returns one MCP "
                "ImageContent block per image plus a JSON summary."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the desired image.",
                    },
                    "model": {"type": "string"},
                    "n": {
                        "type": "integer",
                        "default": 1, "minimum": 1, "maximum": 4,
                    },
                    "size": {
                        "type": "string",
                        "default": "1024x1024",
                        "pattern": "^[0-9]+x[0-9]+$",
                    },
                    "negative_prompt": {"type": "string"},
                    "steps": {
                        "type": "integer", "minimum": 1, "maximum": 100,
                    },
                    "guidance": {
                        "type": "number", "minimum": 0, "maximum": 20,
                    },
                    "seed": {"type": "integer"},
                },
                "required": ["prompt"],
            },
        ),
        handler=_handle_generate_image,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_edit_image",
            description=(
                "Inpaint a region of an existing image. The mask defines "
                "which pixels to regenerate (white = regenerate, black "
                "= keep). Use when the user wants targeted edits rather "
                "than a full new image. Provide image + mask + prompt."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {"type": "string"},
                    "n": {"type": "integer", "default": 1},
                    "size": {"type": "string"},
                    **binary_input_schema("image"),
                    **binary_input_schema("mask"),
                },
                "required": ["prompt"],
            },
        ),
        handler=_handle_edit_image,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_vary_image",
            description=(
                "Generate visually-similar variations of an existing "
                "image (no prompt). Use when the user wants 'something "
                "like this but different'. Provide an image."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "n": {"type": "integer", "default": 1},
                    "size": {"type": "string"},
                    **binary_input_schema("image"),
                },
            },
        ),
        handler=_handle_vary_image,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_upscale_image",
            description=(
                "Upscale an image to higher resolution using a "
                "diffusion-based super-resolution model. Use when the "
                "user wants more pixels (typically 2x or 4x). Optional "
                "prompt steers the upscaler."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "scale": {
                        "type": "integer", "default": 4, "enum": [2, 4],
                    },
                    "prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "steps": {"type": "integer"},
                    "guidance": {"type": "number"},
                    "seed": {"type": "integer"},
                    "n": {"type": "integer", "default": 1},
                    **binary_input_schema("image"),
                },
            },
        ),
        handler=_handle_upscale_image,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_segment_image",
            description=(
                "Run image segmentation (e.g. SAM2). Modes: 'auto' "
                "(automatic mask grid), 'points' (click points), "
                "'boxes' (bounding boxes), 'text' (open-vocab prompt). "
                "Returns mask data per the chosen mask_format."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "points", "boxes", "text"],
                        "default": "auto",
                    },
                    "prompt": {"type": "string"},
                    "points": {
                        "type": "string",
                        "description": (
                            "JSON-encoded list of [x, y] click points "
                            "(used in mode='points')."
                        ),
                    },
                    "boxes": {
                        "type": "string",
                        "description": (
                            "JSON-encoded list of [x0, y0, x1, y1] "
                            "boxes (used in mode='boxes')."
                        ),
                    },
                    "mask_format": {
                        "type": "string",
                        "enum": ["png_b64", "rle"],
                        "default": "png_b64",
                    },
                    "max_masks": {
                        "type": "integer", "default": 16,
                    },
                    **binary_input_schema("image"),
                },
            },
        ),
        handler=_handle_segment_image,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_vectorize_image",
            description=(
                "Convert a raster icon, logo, or diagram into editable "
                "static SVG. Returns the SVG source, dimensions, seed, "
                "and token usage. Useful before animation or diagram "
                "assembly; photographic inputs are a poor fit."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "max_new_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7680,
                        "default": 4096,
                    },
                    "temperature": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 2,
                        "default": 1,
                    },
                    "top_p": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                        "default": 0.9,
                    },
                    "num_beams": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 2,
                    },
                    "seed": {"type": "integer"},
                    **binary_input_schema("image"),
                },
            },
        ),
        handler=_handle_vectorize_image,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_generate_animation",
            description=(
                "Generate a short looping animation from a text prompt "
                "(e.g. AnimateDiff). Returns a JSON envelope with the "
                "encoded animation bytes (webp/gif/mp4) per frame."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {"type": "string"},
                    "n": {"type": "integer", "default": 1},
                    "frames": {"type": "integer"},
                    "fps": {"type": "integer"},
                    "loop": {"type": "boolean"},
                    "negative_prompt": {"type": "string"},
                    "steps": {"type": "integer"},
                    "guidance": {"type": "number"},
                    "seed": {"type": "integer"},
                    "size": {"type": "string"},
                    "image": {
                        "type": "string",
                        "description": (
                            "Optional conditioning image (data URL or "
                            "http URL). For raw bytes, use a data URL."
                        ),
                    },
                    "strength": {"type": "number"},
                    "response_format": {
                        "type": "string",
                        "enum": ["webp", "gif", "mp4", "frames_b64"],
                        "default": "webp",
                    },
                },
                "required": ["prompt"],
            },
        ),
        handler=_handle_generate_animation,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_embed_image",
            description=(
                "Compute dense vector embeddings for one or more images. "
                "Use this for image search, clustering, or "
                "cross-modal similarity. Accepts a list of data: or "
                "http(s) URLs in `input`."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "description": (
                            "Single URL string or array of URLs (data: "
                            "or http(s)://). Use a data URL for raw "
                            "bytes (data:image/png;base64,...)."
                        ),
                    },
                    "model": {"type": "string"},
                    "encoding_format": {
                        "type": "string",
                        "enum": ["float", "base64"],
                        "default": "float",
                    },
                },
                "required": ["input"],
            },
        ),
        handler=_handle_embed_image,
    ),
])
