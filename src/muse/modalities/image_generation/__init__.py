"""Image generation modality: text-to-image.

Wire contract: POST /v1/images/generations with {prompt, model, n?, size?,
response_format? ('b64_json' | 'url'), negative_prompt?, steps?,
guidance?, seed?} returns list of generated images in OpenAI-compatible
shape (b64_json bytes or data URL).

Models declaring `modality = "image/generation"` in their MANIFEST and
satisfying the ImageModel protocol plug into this modality.
"""
from muse.modalities.image_generation.client import GenerationsClient
from muse.modalities.image_generation.protocol import ImageModel, ImageResult
from muse.modalities.image_generation.routes import build_router

MODALITY = "image/generation"

__all__ = [
    "MODALITY",
    "build_router",
    "GenerationsClient",
    "ImageResult",
    "ImageModel",
]
