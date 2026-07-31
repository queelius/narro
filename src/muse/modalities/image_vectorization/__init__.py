"""Raster-to-SVG vectorization.

Wire contract: ``POST /v1/images/vectorize`` with a multipart ``image``
and optional generation controls. JSON output includes the static SVG,
geometry, seed, and token usage; ``response_format=svg`` returns
``image/svg+xml`` directly.

The bundled StarVector checkpoint is intentionally exact-only. Muse
does not register a general Hugging Face resolver for this modality
because StarVector-family repositories use custom executable model code.
"""
from muse.modalities.image_vectorization.client import VectorizationClient
from muse.modalities.image_vectorization.protocol import (
    ImageVectorizationModel,
    VectorizationOutputError,
    VectorizationResult,
)
from muse.modalities.image_vectorization.routes import build_router


MODALITY = "image/vectorization"
MODEL_OPTIONAL_PATHS = ("/v1/images/vectorize",)


def _probe_call(model):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((12, 12, 52, 52), fill="#3867d6")
    return model.vectorize(
        image,
        max_new_tokens=512,
        temperature=0.0,
        num_beams=1,
        seed=0,
    )


PROBE_DEFAULTS = {
    "shape": "1 small 64x64 icon, greedy, max_new_tokens=512",
    "call": _probe_call,
}


__all__ = [
    "MODALITY",
    "MODEL_OPTIONAL_PATHS",
    "PROBE_DEFAULTS",
    "build_router",
    "ImageVectorizationModel",
    "VectorizationClient",
    "VectorizationOutputError",
    "VectorizationResult",
]
