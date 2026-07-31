"""Wire encoding for image/vectorization results."""
from __future__ import annotations

import time
import uuid
from typing import Any

from muse.modalities.image_vectorization.protocol import VectorizationResult


def encode_vectorization(result: VectorizationResult) -> dict[str, Any]:
    """Build the JSON response envelope for ``/v1/images/vectorize``."""
    return {
        "id": f"vec-{uuid.uuid4().hex}",
        "object": "image.vectorization",
        "created": int(time.time()),
        "model": result.model_id,
        "mime_type": "image/svg+xml",
        "svg": result.svg,
        "source_size": [result.source_width, result.source_height],
        "width": result.width,
        "height": result.height,
        "view_box": list(result.view_box) if result.view_box is not None else None,
        "seed": result.seed,
        "usage": {"completion_tokens": result.completion_tokens},
        "metadata": result.metadata,
    }
