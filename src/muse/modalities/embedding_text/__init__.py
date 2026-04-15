"""Embedding text modality: text-to-vector.

Wire contract: POST /v1/embeddings with {input (str | list[str]), model,
encoding_format? ('float' | 'base64'), dimensions?} returns list of
embedding vectors in OpenAI-compatible shape.

Models declaring `modality = "embedding/text"` in their MANIFEST and
satisfying the EmbeddingsModel protocol plug into this modality.
"""
from muse.modalities.embedding_text.client import EmbeddingsClient
from muse.modalities.embedding_text.protocol import (
    EmbeddingResult,
    EmbeddingsModel,
)
from muse.modalities.embedding_text.routes import build_router

MODALITY = "embedding/text"

__all__ = [
    "MODALITY",
    "build_router",
    "EmbeddingsClient",
    "EmbeddingResult",
    "EmbeddingsModel",
]
