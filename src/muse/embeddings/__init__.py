"""Muse embeddings modality: text-to-vector.

Backends implementing the EmbeddingsModel protocol return
EmbeddingResult dataclasses. The /v1/embeddings router serializes
these to OpenAI-compatible JSON (with optional base64 encoding).
"""
from muse.embeddings.client import EmbeddingsClient
from muse.embeddings.protocol import EmbeddingsModel, EmbeddingResult

__all__ = ["EmbeddingsClient", "EmbeddingsModel", "EmbeddingResult"]
