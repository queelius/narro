"""Muse embeddings modality protocol.

Defines EmbeddingsModel (backend contract) and EmbeddingResult
(synthesis return). The modality is flat (no family.op) because
OpenAI's endpoint is /v1/embeddings (single level).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class EmbeddingResult:
    """N input texts in, N embedding vectors out, plus provenance.

    `embeddings` is a plain list[list[float]] at the protocol boundary
    so no numpy dep is needed by consumers. Backends may use numpy
    internally and convert via `.tolist()` before returning.
    """
    embeddings: list[list[float]]
    dimensions: int
    model_id: str
    prompt_tokens: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class EmbeddingsModel(Protocol):
    """Protocol for text-to-embedding backends."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(
        self,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        **kwargs,
    ) -> EmbeddingResult: ...
