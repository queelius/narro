"""Muse images.generations modality protocol.

Defines ImageModel (backend contract) and ImageResult (synthesis return).
No streaming type — diffusion progress is per-step refinement of the
same image, not time-ordered chunks like audio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ImageResult:
    """A synthesized image plus provenance metadata.

    `image` is typed as Any so backends can return PIL.Image, numpy arrays,
    or torch tensors without forcing a common supertype here. Codec-layer
    code is responsible for normalizing to PIL before encoding.
    """
    image: Any
    width: int
    height: int
    seed: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ImageModel(Protocol):
    """Protocol for text-to-image backends."""

    @property
    def model_id(self) -> str: ...

    @property
    def default_size(self) -> tuple[int, int]: ...

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> ImageResult: ...
