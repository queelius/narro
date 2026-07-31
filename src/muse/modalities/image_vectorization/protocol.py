"""Protocol and result types for image/vectorization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VectorizationResult:
    """One static SVG reconstructed from a raster image."""

    svg: str
    model_id: str
    source_width: int
    source_height: int
    completion_tokens: int = 0
    seed: int = -1
    width: float | None = None
    height: float | None = None
    view_box: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ImageVectorizationModel(Protocol):
    """Structural contract implemented by vectorization backends."""

    model_id: str

    def vectorize(
        self,
        image: Any,
        *,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.9,
        num_beams: int = 2,
        seed: int | None = None,
    ) -> VectorizationResult:
        """Convert one PIL image to a validated, static SVG."""
        ...


class VectorizationOutputError(ValueError):
    """The backend generated malformed or unsafe SVG."""
