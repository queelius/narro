"""Narro model registry.

The registry maps model IDs to loaded TTSModel instances.  The server
populates it at startup; request handlers look up models by ID.

Usage::

    from narro.models import registry

    registry.register(my_model)          # at startup
    model = registry.get("soprano-80m")  # per-request
    info  = registry.list_models()       # for /v1/models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from narro.protocol import TTSModel

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Public metadata for a registered model."""
    id: str
    sample_rate: int


class ModelRegistry:
    """Thread-safe (GIL-protected) model registry."""

    def __init__(self) -> None:
        self._models: dict[str, TTSModel] = {}
        self._default: str | None = None

    def register(self, model: TTSModel) -> None:
        """Register a model.  First registered becomes the default."""
        if not isinstance(model, TTSModel):
            raise TypeError(
                f"{type(model).__name__} does not implement TTSModel protocol"
            )
        mid = model.model_id
        self._models[mid] = model
        if self._default is None:
            self._default = mid
        logger.info("Registered model: %s (sample_rate=%d)", mid, model.sample_rate)

    def get(self, model_id: str | None = None) -> TTSModel:
        """Look up a model by ID, or return the default."""
        if model_id is None:
            model_id = self._default
        if model_id is None:
            raise RuntimeError("No models registered")
        if model_id in self._models:
            return self._models[model_id]
        available = ", ".join(sorted(self._models)) or "(none)"
        raise KeyError(f"Unknown model: {model_id!r}. Available: {available}")

    def list_models(self) -> list[ModelInfo]:
        """Return public metadata for all registered models."""
        return [
            ModelInfo(id=m.model_id, sample_rate=m.sample_rate)
            for m in self._models.values()
        ]

    @property
    def default_model_id(self) -> str | None:
        return self._default

    def set_default(self, model_id: str) -> None:
        if model_id not in self._models:
            raise KeyError(f"Cannot set default to unregistered model: {model_id!r}")
        self._default = model_id

    def clear(self) -> None:
        """Remove all models (useful for testing)."""
        self._models.clear()
        self._default = None

    def __len__(self) -> int:
        return len(self._models)


# Module-level singleton used by the server.
registry = ModelRegistry()
