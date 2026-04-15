"""Modality-keyed model registry.

Registry shape: {modality: {model_id: Model}}.
First model registered per modality becomes its default.
Each modality is independent — no shared protocol between audio and image models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelInfo:
    """Registry metadata for a loaded model."""
    modality: str
    model_id: str
    extra: dict = field(default_factory=dict)


class ModalityRegistry:
    """Holds loaded models grouped by modality.

    Each modality namespace has its own default. Modalities are independent:
    looking up `audio/speech` won't find models registered under `image/generation`.
    """

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] = {}
        self._defaults: dict[str, str] = {}

    def register(self, modality: str, model: Any) -> None:
        """Register a model under a modality. First registered becomes default."""
        models = self._models.setdefault(modality, {})
        models[model.model_id] = model
        self._defaults.setdefault(modality, model.model_id)

    def get(self, modality: str, model_id: str | None = None) -> Any:
        if modality not in self._models or not self._models[modality]:
            known = sorted(self._models)
            raise KeyError(
                f"no models registered for modality {modality!r}; "
                f"known modalities: {known}"
            )
        if model_id is None:
            model_id = self._defaults[modality]
        if model_id not in self._models[modality]:
            available = sorted(self._models[modality])
            raise KeyError(
                f"model {model_id!r} not registered under {modality!r}; "
                f"available: {available}"
            )
        return self._models[modality][model_id]

    def set_default(self, modality: str, model_id: str) -> None:
        if model_id not in self._models.get(modality, {}):
            available = sorted(self._models.get(modality, {}))
            raise KeyError(
                f"model {model_id!r} not registered under {modality!r}; "
                f"available: {available}"
            )
        self._defaults[modality] = model_id

    def list_models(self, modality: str) -> list[ModelInfo]:
        return [
            ModelInfo(modality=modality, model_id=mid, extra=_extra(m))
            for mid, m in self._models.get(modality, {}).items()
        ]

    def list_all(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for modality in self._models:
            out.extend(self.list_models(modality))
        return out

    def modalities(self) -> list[str]:
        return list(self._models.keys())



def _extra(model: Any) -> dict:
    """Pull commonly-exposed metadata from a model without assuming a base class."""
    extra: dict = {}
    for attr in ("sample_rate", "default_size", "voices", "description"):
        if hasattr(model, attr):
            extra[attr] = getattr(model, attr)
    return extra


# Module-level singleton. Modalities register into this at server startup.
registry = ModalityRegistry()
