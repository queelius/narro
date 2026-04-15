"""Modality-keyed model registry.

Registry shape: {modality: {model_id: Model}}.
First model registered per modality becomes its default.
Each modality is independent; no shared protocol between audio and image models.

Registration carries a MANIFEST dict (the one declared in the model's
script) that the server surfaces via /v1/models. Capabilities,
description, license, and hf_repo flow straight from the manifest to the
API response; nothing is gathered by hardcoded attribute allowlist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelInfo:
    """Registry metadata for a loaded model.

    `manifest` holds the full MANIFEST dict from the model's script
    (or an empty dict for tests / hand-registered models that skip it).
    Replaces the old `extra` field populated by a hardcoded attribute
    allowlist.
    """
    modality: str
    model_id: str
    manifest: dict = field(default_factory=dict)


class ModalityRegistry:
    """Holds loaded models grouped by modality.

    Each modality namespace has its own default. Modalities are independent:
    looking up `audio/speech` won't find models registered under `image/generation`.
    """

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] = {}
        self._manifests: dict[str, dict[str, dict]] = {}
        self._defaults: dict[str, str] = {}

    def register(
        self,
        modality: str,
        model: Any,
        manifest: dict | None = None,
    ) -> None:
        """Register a model under a modality.

        First registered becomes default. `manifest` is stored verbatim
        and surfaced via list_models / list_all. When omitted a minimal
        stub `{"model_id", "modality"}` is used so ModelInfo consumers
        always see the two required keys.
        """
        models = self._models.setdefault(modality, {})
        models[model.model_id] = model
        self._manifests.setdefault(modality, {})[model.model_id] = (
            manifest if manifest is not None else {
                "model_id": model.model_id,
                "modality": modality,
            }
        )
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
            ModelInfo(
                modality=modality,
                model_id=mid,
                manifest=self._manifests.get(modality, {}).get(mid, {}),
            )
            for mid in self._models.get(modality, {})
        ]

    def list_all(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for modality in self._models:
            out.extend(self.list_models(modality))
        return out

    def modalities(self) -> list[str]:
        return list(self._models.keys())


# Module-level singleton. Modalities register into this at server startup.
registry = ModalityRegistry()
