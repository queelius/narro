"""Model catalog for Narro.

Tracks known models, their metadata, and local pull state.  Model
weights are stored in HuggingFace's cache (via ``huggingface_hub``);
this module manages a lightweight manifest that records which models
the user has explicitly pulled.

Custom voices (e.g. Bark voice clones) are stored separately in
``$NARRO_HOME/voices/<model_id>/``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

def _narro_home() -> Path:
    """Return the narro config directory, creating it if needed."""
    home = Path(os.environ.get("NARRO_HOME", "~/.config/narro")).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    return home


def _catalog_path() -> Path:
    return _narro_home() / "catalog.json"


def voices_dir(model_id: str) -> Path:
    """Return the voice storage directory for a model, creating it if needed."""
    d = _narro_home() / "voices" / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Model entries (the "known models" registry)
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """Metadata for a model that Narro knows how to serve."""
    id: str
    description: str
    hf_repo: str
    backend: str          # fully qualified class: "narro.models.soprano.SopranoModel"
    sample_rate: int
    size_mb: int
    voices: list[str] = field(default_factory=list)


KNOWN_MODELS: dict[str, ModelEntry] = {
    "soprano-80m": ModelEntry(
        id="soprano-80m",
        description="Soprano 1.1 80M: lightweight English TTS (32kHz, CPU-friendly)",
        hf_repo="ekwek/Soprano-1.1-80M",
        backend="narro.models.soprano.SopranoModel",
        sample_rate=32000,
        size_mb=374,
    ),
}


# ---------------------------------------------------------------------------
# Catalog state (pulled models)
# ---------------------------------------------------------------------------

def _read_catalog() -> dict:
    path = _catalog_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_catalog(data: dict) -> None:
    path = _catalog_path()
    path.write_text(json.dumps(data, indent=2) + "\n")


def pulled_models() -> dict[str, dict]:
    """Return a dict of model_id -> pull metadata for all pulled models."""
    return _read_catalog()


def is_pulled(model_id: str) -> bool:
    return model_id in _read_catalog()


# ---------------------------------------------------------------------------
# Pull / remove
# ---------------------------------------------------------------------------

def pull(model_id: str) -> None:
    """Download a model's weights and record it as pulled.

    Uses ``huggingface_hub.snapshot_download`` so all files are cached
    in HuggingFace's standard cache directory.  This function just
    triggers the download and records the pull in catalog.json.
    """
    if model_id not in KNOWN_MODELS:
        available = ", ".join(sorted(KNOWN_MODELS))
        raise ValueError(f"Unknown model: {model_id!r}. Available: {available}")

    entry = KNOWN_MODELS[model_id]
    logger.info("Pulling %s from %s (~%d MB)...", model_id, entry.hf_repo, entry.size_mb)

    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=entry.hf_repo)

    catalog = _read_catalog()
    catalog[model_id] = {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "hf_repo": entry.hf_repo,
    }
    _write_catalog(catalog)
    logger.info("Pulled %s successfully.", model_id)


def remove(model_id: str) -> None:
    """Remove a model from the catalog (does not delete cached weights)."""
    catalog = _read_catalog()
    if model_id not in catalog:
        raise KeyError(f"Model {model_id!r} is not pulled.")
    del catalog[model_id]
    _write_catalog(catalog)
    logger.info("Removed %s from catalog.", model_id)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_backend(model_id: str, **kwargs):
    """Instantiate a model backend from a catalog entry.

    The model must be pulled first.  Keyword arguments (device, compile,
    quantize, etc.) are forwarded to the backend constructor.
    """
    if model_id not in KNOWN_MODELS:
        available = ", ".join(sorted(KNOWN_MODELS))
        raise ValueError(f"Unknown model: {model_id!r}. Available: {available}")
    if not is_pulled(model_id):
        raise RuntimeError(
            f"Model {model_id!r} is not pulled. Run: narro models pull {model_id}"
        )

    entry = KNOWN_MODELS[model_id]
    module_path, cls_name = entry.backend.rsplit(".", 1)

    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls(**kwargs)
