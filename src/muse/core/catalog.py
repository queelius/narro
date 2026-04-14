"""Known-models catalog: what can be pulled, what's been pulled.

Structure:
    KNOWN_MODELS: dict[model_id, CatalogEntry]  — static at import time
    catalog.json (on disk): dict[model_id, {pulled_at, hf_repo, local_dir}]
"""
from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from muse.core.install import check_system_packages, install_pip_extras
from muse.core.venv import create_venv, install_into_venv, venv_python

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogEntry:
    """Static metadata for a known model."""
    model_id: str
    modality: str              # "audio.speech" | "images.generations"
    backend_path: str          # "module.path:ClassName"
    hf_repo: str
    description: str = ""
    pip_extras: tuple[str, ...] = ()
    system_packages: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)  # voices, default_size, etc.


# Seeded with representative models. Expand as new backends land.
KNOWN_MODELS: dict[str, CatalogEntry] = {
    "soprano-80m": CatalogEntry(
        model_id="soprano-80m",
        modality="audio.speech",
        backend_path="muse.audio.speech.backends.soprano:SopranoModel",
        hf_repo="ekwek/Soprano-1.1-80M",
        description="Qwen3 LLM backbone + Vocos decoder, 32kHz, 80M params",
        pip_extras=("transformers>=4.36.0", "scipy", "inflect", "unidecode"),
    ),
    "kokoro-82m": CatalogEntry(
        model_id="kokoro-82m",
        modality="audio.speech",
        backend_path="muse.audio.speech.backends.kokoro:KokoroModel",
        hf_repo="hexgrad/Kokoro-82M",
        description="Lightweight TTS, 54 voices, 24kHz",
        pip_extras=("kokoro", "soundfile", "misaki[en]"),
        system_packages=("espeak-ng",),
    ),
    "bark-small": CatalogEntry(
        model_id="bark-small",
        modality="audio.speech",
        backend_path="muse.audio.speech.backends.bark:BarkModel",
        hf_repo="suno/bark-small",
        description="Multilingual + voice cloning, 24kHz",
        pip_extras=("transformers>=4.36.0", "scipy"),
    ),
    "sd-turbo": CatalogEntry(
        model_id="sd-turbo",
        modality="images.generations",
        backend_path="muse.images.generations.backends.sd_turbo:SDTurboModel",
        hf_repo="stabilityai/sd-turbo",
        description="Stable Diffusion Turbo: 1-step distilled, 512x512",
        pip_extras=("diffusers>=0.27.0", "accelerate", "Pillow", "safetensors"),
        extra={"default_size": (512, 512)},
    ),
}


def _catalog_dir() -> Path:
    env = os.environ.get("MUSE_CATALOG_DIR")
    if env:
        return Path(env)
    return Path.home() / ".muse"


def _catalog_path() -> Path:
    return _catalog_dir() / "catalog.json"


def _read_catalog() -> dict:
    p = _catalog_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        logger.warning("catalog at %s corrupt; resetting", p)
        return {}


def _write_catalog(data: dict) -> None:
    """Atomic write: write to .tmp in same dir, then rename.

    Rename within the same filesystem is atomic on POSIX and near-atomic
    on Windows (Python 3.3+ Path.replace wraps MoveFileEx with REPLACE_EXISTING).
    """
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def is_pulled(model_id: str) -> bool:
    return model_id in _read_catalog()


def list_known(modality: str | None = None) -> list[CatalogEntry]:
    entries = list(KNOWN_MODELS.values())
    if modality is None:
        return entries
    return [e for e in entries if e.modality == modality]


def pull(model_id: str) -> None:
    """Create per-model venv, install deps into it, download weights, record state.

    Each pulled model gets `<MUSE_CATALOG_DIR>/venvs/<model-id>/` with its
    `pip_extras` installed inside. The catalog records the venv's Python path
    so `muse serve` can spawn workers with the right interpreter.
    """
    if model_id not in KNOWN_MODELS:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(KNOWN_MODELS)}")
    entry = KNOWN_MODELS[model_id]

    # Venv lives under the catalog dir so MUSE_CATALOG_DIR controls everything
    venvs_root = _catalog_dir() / "venvs"
    venv_path = venvs_root / model_id

    # Idempotent: if the venv already exists, create_venv reuses it
    if not venv_path.exists():
        create_venv(venv_path)

    # Install pip_extras INTO the venv, not the supervisor's env
    if entry.pip_extras:
        install_into_venv(venv_path, list(entry.pip_extras))

    if entry.system_packages:
        missing = check_system_packages(list(entry.system_packages))
        if missing:
            logger.warning(
                "model %s needs system packages not found on PATH: %s "
                "(install via apt/brew before running)",
                model_id, missing,
            )

    local_dir = snapshot_download(repo_id=entry.hf_repo)

    catalog = _read_catalog()
    catalog[model_id] = {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "hf_repo": entry.hf_repo,
        "local_dir": str(local_dir),
        "venv_path": str(venv_path),
        "python_path": str(venv_python(venv_path)),
    }
    _write_catalog(catalog)


def remove(model_id: str) -> None:
    """Unregister from catalog (does not delete HF cache)."""
    catalog = _read_catalog()
    catalog.pop(model_id, None)
    _write_catalog(catalog)


def load_backend(model_id: str, **kwargs) -> Any:
    """Import backend class and instantiate it.

    `backend_path` has the form "package.module:ClassName". The class
    is expected to accept (hf_repo, local_dir, **kwargs) in its constructor.
    """
    if model_id not in KNOWN_MODELS:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(KNOWN_MODELS)}")
    if not is_pulled(model_id):
        raise RuntimeError(f"model {model_id!r} not pulled; run `muse pull {model_id}`")
    entry = KNOWN_MODELS[model_id]
    module_path, class_name = entry.backend_path.split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    catalog = _read_catalog()
    local_dir = catalog[model_id]["local_dir"]
    return cls(hf_repo=entry.hf_repo, local_dir=local_dir, **kwargs)
