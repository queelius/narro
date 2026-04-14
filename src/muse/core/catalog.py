"""Known-models catalog: what can be pulled, what's been pulled.

Structure:
    KNOWN_MODELS: dict[model_id, CatalogEntry] -- static at import time
    catalog.json (on disk): dict[model_id, {
        pulled_at,                     # ISO 8601 timestamp
        hf_repo,                       # original HF repo id
        local_dir,                     # HF snapshot_download cache path
        venv_path,                     # dedicated venv for this model
        python_path,                   # <venv_path>/bin/python for workers
    }]
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
    "all-minilm-l6-v2": CatalogEntry(
        model_id="all-minilm-l6-v2",
        modality="embeddings",
        backend_path="muse.embeddings.backends.minilm:MiniLMBackend",
        hf_repo="sentence-transformers/all-MiniLM-L6-v2",
        description="MiniLM sentence embeddings: 384 dims, 22MB, CPU-friendly",
        pip_extras=("torch>=2.1.0", "sentence-transformers>=2.2.0"),
        extra={"dimensions": 384},
    ),
    "qwen3-embedding-0.6b": CatalogEntry(
        model_id="qwen3-embedding-0.6b",
        modality="embeddings",
        backend_path="muse.embeddings.backends.qwen3_embedding:Qwen3Embedding06BBackend",
        hf_repo="Qwen/Qwen3-Embedding-0.6B",
        description="Qwen3-Embedding 0.6B: 1024 dims (matryoshka), 32K context, Apache 2.0",
        pip_extras=(
            "torch>=2.1.0",
            "sentence-transformers>=4.0.0",
            "transformers>=4.51.0",
        ),
        extra={"dimensions": 1024, "context_length": 32768, "matryoshka": True},
    ),
    "nv-embed-v2": CatalogEntry(
        model_id="nv-embed-v2",
        modality="embeddings",
        backend_path="muse.embeddings.backends.nv_embed_v2:NVEmbedV2Backend",
        hf_repo="nvidia/NV-Embed-v2",
        description=(
            "NVIDIA NV-Embed-v2: 4096 dims, 32K context, SotA MTEB "
            "(LICENSE: CC-BY-NC-4.0, non-commercial only)"
        ),
        pip_extras=(
            "torch>=2.1.0",
            "transformers>=4.42.4",
            "sentence-transformers>=2.7.0",
            "einops",
        ),
        extra={
            "dimensions": 4096,
            "context_length": 32768,
            "license": "CC-BY-NC-4.0",
        },
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
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        logger.warning("catalog at %s corrupt; resetting", p)
        return {}
    # Backfill enabled=True for pre-enable-flag entries (migration path).
    # Non-destructive: only affects the in-memory dict on read.
    for entry in data.values():
        entry.setdefault("enabled", True)
    return data


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


def _muse_repo_root() -> Path:
    """Resolve the muse source tree that contains this catalog.py.

    `__file__` is at `<repo>/src/muse/core/catalog.py`, so `parents[3]`
    is the repo root with pyproject.toml. Used to install muse itself
    (editable) into each worker venv.
    """
    return Path(__file__).resolve().parents[3]


def pull(model_id: str) -> None:
    """Create per-model venv, install muse + deps into it, download weights.

    Each pulled model gets `<MUSE_CATALOG_DIR>/venvs/<model-id>/`. The venv
    contains:
      - `muse` itself (editable, pointing at the current repo) so that
        `<venv>/bin/python -m muse.cli _worker ...` works when the
        supervisor spawns a worker subprocess.
      - The `[server]` extras (fastapi, uvicorn, sse-starlette, httpx)
        for the worker's FastAPI app.
      - The model's own `pip_extras` (torch, transformers, diffusers, etc).

    The catalog records the venv's Python path so `muse serve` can spawn
    workers with the right interpreter.
    """
    if model_id not in KNOWN_MODELS:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(KNOWN_MODELS)}")
    entry = KNOWN_MODELS[model_id]

    # Venv lives under the catalog dir so MUSE_CATALOG_DIR controls everything
    venvs_root = _catalog_dir() / "venvs"
    venv_path = venvs_root / model_id

    # Skip creation if the venv dir already exists; pip installs below
    # are (re-)run regardless to pick up repo or catalog updates.
    if not venv_path.exists():
        create_venv(venv_path)

    # Install muse itself (editable) + [server] extras so the worker
    # subprocess can `python -m muse.cli _worker`.  Editable means a
    # `git pull` in the repo is reflected in workers on next start.
    install_into_venv(venv_path, ["-e", f"{_muse_repo_root()}[server]"])

    # Install model-specific pip_extras on top
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
        "enabled": True,
    }
    _write_catalog(catalog)


def remove(model_id: str) -> None:
    """Unregister from catalog (does not delete HF cache)."""
    catalog = _read_catalog()
    catalog.pop(model_id, None)
    _write_catalog(catalog)


def is_enabled(model_id: str) -> bool:
    """Return True if model is pulled AND enabled in the catalog."""
    catalog = _read_catalog()
    if model_id not in catalog:
        return False
    return catalog[model_id].get("enabled", True)


def set_enabled(model_id: str, enabled: bool) -> None:
    """Toggle the `enabled` flag for a pulled model.

    Raises KeyError if the model is not in the catalog (not pulled).
    Other catalog fields are preserved.
    """
    catalog = _read_catalog()
    if model_id not in catalog:
        raise KeyError(f"model {model_id!r} is not pulled")
    catalog[model_id]["enabled"] = bool(enabled)
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
