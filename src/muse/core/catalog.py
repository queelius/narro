"""Known-models catalog: what can be pulled, what's been pulled.

The set of known models is not hardcoded. It is discovered at first
access by scanning `src/muse/models/*.py` for scripts that define a
top-level `MANIFEST` dict and a `Model` class (see `muse.core.discovery`).
Each MANIFEST's fields are projected onto the stable `CatalogEntry`
shape that the rest of muse (CLI, server, worker) consumes.

Structure:
    known_models() -> dict[model_id, CatalogEntry]  # cached, discovery-driven
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

from muse.core.discovery import DiscoveredModel, discover_models
from muse.core.install import check_system_packages, install_pip_extras
from muse.core.venv import create_venv, install_into_venv, venv_python

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogEntry:
    """Stable catalog shape derived from a model script's MANIFEST."""
    model_id: str
    modality: str              # MIME-style: "audio/speech", "embedding/text", etc.
    backend_path: str          # "module.path:ClassName"
    hf_repo: str
    description: str = ""
    pip_extras: tuple[str, ...] = ()
    system_packages: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)  # voices, default_size, capabilities


def _bundled_models_dir() -> Path:
    """Path to the in-repo `src/muse/models/` directory."""
    # catalog.py sits at src/muse/core/catalog.py; parents[1] is src/muse/.
    return Path(__file__).resolve().parents[1] / "models"


def _model_dirs() -> list[Path]:
    """Scan order for model discovery (bundled first).

    Task F1 will extend this with `~/.muse/models/` and `$MUSE_MODELS_DIR`.
    For now, only the bundled dir is scanned.
    """
    return [_bundled_models_dir()]


def _manifest_to_catalog_entry(discovered: DiscoveredModel) -> CatalogEntry:
    """Project a DiscoveredModel onto the CatalogEntry shape.

    Manifest -> CatalogEntry mapping:
        model_id        -> model_id                 (required)
        modality        -> modality                 (required)
        hf_repo         -> hf_repo                  (required)
        description     -> description              (optional, defaults "")
        pip_extras      -> pip_extras               (tuple-coerced, defaults ())
        system_packages -> system_packages          (tuple-coerced, defaults ())
        capabilities    -> extra                    (dict-copied, defaults {})
    backend_path is synthesized from the Model class's module and name.
    """
    m = discovered.manifest
    cls = discovered.model_class
    return CatalogEntry(
        model_id=m["model_id"],
        modality=m["modality"],
        backend_path=f"{cls.__module__}:{cls.__name__}",
        hf_repo=m["hf_repo"],
        description=m.get("description", ""),
        pip_extras=tuple(m.get("pip_extras", ())),
        system_packages=tuple(m.get("system_packages", ())),
        extra=dict(m.get("capabilities", {})),
    )


_known_models_cache: dict[str, CatalogEntry] | None = None


def known_models() -> dict[str, CatalogEntry]:
    """Return {model_id: CatalogEntry} for every discovered model.

    Runs `discover_models` over the configured dirs on first call and
    caches the result. Restart the process to pick up new scripts.
    """
    global _known_models_cache
    if _known_models_cache is None:
        discovered = discover_models(_model_dirs())
        _known_models_cache = {
            model_id: _manifest_to_catalog_entry(d)
            for model_id, d in discovered.items()
        }
    return _known_models_cache


def _reset_known_models_cache() -> None:
    """Test hook: clear the cache so discovery re-runs on next call."""
    global _known_models_cache
    _known_models_cache = None


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
    entries = list(known_models().values())
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
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    entry = catalog_known[model_id]

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
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    if not is_pulled(model_id):
        raise RuntimeError(f"model {model_id!r} not pulled; run `muse pull {model_id}`")
    entry = catalog_known[model_id]
    module_path, class_name = entry.backend_path.split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    catalog = _read_catalog()
    local_dir = catalog[model_id]["local_dir"]
    return cls(hf_repo=entry.hf_repo, local_dir=local_dir, **kwargs)


def get_manifest(model_id: str) -> dict:
    """Return the MANIFEST dict from a known model's script.

    Looks up the model's module via its CatalogEntry.backend_path, then
    reads the module-level MANIFEST. Returns a copy so callers can mutate
    without affecting the source.

    Raises KeyError if the model is not in `known_models()`.
    """
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    entry = catalog_known[model_id]
    module_path, _ = entry.backend_path.split(":", 1)
    module = importlib.import_module(module_path)
    return dict(getattr(module, "MANIFEST", {}))
