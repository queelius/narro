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

from muse.core.curated import find_curated
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


def _user_models_dir() -> Path:
    """Path to the per-user `~/.muse/models/` drop-in directory.

    Users can drop `.py` model scripts here to add backends without
    modifying the muse source tree. Resolves via `Path.home()` so
    monkeypatching `$HOME` in tests Just Works.
    """
    return Path.home() / ".muse" / "models"


def _env_models_dir() -> Path | None:
    """Optional extra models dir from the `$MUSE_MODELS_DIR` env var."""
    env = os.environ.get("MUSE_MODELS_DIR")
    return Path(env) if env else None


def _model_dirs() -> list[Path]:
    """Scan order for model discovery: bundled, then user dir, then env.

    First-found-wins on model_id collision, so bundled models shadow
    user and env entries with the same id. This is intentional: users
    cannot silently replace a bundled model by dropping a script with
    the same id. To override, rename or remove the bundled script.
    """
    dirs = [_bundled_models_dir(), _user_models_dir()]
    env = _env_models_dir()
    if env is not None:
        dirs.append(env)
    return dirs


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


def _persisted_manifest_to_catalog_entry(manifest: dict) -> CatalogEntry:
    """Project a catalog-persisted manifest dict onto the CatalogEntry shape.

    Resolver-pulled models persist their full synthesized MANIFEST inside
    catalog.json (under the `manifest` key) so that `known_models()` can
    surface them without rerunning discovery. The persisted dict carries
    `backend_path` directly (it was synthesized from the resolver's
    runtime class path), unlike script-discovered models where backend_path
    is computed from the Model class's `__module__:__name__`.
    """
    return CatalogEntry(
        model_id=manifest["model_id"],
        modality=manifest["modality"],
        backend_path=manifest["backend_path"],
        hf_repo=manifest["hf_repo"],
        description=manifest.get("description", ""),
        pip_extras=tuple(manifest.get("pip_extras", ())),
        system_packages=tuple(manifest.get("system_packages", ())),
        extra=dict(manifest.get("capabilities", {})),
    )


def known_models() -> dict[str, CatalogEntry]:
    """Return {model_id: CatalogEntry} for every discovered model.

    Two sources are merged:
      1. `discover_models` over the configured dirs (script-based models,
         bundled or user-dropped).
      2. catalog.json entries with a `manifest` field (resolver-pulled
         models persisted by Task F2's `_pull_via_resolver`).

    Bundled / discovered scripts win on model_id collision: a user
    cannot silently shadow a script by pulling a same-id resolver
    entry. The persisted entry is dropped from the merge with a
    debug log; the resolver entry can still be removed via
    `muse models remove`.

    Cached on first call; restart the process to pick up new scripts
    or new resolver pulls.
    """
    global _known_models_cache
    if _known_models_cache is None:
        discovered = discover_models(_model_dirs())
        entries = {
            model_id: _manifest_to_catalog_entry(d)
            for model_id, d in discovered.items()
        }
        catalog = _read_catalog()
        for model_id, entry_data in catalog.items():
            manifest = entry_data.get("manifest")
            if not manifest:
                # Legacy entry: pulled via the bare-id path; the
                # corresponding script's discovery already populated
                # `entries`. Nothing to merge.
                continue
            if model_id in entries:
                logger.debug(
                    "skipping persisted manifest for %s: shadowed by bundled script",
                    model_id,
                )
                continue
            entries[model_id] = _persisted_manifest_to_catalog_entry(manifest)
        _known_models_cache = entries
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


def pull(identifier: str) -> None:
    """Pull a model. Dispatch by identifier shape, with curated alias resolution.

    Resolution order:
      1. Curated alias (e.g. "qwen3-8b-q4" from src/muse/curated.yaml):
         expands to the underlying URI or bundled id. The curated id is
         preserved as the catalog key, so the user sees the friendly id
         in `muse models list` rather than a synthesized resolver one.
      2. Resolver URI (contains "://", e.g. "hf://Qwen/Qwen3-8B-GGUF@q4_k_m"):
         routed to the matching resolver, which synthesizes a manifest.
      3. Bare model_id (e.g. "kokoro-82m"): looked up in `known_models()`
         and pulled via the bundled-script path.

    All paths create a per-model venv at `<MUSE_CATALOG_DIR>/venvs/<id>/`,
    install muse[server] (editable) + pip_extras, fetch weights, and
    record the venv's Python path so `muse serve` can spawn workers
    with the right interpreter.
    """
    curated = find_curated(identifier)
    if curated is not None:
        if curated.uri:
            # Resolver-pulled curated entry. Override the synthesized id
            # so the catalog stores the friendly curated id (e.g.
            # qwen3-8b-q4) instead of qwen3-8b-instruct-gguf-q4-k-m.
            _pull_via_resolver(curated.uri, model_id_override=curated.id)
            return
        # Bundled curated entry: id equals an existing bundled script's
        # model_id. Fall through to the bundled path with that id.
        _pull_bundled(curated.id)
        return

    if "://" in identifier:
        _pull_via_resolver(identifier)
    else:
        _pull_bundled(identifier)


def _pull_bundled(model_id: str) -> None:
    """Pull a bundled (script-discovered) model by bare id."""
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    entry = catalog_known[model_id]

    venvs_root = _catalog_dir() / "venvs"
    venv_path = venvs_root / model_id

    if not venv_path.exists():
        create_venv(venv_path)

    install_into_venv(venv_path, ["-e", f"{_muse_repo_root()}[server]"])

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
    _reset_known_models_cache()


def _pull_via_resolver(uri: str, *, model_id_override: str | None = None) -> None:
    """Pull a model via a resolver URI (e.g. hf://Qwen/Qwen3-8B-GGUF@q4_k_m).

    Looks up the resolver for the URI's scheme, calls `resolve(uri)` to
    get a synthesized ResolvedModel (manifest + backend_path + download
    callable), creates the per-model venv, installs deps, downloads the
    weights via `resolved.download()`, persists the synthesized manifest
    plus a `source: <uri>` field into catalog.json, and invalidates the
    known_models cache so the next call sees the new entry.

    `model_id_override` is set when the URI was reached via a curated
    alias (e.g. user typed `qwen3-8b-q4` which expands to
    `hf://Qwen/Qwen3-8B-Instruct-GGUF@q4_k_m`). The override replaces
    the resolver's synthesized model_id so the catalog stores the
    friendly curated id.
    """
    from muse.core.resolvers import resolve

    resolved = resolve(uri)
    manifest = dict(resolved.manifest)
    # Resolver may put backend_path in the manifest itself, or only on
    # the ResolvedModel. Persist it consistently so load_backend can
    # find it without consulting the resolver again.
    manifest.setdefault("backend_path", resolved.backend_path)

    if model_id_override:
        manifest["model_id"] = model_id_override
    model_id = manifest["model_id"]

    venvs_root = _catalog_dir() / "venvs"
    venv_path = venvs_root / model_id

    if not venv_path.exists():
        create_venv(venv_path)

    install_into_venv(venv_path, ["-e", f"{_muse_repo_root()}[server]"])

    pip_extras = manifest.get("pip_extras") or ()
    if pip_extras:
        install_into_venv(venv_path, list(pip_extras))

    system_packages = manifest.get("system_packages") or ()
    if system_packages:
        missing = check_system_packages(list(system_packages))
        if missing:
            logger.warning(
                "model %s needs system packages not found on PATH: %s "
                "(install via apt/brew before running)",
                model_id, missing,
            )

    weights_cache = _catalog_dir() / "weights"
    weights_cache.mkdir(parents=True, exist_ok=True)
    local_dir = resolved.download(weights_cache)

    catalog = _read_catalog()
    catalog[model_id] = {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "hf_repo": manifest["hf_repo"],
        "local_dir": str(local_dir),
        "venv_path": str(venv_path),
        "python_path": str(venv_python(venv_path)),
        "enabled": True,
        "source": uri,
        "manifest": manifest,
    }
    _write_catalog(catalog)
    _reset_known_models_cache()


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

    For resolver-pulled models, manifest.capabilities are merged into the
    kwargs (caller's explicit kwargs win). This lets generic runtimes
    like LlamaCppModel pull `gguf_file`, `chat_template`, `context_length`
    out of the persisted manifest without the worker having to know
    those keys exist. `model_id` is also injected so generic runtimes
    (one class, many models) know which model they're loading.
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
    persisted_manifest = catalog[model_id].get("manifest") or {}
    capabilities = persisted_manifest.get("capabilities") or {}
    merged: dict = {"model_id": model_id, **capabilities, **kwargs}
    return cls(hf_repo=entry.hf_repo, local_dir=local_dir, **merged)


def get_manifest(model_id: str) -> dict:
    """Return the MANIFEST dict for a known model.

    Two sources, in order of preference:
      1. catalog.json's persisted manifest (resolver-pulled models). The
         resolver synthesized this dict at pull time; it's the source of
         truth for that entry.
      2. The model script's module-level MANIFEST (bundled scripts).

    Returns a copy so callers can mutate without affecting the source.

    Raises KeyError if the model is not in `known_models()`.
    """
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    catalog = _read_catalog()
    persisted = catalog.get(model_id, {}).get("manifest")
    if persisted:
        return dict(persisted)
    entry = catalog_known[model_id]
    module_path, _ = entry.backend_path.split(":", 1)
    module = importlib.import_module(module_path)
    return dict(getattr(module, "MANIFEST", {}))
