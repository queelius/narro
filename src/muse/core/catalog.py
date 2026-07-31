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
import contextlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from muse.core import config
from muse.core.curated import find_curated, find_curated_by_uri
from muse.core.discovery import DiscoveredModel, discover_models
from muse.core.install import check_system_packages
from muse.core.venv import (
    _is_verbose,
    create_venv,
    install_into_venv,
    venv_python,
)


@contextlib.contextmanager
def _hf_quiet_if_needed():
    """Suppress huggingface_hub tqdm progress bars when in quiet mode.

    The bars are useful when a 1GB+ download is in progress and the
    user is staring at the terminal, but during `muse pull <id>` they
    interleave with subsequent stages and add noise. Honor the
    `install_output_mode(verbose=...)` flag set by the CLI: quiet =
    bars off; verbose = bars stay on.

    The `HF_HUB_DISABLE_PROGRESS_BARS` env var is read by
    huggingface_hub at module import time, NOT per call, so setting
    it inside this context manager has no effect after huggingface_hub
    is already loaded. Use the runtime API
    `huggingface_hub.utils.disable_progress_bars()` /
    `enable_progress_bars()` instead; both are idempotent and cheap.
    Fall back to the env var only if the runtime API isn't available
    (very old huggingface_hub versions, very rare in practice).
    """
    if _is_verbose():
        yield
        return
    try:
        from huggingface_hub.utils import (
            disable_progress_bars,
            enable_progress_bars,
        )
        disable_progress_bars()
        try:
            yield
        finally:
            # Restore default. Note: this re-enables globally; if a
            # caller above us had already disabled bars for their own
            # reasons, they get re-enabled. Acceptable since the only
            # in-tree disabler is this context manager itself.
            enable_progress_bars()
    except ImportError:
        # Old huggingface_hub: fall back to the env var. It only
        # works if the import happens after we set it; not great,
        # but better than nothing.
        prev = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev

logger = logging.getLogger(__name__)


class CatalogError(RuntimeError):
    """Raised when catalog.json is corrupt and no cached good copy exists.

    A corrupt/truncated catalog.json must never be treated as an empty-
    but-valid catalog: every consumer (get_manifest, known_models,
    /v1/models, is_pulled) would then silently behave as "no models
    installed" -- 404-ing requests and emptying listings -- with no
    signal that the catalog file itself needs attention. When a prior
    good read is cached, `_read_catalog` serves that instead of raising;
    this error is reserved for the case where there is nothing good to
    fall back to.
    """


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
    env = config.get("paths.models_dir")
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


# Discovery projection cache: the importlib-driven scan of model script
# dirs. Built once per process (script imports execute module bodies, and
# bundled/user scripts do not change under a running server); reset via
# _reset_known_models_cache().
_discovered_entries_cache: dict[str, CatalogEntry] | None = None

# Merged known-models cache, keyed by the catalog.json identity
# (path_str, mtime_ns) it was built against. A catalog write from ANY
# process -- the admin pull endpoint's `muse pull` subprocess, an operator
# running `muse pull` / `muse models remove` in a shell beside a running
# supervisor -- bumps the file's mtime, so the next known_models() call
# rebuilds the merge instead of serving a frozen snapshot. Without this,
# the supervisor 404'd "unknown model" on enable/route for anything pulled
# after its cache was built, even though catalog.json and /v1/models both
# showed the entry. _MISSING_CATALOG_KEY keeps the no-catalog-file state
# cacheable (fresh install: nothing to merge, stable until a file appears).
_known_models_cache: tuple[tuple[str, int], dict[str, CatalogEntry]] | None = None
_MISSING_CATALOG_KEY = ("<no-catalog>", -2)

# H5: guards the check-then-populate sequence in known_models() so that
# two threads racing on the first call do not both run discover_models()
# (which does importlib imports -- double-executing user script module
# bodies) and both write the cache. Double-checked locking: outer
# check outside the lock for the common hot path; inner check under the
# lock only when the outer miss suggests a first-time build is needed.
# Lock ordering: _KNOWN_MODELS_LOCK is always acquired BEFORE
# _CATALOG_WRITE_LOCK (see M1 fix). Never hold _CATALOG_WRITE_LOCK when
# acquiring _KNOWN_MODELS_LOCK.
_KNOWN_MODELS_LOCK: threading.Lock = threading.Lock()

# M1: shared lock for ALL catalog read-modify-write sequences.
#
# The atomic write-then-rename in _write_catalog prevents file corruption
# but does NOT prevent lost updates: if thread A reads, thread B reads,
# thread B writes, then thread A writes, B's update is silently erased.
#
# Sites that previously each did their own _read_catalog -> mutate ->
# _write_catalog without coordination:
#   - probe.py (_write_probe_results / run_probe)
#   - admin/operations.py (disable_model's set_enabled + any future
#     RMW that does not already go through catalog helpers)
#   - load_director.py (_observed_peak_writeback, previously guarded by
#     its own _WRITEBACK_LOCK)
#
# All callers must hold _CATALOG_WRITE_LOCK for the full
# read -> mutate -> write sequence. Do NOT hold state.lock or
# _KNOWN_MODELS_LOCK when acquiring _CATALOG_WRITE_LOCK; that would
# invert the documented acquisition order (state.lock -> catalog lock).
#
# Lock ordering (always acquire in this order to prevent deadlocks):
#   1. _KNOWN_MODELS_LOCK  (only when building the known_models cache)
#   2. _CATALOG_WRITE_LOCK (when doing a catalog RMW)
#   state.lock and _CATALOG_WRITE_LOCK are NEVER held simultaneously;
#   the plan-under-state.lock then execute-outside-lock discipline in
#   supervisor and admin operations already keeps catalog writes outside
#   state.lock.
#
# load_director._WRITEBACK_LOCK is an alias to this lock (set when that
# module imports from here) so existing code using the old name keeps
# working without changes.
_CATALOG_WRITE_LOCK: threading.Lock = threading.Lock()


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


_MODEL_ID_CHARSET_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_model_id_for_fs(model_id: str) -> None:
    """Reject a model_id that is unsafe as a filesystem path component.

    model_id becomes `<catalog_dir>/venvs/<model_id>` (and, for
    resolver-pulled models, feeds into the weights cache path too).
    Applied at every pull entry point -- the bare-id branch of pull(),
    the finalized model_id in `_pull_via_resolver`, and the top of
    `_pull_bundled` -- covering every identifier shape (bare id,
    curated alias resolution result, resolver-synthesized id) BEFORE
    any venv/weights path is constructed. A hostile or malformed
    identifier (path traversal via "..", a path separator via "/")
    can therefore never escape the muse-owned catalog directories.
    """
    if model_id in (".", ".."):
        raise ValueError(
            f"invalid model id {model_id!r}: must not be '.' or '..'"
        )
    if not _MODEL_ID_CHARSET_RE.match(model_id):
        raise ValueError(
            f"invalid model id {model_id!r}: must match "
            f"{_MODEL_ID_CHARSET_RE.pattern!r} "
            f"(letters, digits, '.', '_', '-' only)"
        )


def _apply_manifest_overlays(model_id: str, manifest: dict, entry_data: dict) -> dict:
    """Apply the curated-capabilities overlay, then base_override, onto a
    persisted resolver manifest. Returns a fresh dict (never mutates the
    input).

    Shared by `known_models()` (which feeds `load_backend` via
    `entry.extra`) and `get_manifest()` (which `worker.py` uses to
    register/gate the model), so a curated.yaml edit or an operator
    `--base` pin can never diverge between what CONSTRUCTS the backend
    and what ADVERTISES/GATES it. Before this helper existed, only
    known_models() re-applied the curated overlay; get_manifest()
    returned the raw persisted manifest, so a curated.yaml capability
    edit + restart left the two read paths disagreeing.

    Order: curated overlay first (curated.yaml is hand-edited and may
    postdate the persisted manifest, so it wins on capability
    collision), then base_override (an operator `--base` pin, applied
    AFTER curated so it wins over both the tag-declared base and a
    curated base_model -- mirrors `device_override`).
    """
    manifest = dict(manifest)

    curated = find_curated(model_id)
    if curated is None:
        source = entry_data.get("source")
        if source:
            curated = find_curated_by_uri(source)
    if curated is not None and curated.capabilities:
        merged_caps = dict(manifest.get("capabilities") or {})
        merged_caps.update(curated.capabilities)
        manifest["capabilities"] = merged_caps

    base_override = entry_data.get("base_override")
    if base_override:
        merged_caps = dict(manifest.get("capabilities") or {})
        merged_caps["base_model"] = base_override
        manifest["capabilities"] = merged_caps

    return manifest


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

    Caching is two-tier. The discovery scan (importlib over script dirs)
    is cached for the process lifetime: new SCRIPTS still require a
    restart. The merged result is memoized against catalog.json's
    (path, mtime_ns), so catalog changes written by ANY process (the
    admin pull subprocess, an operator's CLI pull/remove beside a running
    supervisor) are picked up on the next call -- no restart, no manual
    cache reset. The stat key is taken BEFORE the catalog read: if the
    file changes between stat and read we cache newer content under an
    older key, which the next call's key mismatch rebuilds away; the
    reverse (stale content under a fresh key) cannot happen.

    H5: double-checked locking around the cache population. The outer
    check is fast and lock-free for the common hot path. The inner check
    under _KNOWN_MODELS_LOCK is the critical section that prevents two
    concurrent threads from both calling discover_models() (importlib
    imports -- double-executing user script module bodies) and both
    writing the cache.
    Lock ordering: _KNOWN_MODELS_LOCK is always acquired BEFORE
    _CATALOG_WRITE_LOCK (never the other way around; see M1 note).
    """
    global _known_models_cache, _discovered_entries_cache
    # Fast path: cache built against the current catalog file identity.
    key = _catalog_cache_key()
    cached = _known_models_cache
    if cached is not None and key is not None and cached[0] == key:
        return cached[1]
    # Slow path: acquire the lock and re-check inside it.
    with _KNOWN_MODELS_LOCK:
        # Re-key under the lock: the catalog may have changed while we
        # waited, and another thread may have already rebuilt for the
        # current key.
        key = _catalog_cache_key()
        cached = _known_models_cache
        if cached is not None and key is not None and cached[0] == key:
            return cached[1]
        if _discovered_entries_cache is None:
            discovered = discover_models(_model_dirs())
            _discovered_entries_cache = {
                model_id: _manifest_to_catalog_entry(d)
                for model_id, d in discovered.items()
            }
        entries = dict(_discovered_entries_cache)
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
            # Re-apply curated capabilities overlay (+ base_override) so
            # edits to curated.yaml / operator pins take effect on next
            # process restart without requiring a re-pull. See
            # _apply_manifest_overlays for the shared order/rationale
            # (also used by get_manifest() so the two never diverge).
            manifest = _apply_manifest_overlays(model_id, manifest, entry_data)
            entries[model_id] = _persisted_manifest_to_catalog_entry(manifest)
        if key is not None:
            _known_models_cache = (key, entries)
        return entries


def _catalog_cache_key() -> tuple[str, int] | None:
    """Identity of the catalog file the known_models merge was built against.

    (path_str, mtime_ns) for an existing file; a stable sentinel when the
    file does not exist (fresh install -- cacheable until a file appears);
    None when the file exists but cannot be stat'ed (never cache: always
    rebuild rather than risk serving a snapshot we cannot validate).
    """
    p = _catalog_path()
    try:
        return (str(p), p.stat().st_mtime_ns)
    except FileNotFoundError:
        return _MISSING_CATALOG_KEY
    except OSError:
        return None


def _reset_known_models_cache() -> None:
    """Clear both known_models caches so discovery re-runs on next call.

    Catalog-side staleness is handled automatically by the mtime key in
    known_models(); this reset exists for the DISCOVERY tier (tests that
    drop new script files into a models dir mid-process) and as a
    belt-and-braces invalidation after in-process catalog mutations.

    Takes _KNOWN_MODELS_LOCK (L9): a lock-free `cache = None` races the
    lock-guarded rebuild in known_models(). Serializing on the same lock
    forces the invalidation to order strictly before or after any in-flight
    rebuild, so the next known_models() call always rebuilds fresh.
    """
    global _known_models_cache, _discovered_entries_cache
    with _KNOWN_MODELS_LOCK:
        _known_models_cache = None
        _discovered_entries_cache = None


def _catalog_dir() -> Path:
    """Resolve the catalog directory identically to `config.py`'s own
    bootstrap resolution (env+default only), so catalog.json always
    co-locates with config.yaml. Delegates to `config._catalog_dir()`
    rather than `config.get("paths.catalog_dir")`: catalog_dir is
    bootstrap state, and a config.yaml value must never be able to
    redirect catalog.json away from where config.yaml itself lives.
    """
    return config._catalog_dir()


def _catalog_path() -> Path:
    return _catalog_dir() / "catalog.json"


# mtime-based cache for _read_catalog. The catalog is consulted on the
# gateway hot path (every request: get_manifest -> _read_catalog) and by
# admin / CLI flows. Cache stores (path_str, mtime_ns, parsed_dict);
# invalidates whenever the file's mtime advances (writes go through
# `_write_catalog`'s atomic rename, which updates mtime). A path-keyed
# lookup means tests using `tmp_path` + the MUSE_CATALOG_DIR env var
# don't accidentally hit cache from a prior run; first read after path
# change is always a fresh disk read.
_read_catalog_cache: tuple[str, int, dict] | None = None


def _read_catalog() -> dict:
    """Return the parsed catalog.json contents, with mtime-based caching.

    Hot path. Returns a fresh dict each call (deep copy of the cached
    parse) so callers can mutate without polluting the cache.

    A corrupt (invalid-JSON) catalog file does NOT silently degrade to
    an empty dict, which would look like a valid "no models" catalog to
    every caller. Instead: if a last-known-good parse is cached for this
    same path, serve that (stale-but-real data, logged as a warning);
    if there is no cache to fall back to, raise CatalogError so the
    caller surfaces the problem instead of masking it.
    """
    global _read_catalog_cache
    p = _catalog_path()
    if not p.exists():
        # Drop any cached state from a prior pulled+removed cycle so a
        # subsequent write produces a cache-miss instead of returning a
        # stale dict.
        _read_catalog_cache = None
        return {}
    try:
        mtime_ns = p.stat().st_mtime_ns
    except OSError:
        mtime_ns = -1
    path_str = str(p)

    cached = _read_catalog_cache
    if (
        cached is not None
        and cached[0] == path_str
        and cached[1] == mtime_ns
        and mtime_ns != -1
    ):
        # Deep copy: callers (`pull`, `remove`, `set_enabled`) mutate the
        # returned dict in-place, then write back. Sharing the cached
        # reference would let those mutations bleed into later cache hits.
        return _deep_copy_catalog(cached[2])

    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        if cached is not None and cached[0] == path_str:
            logger.warning(
                "catalog at %s corrupt; serving last-known-good cached parse", p,
            )
            return _deep_copy_catalog(cached[2])
        logger.error("catalog at %s corrupt; no cached copy to fall back to", p)
        raise CatalogError(
            f"catalog at {p} is corrupt (invalid JSON) and no last-known-"
            f"good cached copy is available; inspect or restore the file "
            f"before continuing"
        )
    # Backfill enabled=True for pre-enable-flag entries (migration path).
    # Non-destructive: only affects the in-memory dict on read.
    for entry in data.values():
        entry.setdefault("enabled", True)
    if mtime_ns != -1:
        _read_catalog_cache = (path_str, mtime_ns, _deep_copy_catalog(data))
    return data


def _deep_copy_catalog(data: dict) -> dict:
    """Shallow-then-shallow copy of the catalog dict-of-dicts.

    Catalog entries are plain JSON shapes (str/number/bool/None plus dicts
    and lists). The persisted `manifest` field is the deepest structure,
    and callers mutate it (e.g. `_pull_via_resolver` does
    `manifest = dict(resolved.manifest)` before storing). So a top-level
    deep copy via `json.loads(json.dumps(data))` is the most defensive
    cheap option; profile if this becomes a hot spot.
    """
    return json.loads(json.dumps(data))


def _reset_read_catalog_cache() -> None:
    """Test hook: clear the catalog read cache."""
    global _read_catalog_cache
    _read_catalog_cache = None


def _write_catalog(data: dict) -> None:
    """Atomic write: write to .tmp in same dir, then rename.

    Rename within the same filesystem is atomic on POSIX and near-atomic
    on Windows (Python 3.3+ Path.replace wraps MoveFileEx with REPLACE_EXISTING).

    Invalidates the read cache so the next `_read_catalog()` sees this
    write. The mtime check would catch this on its own under normal
    filesystems, but explicit invalidation removes a class of race that
    surfaces on coarse-mtime filesystems (e.g. some FAT, network mounts):
    consecutive writes within the same mtime tick would otherwise serve
    stale data on the read between them.
    """
    global _read_catalog_cache
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)
    _read_catalog_cache = None


def is_pulled(model_id: str) -> bool:
    return model_id in _read_catalog()


def list_known(modality: str | None = None) -> list[CatalogEntry]:
    entries = list(known_models().values())
    if modality is None:
        return entries
    return [e for e in entries if e.modality == modality]


# Published distribution name on PyPI. The importable package, CLI, and
# repo are all `muse`, but the wheel is `museq`. A wheel/PyPI install
# installs muse-into-venv from this dist, not from an editable source tree.
_PYPI_DIST = "museq"


def _is_muse_pyproject(pyproject: Path) -> bool:
    """True when pyproject.toml declares the museq project (name = "museq").

    A cheap sniff so `_muse_repo_root` only claims a directory that is
    actually the muse source tree, never an unrelated parent project.
    """
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return False
    return f'name = "{_PYPI_DIST}"' in text or f"name = '{_PYPI_DIST}'" in text


def _muse_repo_root() -> Path | None:
    """Resolve the muse source tree that contains this catalog.py, or None.

    Walks parents of this file for a pyproject.toml that actually declares
    the museq project. Returns None from a wheel/PyPI install (no such
    pyproject in any parent, e.g. under site-packages), so callers install
    the published `museq` distribution instead of editable-installing
    site-packages -- which `pip install -e` rejects ("neither setup.py nor
    pyproject.toml found"), previously making `muse pull` fail outright on a
    PyPI install.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.exists() and _is_muse_pyproject(pyproject):
            return parent
    return None


def _muse_server_install_args() -> list[str]:
    """pip args to install muse[server] itself into a per-model venv.

    From a source checkout: ``-e <root>[server]`` (editable, tracks the
    working tree). From a wheel/PyPI install: ``museq[server]`` (no -e;
    pip resolves museq from PyPI). Mirrors cli_impl.refresh._pip_target_args.
    """
    root = _muse_repo_root()
    if root is not None:
        return ["-e", f"{root}[server]"]
    return [f"{_PYPI_DIST}[server]"]


def _validate_lora_capabilities(manifest: dict) -> None:
    """Reject unservable LoRA manifests at pull time, post-overlay.

    A lora_adapter entry without a base_model can never load; a
    muse-id base that is not pulled would fail at first request with a
    from_pretrained error. Both fail here, BEFORE the expensive venv
    creation and download, with the fix in the message. Runs after the
    curated/--base capabilities overlay merge so a --base override can
    satisfy a tagless adapter repo.
    """
    from muse.core.resolvers import ResolverError

    caps = manifest.get("capabilities") or {}
    if not caps.get("lora_adapter"):
        return
    model_id = manifest.get("model_id", "<unknown>")
    base = caps.get("base_model")
    if not base:
        raise ResolverError(
            f"LoRA adapter {model_id!r} declares no base model and none was "
            f"given; re-run with: muse pull <identifier> --base "
            f"<muse-id-or-hf-repo>"
        )
    if "/" not in base:
        entry = _read_catalog().get(base)
        if not entry or not entry.get("local_dir"):
            raise ResolverError(
                f"LoRA base {base!r} is not pulled; run `muse pull {base}` "
                f"first, then retry"
            )


def pull(identifier: str, *, base_override: str | None = None) -> None:
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
    install museq[server] (editable) + pip_extras, fetch weights, and
    record the venv's Python path so `muse serve` can spawn workers
    with the right interpreter.

    `base_override` applies to LoRA adapter pulls: resolver-URI,
    curated-by-URI/curated-alias, AND resolver-sourced bare-id re-pulls.
    It is threaded to `_pull_via_resolver` as its own `base_override`
    kwarg (a top-level catalog field, mirroring `device_override`), NOT
    merged into the capabilities overlay: see `_pull_via_resolver` for
    why that durability matters. It is warned-and-ignored only for true
    bundled-script pulls, which have no LoRA base to set.
    """
    curated = find_curated(identifier)
    if curated is not None:
        if curated.uri:
            # Resolver-pulled curated entry. Override the synthesized id
            # so the catalog stores the friendly curated id (e.g.
            # qwen3-8b-q4) instead of qwen3-8b-instruct-gguf-q4-k-m.
            # Also forward the curated capabilities overlay so any
            # runtime-specific settings (trust_remote_code, chat_format,
            # context_length) land in the persisted manifest.
            overlay = dict(curated.capabilities or {})
            _pull_via_resolver(
                curated.uri,
                model_id_override=curated.id,
                capabilities_overlay=overlay or None,
                modality_override=curated.modality,
                base_override=base_override,
            )
            return
        # Bundled curated entry: id equals an existing bundled script's
        # model_id. Fall through to the bundled path with that id.
        _pull_bundled(curated.id)
        return

    if "://" in identifier:
        # Inherit curated capabilities for this URI even when the user
        # didn't go through the curated id. Without this, copying a URI
        # out of `muse search` output and pasting it into `muse pull`
        # silently strips overlay fields like `safe_labels` (KoalaAI)
        # or `trust_remote_code` (Qwen3-Embedding) that are required
        # for the model to behave correctly. The curated id, if any,
        # is also preserved so the catalog key stays friendly.
        uri_curated = find_curated_by_uri(identifier)
        if uri_curated is not None:
            overlay = dict(uri_curated.capabilities or {})
            _pull_via_resolver(
                identifier,
                model_id_override=uri_curated.id,
                capabilities_overlay=overlay or None,
                modality_override=uri_curated.modality,
                base_override=base_override,
            )
        else:
            _pull_via_resolver(identifier, base_override=base_override)
        return

    # Bare id: could be a bundled script OR a resolver-pulled model
    # (resolver-pulled ids also show up in known_models() via their
    # persisted manifest). Re-pulling a resolver model by its friendly id
    # must go back through the resolver: _pull_bundled would overwrite the
    # entry with a bundled-shaped dict lacking `manifest`/`source`, and the
    # next known_models() rebuild would then drop the (no-manifest,
    # no-script) entry so the model vanishes with a spurious 'unknown
    # model' error (M3). Detect the resolver case by the `source` URI the
    # resolver persists and route back through _pull_via_resolver, keeping
    # the same catalog id and re-applying any curated overlay.
    from muse.core.curated import load_curated

    _validate_model_id_for_fs(identifier)

    existing = _read_catalog().get(identifier, {}) or {}
    source_uri = existing.get("source")
    if source_uri:
        # Resolver-sourced: thread base_override through instead of
        # warn-and-ignore. The warning below is reserved for pulls with
        # no resolver source at all (true bundled scripts).
        uri_curated = find_curated_by_uri(source_uri)
        if uri_curated is not None:
            _pull_via_resolver(
                source_uri,
                model_id_override=identifier,
                capabilities_overlay=uri_curated.capabilities or None,
                modality_override=uri_curated.modality,
                base_override=base_override,
            )
        else:
            _pull_via_resolver(
                source_uri,
                model_id_override=identifier,
                base_override=base_override,
            )
        return

    if base_override:
        logger.warning(
            "--base only applies to resolver-pulled LoRA adapters; "
            "ignored for %s", identifier,
        )

    catalog_known = known_models()
    if identifier in catalog_known:
        _pull_bundled(identifier)
        return

    curated_ids = [c.id for c in load_curated()]
    all_known = sorted(set(list(catalog_known) + curated_ids))
    from difflib import get_close_matches
    suggestions = get_close_matches(identifier, all_known, n=3, cutoff=0.5)
    msg = f"unknown model {identifier!r}"
    if suggestions:
        msg += f"; did you mean: {', '.join(suggestions)}?"
    else:
        msg += " (run `muse models list` to see all model ids)"
    raise KeyError(msg)


def _pull_bundled(model_id: str) -> None:
    """Pull a bundled (script-discovered) model by bare id.

    Callers (only `pull()`) verify the id is in `known_models()` first
    and produce a user-friendly error for unknown ids; this defensive
    check guards against internal mistakes.
    """
    _validate_model_id_for_fs(model_id)
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(
            f"unknown model {model_id!r} (internal dispatch bug in "
            f"_pull_bundled; use pull() to get a better error)"
        )
    entry = catalog_known[model_id]

    venvs_root = _catalog_dir() / "venvs"
    venv_path = venvs_root / model_id

    if not venv_path.exists():
        create_venv(venv_path)

    install_into_venv(venv_path, _muse_server_install_args())

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

    # Bundled MANIFESTs may declare `capabilities.allow_patterns` to
    # restrict the snapshot_download manifest (mirrors what the resolver
    # plugins do for fp16-shaped or BIN-only repos). This avoids hauling
    # down fp32 siblings, .bin/.h5 dupes, and standalone single-file
    # checkpoints when the diffusers/transformers runtime only needs the
    # subfolder weights.
    allow_patterns = entry.extra.get("allow_patterns")
    revision = entry.extra.get("revision")
    download_kwargs: dict[str, Any] = {"repo_id": entry.hf_repo}
    if allow_patterns:
        download_kwargs["allow_patterns"] = list(allow_patterns)
    if revision:
        download_kwargs["revision"] = revision
    with _hf_quiet_if_needed():
        local_dir = snapshot_download(**download_kwargs)

    # M1: hold _CATALOG_WRITE_LOCK for the full read->mutate->write sequence.
    # The heavy work (venv creation, pip install, HF download) happens above,
    # outside any lock. Only the catalog file RMW is time-sensitive here.
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        catalog[model_id] = {
            "pulled_at": datetime.now(timezone.utc).isoformat(),
            "hf_repo": entry.hf_repo,
            "local_dir": str(local_dir),
            "venv_path": str(venv_path),
            "python_path": str(venv_python(venv_path)),
            "enabled": True,
        }
        if revision:
            catalog[model_id]["revision"] = revision
        _write_catalog(catalog)
    _reset_known_models_cache()


def _pull_via_resolver(
    uri: str,
    *,
    model_id_override: str | None = None,
    capabilities_overlay: dict | None = None,
    modality_override: str | None = None,
    base_override: str | None = None,
) -> None:
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

    `modality_override` is set when the curated alias declared an
    explicit `modality:` field. The priority-based resolver dispatch
    sometimes misclassifies multi-flavor repos (rerankers ship as
    sentence-transformers, so the embedding/text plugin claims them
    even though they're cross-encoders). When the operator declared
    a modality in curated.yaml, we honor it: look up the plugin for
    that modality and resolve via it directly. Bypasses sniff-priority.

    `capabilities_overlay` is set when the URI was reached via a curated
    alias that declared its own `capabilities:` block. It merges into
    the resolver-synthesized manifest's `capabilities` (shallow merge;
    overlay wins on key collision). The merged block ends up in the
    persisted manifest and flows into the runtime constructor via
    `load_backend`.

    `base_override` is the operator's `--base` pin for a LoRA adapter
    pull. Modeled on `device_override`: it is stored as a TOP-LEVEL
    `base_override` field on the catalog entry (not merged into
    `capabilities_overlay`), so it survives `known_models()`'s later
    curated-overlay re-application and `get_manifest()` reads, both of
    which apply it AFTER curated capabilities so the operator wins over
    both the tag-declared base and a curated `base_model`. It is also
    forwarded into `resolve()` so a turbo `--base` pairing re-derives
    generation defaults (steps/guidance) at resolve time (fix I2), and
    written into the manifest's `capabilities.base_model` here so
    `_validate_lora_capabilities` and the persisted manifest agree with
    the top-level field. When omitted (a plain re-pull), any
    `base_override` already recorded on the PRIOR catalog entry for this
    model_id is carried over so re-pulling never silently reverts a
    previously-set operator pin.
    """
    from muse.core.resolvers import resolve

    # `modality_override` is forwarded to resolve() when set; the
    # resolver dispatches via resolve_via_modality (bypassing sniff)
    # so curated yaml's modality declaration beats the priority-based
    # plugin pick. See resolvers.resolve docstring.
    resolved = resolve(uri, modality=modality_override, base_override=base_override)
    manifest = dict(resolved.manifest)
    # Resolver may put backend_path in the manifest itself, or only on
    # the ResolvedModel. Persist it consistently so load_backend can
    # find it without consulting the resolver again.
    manifest.setdefault("backend_path", resolved.backend_path)

    if capabilities_overlay:
        merged_caps = dict(manifest.get("capabilities") or {})
        merged_caps.update(capabilities_overlay)
        manifest["capabilities"] = merged_caps

    if model_id_override:
        manifest["model_id"] = model_id_override
    model_id = manifest["model_id"]
    _validate_model_id_for_fs(model_id)

    # Preserve a prior operator --base pin across a re-pull that omits
    # --base, so re-pulling never silently reverts operator intent.
    effective_base_override = base_override
    if not effective_base_override:
        prior_entry = _read_catalog().get(model_id, {}) or {}
        effective_base_override = prior_entry.get("base_override")

    if effective_base_override:
        merged_caps = dict(manifest.get("capabilities") or {})
        merged_caps["base_model"] = effective_base_override
        manifest["capabilities"] = merged_caps

    _validate_lora_capabilities(manifest)

    venvs_root = _catalog_dir() / "venvs"
    venv_path = venvs_root / model_id

    if not venv_path.exists():
        create_venv(venv_path)

    install_into_venv(venv_path, _muse_server_install_args())

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
    with _hf_quiet_if_needed():
        local_dir = resolved.download(weights_cache)

    # M1: hold _CATALOG_WRITE_LOCK for the full read->mutate->write sequence.
    # The heavy work (resolve, venv creation, pip install, HF download) happens
    # above, outside any lock. Only the catalog file RMW is protected here.
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        entry = {
            "pulled_at": datetime.now(timezone.utc).isoformat(),
            "hf_repo": manifest["hf_repo"],
            "local_dir": str(local_dir),
            "venv_path": str(venv_path),
            "python_path": str(venv_python(venv_path)),
            "enabled": True,
            "source": uri,
            "manifest": manifest,
        }
        if effective_base_override:
            entry["base_override"] = effective_base_override
        catalog[model_id] = entry
        _write_catalog(catalog)
    _reset_known_models_cache()


def remove(model_id: str, *, purge: bool = False) -> None:
    """Unregister `model_id` from the catalog.

    By default this only edits `catalog.json`; the per-model venv at
    `~/.muse/venvs/<model_id>/` stays on disk. Mirrors `apt remove`'s
    "metadata only" semantics.

    When `purge=True`:
      - rmtree the venv directory.
      - rmtree the resolver weights cache at `~/.muse/weights/<dir>/`
        IF the entry's `local_dir` resolves under the muse-owned
        weights tree. Bundled-pulled models that store weights in the
        shared HF cache (`~/.cache/huggingface`) are left alone; muse
        does not own that, and `huggingface-cli delete-cache` is the
        right tool for it.

    Tolerates either path being already gone.

    M1: holds _CATALOG_WRITE_LOCK for the full read->pop->write sequence.
    The rmtree purge steps run OUTSIDE the lock (they are slow and do not
    touch catalog.json).
    """
    import shutil
    venv_path: str | None = None
    local_dir: str | None = None
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        entry = catalog.get(model_id, {}) or {}
        venv_path = entry.get("venv_path")
        local_dir = entry.get("local_dir")
        catalog.pop(model_id, None)
        _write_catalog(catalog)
    # Resolver-pulled entries appear in known_models() via the persisted
    # manifest path; once removed, that cache must drop them too or
    # `muse models list` keeps reporting a model that no longer exists.
    _reset_known_models_cache()
    if not purge:
        return
    if venv_path:
        shutil.rmtree(venv_path, ignore_errors=True)
    if local_dir:
        weights_root = (_catalog_dir() / "weights").resolve()
        try:
            local_path = Path(local_dir).resolve()
            local_path.relative_to(weights_root)
        except (ValueError, OSError):
            # local_dir lives outside the muse-owned weights tree
            # (typically the HF cache). Leave it alone.
            return
        shutil.rmtree(local_path, ignore_errors=True)


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

    M1: holds _CATALOG_WRITE_LOCK for the full read->mutate->write
    sequence so concurrent mutations on different keys do not lose
    each other's updates.
    """
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        if model_id not in catalog:
            raise KeyError(f"model {model_id!r} is not pulled")
        catalog[model_id]["enabled"] = bool(enabled)
        _write_catalog(catalog)
    # `enabled` flows through known_models() into the CatalogEntry
    # consumers see; without the reset, `muse models list` would
    # display the stale state until the process restarts.
    _reset_known_models_cache()


VALID_DEVICE_OVERRIDES = ("auto", "cpu", "cuda", "mps")


def set_device_override(model_id: str, device: str | None) -> None:
    """Set or clear the per-model device override for a pulled model.

    `device` in {auto, cpu, cuda, mps} pins the model's load device,
    overriding both the manifest `capabilities.device` pin and the
    supervisor `--device` flag (see `load_backend`'s precedence). The
    special value "auto" un-pins a cpu-pinned model so the runtime's
    `select_device` picks cuda when a GPU is present. Passing ``None``
    removes the override entirely (revert to manifest pin / --device).

    Catalog state only: takes effect on the model's next cold load. To
    apply it to an already-resident worker, evict or restart that worker.

    Raises ValueError for an unrecognized device label and KeyError when
    the model is not pulled. Holds _CATALOG_WRITE_LOCK for the full
    read->mutate->write so concurrent mutations on different keys do not
    clobber each other (mirrors `set_enabled`).
    """
    if device is not None and device not in VALID_DEVICE_OVERRIDES:
        raise ValueError(
            f"invalid device {device!r}; expected one of "
            f"{', '.join(VALID_DEVICE_OVERRIDES)} or None to clear"
        )
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        if model_id not in catalog:
            raise KeyError(f"model {model_id!r} is not pulled")
        if device is None:
            catalog[model_id].pop("device_override", None)
        else:
            catalog[model_id]["device_override"] = device
        _write_catalog(catalog)
    # device_override is read live from the catalog in load_backend, but
    # `muse models info` surfaces it via known_models()-adjacent reads;
    # reset for display consistency (mirrors set_enabled).
    _reset_known_models_cache()


def set_gpu_layers_override(model_id: str, n: int | None) -> None:
    """Set or clear the per-model llama.cpp GPU-layer pin for a pulled model.

    `n` is the llama.cpp `n_gpu_layers` value: -1 = offload every layer the
    GPU fits, 0 = pure CPU, N > 0 = first N layers on GPU (rest on CPU).
    Stored as the TOP-LEVEL catalog field `gpu_layers_override` (operator
    state, mirroring `device_override` -- NOT part of the manifest). Passing
    ``None`` removes the pin (revert to capabilities.n_gpu_layers / the
    runtime default).

    Catalog state only: takes effect on the model's next cold load. To
    apply it to an already-resident worker, evict or restart that worker.

    Raises ValueError for a non-int or n < -1 and KeyError when the model
    is not pulled. Holds _CATALOG_WRITE_LOCK for the full
    read->mutate->write (mirrors `set_device_override`).
    """
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int) or n < -1:
            raise ValueError(
                f"invalid gpu layers {n!r}; expected an int >= -1 "
                "(-1 = all layers on GPU, 0 = pure CPU) or None to clear"
            )
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        if model_id not in catalog:
            raise KeyError(f"model {model_id!r} is not pulled")
        if n is None:
            catalog[model_id].pop("gpu_layers_override", None)
        else:
            catalog[model_id]["gpu_layers_override"] = n
        _write_catalog(catalog)
    _reset_known_models_cache()


def _import_backend_module(module_path: str):
    """Local indirection for `importlib.import_module`.

    Why this wrapper exists: tests need to stub out the catalog's backend
    import to install a fake module without fetching real ML deps. Patching
    `importlib.import_module` directly (or `muse.core.catalog.importlib.
    import_module`) mutates the shared global `importlib` module object, so
    the stub also intercepts calls from `muse.core.discovery` and anywhere
    else that imports at test time. Indirecting through a catalog-local
    function gives tests a patching target (`muse.core.catalog.
    _import_backend_module`) that only affects catalog's import path.
    """
    return importlib.import_module(module_path)


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
    module_path, class_name = entry.backend_path.split(":", 1)
    module = _import_backend_module(module_path)
    cls = getattr(module, class_name)
    catalog = _read_catalog()
    entry_data = catalog[model_id]
    local_dir = entry_data["local_dir"]
    # entry.extra holds capabilities from either the bundled MANIFEST
    # (read live from source by known_models() each call) or the persisted
    # manifest in catalog.json (resolver-pulled). Reading from entry here
    # means bundled scripts' capabilities (e.g. device: cpu on kokoro) are
    # honored at load time, not just resolver-pulled ones.
    capabilities = entry.extra
    merged: dict = {"model_id": model_id, **capabilities, **kwargs}
    # Device precedence, most authoritative first:
    #   1. catalog `device_override`  (operator, via `muse models set-device`)
    #   2. manifest `capabilities.device` pin (model-author affinity, e.g.
    #      kokoro's "cpu") -- overrides the supervisor --device flag
    #   3. caller kwargs device (the --device flag), already folded into kwargs
    #   4. "auto" (runtime select_device picks cuda if available, else cpu)
    # The override beats even the manifest pin so an operator can force a
    # cpu-pinned model onto cuda (or back to cpu to save VRAM) per deployment
    # without editing the bundled script. override="auto" un-pins a model to
    # auto-detect. Other capability keys still lose to kwargs (the documented
    # contract); device is the exception because it is a placement preference.
    override = entry_data.get("device_override")
    if override:
        merged["device"] = override
    elif "device" in capabilities and capabilities["device"] != "auto":
        merged["device"] = capabilities["device"]
    # GPU-layers precedence (spec 2026-07-08), most authoritative first:
    #   1. catalog `gpu_layers_override` (operator, via
    #      `muse models set-gpu-layers`)
    #   2. manifest `capabilities.n_gpu_layers` (already in `merged` via the
    #      capabilities splat above)
    #   3. runtime default (-1 in LlamaCppModel: everything the GPU fits)
    # Applied AFTER the kwargs merge, like device_override, so the operator
    # pin also beats caller kwargs: it is a placement preference.
    gpu_layers = entry_data.get("gpu_layers_override")
    if gpu_layers is not None:
        merged["n_gpu_layers"] = gpu_layers
    return cls(hf_repo=entry.hf_repo, local_dir=local_dir, **merged)


def _dir_size_bytes(path: str) -> int:
    """Recursive du-style size calc. Returns 0 if path missing/inaccessible.

    Used by `muse models list` to surface on-disk weight size per pulled
    model. Symlinks are not followed, so HuggingFace's snapshot cache
    layout (where snapshots/<sha>/* are symlinks into blobs/*) does not
    double-count blobs already attributed to a sibling pulled model.
    Per-file getsize errors (permissions, race vs deletion) are swallowed.
    """
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path, followlinks=False):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _human_size(b: int) -> str:
    """Format bytes as 'N.N GB' / 'N MB' / 'N KB' for table display.

    Returns '-' for 0 (callers use 0 as the missing-size sentinel).
    GB uses one decimal; MB and KB are integer-rounded to keep the
    column narrow enough for table alignment.
    """
    if b == 0:
        return "-"
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.0f} MB"
    return f"{b / 1024:.0f} KB"


def get_manifest(model_id: str) -> dict:
    """Return the MANIFEST dict for a known model.

    Two sources, in order of preference:
      1. catalog.json's persisted manifest (resolver-pulled models). The
         resolver synthesized this dict at pull time; it's the source of
         truth for that entry. The curated-capabilities overlay and any
         operator base_override are re-applied here via
         `_apply_manifest_overlays`, the SAME helper `known_models()`
         uses, so a curated.yaml edit affects gating (this function) and
         construction (known_models() -> load_backend) identically --
         see `_apply_manifest_overlays` for why that matters.
      2. The model script's module-level MANIFEST (bundled scripts).

    Returns a copy so callers can mutate without affecting the source.

    Raises KeyError if the model is not in `known_models()`.
    """
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    catalog = _read_catalog()
    entry_data = catalog.get(model_id, {})
    persisted = entry_data.get("manifest")
    if persisted:
        return _apply_manifest_overlays(model_id, persisted, entry_data)
    entry = catalog_known[model_id]
    module_path, _ = entry.backend_path.split(":", 1)
    module = _import_backend_module(module_path)
    manifest = getattr(module, "MANIFEST", None)
    # Most bundled scripts define `class Model` in the script itself, so
    # backend_path's module IS the script and carries the MANIFEST. But a
    # script may alias its Model to a shared runtime class (e.g.
    # `from ...runtimes.transformers_vlm import HFVisionLanguageModel as Model`),
    # which makes backend_path point at the runtime module - and that module
    # has no MANIFEST (or one for a different model). In that case the
    # capabilities (supports_vision, etc.) would be silently lost and routes
    # would mis-gate the model. Recover the real MANIFEST from discovery.
    if not manifest or manifest.get("model_id") != model_id:
        discovered = discover_models(_model_dirs()).get(model_id)
        if discovered is not None:
            return dict(discovered.manifest)
    return dict(manifest or {})
