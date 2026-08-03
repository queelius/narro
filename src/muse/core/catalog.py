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
        revision,                      # immutable primary HF commit
        artifact_provenance,           # complete immutable download receipt
        local_dir,                     # HF snapshot_download cache path
        venv_path,                     # dedicated venv for this model
        python_path,                   # <venv_path>/bin/python for workers
    }]
"""
from __future__ import annotations

import copy

import importlib
from importlib import metadata as importlib_metadata
import contextlib
import errno
import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import tempfile
import threading
from collections.abc import Mapping, Sequence
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
    ensure_venv,
    install_into_venv,
    install_python_sources,
    venv_python,
    venv_transaction,
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

_MAX_CATALOG_BYTES = 16 * 1024 * 1024
_MAX_CATALOG_ENTRIES = 10_000
_MAX_CATALOG_DEPTH = 24
_MAX_CATALOG_CONTAINER_ITEMS = 10_000
_MAX_CATALOG_STRING_CHARS = 1_000_000


class CatalogError(RuntimeError):
    """Raised when catalog.json is corrupt and no cached good copy exists.

    Invalid JSON, invalid UTF-8, or a malformed catalog shape must never be
    treated as an empty-but-valid catalog: every consumer (get_manifest,
    known_models,
    /v1/models, is_pulled) would then silently behave as "no models
    installed" -- 404-ing requests and emptying listings -- with no
    signal that the catalog file itself needs attention. When a prior
    good read is cached, `_read_catalog` serves that instead of raising;
    this error is reserved for the case where there is nothing good to
    fall back to.
    """


class ModelInUseError(CatalogError):
    """Raised when a live worker/probe owns a model's mutable resources."""


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
    # Keep new optional fields after `extra`: a few integrations construct
    # CatalogEntry positionally and the original eight-field order is public.
    python_sources: tuple[dict, ...] = ()
    bundled: bool = False


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
        python_sources  -> python_sources            (tuple-coerced, defaults ())
        capabilities    -> extra                    (dict-copied, defaults {})
        revision, allow_patterns, hf_artifacts -> extra (download metadata)
    backend_path is synthesized from the Model class's module and name.
    """
    m = discovered.manifest
    cls = discovered.model_class
    source_path = getattr(discovered, "source_path", None)
    extra = dict(m.get("capabilities", {}))
    for key in ("revision", "allow_patterns", "hf_artifacts"):
        if key in m:
            extra[key] = copy.deepcopy(m[key])
    return CatalogEntry(
        model_id=m["model_id"],
        modality=m["modality"],
        backend_path=f"{cls.__module__}:{cls.__name__}",
        hf_repo=m["hf_repo"],
        description=m.get("description", ""),
        pip_extras=tuple(m.get("pip_extras", ())),
        system_packages=tuple(m.get("system_packages", ())),
        python_sources=tuple(m.get("python_sources", ())),
        extra=extra,
        bundled=(
            source_path is not None
            and Path(source_path).parent.resolve()
            == _bundled_models_dir().resolve()
        ),
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
_CatalogFileKey = tuple[str, int, int, int, int, int]
_known_models_cache: tuple[_CatalogFileKey, dict[str, CatalogEntry]] | None = None
_MISSING_CATALOG_KEY: _CatalogFileKey = ("<no-catalog>", -2, -2, -2, -2, -2)

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


def _acquire_catalog_file_lock(handle) -> None:
    """Take an exclusive advisory lock on an open catalog lock file."""
    if os.name == "nt":  # pragma: no cover - platform-specific fallback
        import msvcrt

        # msvcrt.locking locks bytes from the current file position. Ensure
        # the lock file owns at least one byte, then consistently lock byte 0.
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _try_acquire_catalog_file_lock(handle) -> bool:
    """Try one exclusive advisory lock without waiting."""
    if os.name == "nt":  # pragma: no cover - platform-specific fallback
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _release_catalog_file_lock(handle) -> None:
    """Release an advisory lock acquired by `_acquire_catalog_file_lock`."""
    if os.name == "nt":  # pragma: no cover - platform-specific fallback
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _ensure_owned_directory(path: Path, *, private: bool = False) -> None:
    """Create and validate an owned directory without following its leaf.

    The configured catalog directory may itself intentionally live below a
    symlinked parent.  What must never be accepted is a symlink in place of
    the directory Muse is about to chmod or populate (notably ``locks/``).
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "posix":  # pragma: no cover - Windows fallback
        info = path.lstat()
        is_junction = getattr(path, "is_junction", lambda: False)
        if stat.S_ISLNK(info.st_mode) or is_junction() or not path.is_dir():
            raise CatalogError(f"catalog path is not a safe directory: {path}")
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CatalogError(f"catalog path is not a safe directory: {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise CatalogError(f"catalog path is not a directory: {path}")
        if info.st_uid != os.geteuid():
            raise CatalogError(f"catalog directory is not owned by this user: {path}")
        if private:
            os.fchmod(descriptor, 0o700)
        elif info.st_mode & 0o022:
            raise CatalogError(
                f"catalog directory is group/other writable: {path}"
            )
    finally:
        os.close(descriptor)


def _open_catalog_regular_file(
    path: Path,
    flags: int,
    *,
    mode: int | None = None,
    require_single_link: bool = False,
    require_private: bool = False,
) -> int:
    """Open one owned regular file without following its final component."""
    open_flags = flags | getattr(os, "O_CLOEXEC", 0)
    if os.name == "posix":
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
    else:  # pragma: no cover - Windows fallback
        try:
            before = path.lstat()
        except FileNotFoundError:
            before = None
        if before is not None and stat.S_ISLNK(before.st_mode):
            raise CatalogError(f"refusing symlink catalog file: {path}")
        open_flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = (
            os.open(path, open_flags)
            if mode is None
            else os.open(path, open_flags, mode)
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise CatalogError(f"cannot safely open catalog file {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise CatalogError(f"catalog path is not a regular file: {path}")
        if os.name == "posix":
            if info.st_uid != os.geteuid():
                raise CatalogError(f"catalog file is not owned by this user: {path}")
            if require_private and info.st_mode & 0o022:
                raise CatalogError(
                    f"catalog file is group/other writable: {path}"
                )
        if require_single_link and info.st_nlink != 1:
            raise CatalogError(f"catalog lock has multiple links: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_catalog_lock(path: Path):
    """Return a private binary handle suitable for the advisory lock."""
    descriptor = _open_catalog_regular_file(
        path,
        os.O_CREAT | os.O_RDWR,
        mode=0o600,
        require_single_link=True,
        require_private=True,
    )
    if os.name == "posix":
        os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "a+b", buffering=0)


class _CatalogWriteLock:
    """Re-entrant thread and process lock for catalog transactions.

    The old plain ``threading.Lock`` protected threads in one interpreter,
    but Muse also mutates the catalog from CLI/admin/probe subprocesses. An
    advisory lock file in the catalog directory serializes those processes.
    Re-entrancy lets `_write_catalog` defend standalone writes while callers
    continue to hold this same lock around the complete read-modify-write.
    """

    def __init__(self) -> None:
        self._thread_lock = threading.RLock()
        self._local = threading.local()

    def acquire(self) -> bool:
        self._thread_lock.acquire()
        depth = getattr(self._local, "depth", 0)
        if depth:
            self._local.depth = depth + 1
            return True

        fd: int | None = None
        handle = None
        try:
            catalog_dir = _catalog_dir()
            _ensure_owned_directory(catalog_dir, private=True)
            lock_path = catalog_dir / ".catalog.lock"
            handle = _open_catalog_lock(lock_path)
            _acquire_catalog_file_lock(handle)
            self._local.handle = handle
            self._local.depth = 1

            # Another process may have replaced catalog.json while this one
            # waited. Force the first read in the transaction to hit disk,
            # even on filesystems whose mtimes are too coarse to expose it.
            global _read_catalog_cache
            _read_catalog_cache = None
            return True
        except BaseException:
            if handle is not None:
                handle.close()
            elif fd is not None:
                os.close(fd)
            self._thread_lock.release()
            raise

    def release(self) -> None:
        depth = getattr(self._local, "depth", 0)
        if depth <= 0:
            raise RuntimeError("cannot release an un-acquired catalog lock")
        try:
            if depth > 1:
                self._local.depth = depth - 1
                return

            handle = self._local.handle
            try:
                _release_catalog_file_lock(handle)
            finally:
                handle.close()
                del self._local.handle
                self._local.depth = 0
        finally:
            self._thread_lock.release()

    def __enter__(self) -> "_CatalogWriteLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


_CATALOG_WRITE_LOCK = _CatalogWriteLock()


# The catalog transaction lock protects JSON consistency, but two processes
# installing into the same venv can corrupt site-packages before either one
# writes catalog.json. Pair a per-identity thread lock with an advisory file
# lock that spans the complete pull helper call. Resolver URIs are digested so
# untrusted identifier text never becomes part of a filesystem path.
_PULL_LOCKS_GUARD = threading.Lock()
_PULL_THREAD_LOCKS: dict[str, threading.RLock] = {}
_RESOURCE_LOCKS_GUARD = threading.Lock()
_RESOURCE_THREAD_LOCKS: dict[str, threading.Lock] = {}


class _ModelPullLock:
    """Thread/process lock covering one model's complete pull transaction."""

    def __init__(self, identity: str) -> None:
        self._digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        self._thread_lock: threading.RLock | None = None
        self._handle = None

    def __enter__(self) -> "_ModelPullLock":
        with _PULL_LOCKS_GUARD:
            thread_lock = _PULL_THREAD_LOCKS.setdefault(
                self._digest, threading.RLock(),
            )
        self._thread_lock = thread_lock
        thread_lock.acquire()
        fd: int | None = None
        handle = None
        try:
            locks_dir = _catalog_dir() / "locks"
            _ensure_owned_directory(locks_dir, private=True)
            lock_path = locks_dir / f"pull-{self._digest}.lock"
            handle = _open_catalog_lock(lock_path)
            _acquire_catalog_file_lock(handle)
            self._handle = handle
            return self
        except BaseException:
            if handle is not None:
                handle.close()
            elif fd is not None:
                os.close(fd)
            thread_lock.release()
            self._thread_lock = None
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._handle is not None:
                try:
                    _release_catalog_file_lock(self._handle)
                finally:
                    self._handle.close()
                    self._handle = None
        finally:
            if self._thread_lock is not None:
                self._thread_lock.release()
                self._thread_lock = None


def _model_pull_lock(identity: str) -> _ModelPullLock:
    return _ModelPullLock(identity)


class _ModelResourceLease:
    """Exclusive lifetime lease for one model's venv and weight resources.

    Workers acquire a blocking lease for their whole lifetime. Commands that
    mutate or probe the environment use ``wait=False`` and fail clearly when
    a live worker still owns it. The advisory file lock makes this contract
    work across the supervisor, worker venv, and direct CLI processes.
    """

    def __init__(self, model_id: str, *, wait: bool) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model resource lease requires a non-empty model id")
        self._model_id = model_id
        self._digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()
        self._wait = wait
        self._thread_lock: threading.Lock | None = None
        self._handle = None

    def __enter__(self) -> "_ModelResourceLease":
        with _RESOURCE_LOCKS_GUARD:
            thread_lock = _RESOURCE_THREAD_LOCKS.setdefault(
                self._digest, threading.Lock(),
            )
        if not thread_lock.acquire(blocking=self._wait):
            raise ModelInUseError(
                f"model {self._model_id!r} is in use; stop or unload it and retry"
            )
        self._thread_lock = thread_lock
        handle = None
        try:
            locks_dir = _catalog_dir() / "locks"
            _ensure_owned_directory(locks_dir, private=True)
            handle = _open_catalog_lock(
                locks_dir / f"resource-{self._digest}.lock"
            )
            acquired = (
                (_acquire_catalog_file_lock(handle) is None)
                if self._wait
                else _try_acquire_catalog_file_lock(handle)
            )
            if not acquired:
                raise ModelInUseError(
                    f"model {self._model_id!r} is in use; "
                    "stop or unload it and retry"
                )
            self._handle = handle
            return self
        except BaseException:
            if handle is not None:
                handle.close()
            thread_lock.release()
            self._thread_lock = None
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._handle is not None:
                try:
                    _release_catalog_file_lock(self._handle)
                finally:
                    self._handle.close()
                    self._handle = None
        finally:
            if self._thread_lock is not None:
                self._thread_lock.release()
                self._thread_lock = None


def _model_resource_lease(
    model_id: str, *, wait: bool = False,
) -> _ModelResourceLease:
    return _ModelResourceLease(model_id, wait=wait)


def _validated_persisted_manifest(
    catalog_model_id: str, manifest: object,
) -> dict:
    """Validate one resolver manifest without poisoning unrelated entries."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    required = ("model_id", "modality", "backend_path", "hf_repo")
    for field_name in required:
        value = manifest.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"manifest {field_name} must be a non-empty string")
    if manifest["model_id"] != catalog_model_id:
        raise ValueError(
            f"manifest model_id {manifest['model_id']!r} does not match "
            f"catalog key {catalog_model_id!r}"
        )
    backend_module, separator, backend_name = manifest["backend_path"].partition(":")
    if not separator or not backend_module or not backend_name:
        raise ValueError("manifest backend_path must use 'module:attribute' form")
    description = manifest.get("description", "")
    if not isinstance(description, str):
        raise ValueError("manifest description must be a string")
    for field_name in ("pip_extras", "system_packages"):
        values = manifest.get(field_name, ())
        if not isinstance(values, (list, tuple)) or not all(
            isinstance(item, str) and item for item in values
        ):
            raise ValueError(f"manifest {field_name} must be a list of strings")
    sources = manifest.get("python_sources", ())
    if not isinstance(sources, (list, tuple)) or not all(
        isinstance(item, dict) for item in sources
    ):
        raise ValueError("manifest python_sources must be a list of objects")
    capabilities = manifest.get("capabilities", {})
    if capabilities is None:
        capabilities = {}
    if not isinstance(capabilities, dict):
        raise ValueError("manifest capabilities must be a JSON object")
    validated = dict(manifest)
    validated["capabilities"] = dict(capabilities)
    return validated


def _persisted_manifest_to_catalog_entry(
    catalog_model_id: str, manifest: object,
) -> CatalogEntry:
    """Project a catalog-persisted manifest dict onto the CatalogEntry shape.

    Resolver-pulled models persist their full synthesized MANIFEST inside
    catalog.json (under the `manifest` key) so that `known_models()` can
    surface them without rerunning discovery. The persisted dict carries
    `backend_path` directly (it was synthesized from the resolver's
    runtime class path), unlike script-discovered models where backend_path
    is computed from the Model class's `__module__:__name__`.
    """
    manifest = _validated_persisted_manifest(catalog_model_id, manifest)
    return CatalogEntry(
        model_id=manifest["model_id"],
        modality=manifest["modality"],
        backend_path=manifest["backend_path"],
        hf_repo=manifest["hf_repo"],
        description=manifest.get("description", ""),
        pip_extras=tuple(manifest.get("pip_extras", ())),
        system_packages=tuple(manifest.get("system_packages", ())),
        python_sources=tuple(manifest.get("python_sources", ())),
        extra=dict(manifest.get("capabilities", {})),
    )


_MODEL_ID_CHARSET_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_CONCRETE_HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


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
    """Apply current curated pins/capabilities, then base_override.

    Operates on a persisted resolver manifest and returns a fresh dict
    (never mutates the input).

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
    if curated is not None:
        if curated.capabilities or curated.code_revision:
            merged_caps = dict(manifest.get("capabilities") or {})
            merged_caps.update(curated.capabilities)
            if curated.code_revision is not None:
                merged_caps["code_revision"] = curated.code_revision
            manifest["capabilities"] = merged_caps
        if curated.revision is not None:
            # This is the currently reviewed target, not a claim about the
            # snapshot already on disk. load_backend compares it with the
            # persisted pull revision and requires a re-pull on mismatch.
            manifest["revision"] = curated.revision

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
            try:
                manifest = _apply_manifest_overlays(model_id, manifest, entry_data)
                entries[model_id] = _persisted_manifest_to_catalog_entry(
                    model_id, manifest,
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "skipping invalid persisted manifest for %s: %s",
                    model_id,
                    exc,
                )
        if key is not None:
            _known_models_cache = (key, entries)
        return entries


def _catalog_file_key(path: Path, info: os.stat_result) -> _CatalogFileKey:
    """Return a replacement-sensitive identity for one opened catalog."""
    return (
        str(path),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _catalog_cache_key() -> _CatalogFileKey | None:
    """Identity of the catalog file the known_models merge was built against.

    (path_str, mtime_ns) for an existing file; a stable sentinel when the
    file does not exist (fresh install -- cacheable until a file appears);
    None when the file exists but cannot be stat'ed (never cache: always
    rebuild rather than risk serving a snapshot we cannot validate).
    """
    p = _catalog_path()
    try:
        info = p.lstat()
    except FileNotFoundError:
        return _MISSING_CATALOG_KEY
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return None
    return _catalog_file_key(p, info)


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
_read_catalog_cache: tuple[_CatalogFileKey, dict] | None = None


class _CatalogSchemaError(ValueError):
    """Internal marker for syntactically valid but unusable catalog JSON."""


def _validate_catalog_json_value(value: Any, *, path: str, depth: int = 0) -> None:
    """Bound arbitrary extension data before caching or copying it."""
    if depth > _MAX_CATALOG_DEPTH:
        raise _CatalogSchemaError(
            f"{path} exceeds maximum nesting depth {_MAX_CATALOG_DEPTH}"
        )
    if isinstance(value, dict):
        if len(value) > _MAX_CATALOG_CONTAINER_ITEMS:
            raise _CatalogSchemaError(f"{path} contains too many fields")
        for key, child in value.items():
            if not isinstance(key, str):
                raise _CatalogSchemaError(f"{path} contains a non-string key")
            _validate_catalog_json_value(
                child,
                path=f"{path}.{key}",
                depth=depth + 1,
            )
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_CATALOG_CONTAINER_ITEMS:
            raise _CatalogSchemaError(f"{path} contains too many items")
        for index, child in enumerate(value):
            _validate_catalog_json_value(
                child,
                path=f"{path}[{index}]",
                depth=depth + 1,
            )
        return
    if isinstance(value, str):
        if len(value) > _MAX_CATALOG_STRING_CHARS:
            raise _CatalogSchemaError(f"{path} contains an oversized string")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _CatalogSchemaError(f"{path} must contain finite numbers")
        return
    raise _CatalogSchemaError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _validate_artifact_receipt(value: Any, *, path: str) -> None:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (list, tuple))
        or not value
    ):
        raise _CatalogSchemaError(f"{path} must be a non-empty list of objects")
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            raise _CatalogSchemaError(f"{item_path} must be an object")
        for field_name in ("repo_id", "revision", "subdir"):
            field = item.get(field_name)
            if not isinstance(field, str):
                raise _CatalogSchemaError(
                    f"{item_path}.{field_name} must be a string"
                )
        revision = item["revision"]
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise _CatalogSchemaError(
                f"{item_path}.revision must be a 40-character commit"
            )
        for field_name in ("allow_patterns", "required_patterns"):
            patterns = item.get(field_name)
            if patterns is None:
                continue
            if (
                isinstance(patterns, (str, bytes))
                or not isinstance(patterns, (list, tuple))
                or not patterns
                or not all(isinstance(pattern, str) and pattern for pattern in patterns)
            ):
                raise _CatalogSchemaError(
                    f"{item_path}.{field_name} must be a non-empty list of strings"
                )


def _validate_measurements(value: Any, *, path: str) -> None:
    if not isinstance(value, dict):
        raise _CatalogSchemaError(f"{path} must be an object")
    for device, bucket in value.items():
        bucket_path = f"{path}.{device}"
        if not isinstance(device, str) or not device:
            raise _CatalogSchemaError(f"{path} device keys must be strings")
        if not isinstance(bucket, dict):
            raise _CatalogSchemaError(f"{bucket_path} must be an object")
        for field_name, field in bucket.items():
            if field_name.endswith("_bytes") and (
                type(field) is not int or field < 0
            ):
                raise _CatalogSchemaError(
                    f"{bucket_path}.{field_name} must be a non-negative integer"
                )
        for field_name in ("device", "source", "observed_at", "shape"):
            if field_name in bucket and not isinstance(bucket[field_name], str):
                raise _CatalogSchemaError(
                    f"{bucket_path}.{field_name} must be a string"
                )


def _validate_catalog_entry(
    model_id: str,
    entry: dict,
    *,
    strict_manifest: bool,
) -> None:
    path = f"entry {model_id!r}"
    if "enabled" in entry and type(entry["enabled"]) is not bool:
        raise _CatalogSchemaError(f"{path}.enabled must be a boolean")
    if "device_override" in entry and entry["device_override"] not in {
        "auto", "cpu", "cuda", "mps",
    }:
        raise _CatalogSchemaError(
            f"{path}.device_override must be auto, cpu, cuda, or mps"
        )
    if "gpu_layers_override" in entry and (
        type(entry["gpu_layers_override"]) is not int
        or entry["gpu_layers_override"] < -1
    ):
        raise _CatalogSchemaError(
            f"{path}.gpu_layers_override must be an integer >= -1"
        )
    for field_name in (
        "pulled_at",
        "hf_repo",
        "local_dir",
        "venv_path",
        "python_path",
        "source",
        "revision",
        "code_revision",
        "base_override",
    ):
        if field_name in entry and (
            not isinstance(entry[field_name], str) or not entry[field_name]
        ):
            raise _CatalogSchemaError(
                f"{path}.{field_name} must be a non-empty string"
            )
    if "measurements" in entry:
        _validate_measurements(entry["measurements"], path=f"{path}.measurements")
    if "artifact_provenance" in entry:
        _validate_artifact_receipt(
            entry["artifact_provenance"],
            path=f"{path}.artifact_provenance",
        )
    if strict_manifest and "manifest" in entry:
        try:
            manifest = _validated_persisted_manifest(model_id, entry["manifest"])
        except ValueError as exc:
            raise _CatalogSchemaError(f"{path}.manifest is invalid: {exc}") from exc
        if "artifact_provenance" in manifest:
            _validate_artifact_receipt(
                manifest["artifact_provenance"],
                path=f"{path}.manifest.artifact_provenance",
            )


def _validate_catalog_shape(
    data: Any,
    *,
    strict_manifests: bool = False,
) -> dict:
    """Validate stable semantics while retaining unknown extension fields.

    Internal writes validate persisted manifests eagerly. Reads leave those
    model-specific semantics to `_validated_persisted_manifest`, whose callers
    quarantine only the bad resolver entry instead of making every otherwise
    valid model unavailable because one legacy/external entry is malformed.
    """
    if not isinstance(data, dict):
        raise _CatalogSchemaError("top level must be a JSON object")
    if len(data) > _MAX_CATALOG_ENTRIES:
        raise _CatalogSchemaError(
            f"catalog cannot contain more than {_MAX_CATALOG_ENTRIES} entries"
        )
    for model_id, entry in data.items():
        if not isinstance(model_id, str) or not model_id:
            raise _CatalogSchemaError(
                "model identifiers must be non-empty strings"
            )
        if not isinstance(entry, dict):
            raise _CatalogSchemaError(
                f"entry for model {model_id!r} must be a JSON object"
            )
        _validate_catalog_json_value(entry, path=f"entry {model_id!r}")
        _validate_catalog_entry(
            model_id,
            entry,
            strict_manifest=strict_manifests,
        )
    return data


def _catalog_read_failure(
    path: Path,
    cached: tuple[_CatalogFileKey, dict] | None,
    reason: str,
) -> dict:
    """Serve last-known-good data or raise the public corruption error."""
    if cached is not None and cached[0][0] == str(path):
        logger.warning(
            "catalog at %s corrupt (%s); serving last-known-good cached parse",
            path,
            reason,
        )
        return _deep_copy_catalog(cached[1])
    logger.error(
        "catalog at %s corrupt (%s); no cached copy to fall back to",
        path,
        reason,
    )
    raise CatalogError(
        f"catalog at {path} is corrupt ({reason}) and no last-known-good "
        "cached copy is available; inspect or restore the file before continuing"
    )


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
    try:
        path_info = p.lstat()
    except FileNotFoundError:
        # Drop any cached state from a prior pulled+removed cycle so a
        # subsequent write produces a cache-miss instead of returning a
        # stale dict.
        _read_catalog_cache = None
        return {}
    except OSError as exc:
        return _catalog_read_failure(p, _read_catalog_cache, f"cannot inspect file: {exc}")
    if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISREG(path_info.st_mode):
        return _catalog_read_failure(
            p, _read_catalog_cache, "path is not a regular non-symlink file",
        )
    if path_info.st_size > _MAX_CATALOG_BYTES:
        return _catalog_read_failure(
            p,
            _read_catalog_cache,
            f"file exceeds {_MAX_CATALOG_BYTES} bytes",
        )
    path_key = _catalog_file_key(p, path_info)

    cached = _read_catalog_cache
    if (
        cached is not None
        and cached[0] == path_key
    ):
        # Deep copy: callers (`pull`, `remove`, `set_enabled`) mutate the
        # returned dict in-place, then write back. Sharing the cached
        # reference would let those mutations bleed into later cache hits.
        return _deep_copy_catalog(cached[1])

    try:
        descriptor = _open_catalog_regular_file(p, os.O_RDONLY)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            opened_info = os.fstat(handle.fileno())
            if opened_info.st_size > _MAX_CATALOG_BYTES:
                return _catalog_read_failure(
                    p,
                    cached,
                    f"file exceeds {_MAX_CATALOG_BYTES} bytes",
                )
            text = handle.read(_MAX_CATALOG_BYTES + 1)
            if len(text.encode("utf-8")) > _MAX_CATALOG_BYTES:
                return _catalog_read_failure(
                    p,
                    cached,
                    f"file exceeds {_MAX_CATALOG_BYTES} bytes",
                )
            opened_key = _catalog_file_key(p, opened_info)
        data = _validate_catalog_shape(json.loads(text))
    except CatalogError as exc:
        return _catalog_read_failure(p, cached, str(exc))
    except json.JSONDecodeError:
        return _catalog_read_failure(p, cached, "invalid JSON")
    except RecursionError:
        return _catalog_read_failure(p, cached, "JSON nesting is too deep")
    except _CatalogSchemaError as exc:
        return _catalog_read_failure(p, cached, f"invalid schema: {exc}")
    except UnicodeError:
        return _catalog_read_failure(p, cached, "invalid UTF-8")
    except OSError as exc:
        return _catalog_read_failure(p, cached, f"cannot read file: {exc}")
    # Backfill enabled=True for pre-enable-flag entries (migration path).
    # Non-destructive: only affects the in-memory dict on read.
    for entry in data.values():
        entry.setdefault("enabled", True)
    _read_catalog_cache = (opened_key, _deep_copy_catalog(data))
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


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync after atomic replacement."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems/platforms do not support directory fsync. The
        # file itself was still fsynced before replace, so retain portability.
        pass
    finally:
        os.close(fd)


def _write_catalog(data: dict) -> None:
    """Durably write catalog JSON through a unique same-directory temp file.

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
    try:
        _validate_catalog_shape(data, strict_manifests=True)
    except (_CatalogSchemaError, RecursionError) as exc:
        raise CatalogError(f"invalid catalog schema: {exc}") from exc
    try:
        serialized = json.dumps(data, indent=2, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise CatalogError(f"catalog contains non-JSON data: {exc}") from exc
    if len(serialized.encode("utf-8")) > _MAX_CATALOG_BYTES:
        raise CatalogError(
            f"catalog cannot exceed {_MAX_CATALOG_BYTES} bytes"
        )
    with _CATALOG_WRITE_LOCK:
        p = _catalog_path()
        _ensure_owned_directory(p.parent)
        fd, tmp_name = tempfile.mkstemp(
            dir=p.parent,
            prefix=f".{p.name}.",
            suffix=".tmp",
            text=True,
        )
        tmp = Path(tmp_name)
        try:
            if os.name == "posix":
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, p)
            _fsync_directory(p.parent)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
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
    working tree). From a wheel/PyPI install, pin the exact invoking release
    so the worker cannot silently run newer protocol/catalog code than its
    host. Mirrors cli_impl.refresh._pip_target_args.
    """
    root = _muse_repo_root()
    if root is not None:
        return ["-e", f"{root}[server]"]
    return [_installed_muse_requirement(["server"])]


def _installed_muse_requirement(extras: list[str]) -> str:
    """Return an exact requirement for the invoking installed Muse release."""
    try:
        installed_version = importlib_metadata.version(_PYPI_DIST)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"cannot create a compatible worker environment: installed "
            f"distribution metadata for {_PYPI_DIST!r} is unavailable"
        ) from exc
    if not installed_version or any(char.isspace() for char in installed_version):
        raise RuntimeError(
            f"cannot create a compatible worker environment: invalid installed "
            f"{_PYPI_DIST} version {installed_version!r}"
        )
    extras_spec = f"[{','.join(extras)}]" if extras else ""
    return f"{_PYPI_DIST}{extras_spec}=={installed_version}"


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
            with _model_pull_lock(curated.id):
                _pull_via_resolver(
                    curated.uri,
                    model_id_override=curated.id,
                    capabilities_overlay=overlay or None,
                    modality_override=curated.modality,
                    base_override=base_override,
                    revision_override=curated.revision,
                    code_revision_override=curated.code_revision,
                )
            return
        # Bundled curated entry: id equals an existing bundled script's
        # model_id. Fall through to the bundled path with that id.
        with _model_pull_lock(curated.id):
            _pull_bundled(curated.id)
        return

    if "://" in identifier:
        # Inherit curated capabilities for this URI even when the user
        # didn't go through the curated id. Without this, copying a URI
        # out of `muse search` output and pasting it into `muse pull`
        # silently strips overlay fields like `safe_labels` (KoalaAI)
        # or `trust_remote_code` (Nomic/Jina) that are required
        # for the model to behave correctly. The curated id, if any,
        # is also preserved so the catalog key stays friendly.
        uri_curated = find_curated_by_uri(identifier)
        existing_source_id = next(
            (
                model_id
                for model_id, entry in _read_catalog().items()
                if isinstance(entry, dict) and entry.get("source") == identifier
            ),
            None,
        )
        # Before resolution we do not yet know the synthesized model id.
        # Two spelling variants can resolve to the same id, so serialize all
        # first-time, non-curated resolver pulls under one identity. Re-pulls
        # use their stable catalog id and retain per-model concurrency.
        lock_identity = (
            uri_curated.id if uri_curated is not None
            else existing_source_id or "__unresolved_resolver_pull__"
        )
        with _model_pull_lock(lock_identity):
            if uri_curated is not None:
                overlay = dict(uri_curated.capabilities or {})
                _pull_via_resolver(
                    identifier,
                    model_id_override=uri_curated.id,
                    capabilities_overlay=overlay or None,
                    modality_override=uri_curated.modality,
                    base_override=base_override,
                    revision_override=uri_curated.revision,
                    code_revision_override=uri_curated.code_revision,
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
        with _model_pull_lock(identifier):
            if uri_curated is not None:
                _pull_via_resolver(
                    source_uri,
                    model_id_override=identifier,
                    capabilities_overlay=uri_curated.capabilities or None,
                    modality_override=uri_curated.modality,
                    base_override=base_override,
                    revision_override=uri_curated.revision,
                    code_revision_override=uri_curated.code_revision,
                )
            else:
                _pull_via_resolver(
                    source_uri,
                    model_id_override=identifier,
                    base_override=base_override,
                    revision_override=existing.get("revision"),
                    code_revision_override=existing.get("code_revision"),
                )
        return

    if base_override:
        logger.warning(
            "--base only applies to resolver-pulled LoRA adapters; "
            "ignored for %s", identifier,
        )

    catalog_known = known_models()
    if identifier in catalog_known:
        with _model_pull_lock(identifier):
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
    """Pull one bundled model while excluding live resource users."""
    with _model_resource_lease(model_id):
        _pull_bundled_with_lease(model_id)


def _pull_bundled_with_lease(model_id: str) -> None:
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
    _ensure_owned_directory(venvs_root, private=True)
    venv_path = venvs_root / model_id

    with venv_transaction(venv_path) as tx:
        _pull_bundled_transaction(model_id, entry, tx.path)
        tx.commit()


@dataclass(frozen=True)
class _BundledDownloadSpec:
    """Validated immutable inputs and receipt for one script-based model."""

    revision: str | None
    allow_patterns: tuple[str, ...] | None
    artifacts: tuple[Any, ...] | None
    artifact_provenance: tuple[dict[str, Any], ...] | None


def _artifact_provenance_item(
    *,
    repo_id: str,
    revision: str,
    subdir: str,
    allow_patterns: tuple[str, ...] | None,
    required_patterns: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe canonical artifact receipt item."""
    item = {
        "repo_id": repo_id,
        "revision": revision,
        "subdir": subdir,
        "allow_patterns": (
            list(allow_patterns) if allow_patterns is not None else None
        ),
    }
    if required_patterns is not None:
        item["required_patterns"] = list(required_patterns)
    return item


def _bundled_download_spec(
    model_id: str,
    entry: CatalogEntry,
) -> _BundledDownloadSpec:
    """Validate script download metadata before any environment mutation.

    Packaged manifests must be immutable. User drop-in scripts retain the
    historical option of an unpinned snapshot, but receive the same receipt
    protection whenever they opt into a concrete revision.
    """
    raw_revision = entry.extra.get("revision")
    if raw_revision is None:
        if entry.bundled:
            raise CatalogError(
                f"bundled model {model_id!r} must declare an immutable "
                "40-character revision"
            )
        revision = None
    elif (
        not isinstance(raw_revision, str)
        or _CONCRETE_HF_REVISION_RE.fullmatch(raw_revision) is None
    ):
        raise CatalogError(
            f"bundled model {model_id!r} has an invalid immutable revision"
        )
    else:
        revision = raw_revision

    raw_allow_patterns = entry.extra.get("allow_patterns")
    if raw_allow_patterns is not None and (
        isinstance(raw_allow_patterns, (str, bytes))
        or not isinstance(raw_allow_patterns, (list, tuple))
        or not all(
            isinstance(pattern, str) and pattern
            for pattern in raw_allow_patterns
        )
    ):
        raise CatalogError(
            f"bundled model {model_id!r} has invalid allow_patterns"
        )
    allow_patterns = (
        tuple(raw_allow_patterns) if raw_allow_patterns is not None else None
    )

    if "hf_artifacts" not in entry.extra:
        provenance = None
        if revision is not None:
            provenance = (
                _artifact_provenance_item(
                    repo_id=entry.hf_repo,
                    revision=revision,
                    subdir="",
                    allow_patterns=allow_patterns,
                ),
            )
        return _BundledDownloadSpec(
            revision=revision,
            allow_patterns=allow_patterns,
            artifacts=None,
            artifact_provenance=provenance,
        )

    from muse.core.artifacts import ArtifactBundleError, normalize_hf_artifacts

    try:
        artifacts = normalize_hf_artifacts(entry.extra.get("hf_artifacts"))
    except ArtifactBundleError as exc:
        raise CatalogError(
            f"invalid hf_artifacts for bundled model {model_id!r}: {exc}"
        ) from exc
    primary = artifacts[0]
    if primary.repo_id != entry.hf_repo:
        raise CatalogError(
            f"bundled model {model_id!r} primary artifact repo "
            f"{primary.repo_id!r} does not match hf_repo {entry.hf_repo!r}"
        )
    if revision is not None and revision != primary.revision:
        raise CatalogError(
            f"bundled model {model_id!r} revision does not match its "
            "primary artifact revision"
        )
    revision = primary.revision
    provenance = tuple(
        _artifact_provenance_item(
            repo_id=artifact.repo_id,
            revision=artifact.revision,
            subdir=artifact.subdir,
            allow_patterns=artifact.allow_patterns,
            required_patterns=artifact.required_patterns,
        )
        for artifact in artifacts
    )
    return _BundledDownloadSpec(
        revision=revision,
        allow_patterns=allow_patterns,
        artifacts=artifacts,
        artifact_provenance=provenance,
    )


_OPERATOR_STATE_FIELDS = (
    "enabled",
    "device_override",
    "gpu_layers_override",
)


def _preserve_operator_state(
    replacement: dict[str, Any],
    current: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge live operator controls into freshly pulled model metadata.

    Pull work deliberately runs outside the catalog lock.  Re-read and merge
    these fields only at the final locked commit so an admin edit made while a
    download is running cannot be replaced by the pull's stale defaults.
    """
    if current is None:
        return replacement
    for field in _OPERATOR_STATE_FIELDS:
        if field in current:
            replacement[field] = copy.deepcopy(current[field])
    return replacement


def _pull_bundled_transaction(
    model_id: str,
    entry: CatalogEntry,
    venv_path: Path,
) -> None:
    """Build and register one bundled model inside a venv transaction."""

    download_spec = _bundled_download_spec(model_id, entry)

    ensure_venv(venv_path, creator=create_venv)

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

    # Synchronize even when empty: removing a reviewed source declaration must
    # revoke its old .pth activation from a cloned/reused environment.
    install_python_sources(venv_path, entry.python_sources)

    # Bundled MANIFESTs may declare `capabilities.allow_patterns` to
    # restrict the snapshot_download manifest (mirrors what the resolver
    # plugins do for fp16-shaped or BIN-only repos). This avoids hauling
    # down fp32 siblings, .bin/.h5 dupes, and standalone single-file
    # checkpoints when the diffusers/transformers runtime only needs the
    # subfolder weights.
    allow_patterns = download_spec.allow_patterns
    revision = download_spec.revision
    if download_spec.artifacts is not None:
        from muse.core.artifacts import download_hf_artifact_bundle

        weights_cache = _catalog_dir() / "weights"
        _ensure_owned_directory(weights_cache, private=True)
        with _hf_quiet_if_needed():
            local_dir = download_hf_artifact_bundle(
                weights_cache,
                bundle_name=model_id,
                artifacts=download_spec.artifacts,
                snapshot_download_fn=snapshot_download,
            )
    else:
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
        replacement = {
            "pulled_at": datetime.now(timezone.utc).isoformat(),
            "hf_repo": entry.hf_repo,
            "local_dir": str(local_dir),
            "venv_path": str(venv_path),
            "python_path": str(venv_python(venv_path)),
            "enabled": True,
        }
        if revision:
            replacement["revision"] = revision
        if download_spec.artifact_provenance is not None:
            replacement["artifact_provenance"] = copy.deepcopy(
                list(download_spec.artifact_provenance)
            )
        catalog[model_id] = _preserve_operator_state(
            replacement,
            catalog.get(model_id),
        )
        _write_catalog(catalog)
    _reset_known_models_cache()


def _pull_via_resolver(
    uri: str,
    *,
    model_id_override: str | None = None,
    capabilities_overlay: dict | None = None,
    modality_override: str | None = None,
    base_override: str | None = None,
    revision_override: str | None = None,
    code_revision_override: str | None = None,
    _resolved: object | None = None,
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

    `revision_override` pins the Hugging Face model snapshot selected by a
    curated entry. `code_revision_override` separately pins a repository
    referenced by Transformers `auto_map`; it is persisted as a runtime
    capability so the loader can pass `code_revision` to Transformers.

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
    resolved = _resolved
    if resolved is None:
        resolved = resolve(
            uri,
            modality=modality_override,
            base_override=base_override,
            revision=revision_override,
        )
    manifest = dict(resolved.manifest)
    # Resolver may put backend_path in the manifest itself, or only on
    # the ResolvedModel. Persist it consistently so load_backend can
    # find it without consulting the resolver again.
    manifest.setdefault("backend_path", resolved.backend_path)

    if capabilities_overlay:
        merged_caps = dict(manifest.get("capabilities") or {})
        merged_caps.update(capabilities_overlay)
        manifest["capabilities"] = merged_caps

    if code_revision_override is not None:
        merged_caps = dict(manifest.get("capabilities") or {})
        merged_caps["code_revision"] = code_revision_override
        manifest["capabilities"] = merged_caps

    if model_id_override:
        manifest["model_id"] = model_id_override
    model_id = manifest["model_id"]
    _validate_model_id_for_fs(model_id)

    if _resolved is None:
        # Resolver URIs do not reveal their synthesized model id until after
        # resolution. Re-enter with that immutable result so the complete
        # venv/download/catalog mutation runs under the model's lifetime
        # resource lease without resolving or downloading twice.
        with _model_resource_lease(model_id):
            return _pull_via_resolver(
                uri,
                model_id_override=model_id_override,
                capabilities_overlay=capabilities_overlay,
                modality_override=modality_override,
                base_override=base_override,
                revision_override=revision_override,
                code_revision_override=code_revision_override,
                _resolved=resolved,
            )

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

    capabilities = manifest.get("capabilities") or {}
    if capabilities.get("trust_remote_code"):
        resolved_revision = manifest.get("revision")
        if not isinstance(resolved_revision, str) or not (
            _CONCRETE_HF_REVISION_RE.fullmatch(resolved_revision)
        ):
            from muse.core.resolvers import ResolverError

            raise ResolverError(
                f"refusing to enable trust_remote_code for {model_id!r} "
                "without an immutable Hugging Face commit"
            )
        if revision_override is not None and resolved_revision != revision_override:
            from muse.core.resolvers import ResolverError

            raise ResolverError(
                f"resolver did not honor reviewed revision {revision_override} "
                f"for {model_id!r}; got {resolved_revision!r}"
            )
        code_revision = capabilities.get("code_revision")
        if code_revision is not None and (
            not isinstance(code_revision, str)
            or not _CONCRETE_HF_REVISION_RE.fullmatch(code_revision)
        ):
            from muse.core.resolvers import ResolverError

            raise ResolverError(
                f"refusing to enable external remote code for {model_id!r} "
                "without an immutable code commit"
            )
        if revision_override is None:
            logger.warning(
                "model %s enables trust_remote_code at discovered commit %s; "
                "this code was not selected by Muse's curated review pins",
                model_id,
                resolved_revision,
            )

    _validate_lora_capabilities(manifest)

    venvs_root = _catalog_dir() / "venvs"
    _ensure_owned_directory(venvs_root, private=True)
    venv_path = venvs_root / model_id

    with venv_transaction(venv_path) as tx:
        _pull_resolved_transaction(
            uri=uri,
            model_id=model_id,
            manifest=manifest,
            effective_base_override=effective_base_override,
            resolved=resolved,
            venv_path=tx.path,
        )
        tx.commit()


def _pull_resolved_transaction(
    *,
    uri: str,
    model_id: str,
    manifest: dict[str, Any],
    effective_base_override: str | None,
    resolved: Any,
    venv_path: Path,
) -> None:
    """Build and register one resolved model inside a venv transaction."""

    ensure_venv(venv_path, creator=create_venv)

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

    python_sources = manifest.get("python_sources") or ()
    # Synchronize even when empty so a re-pull revokes removed source hooks.
    install_python_sources(venv_path, python_sources)

    raw_receipt = getattr(resolved, "artifact_provenance", ())
    if isinstance(raw_receipt, (str, bytes)) or not isinstance(raw_receipt, Sequence):
        from muse.core.resolvers import ResolverError

        raise ResolverError("resolver artifact provenance must be a sequence")
    artifact_provenance: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_receipt):
        if not isinstance(raw_item, Mapping):
            from muse.core.resolvers import ResolverError

            raise ResolverError(
                f"resolver artifact provenance item {index} is not an object"
            )
        artifact_provenance.append(copy.deepcopy(dict(raw_item)))
    if not artifact_provenance:
        resolved_revision = manifest.get("revision")
        if resolved_revision:
            artifact_provenance.append({
                "repo_id": manifest["hf_repo"],
                "revision": resolved_revision,
                "subdir": ".",
            })
    if artifact_provenance:
        manifest["artifact_provenance"] = copy.deepcopy(artifact_provenance)

    weights_cache = _catalog_dir() / "weights"
    _ensure_owned_directory(weights_cache, private=True)
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
        resolved_revision = manifest.get("revision")
        if resolved_revision:
            entry["revision"] = resolved_revision
        if artifact_provenance:
            entry["artifact_provenance"] = copy.deepcopy(artifact_provenance)
        code_revision = (manifest.get("capabilities") or {}).get("code_revision")
        if code_revision:
            entry["code_revision"] = code_revision
        if effective_base_override:
            entry["base_override"] = effective_base_override
        catalog[model_id] = _preserve_operator_state(
            entry,
            catalog.get(model_id),
        )
        _write_catalog(catalog)
    _reset_known_models_cache()


def _strict_owned_purge_path(
    raw_path: str,
    root: Path,
    *,
    field_name: str,
    model_id: str,
) -> Path:
    """Resolve a venv target and require this model's exact owned directory."""
    try:
        resolved_root = root.expanduser().resolve()
        resolved_path = Path(raw_path).expanduser().resolve()
        relative = resolved_path.relative_to(resolved_root)
    except (TypeError, ValueError, OSError) as exc:
        raise CatalogError(
            f"refusing to purge model {model_id!r}: catalog {field_name} "
            "does not resolve inside the owned venv root"
        ) from exc
    if relative.parts != (model_id,):
        expected_path = resolved_root / model_id
        raise CatalogError(
            f"refusing to purge model {model_id!r}: catalog {field_name} "
            f"does not point at expected model directory {expected_path}"
        )
    return resolved_path


def _owned_weights_purge_path(
    raw_path: str,
    weights_root: Path,
    *,
    model_id: str,
) -> Path | None:
    """Return an owned weights target; preserve legitimate external caches."""
    try:
        resolved_root = weights_root.expanduser().resolve()
        resolved_path = Path(raw_path).expanduser().resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise CatalogError(
            f"refusing to purge model {model_id!r}: invalid catalog local_dir"
        ) from exc
    if resolved_path == resolved_root:
        raise CatalogError(
            f"refusing to purge model {model_id!r}: catalog local_dir points "
            f"at owned root {resolved_root}, not a model directory"
        )
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        # Bundled pulls normally live in Hugging Face's shared cache. Muse
        # does not own that location, so unregister while preserving it.
        return None
    return resolved_path


def _other_catalog_path_reference(
    catalog: dict,
    *,
    model_id: str,
    field_name: str,
    target: Path,
) -> str | None:
    """Return another entry whose resource path overlaps ``target``.

    Recursive deletion of a parent damages a child reference, while deleting
    a child can damage a model that treats the parent as its snapshot root.
    Preserve either relationship, not only byte-for-byte path equality.
    """
    for other_model_id, other_entry in catalog.items():
        if other_model_id == model_id or not isinstance(other_entry, dict):
            continue
        raw_path = other_entry.get(field_name)
        if not raw_path:
            continue
        try:
            other_target = Path(raw_path).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            continue
        try:
            target.relative_to(other_target)
            overlaps = True
        except ValueError:
            try:
                other_target.relative_to(target)
                overlaps = True
            except ValueError:
                overlaps = False
        if overlaps:
            return other_model_id
    return None


def _purge_owned_directory(
    target: Path,
    *,
    model_id: str,
    label: str,
) -> None:
    """Remove one validated owned path without hiding cleanup failures.

    Symlinks and non-directories are unlinked as one entry. Recursive
    deletion is allowed only when the platform's `shutil.rmtree` advertises
    its fd-safe implementation, matching artifact/installer staging cleanup.
    """
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        return
    except OSError as exc:
        raise CatalogError(
            f"could not inspect {label} for model {model_id!r}: {target}: {exc}"
        ) from exc

    try:
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            target.unlink()
            return
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            raise CatalogError(
                f"refusing to purge {label} for model {model_id!r} without "
                f"fd-safe recursive deletion support: {target}"
            )
        shutil.rmtree(target)
    except FileNotFoundError:
        # Another cleanup owner may already have removed the exact path.
        return
    except CatalogError:
        raise
    except OSError as exc:
        raise CatalogError(
            f"could not purge {label} for model {model_id!r}: {target}: {exc}"
        ) from exc


def remove(model_id: str, *, purge: bool = False) -> None:
    """Serialize unregister/purge against every pull of the same model."""
    with _model_pull_lock(model_id):
        with _model_resource_lease(model_id):
            _remove_locked(model_id, purge=purge)


def _remove_locked(model_id: str, *, purge: bool = False) -> None:
    """Unregister `model_id` from the catalog.

    By default this only edits `catalog.json`; the per-model venv at
    `~/.muse/venvs/<model_id>/` stays on disk. Mirrors `apt remove`'s
    "metadata only" semantics.

    When `purge=True`:
      - rmtree the venv directory unless another catalog entry references it.
      - rmtree the resolver weights cache at `~/.muse/weights/<dir>/`
        IF the entry's `local_dir` resolves under the muse-owned
        weights tree and no other catalog entry references it.
        Bundled-pulled models that store weights in the shared HF cache
        (`~/.cache/huggingface`) are left alone; muse does not own that,
        and `huggingface-cli delete-cache` is the right tool for it.

    Tolerates either path being already gone. Once catalog removal commits,
    every independent owned target is attempted; unsafe or failed filesystem
    cleanup raises ``CatalogError`` instead of reporting a false success.

    Holds _CATALOG_WRITE_LOCK for the full read->pop->write sequence. The
    rmtree steps run outside that transaction lock but remain under the
    per-model pull lock held by `remove`, so a re-pull cannot recreate a
    directory between unregister and deletion.
    """
    venv_target: Path | None = None
    weights_target: Path | None = None
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        entry = catalog.get(model_id, {}) or {}
        if purge:
            venv_path = entry.get("venv_path")
            local_dir = entry.get("local_dir")
            if venv_path:
                venv_target = _strict_owned_purge_path(
                    venv_path,
                    _catalog_dir() / "venvs",
                    field_name="venv_path",
                    model_id=model_id,
                )
            if local_dir:
                weights_target = _owned_weights_purge_path(
                    local_dir,
                    _catalog_dir() / "weights",
                    model_id=model_id,
                )
            if venv_target is not None:
                other_model_id = _other_catalog_path_reference(
                    catalog,
                    model_id=model_id,
                    field_name="venv_path",
                    target=venv_target,
                )
                if other_model_id is not None:
                    logger.info(
                        "preserving venv for model %s because model %s "
                        "also references %s",
                        model_id,
                        other_model_id,
                        venv_target,
                    )
                    venv_target = None
            if weights_target is not None:
                other_model_id = _other_catalog_path_reference(
                    catalog,
                    model_id=model_id,
                    field_name="local_dir",
                    target=weights_target,
                )
                if other_model_id is not None:
                    logger.info(
                        "preserving weights for model %s because model %s "
                        "also references %s",
                        model_id,
                        other_model_id,
                        weights_target,
                    )
                    weights_target = None
        catalog.pop(model_id, None)
        _write_catalog(catalog)
    # Resolver-pulled entries appear in known_models() via the persisted
    # manifest path; once removed, that cache must drop them too or
    # `muse models list` keeps reporting a model that no longer exists.
    _reset_known_models_cache()
    if not purge:
        return
    failures: list[str] = []
    for label, target in (
        ("venv", venv_target),
        ("weights", weights_target),
    ):
        if target is None:
            continue
        try:
            _purge_owned_directory(
                target,
                model_id=model_id,
                label=label,
            )
        except CatalogError as exc:
            # Attempt every independently-owned target so one filesystem
            # failure does not strand the other resource unnecessarily.
            failures.append(str(exc))
    if failures:
        raise CatalogError("; ".join(failures))


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


def _validate_remote_code_catalog_pins(
    model_id: str,
    manifest: dict,
    entry_data: dict,
) -> None:
    """Refuse a local remote-code snapshot that predates current review pins."""
    capabilities = manifest.get("capabilities") or {}
    if not capabilities.get("trust_remote_code"):
        return

    mismatches: list[str] = []
    expected_revision = manifest.get("revision")
    if expected_revision and entry_data.get("revision") != expected_revision:
        mismatches.append("model revision")
    expected_code_revision = capabilities.get("code_revision")
    if (
        expected_code_revision
        and entry_data.get("code_revision") != expected_code_revision
    ):
        mismatches.append("external code revision")
    if mismatches:
        raise RuntimeError(
            f"model {model_id!r} does not match the current reviewed "
            f"remote-code pin ({', '.join(mismatches)}); re-run "
            f"`muse pull {model_id}` before loading it"
        )


def _validate_bundled_catalog_provenance(
    model_id: str,
    entry: CatalogEntry,
    entry_data: dict,
) -> None:
    """Reject script snapshots that do not match their complete receipt."""
    # Resolver downloads persist their own manifest/source and use resolver
    # pin validation. A receipt is written only by the script-based pull path.
    if "manifest" in entry_data:
        return
    if not entry.bundled and "artifact_provenance" not in entry_data:
        return

    spec = _bundled_download_spec(model_id, entry)
    expected = (
        list(spec.artifact_provenance)
        if spec.artifact_provenance is not None
        else None
    )
    mismatches: list[str] = []
    if entry_data.get("revision") != spec.revision:
        mismatches.append("primary revision")
    if entry_data.get("artifact_provenance") != expected:
        mismatches.append("artifact receipt")
    if mismatches:
        raise RuntimeError(
            f"model {model_id!r} does not match the current immutable "
            f"artifact provenance ({', '.join(mismatches)}); re-run "
            f"`muse pull {model_id}` before loading it"
        )


def _validate_resolver_catalog_provenance(
    model_id: str,
    manifest: dict,
    entry_data: dict,
) -> None:
    """Reject resolver snapshots stale against current pins or their receipt."""
    if "manifest" not in entry_data or not entry_data.get("source"):
        return

    mismatches: list[str] = []
    expected_revision = manifest.get("revision")
    if expected_revision and entry_data.get("revision") != expected_revision:
        mismatches.append("primary revision")

    expected_receipt = manifest.get("artifact_provenance")
    if expected_receipt is not None:
        if entry_data.get("artifact_provenance") != expected_receipt:
            mismatches.append("artifact receipt")
        if expected_revision:
            primary_repo = manifest.get("hf_repo")
            primary_revisions = {
                item.get("revision")
                for item in expected_receipt
                if isinstance(item, dict) and item.get("repo_id") == primary_repo
            } if isinstance(expected_receipt, list) else set()
            if expected_revision not in primary_revisions:
                mismatches.append("manifest artifact receipt")
    if mismatches:
        raise RuntimeError(
            f"model {model_id!r} does not match the current immutable resolver "
            f"provenance ({', '.join(mismatches)}); re-run "
            f"`muse pull {model_id}` before loading it"
        )


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
    catalog = _read_catalog()
    entry_data = catalog[model_id]
    manifest = get_manifest(model_id)
    _validate_remote_code_catalog_pins(
        model_id,
        manifest,
        entry_data,
    )
    _validate_resolver_catalog_provenance(model_id, manifest, entry_data)
    _validate_bundled_catalog_provenance(model_id, entry, entry_data)
    module_path, class_name = entry.backend_path.split(":", 1)
    module = _import_backend_module(module_path)
    cls = getattr(module, class_name)
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
    catalog = _read_catalog()
    entry_data = catalog.get(model_id, {})
    persisted = entry_data.get("manifest")
    with _KNOWN_MODELS_LOCK:
        is_discovered = (
            _discovered_entries_cache is not None
            and model_id in _discovered_entries_cache
        )
    if model_id not in catalog_known:
        if persisted is not None:
            try:
                _validated_persisted_manifest(model_id, persisted)
            except (TypeError, ValueError) as exc:
                raise CatalogError(
                    f"catalog entry for {model_id!r} has an invalid manifest: {exc}"
                ) from exc
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    if persisted is not None and not is_discovered:
        try:
            overlaid = _apply_manifest_overlays(model_id, persisted, entry_data)
            return _validated_persisted_manifest(model_id, overlaid)
        except (KeyError, TypeError, ValueError) as exc:
            raise CatalogError(
                f"catalog entry for {model_id!r} has an invalid manifest: {exc}"
            ) from exc
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
