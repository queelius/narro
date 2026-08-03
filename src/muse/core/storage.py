"""Storage inventory and conservative garbage collection for Muse.

The catalog is authoritative for liveness.  Paths in Muse-owned roots that
are not catalog-referenced may be intentional retained caches, so they are
reported but never deleted unless the caller explicitly opts in.  Shared
Hugging Face and pip caches are accounting-only and are never cleanup targets.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from muse.core.catalog import (
    _CATALOG_WRITE_LOCK,
    _catalog_dir,
    _model_pull_lock,
    _model_resource_lease,
    _owned_weights_purge_path,
    _purge_owned_directory,
    _read_catalog,
    _storage_cache_lock,
    _validate_model_id_for_fs,
    CatalogError,
    ModelInUseError,
    StorageBusyError,
)
_VENV_STAGING_RE = re.compile(
    r"^\.(?P<model>[A-Za-z0-9._-]+)\."
    r"(?P<transaction>staging|transaction)-[A-Za-z0-9._-]+$"
)
_WEIGHTS_STAGING_RE = re.compile(r"^\.[^.].*\.staging-[A-Za-z0-9._-]+$")


class StorageInspectionError(RuntimeError):
    """Raised when storage cannot be classified safely."""


@dataclass(frozen=True)
class PathIdentity:
    device: int
    inode: int
    mode: int


@dataclass(frozen=True)
class StorageItem:
    kind: str
    status: str
    path: Path
    physical_bytes: int
    latest_mtime: float
    identity: PathIdentity
    model_id: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class StorageIssue:
    kind: str
    model_id: str | None
    path: Path | None
    detail: str


@dataclass(frozen=True)
class StorageReport:
    catalog_dir: Path
    scanned_at: float
    filesystem_total_bytes: int
    filesystem_free_bytes: int
    muse_bytes: int
    venv_bytes: int
    weights_bytes: int
    referenced_venvs: tuple[StorageItem, ...]
    unreferenced_venvs: tuple[StorageItem, ...]
    referenced_weights: tuple[StorageItem, ...]
    unreferenced_weights: tuple[StorageItem, ...]
    incomplete_downloads: tuple[StorageItem, ...]
    abandoned_staging: tuple[StorageItem, ...]
    recovery_workspaces: tuple[StorageItem, ...]
    issues: tuple[StorageIssue, ...]
    venv_references_safe: bool
    weights_references_safe: bool
    owned_hf_recognized_bytes: int
    owned_hf_warnings: tuple[str, ...]
    shared_hf_cache: Path | None
    shared_hf_bytes: int
    shared_hf_referenced_bytes: int
    pip_cache: Path | None
    pip_cache_bytes: int


@dataclass(frozen=True)
class PrunePlan:
    created_at: float
    older_than_seconds: float
    include_unreferenced: bool
    candidates: tuple[StorageItem, ...]
    notices: tuple[str, ...] = ()

    @property
    def estimated_bytes(self) -> int:
        return sum(item.physical_bytes for item in self.candidates)


@dataclass(frozen=True)
class PruneOutcome:
    item: StorageItem
    action: str
    detail: str
    reclaimed_bytes: int = 0


@dataclass(frozen=True)
class PruneResult:
    dry_run: bool
    outcomes: tuple[PruneOutcome, ...]

    @property
    def reclaimed_bytes(self) -> int:
        return sum(outcome.reclaimed_bytes for outcome in self.outcomes)

    @property
    def failures(self) -> tuple[PruneOutcome, ...]:
        return tuple(
            outcome for outcome in self.outcomes
            if outcome.action in {"failed", "busy"}
        )


@dataclass(frozen=True)
class AutomaticPruneResult:
    """One low-space pre-pull maintenance pass."""

    threshold_bytes: int
    free_bytes_before: int
    free_bytes_after: int
    plan: PrunePlan
    result: PruneResult


def _identity(info: os.stat_result) -> PathIdentity:
    return PathIdentity(info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))


def _allocated_bytes(info: os.stat_result) -> int:
    blocks = getattr(info, "st_blocks", None)
    return int(blocks * 512) if blocks is not None else int(info.st_size)


def _measure_path(
    path: Path,
    *,
    seen: set[tuple[int, int]] | None = None,
    warnings: list[str] | None = None,
    fail_on_error: bool = False,
) -> tuple[int, float]:
    """Return allocated bytes and newest descendant mtime without links."""
    if seen is None:
        seen = set()
    total = 0
    latest = 0.0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            if fail_on_error:
                raise StorageInspectionError(
                    f"path changed while inspecting {current}: {exc}"
                ) from exc
            continue
        except OSError as exc:
            if fail_on_error:
                raise StorageInspectionError(
                    f"could not inspect every entry below {path}: {exc}"
                ) from exc
            if warnings is not None:
                warnings.append(f"could not inspect {current}: {exc}")
            continue
        key = (info.st_dev, info.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += _allocated_bytes(info)
        latest = max(latest, float(info.st_mtime))
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            continue
        try:
            with os.scandir(current) as entries:
                children = [Path(entry.path) for entry in entries]
        except OSError as exc:
            if fail_on_error:
                raise StorageInspectionError(
                    f"could not inspect every entry below {path}: {exc}"
                ) from exc
            if warnings is not None:
                warnings.append(f"could not list {current}: {exc}")
            continue
        stack.extend(children)
    return total, latest


def _du_sizes(paths: Iterable[Path], warnings: list[str]) -> dict[str, int]:
    """Use the platform du for fast allocated-byte accounting, with fallback."""
    ordered = list(paths)
    if not ordered:
        return {}
    command = ["du", "-sk", "--", *(str(path) for path in ordered)]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        warnings.append(f"could not run du for storage accounting: {exc}")
        completed = None
    sizes: dict[str, int] = {}
    if completed is not None:
        if completed.returncode != 0 and completed.stderr.strip():
            warnings.append(f"du reported an inspection error: {completed.stderr.strip()}")
        for line in completed.stdout.splitlines():
            fields = line.split(None, 1)
            if len(fields) != 2:
                continue
            try:
                sizes[_path_key(fields[1])] = int(fields[0]) * 1024
            except (OSError, TypeError, ValueError):
                continue
    for path in ordered:
        key = _path_key(path)
        if key not in sizes:
            size, _ = _measure_path(path, warnings=warnings)
            sizes[key] = size
    return sizes


def _direct_items(
    root: Path,
    *,
    kind: str,
    warnings: list[str],
) -> tuple[list[StorageItem], int]:
    try:
        root_info = root.lstat()
    except FileNotFoundError:
        return [], 0
    except OSError as exc:
        warnings.append(f"could not inspect {root}: {exc}")
        return [], 0
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        warnings.append(f"managed root is not a real directory: {root}")
        return [], _allocated_bytes(root_info)

    root_bytes = _allocated_bytes(root_info)
    try:
        with os.scandir(root) as entries:
            paths = sorted((Path(entry.path) for entry in entries), key=lambda p: p.name)
    except OSError as exc:
        warnings.append(f"could not list {root}: {exc}")
        return [], root_bytes

    sizes = _du_sizes(paths, warnings)
    items: list[StorageItem] = []
    for path in paths:
        try:
            info = path.lstat()
        except OSError as exc:
            warnings.append(f"could not inspect {path}: {exc}")
            continue
        items.append(StorageItem(
            kind=kind,
            status="unknown",
            path=path,
            physical_bytes=sizes[_path_key(path)],
            latest_mtime=float(info.st_mtime),
            identity=_identity(info),
        ))
    return items, root_bytes + sum(item.physical_bytes for item in items)


def _path_key(path: str | os.PathLike[str] | Path) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def _top_level_owned_path(raw_path: object, root: Path) -> Path | None:
    if not isinstance(raw_path, (str, os.PathLike)) or not raw_path:
        return None
    try:
        relative = Path(_path_key(raw_path)).relative_to(Path(_path_key(root)))
    except (TypeError, ValueError, OSError):
        return None
    if not relative.parts:
        return None
    return root / relative.parts[0]


def _replace_item(item: StorageItem, *, status: str, reason: str) -> StorageItem:
    return StorageItem(
        kind=item.kind,
        status=status,
        path=item.path,
        physical_bytes=item.physical_bytes,
        latest_mtime=item.latest_mtime,
        identity=item.identity,
        model_id=item.model_id,
        reason=reason,
    )


def _with_newest_mtime(
    item: StorageItem,
    warnings: list[str],
) -> StorageItem | None:
    try:
        _, latest = _measure_path(item.path, fail_on_error=True)
    except StorageInspectionError as exc:
        warnings.append(str(exc))
        return None
    return StorageItem(
        kind=item.kind,
        status=item.status,
        path=item.path,
        physical_bytes=item.physical_bytes,
        latest_mtime=latest,
        identity=item.identity,
        model_id=item.model_id,
        reason=item.reason,
    )


def _find_incomplete_files(root: Path, warnings: list[str]) -> list[StorageItem]:
    """Find Hugging Face blob partials, not arbitrary suffix matches.

    ``*.incomplete`` is an HF cache implementation detail only for direct
    files below ``models--*/blobs``. A flat/artifact model may legitimately
    own a payload with that suffix, so a recursive name search is unsafe.
    """
    items: list[StorageItem] = []
    try:
        root_info = root.lstat()
    except FileNotFoundError:
        return items
    except OSError as exc:
        warnings.append(f"could not inspect {root}: {exc}")
        return items
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        return items
    try:
        with os.scandir(root) as entries:
            repositories = [
                Path(entry.path) for entry in entries
                if entry.name.startswith("models--")
            ]
    except OSError as exc:
        warnings.append(f"could not list {root}: {exc}")
        return items
    for repository in repositories:
        try:
            repository_info = repository.lstat()
        except OSError as exc:
            warnings.append(f"could not inspect {repository}: {exc}")
            continue
        if (
            stat.S_ISLNK(repository_info.st_mode)
            or not stat.S_ISDIR(repository_info.st_mode)
        ):
            continue
        blobs = repository / "blobs"
        try:
            blobs_info = blobs.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            warnings.append(f"could not inspect {blobs}: {exc}")
            continue
        if stat.S_ISLNK(blobs_info.st_mode) or not stat.S_ISDIR(blobs_info.st_mode):
            continue
        try:
            with os.scandir(blobs) as entries:
                children = list(entries)
        except OSError as exc:
            warnings.append(f"could not list {blobs}: {exc}")
            continue
        for entry in children:
            path = Path(entry.path)
            try:
                info = path.lstat()
            except OSError as exc:
                warnings.append(f"could not inspect {path}: {exc}")
                continue
            if stat.S_ISREG(info.st_mode) and path.name.endswith(".incomplete"):
                items.append(StorageItem(
                    kind="incomplete-download",
                    status="safe-garbage",
                    path=path,
                    physical_bytes=_allocated_bytes(info),
                    latest_mtime=float(info.st_mtime),
                    identity=_identity(info),
                    reason="unfinished download in Muse's owned weights cache",
                ))
    return sorted(items, key=lambda item: str(item.path))


def _complete_venv(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        return False
    python = path / "bin" / "python"
    return python.is_file() and os.access(python, os.X_OK)


def _shared_cache_roots() -> tuple[Path | None, Path | None]:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        hf_cache = Path(HF_HUB_CACHE).expanduser()
    except Exception:  # noqa: BLE001 - diagnostics must survive optional drift
        hf_cache = None
    pip_raw = os.environ.get("PIP_CACHE_DIR")
    if pip_raw:
        pip_cache = Path(pip_raw).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        pip_cache = base / "pip"
    return hf_cache, pip_cache


def _filesystem_usage(path: Path) -> tuple[int, int]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return int(usage.total), int(usage.free)


def inspect_storage(*, include_shared: bool = True) -> StorageReport:
    """Inventory Muse-owned storage without creating or mutating anything."""
    scanned_at = time.time()
    catalog_dir = _catalog_dir()
    venvs_root = catalog_dir / "venvs"
    weights_root = catalog_dir / "weights"
    warnings: list[str] = []
    issues: list[StorageIssue] = []
    try:
        catalog = _read_catalog()
    except CatalogError as exc:
        raise StorageInspectionError(str(exc)) from exc

    filesystem_total, filesystem_free = _filesystem_usage(catalog_dir)
    venv_items, venv_bytes = _direct_items(
        venvs_root, kind="venv", warnings=warnings,
    )
    weight_items, weights_bytes = _direct_items(
        weights_root, kind="weights", warnings=warnings,
    )

    muse_bytes = venv_bytes + weights_bytes
    try:
        catalog_info = catalog_dir.lstat()
    except FileNotFoundError:
        catalog_info = None
    except OSError as exc:
        warnings.append(f"could not inspect {catalog_dir}: {exc}")
        catalog_info = None
    if catalog_info is not None:
        muse_bytes += _allocated_bytes(catalog_info)
        try:
            with os.scandir(catalog_dir) as entries:
                other_paths = sorted(
                    (
                        Path(entry.path) for entry in entries
                        if entry.name not in {"venvs", "weights"}
                    ),
                    key=lambda path: path.name,
                )
        except OSError as exc:
            warnings.append(f"could not list {catalog_dir}: {exc}")
            other_paths = []
        muse_bytes += sum(_du_sizes(other_paths, warnings).values())

    venv_refs: set[str] = set()
    weight_refs: set[str] = set()
    venv_references_safe = True
    weights_references_safe = True
    hf_cache, pip_cache = _shared_cache_roots()
    shared_repo_refs: set[Path] = set()

    for model_id, entry in sorted(catalog.items()):
        raw_venv = entry.get("venv_path") if isinstance(entry, dict) else None
        expected_venv = venvs_root / model_id
        if not raw_venv:
            venv_references_safe = False
            issues.append(StorageIssue(
                "missing-reference", model_id, expected_venv,
                "catalog entry has no venv_path",
            ))
        else:
            protected = _top_level_owned_path(raw_venv, venvs_root)
            if protected is not None:
                venv_refs.add(_path_key(protected))
            if _path_key(raw_venv) != _path_key(expected_venv):
                venv_references_safe = False
                issues.append(StorageIssue(
                    "unsafe-reference", model_id, Path(str(raw_venv)),
                    f"venv_path is not the expected {expected_venv}",
                ))
            if not Path(str(raw_venv)).exists():
                issues.append(StorageIssue(
                    "missing-reference", model_id, Path(str(raw_venv)),
                    "catalog venv_path is missing",
                ))

        raw_weights = entry.get("local_dir") if isinstance(entry, dict) else None
        if not raw_weights:
            weights_references_safe = False
            issues.append(StorageIssue(
                "missing-reference", model_id, None,
                "catalog entry has no local_dir",
            ))
            continue
        raw_weights_path = Path(str(raw_weights)).expanduser()
        try:
            owned_target = _owned_weights_purge_path(
                str(raw_weights), weights_root, model_id=model_id,
            )
        except CatalogError as exc:
            weights_references_safe = False
            protected = _top_level_owned_path(raw_weights, weights_root)
            if protected is not None:
                weight_refs.add(_path_key(protected))
            issues.append(StorageIssue(
                "unsafe-reference", model_id, raw_weights_path, str(exc),
            ))
        else:
            if owned_target is not None:
                protected = _top_level_owned_path(owned_target, weights_root)
                if protected is not None:
                    weight_refs.add(_path_key(protected))
            elif hf_cache is not None:
                shared_root = _top_level_owned_path(raw_weights, hf_cache)
                if shared_root is not None and shared_root.name.startswith("models--"):
                    shared_repo_refs.add(shared_root)
        if not raw_weights_path.exists():
            issues.append(StorageIssue(
                "missing-reference", model_id, raw_weights_path,
                "catalog local_dir is missing",
            ))

    referenced_venvs: list[StorageItem] = []
    unreferenced_venvs: list[StorageItem] = []
    abandoned_staging: list[StorageItem] = []
    recovery_workspaces: list[StorageItem] = []
    for item in venv_items:
        try:
            mode = item.path.lstat().st_mode
        except OSError:
            continue
        staging_match = _VENV_STAGING_RE.fullmatch(item.path.name)
        if staging_match and stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            model_id = staging_match.group("model")
            try:
                _validate_model_id_for_fs(model_id)
            except ValueError as exc:
                warnings.append(f"unsafe venv staging name {item.path}: {exc}")
                continue
            item = _with_newest_mtime(item, warnings)
            if item is None:
                continue
            transaction = staging_match.group("transaction") == "transaction"
            safe = not transaction or _complete_venv(venvs_root / model_id)
            classified = StorageItem(
                kind="venv-staging",
                status="safe-garbage" if safe else "recovery",
                path=item.path,
                physical_bytes=item.physical_bytes,
                latest_mtime=item.latest_mtime,
                identity=item.identity,
                model_id=model_id,
                reason=(
                    "abandoned venv staging workspace"
                    if safe else
                    "transaction may contain the only recoverable prior venv"
                ),
            )
            (abandoned_staging if safe else recovery_workspaces).append(classified)
            continue
        if (
            item.path.name.startswith(".")
            and _path_key(item.path) not in venv_refs
        ):
            continue
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            issues.append(StorageIssue(
                "unsafe-owned-path", None, item.path,
                "unexpected non-directory or symlink in venv root",
            ))
            continue
        if _path_key(item.path) in venv_refs:
            referenced_venvs.append(_replace_item(
                item, status="referenced", reason="referenced by catalog",
            ))
        else:
            item = _with_newest_mtime(item, warnings)
            if item is None:
                continue
            unreferenced_venvs.append(_replace_item(
                item,
                status="unreferenced",
                reason="not catalog-referenced; may be intentionally retained",
            ))

    referenced_weights: list[StorageItem] = []
    unreferenced_weights: list[StorageItem] = []
    for item in weight_items:
        try:
            mode = item.path.lstat().st_mode
        except OSError:
            continue
        if (
            _WEIGHTS_STAGING_RE.fullmatch(item.path.name)
            and stat.S_ISDIR(mode)
            and not stat.S_ISLNK(mode)
        ):
            item = _with_newest_mtime(item, warnings)
            if item is None:
                continue
            abandoned_staging.append(StorageItem(
                kind="weights-staging",
                status="safe-garbage",
                path=item.path,
                physical_bytes=item.physical_bytes,
                latest_mtime=item.latest_mtime,
                identity=item.identity,
                reason="abandoned artifact download staging workspace",
            ))
            continue
        if item.path.name.startswith("."):
            continue
        if item.path.name == "CACHEDIR.TAG" and stat.S_ISREG(mode):
            continue
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            issues.append(StorageIssue(
                "unsafe-owned-path", None, item.path,
                "unexpected non-directory or symlink in weights root",
            ))
            continue
        if _path_key(item.path) in weight_refs:
            referenced_weights.append(_replace_item(
                item, status="referenced", reason="referenced by catalog",
            ))
        else:
            item = _with_newest_mtime(item, warnings)
            if item is None:
                continue
            unreferenced_weights.append(_replace_item(
                item,
                status="unreferenced",
                reason="not catalog-referenced; may be intentionally retained",
            ))

    incomplete_downloads = _find_incomplete_files(weights_root, warnings)

    owned_hf_recognized_bytes = 0
    owned_hf_warnings: list[str] = []
    if weights_root.exists():
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir(weights_root)
            owned_hf_recognized_bytes = int(cache_info.size_on_disk)
            owned_hf_warnings.extend(str(warning) for warning in cache_info.warnings)
        except Exception as exc:  # noqa: BLE001 - report, never guess
            owned_hf_warnings.append(f"could not scan owned HF cache: {exc}")

    shared_hf_bytes = 0
    shared_hf_referenced_bytes = 0
    pip_cache_bytes = 0
    if include_shared:
        if (
            hf_cache is not None
            and hf_cache.exists()
            and _path_key(hf_cache) != _path_key(weights_root)
        ):
            shared_hf_bytes = _du_sizes([hf_cache], warnings).get(
                _path_key(hf_cache), 0,
            )
            existing_shared_refs = [
                path for path in sorted(shared_repo_refs, key=str)
                if path.exists()
            ]
            shared_hf_referenced_bytes = sum(
                _du_sizes(existing_shared_refs, warnings).values()
            )
        if pip_cache is not None and pip_cache.exists():
            pip_cache_bytes = _du_sizes([pip_cache], warnings).get(
                _path_key(pip_cache), 0,
            )

    for warning in warnings:
        issues.append(StorageIssue("inspection-warning", None, None, warning))

    return StorageReport(
        catalog_dir=catalog_dir,
        scanned_at=scanned_at,
        filesystem_total_bytes=filesystem_total,
        filesystem_free_bytes=filesystem_free,
        muse_bytes=muse_bytes,
        venv_bytes=venv_bytes,
        weights_bytes=weights_bytes,
        referenced_venvs=tuple(referenced_venvs),
        unreferenced_venvs=tuple(unreferenced_venvs),
        referenced_weights=tuple(referenced_weights),
        unreferenced_weights=tuple(unreferenced_weights),
        incomplete_downloads=tuple(incomplete_downloads),
        abandoned_staging=tuple(sorted(abandoned_staging, key=lambda item: str(item.path))),
        recovery_workspaces=tuple(sorted(recovery_workspaces, key=lambda item: str(item.path))),
        issues=tuple(issues),
        venv_references_safe=venv_references_safe,
        weights_references_safe=weights_references_safe,
        owned_hf_recognized_bytes=owned_hf_recognized_bytes,
        owned_hf_warnings=tuple(owned_hf_warnings),
        shared_hf_cache=hf_cache,
        shared_hf_bytes=shared_hf_bytes,
        shared_hf_referenced_bytes=shared_hf_referenced_bytes,
        pip_cache=pip_cache,
        pip_cache_bytes=pip_cache_bytes,
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(_path_key(path)).relative_to(Path(_path_key(parent)))
    except ValueError:
        return False
    return True


def _validated_age(value: float) -> float:
    if (
        isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("older_than_seconds must be a finite non-negative number")
    return float(value)


def plan_prune(
    report: StorageReport,
    *,
    older_than_seconds: float = 24 * 3600,
    include_unreferenced: bool = False,
) -> PrunePlan:
    """Build a deterministic cleanup plan; mutation is a separate operation."""
    age = _validated_age(older_than_seconds)
    cutoff = report.scanned_at - age
    notices: list[str] = []
    selected_dirs: list[StorageItem] = []

    for item in report.abandoned_staging:
        if item.latest_mtime <= cutoff:
            selected_dirs.append(item)

    if include_unreferenced:
        if report.venv_references_safe:
            selected_dirs.extend(
                item for item in report.unreferenced_venvs
                if item.latest_mtime <= cutoff
            )
        else:
            notices.append(
                "unreferenced venv cleanup disabled by unsafe catalog references"
            )
        if report.weights_references_safe:
            selected_dirs.extend(
                item for item in report.unreferenced_weights
                if item.latest_mtime <= cutoff
            )
        else:
            notices.append(
                "unreferenced weights cleanup disabled by unsafe catalog references"
            )

    candidates = list(selected_dirs)
    for item in report.incomplete_downloads:
        if item.latest_mtime > cutoff:
            continue
        if any(_is_within(item.path, directory.path) for directory in selected_dirs):
            continue
        candidates.append(item)
    candidates.sort(key=lambda item: (item.kind, str(item.path)))
    return PrunePlan(
        created_at=report.scanned_at,
        older_than_seconds=age,
        include_unreferenced=include_unreferenced,
        candidates=tuple(candidates),
        notices=tuple(notices),
    )


def _same_identity(path: Path, expected: PathIdentity) -> bool:
    try:
        return _identity(path.lstat()) == expected
    except OSError:
        return False


def _current_owned_references() -> tuple[set[str], set[str]]:
    """Return current top-level venv/weights references under catalog lock."""
    catalog_dir = _catalog_dir()
    venv_root = catalog_dir / "venvs"
    weights_root = catalog_dir / "weights"
    venv_refs: set[str] = set()
    weight_refs: set[str] = set()
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        for model_id, entry in catalog.items():
            if not isinstance(entry, dict):
                continue
            protected_venv = _top_level_owned_path(entry.get("venv_path"), venv_root)
            if protected_venv is not None:
                venv_refs.add(_path_key(protected_venv))
            raw_weights = entry.get("local_dir")
            if not raw_weights:
                continue
            try:
                target = _owned_weights_purge_path(
                    str(raw_weights), weights_root, model_id=model_id,
                )
            except CatalogError:
                target = _top_level_owned_path(raw_weights, weights_root)
            if target is not None:
                protected_weights = _top_level_owned_path(target, weights_root)
                if protected_weights is not None:
                    weight_refs.add(_path_key(protected_weights))
    return venv_refs, weight_refs


def _refresh_item(item: StorageItem) -> StorageItem | None:
    if not _same_identity(item.path, item.identity):
        return None
    try:
        info = item.path.lstat()
    except OSError:
        return None
    size, latest = _measure_path(item.path, fail_on_error=True)
    return StorageItem(
        kind=item.kind,
        status=item.status,
        path=item.path,
        physical_bytes=size,
        latest_mtime=latest,
        identity=_identity(info),
        model_id=item.model_id,
        reason=item.reason,
    )


def _delete_item(item: StorageItem) -> int:
    if item.kind == "incomplete-download":
        info = item.path.lstat()
        if not stat.S_ISREG(info.st_mode) or not item.path.name.endswith(".incomplete"):
            raise StorageInspectionError(f"refusing changed partial file {item.path}")
        item.path.unlink()
        return item.physical_bytes
    _purge_owned_directory(
        item.path,
        model_id=item.model_id or "storage-prune",
        label=item.kind,
    )
    return item.physical_bytes


def execute_prune(plan: PrunePlan, *, dry_run: bool = False) -> PruneResult:
    """Apply a plan after lock-, catalog-, age-, and inode revalidation."""
    if dry_run:
        return PruneResult(
            dry_run=True,
            outcomes=tuple(
                PruneOutcome(item, "would-delete", item.reason)
                for item in plan.candidates
            ),
        )

    cutoff = time.time() - plan.older_than_seconds
    outcomes: list[PruneOutcome] = []
    venv_candidates = [
        item for item in plan.candidates
        if item.kind in {"venv", "venv-staging"}
    ]
    weight_candidates = [
        item for item in plan.candidates
        if item.kind not in {"venv", "venv-staging"}
    ]

    for original in venv_candidates:
        model_id = original.model_id or original.path.name
        try:
            with _model_pull_lock(model_id, wait=False):
                with _model_resource_lease(model_id, wait=False):
                    current = _refresh_item(original)
                    if current is None:
                        outcomes.append(PruneOutcome(
                            original, "skipped", "path identity changed since planning",
                        ))
                        continue
                    if current.latest_mtime > cutoff:
                        outcomes.append(PruneOutcome(
                            current, "skipped", "path changed inside the age grace period",
                        ))
                        continue
                    if current.kind == "venv-staging":
                        match = _VENV_STAGING_RE.fullmatch(current.path.name)
                        if match is None:
                            outcomes.append(PruneOutcome(
                                current, "skipped", "staging name no longer validates",
                            ))
                            continue
                        if (
                            match.group("transaction") == "transaction"
                            and not _complete_venv(_catalog_dir() / "venvs" / model_id)
                        ):
                            outcomes.append(PruneOutcome(
                                current, "skipped", "workspace may be needed for recovery",
                            ))
                            continue
                    venv_refs, _ = _current_owned_references()
                    if (
                        current.kind == "venv"
                        and _path_key(current.path) in venv_refs
                    ):
                        outcomes.append(PruneOutcome(
                            current, "skipped", "catalog acquired a reference",
                        ))
                        continue
                    reclaimed = _delete_item(current)
                    outcomes.append(PruneOutcome(
                        current, "deleted", current.reason, reclaimed,
                    ))
        except ModelInUseError as exc:
            outcomes.append(PruneOutcome(original, "busy", str(exc)))
        except (CatalogError, OSError, StorageInspectionError) as exc:
            outcomes.append(PruneOutcome(original, "failed", str(exc)))

    if weight_candidates:
        try:
            with _storage_cache_lock(wait=False):
                _, weight_refs = _current_owned_references()
                for original in weight_candidates:
                    try:
                        current = _refresh_item(original)
                        if current is None:
                            outcomes.append(PruneOutcome(
                                original, "skipped", "path identity changed since planning",
                            ))
                            continue
                        if current.latest_mtime > cutoff:
                            outcomes.append(PruneOutcome(
                                current, "skipped", "path changed inside the age grace period",
                            ))
                            continue
                        if (
                            current.kind == "weights"
                            and _path_key(current.path) in weight_refs
                        ):
                            outcomes.append(PruneOutcome(
                                current, "skipped", "catalog acquired a reference",
                            ))
                            continue
                        if (
                            current.kind == "weights-staging"
                            and _WEIGHTS_STAGING_RE.fullmatch(current.path.name) is None
                        ):
                            outcomes.append(PruneOutcome(
                                current, "skipped", "staging name no longer validates",
                            ))
                            continue
                        reclaimed = _delete_item(current)
                        outcomes.append(PruneOutcome(
                            current, "deleted", current.reason, reclaimed,
                        ))
                    except (CatalogError, OSError, StorageInspectionError) as exc:
                        outcomes.append(PruneOutcome(original, "failed", str(exc)))
        except StorageBusyError as exc:
            outcomes.extend(
                PruneOutcome(item, "busy", str(exc)) for item in weight_candidates
            )
        except CatalogError as exc:
            outcomes.extend(
                PruneOutcome(item, "failed", str(exc)) for item in weight_candidates
            )

    return PruneResult(dry_run=False, outcomes=tuple(outcomes))


def prune_storage(
    *,
    dry_run: bool = False,
    include_unreferenced: bool = False,
    older_than_seconds: float = 24 * 3600,
) -> tuple[PrunePlan, PruneResult]:
    """Convenience entry point used by the CLI."""
    age = _validated_age(older_than_seconds)
    report = inspect_storage(include_shared=False)
    plan = plan_prune(
        report,
        older_than_seconds=age,
        include_unreferenced=include_unreferenced,
    )
    return plan, execute_prune(plan, dry_run=dry_run)


def automatic_prune_before_pull() -> AutomaticPruneResult | None:
    """Prune only old transient data when either low-space limit is crossed.

    This is intentionally narrower than explicit ``storage prune`` usage: it
    never includes unreferenced resources. The absolute limit protects small
    filesystems while the percentage limit detects pressure on larger ones.
    """
    from muse.core import config

    if not config.get("storage.auto_prune_before_pull"):
        return None
    catalog_dir = _catalog_dir()
    if not catalog_dir.exists():
        return None
    total_bytes, free_bytes = _filesystem_usage(catalog_dir)
    min_free_gb = float(config.get("storage.auto_prune_min_free_gb"))
    min_free_percent = float(
        config.get("storage.auto_prune_min_free_percent")
    )
    absolute_threshold = int(min_free_gb * 1024**3)
    percentage_threshold = int(total_bytes * min_free_percent / 100.0)
    # Either an absolute shortage (important on small filesystems) or a low
    # free-space ratio (important on large filesystems) should trigger the
    # narrow safe cleanup pass.
    threshold = max(absolute_threshold, percentage_threshold)
    if threshold <= 0 or free_bytes >= threshold:
        return None

    grace_hours = float(config.get("storage.auto_prune_grace_hours"))
    plan, result = prune_storage(
        dry_run=False,
        include_unreferenced=False,
        older_than_seconds=grace_hours * 3600.0,
    )
    _, free_after = _filesystem_usage(catalog_dir)
    return AutomaticPruneResult(
        threshold_bytes=threshold,
        free_bytes_before=free_bytes,
        free_bytes_after=free_after,
        plan=plan,
        result=result,
    )
