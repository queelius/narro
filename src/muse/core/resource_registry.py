"""Persistent identity records for processes started and owned by Muse.

The registry is deliberately *not* a process scanner.  It contains only
records written by Muse at process creation time, including the operating
system's process creation timestamp so a recycled PID cannot be mistaken for
the original child.  Normal shutdown uses the in-memory ``Popen`` handles;
this file exists for diagnosis and recovery after an unclean parent exit.

Automatic recovery targets only these registered process leaders through
identity-bound OS handles. It deliberately does not scan for or numerically
signal unregistered descendants, because doing so could target a reused
process or process-group identifier. Normal in-memory shutdown retains the
stronger exact ownership needed to clean up isolated child groups.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import stat
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from muse.core import config


_SCHEMA_VERSION = 1
_MAX_REGISTRY_BYTES = 8 * 1024 * 1024
_MAX_RESOURCE_RECORDS = 4096
_PROCESS_LOCK = threading.RLock()
_REPAIRABLE_KINDS = frozenset({"worker", "admin_job", "supervisor"})


class ResourceRegistryError(RuntimeError):
    """The persisted resource registry could not be read or written safely."""


class ResourceIdentityUnavailable(ResourceRegistryError):
    """The OS cannot safely verify whether a PID still has its old identity."""


@dataclass(frozen=True)
class ResourceRecord:
    resource_id: str
    kind: str
    pid: int
    create_time: float
    owner_pid: int | None
    owner_create_time: float | None
    process_group: int | None
    port: int | None
    models: tuple[str, ...]
    created_at: float

    @classmethod
    def from_dict(cls, value: object) -> "ResourceRecord":
        if not isinstance(value, dict):
            raise ResourceRegistryError("resource entry must be an object")
        try:
            resource_id = value["resource_id"]
            kind = value["kind"]
            pid = value["pid"]
            create_time = value["create_time"]
            owner_pid = value.get("owner_pid")
            owner_create_time = value.get("owner_create_time")
            process_group = value.get("process_group")
            port = value.get("port")
            models = value.get("models", [])
            created_at = value["created_at"]
        except KeyError as exc:
            raise ResourceRegistryError(
                f"resource entry is missing {exc.args[0]!r}"
            ) from exc
        if not isinstance(resource_id, str) or not resource_id:
            raise ResourceRegistryError("resource_id must be a non-empty string")
        if not isinstance(kind, str) or not kind:
            raise ResourceRegistryError("resource kind must be a non-empty string")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
            raise ResourceRegistryError("resource pid must be an integer greater than 1")
        if not isinstance(create_time, (int, float)) or isinstance(create_time, bool):
            raise ResourceRegistryError("resource create_time must be numeric")
        if not math.isfinite(float(create_time)):
            raise ResourceRegistryError("resource create_time must be finite")
        if owner_pid is not None and (
            not isinstance(owner_pid, int) or isinstance(owner_pid, bool) or owner_pid <= 1
        ):
            raise ResourceRegistryError("resource owner_pid must be null or greater than 1")
        if owner_create_time is not None and (
            not isinstance(owner_create_time, (int, float))
            or isinstance(owner_create_time, bool)
        ):
            raise ResourceRegistryError("resource owner_create_time must be numeric or null")
        if owner_create_time is not None and not math.isfinite(float(owner_create_time)):
            raise ResourceRegistryError("resource owner_create_time must be finite or null")
        if process_group is not None and (
            not isinstance(process_group, int)
            or isinstance(process_group, bool)
            or process_group <= 1
        ):
            raise ResourceRegistryError("resource process_group must be null or greater than 1")
        if port is not None and (
            not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535
        ):
            raise ResourceRegistryError("resource port must be null or between 1 and 65535")
        if not isinstance(models, list) or not all(isinstance(item, str) for item in models):
            raise ResourceRegistryError("resource models must be a list of strings")
        if not isinstance(created_at, (int, float)) or isinstance(created_at, bool):
            raise ResourceRegistryError("resource created_at must be numeric")
        if not math.isfinite(float(created_at)):
            raise ResourceRegistryError("resource created_at must be finite")
        return cls(
            resource_id=resource_id,
            kind=kind,
            pid=pid,
            create_time=float(create_time),
            owner_pid=owner_pid,
            owner_create_time=(
                float(owner_create_time) if owner_create_time is not None else None
            ),
            process_group=process_group,
            port=port,
            models=tuple(models),
            created_at=float(created_at),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["models"] = list(self.models)
        return value


@dataclass(frozen=True)
class ResourceStatus:
    record: ResourceRecord
    state: str
    detail: str


@dataclass(frozen=True)
class RepairResult:
    resource_id: str
    action: str
    detail: str


def _runtime_dir(catalog_dir: Path | None = None) -> Path:
    base = (
        catalog_dir
        if catalog_dir is not None
        else Path(config.get("paths.catalog_dir")).expanduser()
    )
    try:
        return base.resolve() / "runtime"
    except (OSError, RuntimeError) as exc:
        raise ResourceRegistryError(
            f"cannot resolve resource catalog directory {base}: {exc}"
        ) from exc


def registry_path(catalog_dir: Path | None = None) -> Path:
    """Return the configured persistent resource-registry path."""
    return _runtime_dir(catalog_dir) / "resources.json"


def _validate_runtime_dir(
    directory: Path, *, missing_ok: bool = False, set_private_mode: bool = False,
) -> bool:
    """Validate the runtime directory itself without following a symlink."""
    if os.name == "posix":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            directory_fd = os.open(directory, flags)
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        except OSError as exc:
            raise ResourceRegistryError(
                f"resource runtime path is not a safe directory: {directory}: {exc}"
            ) from exc
        try:
            if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
                raise ResourceRegistryError(
                    f"resource runtime path is not a directory: {directory}"
                )
            info = os.fstat(directory_fd)
            if info.st_uid != os.geteuid():
                raise ResourceRegistryError(
                    f"resource runtime directory is not owned by this user: {directory}"
                )
            if set_private_mode:
                os.fchmod(directory_fd, 0o700)
            elif info.st_mode & 0o022:
                raise ResourceRegistryError(
                    f"resource runtime directory is group/other writable: {directory}"
                )
        finally:
            os.close(directory_fd)
        return True

    try:  # pragma: no cover - exercised on Windows CI
        info = directory.lstat()
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    except OSError as exc:
        raise ResourceRegistryError(f"cannot inspect {directory}: {exc}") from exc
    is_junction = getattr(directory, "is_junction", lambda: False)
    if stat.S_ISLNK(info.st_mode) or is_junction() or not directory.is_dir():
        raise ResourceRegistryError(
            f"resource runtime path is not a safe directory: {directory}"
        )
    if set_private_mode:
        directory.chmod(0o700)
    return True


def _ensure_runtime_dir(catalog_dir: Path | None = None) -> Path:
    directory = _runtime_dir(catalog_dir)
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _validate_runtime_dir(directory, set_private_mode=True)
    except ResourceRegistryError:
        raise
    except OSError as exc:
        raise ResourceRegistryError(
            f"cannot prepare resource runtime directory {directory}: {exc}"
        ) from exc
    return directory


def _open_regular_file(
    path: Path, flags: int, *, mode: int | None = None, require_single_link: bool = False,
) -> int:
    """Open one regular file without following a final-component symlink."""
    open_flags = flags | getattr(os, "O_CLOEXEC", 0)
    if os.name == "posix":
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
    else:  # pragma: no cover - exercised on Windows CI
        try:
            info = path.lstat()
        except FileNotFoundError:
            info = None
        if info is not None and stat.S_ISLNK(info.st_mode):
            raise ResourceRegistryError(f"refusing symlink resource file {path}")
        open_flags |= getattr(os, "O_BINARY", 0)
    try:
        if mode is None:
            descriptor = os.open(path, open_flags)
        else:
            descriptor = os.open(path, open_flags, mode)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ResourceRegistryError(f"cannot safely open {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ResourceRegistryError(f"resource path is not a regular file: {path}")
        if os.name == "posix":
            if info.st_uid != os.geteuid():
                raise ResourceRegistryError(
                    f"resource file is not owned by this user: {path}"
                )
            if info.st_mode & 0o022:
                raise ResourceRegistryError(
                    f"resource file is group/other writable: {path}"
                )
        if require_single_link and info.st_nlink != 1:
            raise ResourceRegistryError(f"resource lock has multiple links: {path}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _acquire_file_lock(handle: Any, *, exclusive: bool) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import msvcrt

        handle.seek(0)
        operation = msvcrt.LK_LOCK if exclusive else msvcrt.LK_RLCK
        msvcrt.locking(handle.fileno(), operation, 1)
        return
    import fcntl

    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(handle.fileno(), operation)


def _release_file_lock(handle: Any) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _registry_lock(
    catalog_dir: Path | None = None, *, create: bool = True, exclusive: bool = True,
) -> Iterator[None]:
    directory = (
        _ensure_runtime_dir(catalog_dir) if create else _runtime_dir(catalog_dir)
    )
    if not create:
        _validate_runtime_dir(directory)
    lock_path = directory / "resources.lock"
    with _PROCESS_LOCK:
        flags = os.O_RDWR | os.O_CREAT if create else os.O_RDONLY
        descriptor = _open_regular_file(
            lock_path,
            flags,
            mode=0o600 if create else None,
            require_single_link=True,
        )
        mode = "r+b" if create else "rb"
        with os.fdopen(descriptor, mode) as handle:
            try:
                if create:
                    if os.name == "posix":
                        os.fchmod(handle.fileno(), 0o600)
                    # Windows byte-range locking needs a concrete byte. POSIX
                    # also writes it so a registry can be moved across platforms.
                    handle.seek(0, os.SEEK_END)
                    if handle.tell() == 0:
                        handle.write(b"\0")
                        handle.flush()
                        os.fsync(handle.fileno())
                _acquire_file_lock(handle, exclusive=exclusive)
            except (ImportError, OSError) as exc:
                raise ResourceRegistryError(
                    f"cannot lock resource registry {lock_path}: {exc}"
                ) from exc
            try:
                yield
            finally:
                try:
                    _release_file_lock(handle)
                except (ImportError, OSError) as exc:
                    raise ResourceRegistryError(
                        f"cannot unlock resource registry {lock_path}: {exc}"
                    ) from exc


def _regular_file_exists(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ResourceRegistryError(f"cannot inspect {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ResourceRegistryError(f"resource path is not a regular file: {path}")
    return True


def _read_unlocked(catalog_dir: Path | None = None) -> dict[str, ResourceRecord]:
    path = registry_path(catalog_dir)
    try:
        descriptor = _open_regular_file(path, os.O_RDONLY)
    except FileNotFoundError:
        return {}
    try:
        with os.fdopen(descriptor, "rb") as handle:
            raw = handle.read(_MAX_REGISTRY_BYTES + 1)
        if len(raw) > _MAX_REGISTRY_BYTES:
            raise ResourceRegistryError(
                f"resource registry exceeds {_MAX_REGISTRY_BYTES} bytes: {path}"
            )
        data = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResourceRegistryError(f"cannot read {path}: {exc}") from exc
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data["version"] != _SCHEMA_VERSION
    ):
        raise ResourceRegistryError(
            f"{path} is not a supported Muse resource registry"
        )
    resources = data.get("resources")
    if not isinstance(resources, dict):
        raise ResourceRegistryError(f"{path} resources must be an object")
    if len(resources) > _MAX_RESOURCE_RECORDS:
        raise ResourceRegistryError(
            f"{path} contains more than {_MAX_RESOURCE_RECORDS} resources"
        )
    parsed: dict[str, ResourceRecord] = {}
    for key, value in resources.items():
        record = ResourceRecord.from_dict(value)
        if key != record.resource_id:
            raise ResourceRegistryError("resource key does not match resource_id")
        parsed[key] = record
    return parsed


def _write_unlocked(
    records: dict[str, ResourceRecord], catalog_dir: Path | None = None,
) -> None:
    directory = _ensure_runtime_dir(catalog_dir)
    target = registry_path(catalog_dir)
    if len(records) > _MAX_RESOURCE_RECORDS:
        raise ResourceRegistryError(
            f"resource registry cannot exceed {_MAX_RESOURCE_RECORDS} records"
        )
    payload = {
        "version": _SCHEMA_VERSION,
        "resources": {
            key: value.to_dict() for key, value in sorted(records.items())
        },
    }
    try:
        serialized = json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ResourceRegistryError(f"cannot encode resource registry: {exc}") from exc
    if len(serialized.encode("utf-8")) > _MAX_REGISTRY_BYTES:
        raise ResourceRegistryError(
            f"resource registry cannot exceed {_MAX_REGISTRY_BYTES} bytes"
        )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix="resources.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            os.chmod(temp_path, 0o600)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        try:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            directory_fd = os.open(directory, flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except OSError as exc:
        raise ResourceRegistryError(f"cannot write {target}: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _process_create_time(pid: int) -> float | None:
    if pid <= 1:
        return None
    try:
        import psutil
    except ImportError as exc:
        raise ResourceIdentityUnavailable(
            "psutil is required to verify process creation time"
        ) from exc
    try:
        process = psutil.Process(pid)
        create_time = float(process.create_time())
        try:
            if process.status() == getattr(psutil, "STATUS_ZOMBIE", object()):
                return None
        except psutil.ZombieProcess:
            return None
        except (psutil.AccessDenied, OSError):
            # Creation time is the identity authority. Status is queried only
            # to avoid trying to signal an already-resource-free zombie.
            pass
        return create_time
    except psutil.NoSuchProcess:
        return None
    except (psutil.AccessDenied, OSError) as exc:
        raise ResourceIdentityUnavailable(
            f"cannot inspect creation time for pid {pid}: {exc}"
        ) from exc
    except ValueError:
        return None


def _isolated_process_group(pid: int) -> int | None:
    if os.name != "posix" or pid <= 1:
        return None
    try:
        process_group = os.getpgid(pid)
    except OSError:
        return None
    if process_group != pid or process_group == os.getpgrp():
        return None
    return process_group


def _same_identity(expected: float, actual: float | None) -> bool:
    return actual is not None and abs(expected - actual) < 0.001


def register_process(
    *,
    kind: str,
    pid: int,
    owner_pid: int | None = None,
    port: int | None = None,
    models: list[str] | tuple[str, ...] = (),
    catalog_dir: Path | None = None,
) -> str:
    """Persist an identity record for a process Muse just created."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise ResourceRegistryError(f"refusing unsafe process pid {pid!r}")
    if not isinstance(kind, str) or not kind:
        raise ResourceRegistryError("resource kind must be a non-empty string")
    if owner_pid is not None and (
        not isinstance(owner_pid, int) or isinstance(owner_pid, bool) or owner_pid <= 1
    ):
        raise ResourceRegistryError(f"refusing unsafe owner pid {owner_pid!r}")
    if port is not None and (
        not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535
    ):
        raise ResourceRegistryError("resource port must be null or between 1 and 65535")
    if not isinstance(models, (list, tuple)) or not all(
        isinstance(model, str) for model in models
    ):
        raise ResourceRegistryError("resource models must be a list or tuple of strings")
    create_time = _process_create_time(pid)
    if create_time is None:
        raise ResourceRegistryError(f"cannot verify process identity for pid {pid}")
    owner_create_time: float | None = None
    if owner_pid is not None:
        owner_create_time = _process_create_time(owner_pid)
        if owner_create_time is None:
            raise ResourceRegistryError(
                f"cannot verify owner process identity for pid {owner_pid}"
            )
    record = ResourceRecord.from_dict(ResourceRecord(
        resource_id=uuid.uuid4().hex,
        kind=kind,
        pid=pid,
        create_time=create_time,
        owner_pid=owner_pid,
        owner_create_time=owner_create_time,
        process_group=_isolated_process_group(pid),
        port=port,
        models=tuple(models),
        created_at=time.time(),
    ).to_dict())
    with _registry_lock(catalog_dir):
        records = _read_unlocked(catalog_dir)
        records[record.resource_id] = record
        _write_unlocked(records, catalog_dir)
    return record.resource_id


def unregister_process(
    resource_id: str | None, *, catalog_dir: Path | None = None,
    expected: ResourceRecord | None = None,
) -> bool:
    """Remove one matching registry record. This function never signals."""
    if not resource_id:
        return False
    with _registry_lock(catalog_dir):
        records = _read_unlocked(catalog_dir)
        current = records.get(resource_id)
        removed = current is not None and (expected is None or current == expected)
        if removed:
            records.pop(resource_id)
            _write_unlocked(records, catalog_dir)
        return removed


def list_resources(*, catalog_dir: Path | None = None) -> list[ResourceRecord]:
    """Read all Muse-owned records without scanning the host process table."""
    # A read-only doctor invocation on a machine that has never started Muse
    # must not create ~/.muse/runtime merely to report an empty registry.
    directory = _runtime_dir(catalog_dir)
    if not _validate_runtime_dir(directory, missing_ok=True):
        return []
    path = registry_path(catalog_dir)
    if not _regular_file_exists(path):
        return []
    try:
        with _registry_lock(catalog_dir, create=False, exclusive=False):
            records = _read_unlocked(catalog_dir)
    except FileNotFoundError:
        # A writer always creates the lock before the registry. If the data
        # file still exists, silently reading without a lock could race an
        # uncoordinated/tampered path, so fail closed without creating it.
        if _regular_file_exists(path):
            raise ResourceRegistryError(
                f"resource registry lock is missing beside {path}"
            )
        return []
    return sorted(records.values(), key=lambda value: value.created_at)


def inspect_resource(record: ResourceRecord) -> ResourceStatus:
    """Classify one explicit record using PID creation-time identity."""
    try:
        actual = _process_create_time(record.pid)
    except ResourceIdentityUnavailable as exc:
        return ResourceStatus(record, "unverifiable", str(exc))
    if actual is None:
        return ResourceStatus(record, "dead", "recorded process no longer exists")
    if not _same_identity(record.create_time, actual):
        return ResourceStatus(record, "pid_reused", "PID now belongs to another process")
    if record.owner_pid is None:
        return ResourceStatus(record, "running", "process identity matches")
    if record.owner_create_time is None:
        return ResourceStatus(record, "unverifiable", "owner identity was not recorded")
    try:
        owner_actual = _process_create_time(record.owner_pid)
    except ResourceIdentityUnavailable as exc:
        return ResourceStatus(record, "unverifiable", str(exc))
    if not _same_identity(record.owner_create_time, owner_actual):
        return ResourceStatus(record, "orphaned", "recorded owner no longer exists")
    return ResourceStatus(record, "running", "process and owner identities match")


def inspect_resources(
    *, catalog_dir: Path | None = None,
) -> list[ResourceStatus]:
    return [inspect_resource(record) for record in list_resources(catalog_dir=catalog_dir)]


def _wait_until_identity_gone(record: ResourceRecord, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if not _same_identity(record.create_time, _process_create_time(record.pid)):
            return True
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    return not _same_identity(record.create_time, _process_create_time(record.pid))


def _pidfd_open(pid: int) -> int:
    """Open an identity-bound Linux process handle or fail closed."""
    opener = getattr(os, "pidfd_open", None)
    sender = getattr(signal, "pidfd_send_signal", None)
    if os.name != "posix" or not callable(opener) or not callable(sender):
        raise ResourceIdentityUnavailable(
            "safe automatic repair requires Linux pidfd support"
        )
    try:
        descriptor = opener(pid, 0)
        if type(descriptor) is not int or descriptor < 0:
            raise ValueError(f"invalid pidfd {descriptor!r}")
        return descriptor
    except ProcessLookupError as exc:
        raise ResourceRegistryError("resource disappeared before repair") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ResourceIdentityUnavailable(
            f"cannot bind a safe process handle for pid {pid}: {exc}"
        ) from exc


def _pidfd_signal(descriptor: int, sig: signal.Signals) -> None:
    sender = getattr(signal, "pidfd_send_signal", None)
    if not callable(sender):
        raise ResourceIdentityUnavailable(
            "safe automatic repair requires Linux pidfd support"
        )
    sender(descriptor, sig, None, 0)


def _pidfd_close(descriptor: int) -> None:
    os.close(descriptor)


def _signal_verified(record: ResourceRecord, sig: signal.Signals) -> None:
    """Signal one exact orphan through an identity-bound process handle.

    Numeric PID/PGID signalling is deliberately forbidden here. Even after a
    creation-time check, the process or group can exit and its number can be
    reused before ``kill``/``killpg`` runs. A pidfd remains bound to the old
    process across that race and can never target the supervisor, SSH group,
    or another process that later receives the same numeric ID.
    """
    if record.kind not in _REPAIRABLE_KINDS:
        raise ResourceRegistryError(f"automatic repair refuses kind {record.kind!r}")
    if record.pid <= 1 or record.pid == os.getpid():
        raise ResourceRegistryError(f"refusing unsafe target pid {record.pid}")
    descriptor = _pidfd_open(record.pid)
    try:
        # Bind the handle *before* re-reading creation times. If the PID was
        # already reused, the check rejects it; if it changes afterward, the
        # pidfd still denotes only the process opened above.
        status = inspect_resource(record)
        if status.state != "orphaned":
            raise ResourceRegistryError(
                f"resource changed state before signal: {status.state}"
            )
        _pidfd_signal(descriptor, sig)
    finally:
        _pidfd_close(descriptor)


def _terminate_verified(record: ResourceRecord, grace: float) -> str:
    _signal_verified(record, signal.SIGTERM)
    if _wait_until_identity_gone(record, grace):
        return "terminated"
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    _signal_verified(record, kill_signal)
    if _wait_until_identity_gone(record, min(max(grace, 0.1), 2.0)):
        return "killed"
    raise ResourceRegistryError("process remained alive after verified termination")


def repair_stale_resources(
    *, grace: float = 5.0, catalog_dir: Path | None = None,
) -> list[RepairResult]:
    """Repair records conservatively; never discover or target unknown PIDs."""
    if (
        not isinstance(grace, (int, float))
        or isinstance(grace, bool)
        or not math.isfinite(float(grace))
        or grace < 0
    ):
        raise ValueError("grace must be a finite non-negative number")
    results: list[RepairResult] = []
    # Snapshot immutable records, not their state. Earlier repairs can change
    # the ownership state of later records (terminating an orphan supervisor
    # makes its workers orphaned), so inspect each one only when its turn
    # arrives. ``list_resources`` is ordered by creation time, which naturally
    # places a supervisor before children it later spawned.
    for record in list_resources(catalog_dir=catalog_dir):
        status = inspect_resource(record)
        if status.state in {"dead", "pid_reused"}:
            removed = unregister_process(
                record.resource_id, catalog_dir=catalog_dir, expected=record,
            )
            results.append(RepairResult(
                record.resource_id,
                "removed_record" if removed else "refused",
                status.detail if removed else "resource record changed during repair",
            ))
            continue
        if status.state == "unverifiable":
            results.append(
                RepairResult(record.resource_id, "refused", status.detail)
            )
            continue
        if status.state != "orphaned":
            results.append(RepairResult(record.resource_id, "unchanged", status.detail))
            continue
        if record.kind not in _REPAIRABLE_KINDS:
            results.append(
                RepairResult(
                    record.resource_id,
                    "refused",
                    f"automatic repair does not terminate {record.kind!r}",
                )
            )
            continue
        try:
            action = _terminate_verified(record, grace)
        except (OSError, ResourceRegistryError) as exc:
            results.append(RepairResult(record.resource_id, "refused", str(exc)))
            continue
        removed = unregister_process(
            record.resource_id, catalog_dir=catalog_dir, expected=record,
        )
        if not removed:
            results.append(RepairResult(
                record.resource_id,
                "refused",
                "process was terminated but resource record changed during repair",
            ))
        else:
            results.append(RepairResult(record.resource_id, action, status.detail))
    return results
