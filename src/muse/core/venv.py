"""Venv management helpers.

Each pulled model gets its own venv under ~/.muse/venvs/<model-id>/.
This module handles creation and pip-install-into-venv; the catalog
records the resulting Python interpreter path per model.

Output discipline (v0.40.3+): `install_into_venv` defaults to quiet
mode (`pip install -q` + captured stdout/stderr, only emitted on
non-zero exit). The CLI's `muse pull -v` / `--verbose` flips into
pass-through mode via the `install_output_mode` context manager.
The verbose mode preserves the v0.40.2 firehose for dep-resolution
debugging.
"""
from __future__ import annotations

import collections.abc
import contextlib
import configparser
import errno
import locale
import logging
import os
import re
import signal
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Sequence

logger = logging.getLogger(__name__)


# 30 minutes: long enough for a slow PyPI mirror to finish a torch
# install on a fresh venv; short enough to detect a hung mirror before
# the user gives up. TimeoutExpired propagates to the catalog.pull
# caller, which surfaces it as a pull failure.
_PIP_TIMEOUT = 1800
_VENV_CREATE_TIMEOUT = 120
_SOURCE_INSTALL_TIMEOUT = 1800

_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GITHUB_GIT_URL_RE = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$"
)
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_SAFE_CONFIG = (
    "-c", "protocol.allow=never",
    "-c", "protocol.https.allow=always",
    "-c", "credential.interactive=false",
    "-c", "credential.helper=",
    "-c", f"core.hooksPath={os.devnull}",
    "-c", "core.fsmonitor=false",
    "-c", "core.untrackedCache=false",
)

# Quiet installs drain both pipes continuously so a chatty installer cannot
# deadlock, but retain only a bounded tail for diagnostics.  The limit is per
# stream, so one child retains at most 1 MiB plus small bookkeeping overhead.
_CAPTURE_LIMIT_BYTES = 512 * 1024
_CAPTURE_READ_BYTES = 64 * 1024
_CAPTURE_JOIN_TIMEOUT = 1.0

# Cleanup adds at most this much wall time after the command's own deadline.
# TERM and KILL share the budget instead of each receiving a full timeout.
_CLEANUP_TIMEOUT = 5.0
_OS_NAME = os.name
_REAL_POPEN_TYPE = subprocess.Popen
_MANAGED_JOB_GROUP_ENV = "MUSE_MANAGED_JOB_PROCESS_GROUP"


# Thread-local verbose flag. The CLI flips this via
# `install_output_mode(verbose=True)` when `muse pull -v` is set;
# default behavior (no flag) is quiet. Thread-local so concurrent
# pulls in different threads don't stomp each other (not a current
# muse use-case but cheap correctness insurance).
_local = threading.local()


def _is_verbose() -> bool:
    return bool(getattr(_local, "verbose", False))


@contextlib.contextmanager
def install_output_mode(verbose: bool):
    """Toggle pip + HF download output for everything called inside.

    Used by the CLI's `pull` command:

        with install_output_mode(verbose=args.verbose):
            pull(identifier)

    Inside the block: `install_into_venv` runs pip without `-q` and
    streams pip's stdout/stderr to the user; outside (or with
    verbose=False), pip is run with `-q` and stdout/stderr is
    captured, only emitted on non-zero exit.

    Catalog.py wraps snapshot_download similarly using the
    `HF_HUB_DISABLE_PROGRESS_BARS` env var whose value is read from
    `_is_verbose()`.
    """
    prev = getattr(_local, "verbose", False)
    _local.verbose = bool(verbose)
    try:
        yield
    finally:
        _local.verbose = prev


def venv_python(venv_path: Path) -> Path:
    """Return the Python interpreter path inside a venv.

    POSIX layout only (bin/python). The Windows layout (Scripts/python.exe)
    is not supported because muse is Linux/macOS-focused.
    """
    return venv_path / "bin" / "python"


@dataclass
class _OwnedProcess:
    """One exact child plus its positively-validated isolated group.

    On supported POSIX platforms the leader is observed with waitid(WNOWAIT),
    which deliberately keeps it unreaped while the owned group is cleaned.
    That live/zombie leader pins both its PID and equal PGID against reuse.
    """

    process: subprocess.Popen
    pid: int | None
    process_group: int | None
    leader_exit_observed: bool = False
    leader_reaped: bool = False


class _OwnedProcessCleanupError(RuntimeError):
    """The runner could not safely complete its required cleanup sequence."""


@dataclass(frozen=True)
class _GitSubmoduleSpec:
    path: str
    url: str
    revision: str


@dataclass(frozen=True)
class _GitPythonSourceSpec:
    name: str
    url: str
    revision: str
    sparse_paths: tuple[str, ...]
    required_paths: tuple[str, ...]
    pth_path: str
    submodules: tuple[_GitSubmoduleSpec, ...]


class _BoundedCapture:
    """Keep the tail of one binary pipe while accounting for truncation."""

    def __init__(self, limit: int = _CAPTURE_LIMIT_BYTES) -> None:
        self._limit = max(0, int(limit))
        self._buffer = bytearray()
        self._discarded = 0
        self._lock = threading.Lock()

    def append(self, chunk: bytes | str) -> None:
        if isinstance(chunk, str):
            data = chunk.encode(locale.getpreferredencoding(False), errors="replace")
        else:
            data = bytes(chunk)
        if not data:
            return
        with self._lock:
            self._buffer.extend(data)
            excess = len(self._buffer) - self._limit
            if excess > 0:
                del self._buffer[:excess]
                self._discarded += excess

    def text(self) -> str:
        encoding = locale.getpreferredencoding(False)
        with self._lock:
            body = self._buffer.decode(encoding, errors="replace")
            if not self._discarded:
                return body
            return f"[... {self._discarded} output bytes truncated ...]\n{body}"


@dataclass
class _CaptureReader:
    stream: BinaryIO
    capture: _BoundedCapture
    thread: threading.Thread


def _concrete_safe_id(value: Any) -> int | None:
    """Return a concrete signal-safe PID/PGID, rejecting bools and mocks."""
    if type(value) is not int or value <= 1:
        return None
    return value


def _validated_isolated_group(pid: int | None) -> int | None:
    """Return a child-owned POSIX process group only when fully proven."""
    if _OS_NAME != "posix" or pid is None:
        return None
    try:
        process_group = _concrete_safe_id(os.getpgid(pid))
        own_group = _concrete_safe_id(os.getpgrp())
    except OSError as exc:
        logger.warning("could not validate installer process group for pid %s: %s", pid, exc)
        return None
    if process_group != pid or own_group is None or process_group == own_group:
        logger.error(
            "refusing installer process-group ownership for pid=%r pgid=%r own_pgid=%r",
            pid, process_group, own_group,
        )
        return None
    return process_group


def _spawn_owned(
    cmd: Sequence[str],
    *,
    capture_output: bool,
    env: Mapping[str, str] | None = None,
) -> _OwnedProcess:
    """Start one installer in an isolated group and retain its exact handle."""
    popen_kwargs: dict[str, Any] = {}
    if (
        _OS_NAME == "posix"
        and os.environ.get(_MANAGED_JOB_GROUP_ENV) != "1"
    ):
        popen_kwargs["start_new_session"] = True
    elif _OS_NAME == "nt":
        create_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if create_group:
            popen_kwargs["creationflags"] = create_group
    if capture_output:
        # Unbuffered FileIO lets cleanup close the stream object itself. A raw
        # os.close(fd) would leave a Python wrapper holding the old number and
        # could later close an unrelated descriptor after numeric reuse.
        popen_kwargs.update(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    if env is not None:
        popen_kwargs["env"] = dict(env)

    process = subprocess.Popen(list(cmd), **popen_kwargs)
    pid = _concrete_safe_id(getattr(process, "pid", None))
    return _OwnedProcess(
        process=process,
        pid=pid,
        process_group=_validated_isolated_group(pid),
    )


def _drain_pipe(stream: BinaryIO, capture: _BoundedCapture) -> None:
    """Continuously drain one child pipe, retaining only its bounded tail."""
    try:
        while True:
            chunk = stream.read(_CAPTURE_READ_BYTES)
            if not chunk:
                break
            capture.append(chunk)
    except (OSError, ValueError):
        # Cleanup may close a pipe to release a reader whose writer did not
        # exit cleanly. The captured prefix/tail remains useful diagnostics.
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _start_capture(target: _OwnedProcess) -> list[_CaptureReader]:
    readers: list[_CaptureReader] = []
    try:
        for name in ("stdout", "stderr"):
            stream = getattr(target.process, name, None)
            if stream is None:
                continue
            capture = _BoundedCapture()
            thread = threading.Thread(
                target=_drain_pipe,
                args=(stream, capture),
                name=f"muse-installer-{name}",
                daemon=True,
            )
            try:
                thread.start()
            except BaseException:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
                raise
            readers.append(_CaptureReader(stream=stream, capture=capture, thread=thread))
    except BaseException:
        _finish_capture(readers)
        raise
    return readers


def _finish_capture(readers: list[_CaptureReader]) -> tuple[str, str]:
    """Bound pipe-reader shutdown so inherited descriptors cannot hang Muse."""
    stuck: list[_CaptureReader] = []
    for reader in readers:
        reader.thread.join(timeout=_CAPTURE_JOIN_TIMEOUT)
        if reader.thread.is_alive():
            stuck.append(reader)

    if stuck:
        logger.warning(
            "installer cleanup left %d capture reader(s) still open; "
            "releasing Muse-owned streams",
            len(stuck),
        )

    # Supported POSIX cleanup has already signaled the group while its leader
    # pinned the identity. Unsupported-platform fallback may leave a surviving
    # descendant with a writer. At this point no numeric identity is safe to
    # revisit, so close only Muse's exact read descriptors and give the daemon
    # drain threads one final bounded opportunity to unwind.
    for reader in stuck:
        try:
            reader.stream.close()
        except (OSError, ValueError):
            pass

    for reader in stuck:
        reader.thread.join(timeout=_CAPTURE_JOIN_TIMEOUT)
        if reader.thread.is_alive():
            logger.warning(
                "installer %s reader did not close after its read fd was released",
                reader.thread.name,
            )

    values = [reader.capture.text() for reader in readers]
    while len(values) < 2:
        values.append("")
    return values[0], values[1]


def _target_alive(target: _OwnedProcess) -> bool:
    """Best-effort exact-handle liveness for unsupported platforms/tests."""
    if target.leader_reaped:
        return False
    try:
        returncode = target.process.poll()
    except Exception as exc:  # noqa: BLE001 - platform Popen handles vary
        logger.warning("could not prove installer leader is alive: %s", exc)
        return False
    if returncode is not None:
        # Popen.poll() reaps a POSIX child. Record that identity is no longer
        # safe before any caller could consider a numeric group signal.
        target.leader_reaped = True
        return False
    return True


def _supports_pinned_leader(target: _OwnedProcess) -> bool:
    """Whether WNOWAIT can pin this exact real POSIX child identity."""
    return (
        _OS_NAME == "posix"
        and target.pid is not None
        and target.process_group == target.pid
        and isinstance(target.process, _REAL_POPEN_TYPE)
        and callable(getattr(os, "waitid", None))
        and getattr(os, "P_PID", None) is not None
        and getattr(os, "WEXITED", None) is not None
        and getattr(os, "WNOHANG", None) is not None
        and getattr(os, "WNOWAIT", None) is not None
    )


def _observe_owned_leader(
    target: _OwnedProcess,
    *,
    timeout: float,
    command: Sequence[str],
) -> None:
    """Observe child exit without reaping, preserving PID/PGID ownership."""
    if not _supports_pinned_leader(target):
        raise _OwnedProcessCleanupError(
            "WNOWAIT requested for an unsupported installer child"
        )
    deadline = time.monotonic() + max(0.0, timeout)
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    while True:
        try:
            status = os.waitid(os.P_PID, target.pid, flags)
        except InterruptedError:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(list(command), timeout)
            continue
        except ChildProcessError as exc:
            # Another reaper has released the identity; fail closed and never
            # use the stored numeric PID/PGID again.
            target.leader_reaped = True
            raise _OwnedProcessCleanupError(
                f"installer leader {target.pid} was reaped outside its owner"
            ) from exc
        except OSError as exc:
            raise _OwnedProcessCleanupError(
                f"could not observe installer leader {target.pid} safely"
            ) from exc
        if status is not None and getattr(status, "si_pid", target.pid) == target.pid:
            target.leader_exit_observed = True
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(list(command), timeout)
        time.sleep(min(0.01, remaining))


def _wait_owned_leader(
    target: _OwnedProcess,
    *,
    timeout: float,
    command: Sequence[str],
) -> int | None:
    """Wait for a leader, retaining a supported POSIX child as a zombie."""
    if _supports_pinned_leader(target):
        _observe_owned_leader(target, timeout=timeout, command=command)
        return None
    returncode = target.process.wait(timeout=timeout)
    target.leader_reaped = True
    return returncode


def _reap_owned_leader(target: _OwnedProcess, *, deadline: float) -> int:
    """Reap only the exact child after the final safe group signal."""
    if target.leader_reaped:
        returncode = getattr(target.process, "returncode", None)
        if type(returncode) is int:
            return returncode
        raise _OwnedProcessCleanupError(
            f"installer leader {target.pid} was already reaped without status"
        )
    try:
        returncode = target.process.wait(
            timeout=max(0.0, deadline - time.monotonic())
        )
    except subprocess.TimeoutExpired as exc:
        raise _OwnedProcessCleanupError(
            f"installer leader {target.pid} could not be reaped before "
            "the cleanup deadline"
        ) from exc
    except (OSError, ValueError, ChildProcessError) as exc:
        raise _OwnedProcessCleanupError(
            f"could not reap installer leader {target.pid}"
        ) from exc
    target.leader_reaped = True
    return returncode


def _signal_exact_process(target: _OwnedProcess, *, force: bool) -> bool:
    if target.pid is None or target.leader_reaped:
        logger.error("refusing installer child signal with unsafe pid=%r", target.pid)
        return False
    try:
        if force:
            target.process.kill()
        else:
            target.process.terminate()
    except ProcessLookupError:
        return True
    except Exception as exc:  # noqa: BLE001 - platform Popen implementations vary
        logger.warning("could not signal installer child %s: %s", target.pid, exc)
        return False
    return True


def _signal_owned(target: _OwnedProcess, sig: signal.Signals, *, force: bool) -> bool:
    """Signal a pinned group, or use a documented best-effort fallback."""
    process_group = target.process_group
    pinned = _supports_pinned_leader(target) and not target.leader_reaped
    # Mock children retain the legacy double-poll path so safety behavior is
    # deterministic in unit tests. Real platforms without WNOWAIT deliberately
    # avoid numeric group signaling and fall back to the exact Popen handle.
    mocked_group = (
        process_group is not None
        and _OS_NAME == "posix"
        and not isinstance(target.process, _REAL_POPEN_TYPE)
    )
    if pinned or mocked_group:
        if mocked_group and not _target_alive(target):
            logger.warning(
                "refusing installer signal after exact leader exit "
                "(pid=%r pgid=%r)",
                target.pid, process_group,
            )
            return False
        try:
            own_group = _concrete_safe_id(os.getpgrp())
        except OSError:
            own_group = None
        group_is_safe = (
            own_group is not None
            and process_group > 1
            and process_group == target.pid
            and process_group != own_group
        )
        if group_is_safe:
            if mocked_group and not _target_alive(target):
                logger.warning(
                    "refusing installer signal after leader exit during "
                    "group revalidation (pid=%r pgid=%r)",
                    target.pid, process_group,
                )
                return False
            try:
                os.killpg(process_group, sig)
            except ProcessLookupError:
                return True
            except OSError as exc:
                logger.warning("could not signal installer process group %s: %s", process_group, exc)
            else:
                return True
        else:
            logger.error(
                "refusing unsafe installer group signal pid=%r pgid=%r own_pgid=%r",
                target.pid, process_group, own_group,
            )
        # A verified group that failed signaling must not fall through to a
        # second numeric identity. The caller reports cleanup failure.
        return False

    if not _target_alive(target):
        logger.warning(
            "refusing installer signal after exact leader exit (pid=%r pgid=%r)",
            target.pid, target.process_group,
        )
        return False
    return _signal_exact_process(target, force=force)


def _wait_until_gone(target: _OwnedProcess, deadline: float) -> bool:
    """Best-effort exact-child wait for unsupported platforms/tests."""
    while _target_alive(target):
        delay = min(0.05, deadline - time.monotonic())
        if delay <= 0:
            return False
        time.sleep(delay)
    return True


def _terminate_owned_fallback(
    target: _OwnedProcess, *, deadline: float, term_deadline: float,
) -> None:
    """Bounded exact-handle cleanup when WNOWAIT is unavailable."""
    if _target_alive(target):
        _signal_owned(target, signal.SIGTERM, force=False)
        if not _wait_until_gone(target, term_deadline):
            kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
            _signal_owned(target, kill_signal, force=True)
            if not _wait_until_gone(target, deadline):
                raise _OwnedProcessCleanupError(
                    f"installer child {target.pid} did not exit before "
                    "the cleanup deadline"
                )
    try:
        target.process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired as exc:
        raise _OwnedProcessCleanupError(
            f"installer child {target.pid} could not be reaped before "
            "the cleanup deadline"
        ) from exc
    except (OSError, ValueError) as exc:
        raise _OwnedProcessCleanupError(
            f"could not reap installer child {target.pid}"
        ) from exc
    target.leader_reaped = True


def _terminate_owned(target: _OwnedProcess, *, timeout: float | None = None) -> None:
    """TERM then KILL an owned install tree within one shared time budget.

    WNOWAIT-capable POSIX systems keep the leader unreaped until both signals
    have been sent to its validated group, eliminating PID/PGID reuse. Other
    platforms retain bounded best-effort cleanup through the exact Popen
    handle; descendant cleanup cannot be guaranteed there.
    """
    if timeout is None:
        timeout = _CLEANUP_TIMEOUT
    started = time.monotonic()
    deadline = started + max(0.0, timeout)
    term_deadline = started + max(0.0, timeout) * 0.6
    if not _supports_pinned_leader(target):
        _terminate_owned_fallback(
            target, deadline=deadline, term_deadline=term_deadline,
        )
        return

    term_ok = _signal_owned(target, signal.SIGTERM, force=False)
    observed = target.leader_exit_observed
    if not observed:
        try:
            _observe_owned_leader(
                target,
                timeout=max(0.0, term_deadline - time.monotonic()),
                command=getattr(target.process, "args", ()),
            )
            observed = True
        except subprocess.TimeoutExpired:
            pass

    # Always send the terminal signal while the leader identity is pinned:
    # the leader may have honored TERM while a descendant ignored it.
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    kill_ok = _signal_owned(target, kill_signal, force=True)
    if not term_ok:
        logger.warning(
            "could not deliver TERM to installer group pid=%r pgid=%r",
            target.pid, target.process_group,
        )
    if not kill_ok:
        raise _OwnedProcessCleanupError(
            f"could not deliver final cleanup signal to installer group "
            f"{target.process_group}"
        )
    if not observed:
        try:
            _observe_owned_leader(
                target,
                timeout=max(0.0, deadline - time.monotonic()),
                command=getattr(target.process, "args", ()),
            )
        except subprocess.TimeoutExpired as exc:
            raise _OwnedProcessCleanupError(
                f"installer group {target.process_group} did not terminate "
                "before the cleanup deadline"
            ) from exc
    _reap_owned_leader(target, deadline=deadline)


def _run_owned(
    cmd: Sequence[str], *, timeout: float, capture_output: bool,
    check: bool = True, env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded installer command and clean its tree on every failure."""
    target = _spawn_owned(cmd, capture_output=capture_output, env=env)
    readers: list[_CaptureReader] = []
    try:
        if capture_output:
            readers = _start_capture(target)
        returncode = _wait_owned_leader(
            target, timeout=timeout, command=cmd,
        )
        if returncode is None:
            # The exited leader remains an unreaped zombie, pinning the
            # validated equal PID/PGID. Kill any descendants that survived
            # their leader before releasing that identity.
            kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
            if not _signal_owned(target, kill_signal, force=True):
                raise _OwnedProcessCleanupError(
                    f"could not clean completed installer group "
                    f"{target.process_group}"
                )
            returncode = _reap_owned_leader(
                target, deadline=time.monotonic() + _CLEANUP_TIMEOUT,
            )
    except subprocess.TimeoutExpired as exc:
        try:
            _terminate_owned(target)
        except BaseException as cleanup_exc:
            stdout, stderr = _finish_capture(readers)
            exc.stdout = stdout
            exc.stderr = stderr
            raise cleanup_exc from exc
        stdout, stderr = _finish_capture(readers)
        exc.stdout = stdout
        exc.stderr = stderr
        raise
    except BaseException as exc:
        try:
            _terminate_owned(target)
        except BaseException as cleanup_exc:
            _finish_capture(readers)
            raise cleanup_exc from exc
        _finish_capture(readers)
        raise

    # The exact leader is reaped only after the final group signal. No numeric
    # PID/PGID is inspected or signaled beyond this point.
    stdout, stderr = _finish_capture(readers)
    result = subprocess.CompletedProcess(list(cmd), returncode, stdout, stderr)
    if returncode != 0 and check:
        raise subprocess.CalledProcessError(
            returncode, list(cmd), output=stdout, stderr=stderr,
        )
    return result


def run_owned_command(
    cmd: Sequence[str],
    *,
    timeout: float,
    capture_output: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command in an isolated group with bounded ownership-safe cleanup.

    This is the ownership-safe analogue of the small subprocess.run subset
    used by Muse. WNOWAIT-capable POSIX platforms clean the validated group
    while its leader pins the PID/PGID; unsupported platforms retain a
    bounded exact-child best-effort fallback. Captured output keeps a bounded
    tail; check controls whether a non-zero exit is returned or raised.
    """
    return _run_owned(
        cmd,
        timeout=timeout,
        capture_output=capture_output,
        check=check,
    )


def _emit_failure_output(exc: subprocess.CalledProcessError | subprocess.TimeoutExpired) -> None:
    stdout = getattr(exc, "stdout", None) or getattr(exc, "output", None)
    stderr = getattr(exc, "stderr", None)
    if stdout:
        sys.stderr.write(stdout)
    if stderr:
        sys.stderr.write(stderr)


def _cleanup_venv_staging(staging: Path, parent: Path) -> None:
    """Remove exactly one private venv staging entry without following links."""
    if staging.parent != parent:
        logger.error("refusing venv cleanup outside its parent: %s", staging)
        return
    try:
        mode = staging.lstat().st_mode
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("could not inspect venv staging path %s: %s", staging, exc)
        return

    try:
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            staging.unlink()
            return
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            logger.error(
                "refusing recursive venv cleanup without fd-safe rmtree: %s",
                staging,
            )
            return
        shutil.rmtree(staging)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("could not clean venv staging path %s: %s", staging, exc)


def _validate_created_venv(staging: Path) -> None:
    """Require the staged environment to contain a usable interpreter."""
    python = venv_python(staging)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(
            f"venv creation completed without an executable interpreter: {python}"
        )


def create_venv(target: Path) -> None:
    """Create a fresh venv at `target`, using the same Python that muse runs on.

    Using `sys.executable` guarantees ABI compatibility: the venv's Python
    is the same version as the muse-supervisor's Python, so torch/CUDA
    wheels built for one will load in the other.

    Honors `install_output_mode(verbose=...)`: in quiet mode captures
    stdout/stderr and only emits on error.
    """
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        target.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"refusing to replace existing venv path: {target}")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
    logger.info("creating venv at %s", target)
    cmd = [sys.executable, "-m", "venv", str(staging)]
    quiet = not _is_verbose()
    try:
        run_owned_command(
            cmd, timeout=_VENV_CREATE_TIMEOUT,
            capture_output=quiet, check=True,
        )
        _validate_created_venv(staging)
        staging.rename(target)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        if quiet:
            _emit_failure_output(exc)
        raise
    finally:
        _cleanup_venv_staging(staging, parent)


def ensure_venv(
    target: Path,
    *,
    creator: collections.abc.Callable[[Path], None] = create_venv,
) -> None:
    """Create a missing venv or reject an unsafe/incomplete existing one."""
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        creator(target)
        return
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise RuntimeError(f"existing venv path is not a regular directory: {target}")
    python = venv_python(target)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(
            f"existing venv is incomplete (missing executable {python}); "
            "move it aside or purge it, then re-pull the model"
        )


def _venv_clone_bytes(source: Path) -> int:
    """Conservatively estimate bytes needed to copy one venv tree."""
    total = 0
    try:
        for root, directories, files in os.walk(source, followlinks=False):
            entries = [Path(root), *(Path(root) / name for name in directories)]
            entries.extend(Path(root) / name for name in files)
            for entry in entries:
                metadata = entry.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    continue
                allocated = int(getattr(metadata, "st_blocks", 0)) * 512
                total += max(4096, int(metadata.st_size), allocated)
    except OSError as exc:
        raise RuntimeError(
            f"could not size existing venv for transactional update: {source}"
        ) from exc
    return total


def _insufficient_venv_space(
    target: Path, *, required: int, available: int | None = None,
) -> RuntimeError:
    detail = f"requires approximately {required} bytes"
    if available is not None:
        detail += f", but only {available} bytes are available"
    return RuntimeError(
        f"insufficient disk space to transactionally update venv {target}: "
        f"{detail}. Free space or purge unused models, then retry"
    )


def _is_venv_space_error(exc: BaseException) -> bool:
    """Recognize direct and copytree-aggregated disk exhaustion errors."""
    disk_errnos = {errno.ENOSPC, getattr(errno, "EDQUOT", -1)}
    if isinstance(exc, OSError) and exc.errno in disk_errnos:
        return True
    if isinstance(exc, shutil.Error):
        details = exc.args[0] if exc.args else ()
        for detail in details if isinstance(details, list) else ():
            cause = detail[2] if isinstance(detail, tuple) and len(detail) >= 3 else detail
            if isinstance(cause, BaseException) and _is_venv_space_error(cause):
                return True
            text = str(cause)
            if any(f"[Errno {number}]" in text for number in disk_errnos):
                return True
    return False


@dataclass
class _VenvTransaction:
    """Rollback-safe mutation of one canonical model venv.

    An existing venv is renamed into a private sibling workspace and cloned
    back to its original path. Callers therefore mutate the same canonical
    path embedded in venv scripts and .pth files while the prior environment
    remains available for rollback. This temporarily needs roughly one extra
    venv's worth of disk space.

    Exceptions, cancellation, and ordinary exits without commit restore the
    prior venv. A SIGKILL or power loss between filesystem and catalog commits
    cannot be made unambiguous without a cross-resource transaction; private
    transaction directories are therefore never heuristically promoted or
    restored on a later run.
    """

    path: Path
    _workspace: Path | None = None
    _prior: Path | None = None
    _entered: bool = False
    _committed: bool = False

    def __enter__(self) -> "_VenvTransaction":
        if self._entered:
            raise RuntimeError("venv transaction cannot be entered more than once")
        self._entered = True
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)

        try:
            self.path.lstat()
        except FileNotFoundError:
            return self

        # Reject symlinks and incomplete environments before moving anything.
        ensure_venv(self.path)
        required = _venv_clone_bytes(self.path)
        try:
            available = int(shutil.disk_usage(parent).free)
        except OSError as exc:
            raise RuntimeError(
                f"could not determine free space for transactional venv update: "
                f"{parent}"
            ) from exc
        if available < required:
            raise _insufficient_venv_space(
                self.path, required=required, available=available,
            )

        workspace = Path(
            tempfile.mkdtemp(prefix=f".{self.path.name}.transaction-", dir=parent)
        )
        prior = workspace / "prior"
        self._workspace = workspace
        self._prior = prior
        prior_exists = False
        try:
            self.path.rename(prior)
            prior_exists = True
            shutil.copytree(
                prior,
                self.path,
                symlinks=True,
                copy_function=shutil.copy2,
            )
            ensure_venv(self.path)
        except BaseException as exc:
            try:
                if prior_exists:
                    self._restore()
                else:
                    self._prior = None
                    _cleanup_venv_staging(workspace, parent)
                    self._workspace = None
            except BaseException as rollback_exc:
                raise RuntimeError(
                    f"failed to clone venv {self.path} and could not restore "
                    "the prior environment"
                ) from rollback_exc
            if _is_venv_space_error(exc):
                raise _insufficient_venv_space(
                    self.path, required=required, available=available,
                ) from exc
            raise
        return self

    def commit(self) -> None:
        """Keep the mutated canonical venv when the context exits."""
        if not self._entered:
            raise RuntimeError("venv transaction has not been entered")
        self._committed = True

    def _move_current_aside(self) -> Path | None:
        workspace = self._workspace
        if workspace is None:
            # A first pull has no backup workspace. Reserve one only when
            # rollback actually has a newly-created target to remove.
            try:
                self.path.lstat()
            except FileNotFoundError:
                return None
            workspace = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.path.name}.transaction-", dir=self.path.parent,
                )
            )
            self._workspace = workspace
        discard = workspace / "discard"
        try:
            self.path.rename(discard)
        except FileNotFoundError:
            return None
        return discard

    def _restore(self) -> None:
        """Atomically restore the prior path, then remove failed state."""
        self._move_current_aside()
        prior = self._prior
        if prior is not None:
            prior.rename(self.path)
            self._prior = None
        workspace = self._workspace
        if workspace is not None:
            _cleanup_venv_staging(workspace, self.path.parent)
            self._workspace = None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._committed:
            try:
                self._restore()
            except BaseException as rollback_exc:
                raise RuntimeError(
                    f"could not restore venv {self.path} after failed update"
                ) from rollback_exc
            return False

        workspace = self._workspace
        if workspace is not None:
            _cleanup_venv_staging(workspace, self.path.parent)
            self._workspace = None
            self._prior = None
        return False


def venv_transaction(target: Path) -> _VenvTransaction:
    """Return a rollback-safe transaction for one canonical model venv."""
    return _VenvTransaction(Path(target))


def install_into_venv(venv_path: Path, packages: list[str]) -> None:
    """pip-install `packages` using the venv's own pip.

    Shells out to `<venv>/bin/python -m pip install ...` so installs
    land in the target venv, not the supervisor's env.

    Output mode (v0.40.3+):
      - Quiet (default): adds `-q` to pip and captures stdout/stderr.
        On non-zero exit, both are emitted to stderr so the user can
        diagnose. On success, the user sees only the stage marker
        from the caller's `logger.info`.
      - Verbose (`muse pull -v`): no `-q`, no capture; pip's full
        output streams to the user's terminal as before.

    The mode flips via the `install_output_mode(verbose=...)` context
    manager (set by the CLI on each pull invocation).
    """
    if not packages:
        return
    py = venv_python(venv_path)
    verbose = _is_verbose()
    logger.info("installing %s into %s", packages, venv_path)
    cmd = [str(py), "-m", "pip", "install"]
    if not verbose:
        cmd.append("-q")
    cmd.extend(packages)

    try:
        run_owned_command(
            cmd, timeout=_PIP_TIMEOUT,
            capture_output=not verbose, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        if not verbose:
            _emit_failure_output(exc)
        raise


def _source_relative_path(
    value: object, *, field: str, allow_dot: bool = False,
) -> str:
    """Validate one repository-relative POSIX path from a manifest."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"python source {field} must be a non-empty string")
    if "\\" in value or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"python source {field} contains unsafe characters")
    if allow_dot and value == ".":
        return value
    path = PurePosixPath(value)
    if path.is_absolute() or value != str(path):
        raise ValueError(f"python source {field} must be a normalized relative path")
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"python source {field} escapes its checkout")
    return value


def _source_string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"python source {field} must be a non-empty list")
    paths = tuple(
        _source_relative_path(item, field=f"{field}[]") for item in value
    )
    if len(paths) != len(set(paths)):
        raise ValueError(f"python source {field} contains duplicates")
    return paths


def _parse_git_python_source(raw: object) -> _GitPythonSourceSpec:
    """Parse and fail closed on one JSON-serializable source declaration."""
    if not isinstance(raw, Mapping):
        raise ValueError("python source declaration must be a mapping")
    allowed = {
        "type", "name", "url", "revision", "sparse_paths",
        "required_paths", "pth_path", "submodules",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown python source fields: {sorted(unknown)}")
    if raw.get("type") != "git":
        raise ValueError("python source type must be 'git'")

    name = raw.get("name")
    if not isinstance(name, str) or not _SOURCE_NAME_RE.fullmatch(name):
        raise ValueError("python source name must be a filesystem-safe identifier")
    url = raw.get("url")
    if not isinstance(url, str) or not _GITHUB_GIT_URL_RE.fullmatch(url):
        raise ValueError(
            "python source url must be an HTTPS github.com repository ending in .git"
        )
    revision = raw.get("revision")
    if not isinstance(revision, str) or not _GIT_COMMIT_RE.fullmatch(revision):
        raise ValueError("python source revision must be a full lowercase commit SHA")

    sparse_paths = _source_string_sequence(
        raw.get("sparse_paths"), field="sparse_paths",
    )
    required_paths = _source_string_sequence(
        raw.get("required_paths"), field="required_paths",
    )
    pth_path = _source_relative_path(
        raw.get("pth_path", "."), field="pth_path", allow_dot=True,
    )

    raw_submodules = raw.get("submodules", ())
    if not isinstance(raw_submodules, (list, tuple)):
        raise ValueError("python source submodules must be a list")
    submodules: list[_GitSubmoduleSpec] = []
    seen_submodule_paths: set[str] = set()
    for item in raw_submodules:
        if not isinstance(item, Mapping) or set(item) != {"path", "url", "revision"}:
            raise ValueError(
                "each python source submodule requires path, url, and revision"
            )
        path = _source_relative_path(item.get("path"), field="submodules[].path")
        sub_url = item.get("url")
        if not isinstance(sub_url, str) or not _GITHUB_GIT_URL_RE.fullmatch(sub_url):
            raise ValueError("python source submodule url must be HTTPS GitHub .git")
        sub_revision = item.get("revision")
        if (
            not isinstance(sub_revision, str)
            or not _GIT_COMMIT_RE.fullmatch(sub_revision)
        ):
            raise ValueError(
                "python source submodule revision must be a full lowercase commit SHA"
            )
        if path in seen_submodule_paths:
            raise ValueError("python source submodule paths must be unique")
        if not any(path == root or path.startswith(f"{root}/") for root in sparse_paths):
            raise ValueError("python source submodule must be inside a sparse path")
        seen_submodule_paths.add(path)
        submodules.append(_GitSubmoduleSpec(path, sub_url, sub_revision))

    return _GitPythonSourceSpec(
        name=name,
        url=url,
        revision=revision,
        sparse_paths=sparse_paths,
        required_paths=required_paths,
        pth_path=pth_path,
        submodules=tuple(submodules),
    )


def _source_timeout(deadline: float, command: Sequence[str]) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(
            cmd=list(command), timeout=_SOURCE_INSTALL_TIMEOUT,
        )
    return remaining


def _run_source_git(
    checkout: Path,
    args: Sequence[str],
    *,
    deadline: float,
) -> subprocess.CompletedProcess[str]:
    command = ["git", *_GIT_SAFE_CONFIG, "-C", str(checkout), *args]
    # A reviewed URL is meaningless if inherited user configuration can
    # rewrite it, redirect the worktree/object database, or register checkout
    # filters. Preserve ordinary process/TLS/proxy environment while removing
    # every Git-specific override, then install only our non-interactive global
    # config boundary.
    git_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    git_env.update({
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "Never",
    })
    try:
        return _run_owned(
            command,
            timeout=_source_timeout(deadline, command),
            capture_output=True,
            env=git_env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git is required to install a model's reviewed Python source"
        ) from exc


def _source_git_output(
    checkout: Path,
    args: Sequence[str],
    *,
    deadline: float,
) -> str:
    result = _run_source_git(checkout, args, deadline=deadline)
    return (result.stdout or "").strip()


def _require_path_inside(path: Path, root: Path, *, label: str) -> Path:
    try:
        resolved_root = root.resolve()
        resolved = path.resolve()
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"{label} escapes the model-owned venv") from exc
    return resolved


def _validate_submodule_metadata(
    checkout: Path, submodules: tuple[_GitSubmoduleSpec, ...],
) -> None:
    if not submodules:
        return
    metadata = checkout / ".gitmodules"
    if not metadata.is_file() or metadata.is_symlink():
        raise RuntimeError("reviewed Python source is missing a regular .gitmodules file")
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with metadata.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, UnicodeError, configparser.Error) as exc:
        raise RuntimeError("could not parse reviewed Python source submodules") from exc
    for submodule in submodules:
        section = f'submodule "{submodule.path}"'
        if not parser.has_section(section):
            raise RuntimeError(f"reviewed submodule {submodule.path!r} is not declared")
        if parser.get(section, "path", fallback=None) != submodule.path:
            raise RuntimeError(f"reviewed submodule {submodule.path!r} path changed")
        if parser.get(section, "url", fallback=None) != submodule.url:
            raise RuntimeError(f"reviewed submodule {submodule.path!r} URL changed")
        if set(parser.options(section)) != {"path", "url"}:
            raise RuntimeError(
                f"reviewed submodule {submodule.path!r} options changed"
            )


def _validate_git_checkout(
    checkout: Path,
    spec: _GitPythonSourceSpec,
    *,
    deadline: float,
) -> None:
    actual_url = _source_git_output(
        checkout, ("remote", "get-url", "origin"), deadline=deadline,
    )
    if actual_url != spec.url:
        raise RuntimeError(
            f"Python source {spec.name!r} origin changed: {actual_url!r}"
        )
    actual_revision = _source_git_output(
        checkout, ("rev-parse", "HEAD"), deadline=deadline,
    )
    if actual_revision != spec.revision:
        raise RuntimeError(
            f"Python source {spec.name!r} is at unexpected commit "
            f"{actual_revision!r}"
        )
    dirty = _source_git_output(
        checkout,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        deadline=deadline,
    )
    if dirty:
        raise RuntimeError(f"Python source {spec.name!r} contains local changes")

    _validate_submodule_metadata(checkout, spec.submodules)
    for submodule in spec.submodules:
        submodule_root = checkout.joinpath(*PurePosixPath(submodule.path).parts)
        _require_path_inside(
            submodule_root, checkout, label=f"submodule {submodule.path!r}",
        )
        actual = _source_git_output(
            submodule_root, ("rev-parse", "HEAD"), deadline=deadline,
        )
        if actual != submodule.revision:
            raise RuntimeError(
                f"Python source submodule {submodule.path!r} is at "
                f"unexpected commit {actual!r}"
            )
        submodule_dirty = _source_git_output(
            submodule_root,
            ("status", "--porcelain=v1", "--untracked-files=all"),
            deadline=deadline,
        )
        if submodule_dirty:
            raise RuntimeError(
                f"Python source submodule {submodule.path!r} contains local changes"
            )

    for relative in spec.required_paths:
        required = checkout.joinpath(*PurePosixPath(relative).parts)
        _require_path_inside(required, checkout, label=f"required path {relative!r}")
        if not required.is_file() or required.is_symlink():
            raise RuntimeError(
                f"Python source {spec.name!r} is missing regular file {relative!r}"
            )


def _materialize_git_source(
    source_root: Path,
    spec: _GitPythonSourceSpec,
    *,
    deadline: float,
) -> Path:
    name_root = source_root / spec.name
    name_root.mkdir(parents=True, exist_ok=True)
    _require_path_inside(name_root, source_root, label="Python source directory")
    if name_root.is_symlink():
        raise RuntimeError("Python source directory must not be a symlink")
    target = name_root / spec.revision

    if target.exists():
        if not target.is_dir() or target.is_symlink():
            raise RuntimeError("existing Python source target is not a regular directory")
        _validate_git_checkout(target, spec, deadline=deadline)
        return target

    temp = Path(tempfile.mkdtemp(prefix=f".{spec.revision}.", dir=name_root))
    try:
        _run_source_git(temp, ("init", "--quiet"), deadline=deadline)
        _run_source_git(
            temp, ("remote", "add", "origin", spec.url), deadline=deadline,
        )
        _run_source_git(
            temp, ("sparse-checkout", "init", "--cone"), deadline=deadline,
        )
        _run_source_git(
            temp, ("sparse-checkout", "set", *spec.sparse_paths), deadline=deadline,
        )
        _run_source_git(
            temp,
            (
                "fetch", "--depth", "1", "--filter=blob:none", "--no-tags",
                "origin", spec.revision,
            ),
            deadline=deadline,
        )
        _run_source_git(
            temp, ("checkout", "--quiet", "--detach", "FETCH_HEAD"),
            deadline=deadline,
        )
        _validate_submodule_metadata(temp, spec.submodules)
        for submodule in spec.submodules:
            _run_source_git(
                temp,
                (
                    "submodule", "update", "--init", "--depth", "1",
                    "--filter=blob:none", "--", submodule.path,
                ),
                deadline=deadline,
            )
        _validate_git_checkout(temp, spec, deadline=deadline)
        try:
            temp.rename(target)
        except OSError as exc:
            try:
                target_mode = target.lstat().st_mode
            except FileNotFoundError:
                target_is_regular_dir = False
            else:
                target_is_regular_dir = (
                    stat.S_ISDIR(target_mode) and not stat.S_ISLNK(target_mode)
                )
            if (
                exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}
                or not target_is_regular_dir
            ):
                raise
            # A concurrent refresh may have won promotion. Validate its exact
            # identity before discarding our private, still-owned temporary.
            _validate_git_checkout(target, spec, deadline=deadline)
        return target
    finally:
        _cleanup_source_staging(temp, name_root)


def _cleanup_source_staging(temp: Path, name_root: Path) -> None:
    """Remove exactly one installer-owned staging path without traversal risk."""
    if temp.parent != name_root:
        logger.error("refusing Python source cleanup outside its name directory: %s", temp)
        return
    try:
        mode = temp.lstat().st_mode
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("could not inspect Python source staging path %s: %s", temp, exc)
        return

    try:
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            # A substituted symlink/file is removed as one directory entry;
            # its target is never traversed.
            temp.unlink()
            return
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            logger.error(
                "refusing recursive source cleanup without fd-safe rmtree: %s",
                temp,
            )
            return
        shutil.rmtree(temp)
    except OSError as exc:
        # Cleanup must not mask the source/Git failure that led here. Leaving
        # an inert staging directory is safer than widening deletion scope.
        logger.warning("could not remove Python source staging path %s: %s", temp, exc)


def _venv_site_packages(venv_path: Path) -> Path:
    if _OS_NAME == "nt":
        return venv_path / "Lib" / "site-packages"
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return venv_path / "lib" / version / "site-packages"


def _source_pth_target(
    venv_path: Path,
    spec: _GitPythonSourceSpec,
) -> Path:
    site_packages = _venv_site_packages(venv_path)
    if not site_packages.is_dir() or site_packages.is_symlink():
        raise RuntimeError(
            f"venv site-packages is missing or unsafe: {site_packages}"
        )
    _require_path_inside(site_packages, venv_path, label="venv site-packages")
    return site_packages / f"muse-source-{spec.name}.pth"


def _disable_source_pth(venv_path: Path, spec: _GitPythonSourceSpec) -> None:
    """Fail closed before revalidating a previously exposed source tree."""
    target = _source_pth_target(venv_path, spec)
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        raise RuntimeError(f"Python source .pth target is an unsafe directory: {target}")
    try:
        target.unlink()
    except FileNotFoundError:
        pass


def _disable_all_source_pths(venv_path: Path) -> None:
    """Remove every hook in Muse's reserved reviewed-source namespace."""
    site_packages = _venv_site_packages(venv_path)
    try:
        site_mode = site_packages.lstat().st_mode
    except FileNotFoundError:
        # No site-packages means there is no executable .pth surface to revoke.
        return
    if not stat.S_ISDIR(site_mode) or stat.S_ISLNK(site_mode):
        raise RuntimeError(
            f"venv site-packages is missing or unsafe: {site_packages}"
        )
    _require_path_inside(site_packages, venv_path, label="venv site-packages")

    unsafe_directories: list[Path] = []
    for target in sorted(site_packages.glob("muse-source-*.pth")):
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            unsafe_directories.append(target)
            continue
        try:
            target.unlink()
        except FileNotFoundError:
            pass
    if unsafe_directories:
        raise RuntimeError(
            "Python source .pth target is an unsafe directory: "
            f"{unsafe_directories[0]}"
        )


def _write_source_pth(
    venv_path: Path,
    checkout: Path,
    spec: _GitPythonSourceSpec,
) -> None:
    target = _source_pth_target(venv_path, spec)
    python_root = checkout.joinpath(*PurePosixPath(spec.pth_path).parts)
    python_root = _require_path_inside(
        python_root, checkout, label="Python source import path",
    )
    if not python_root.is_dir():
        raise RuntimeError("Python source import path is not a directory")

    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
    )
    temp = Path(raw_temp)
    open_fd: int | None = fd
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            open_fd = None
            handle.write(f"{python_root}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    except BaseException:
        if open_fd is not None:
            os.close(open_fd)
        temp.unlink(missing_ok=True)
        raise


def install_python_sources(
    venv_path: Path,
    sources: Sequence[Mapping[str, object]],
) -> tuple[Path, ...]:
    """Install immutable non-packaged Python sources into one model venv.

    Each declaration is validated before network work. Git operations share
    one wall-clock deadline and use the same exact process-group ownership as
    venv/pip commands. Only reviewed sparse paths and explicit submodules are
    materialized; a plain `.pth` then exposes the checkout to that venv.
    """
    if isinstance(sources, (str, bytes)):
        raise ValueError("python sources must be a sequence of mappings")

    venv_path = Path(venv_path)
    try:
        venv_mode = venv_path.lstat().st_mode
    except FileNotFoundError:
        if not sources:
            return ()
        raise RuntimeError(f"model venv is missing or unsafe: {venv_path}")
    if not stat.S_ISDIR(venv_mode) or stat.S_ISLNK(venv_mode):
        raise RuntimeError(f"model venv is missing or unsafe: {venv_path}")
    # Revoke every old hook before parsing/fetching the new declaration set.
    # This covers removals, renames, empty sets, and invalid replacements.
    _disable_all_source_pths(venv_path)

    specs = tuple(_parse_git_python_source(source) for source in sources)
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("python source names must be unique per model")
    if not specs:
        return ()

    source_root = venv_path / "muse-sources"
    source_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_path_inside(source_root, venv_path, label="Python source root")
    if source_root.is_symlink():
        raise RuntimeError("Python source root must not be a symlink")

    deadline = time.monotonic() + _SOURCE_INSTALL_TIMEOUT
    installed: list[Path] = []
    activated = False
    try:
        try:
            for spec in specs:
                installed.append(_materialize_git_source(
                    source_root, spec, deadline=deadline,
                ))
            # Activate only after every checkout has passed validation.
            for spec, checkout in zip(specs, installed):
                _write_source_pth(venv_path, checkout, spec)
            activated = True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if not _is_verbose():
                _emit_failure_output(exc)
            raise
    finally:
        if not activated:
            for spec in specs:
                try:
                    _disable_source_pth(venv_path, spec)
                except (OSError, RuntimeError) as exc:
                    logger.error(
                        "could not fail-closed Python source %r: %s",
                        spec.name, exc,
                    )
    return tuple(installed)


def find_free_port(start: int = 9001, end: int = 9999) -> int:
    """Find an unbound local port in [start, end]. Raises RuntimeError if exhausted.

    The port is probed by briefly binding then releasing it; the returned
    number is a *hint*, not a reservation. A TOCTOU window exists between
    this function returning and the caller binding the port, so the caller
    MUST verify the worker actually bound it (e.g., via /health check)
    and retry with a different port on startup failure.
    """
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"no free port in range [{start}, {end}]")
