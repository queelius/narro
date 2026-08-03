"""In-memory async-job tracker for admin operations.

Each enable / pull / probe call returns a Job; the caller polls
GET /v1/admin/jobs/{id} to observe progression. Jobs persist for ten
minutes after `finished_at`; older jobs are reaped on every list call
(lazy reap) to keep memory bounded without a dedicated reaper thread.

The job_id is a uuid4 hex string. Jobs go through:
  pending -> running -> (done | failed)
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from muse.core.resource_registry import (
    ResourceRegistryError,
    register_process,
    unregister_process,
)

logger = logging.getLogger(__name__)

_RETENTION_SECONDS = 600.0  # ten minutes
_MAX_JOBS = 100
_PROCESS_WAIT_SLICE_SECONDS = 0.1
_REGISTRATION_ROLLBACK_SECONDS = 5.0
_MANAGED_JOB_GROUP_ENV = "MUSE_MANAGED_JOB_PROCESS_GROUP"
_REAL_POPEN_TYPE = subprocess.Popen


class JobStoreFullError(RuntimeError):
    """Raised when every bounded job slot is still active."""


class JobStoreShuttingDownError(RuntimeError):
    """Raised when work is submitted after shutdown has begun."""


@dataclass
class Job:
    """One async admin operation.

    `thread` is the daemon worker that runs the operation; tracked so
    the gateway can join it on shutdown. Not serialized into to_dict.
    `finished_at_monotonic` is for lazy expiry; not serialized either.
    """
    job_id: str
    op: str
    model_id: str
    state: str = "pending"
    started_at: str = ""
    finished_at: str | None = None
    result: dict | None = None
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    thread: Any = field(default=None, repr=False)
    finished_at_monotonic: float | None = field(default=None, repr=False)
    process: Any = field(default=None, repr=False)
    process_pid: int | None = field(default=None, repr=False)
    process_group_id: int | None = field(default=None, repr=False)
    process_group_released: bool = field(default=False, repr=False)
    resource_id: str | None = field(default=None, repr=False)
    process_cancel_requested: bool = field(default=False, repr=False)
    process_lock: Any = field(default_factory=threading.RLock, repr=False)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "op": self.op,
            "model_id": self.model_id,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "log_lines": list(self.log_lines),
        }


class JobStore:
    """Thread-safe in-memory job map with lazy expiry.

    `retention_seconds` controls how long a finished job stays
    addressable via `get`/`list_recent`. The default is 10 minutes,
    matching the spec.

    `max_jobs` is a hard bound on both the map and its ordering deque.
    Creating a job evicts the oldest terminal entry when necessary. If
    every slot is still active, creation raises `JobStoreFullError`
    rather than forgetting a live job (and any subprocess it owns).

    Subprocess jobs are spawned through `spawn_process`, which registers
    the Popen object while holding the same lock used by `shutdown`.
    Shutdown first closes the store to new subprocesses, then signals only
    those positively-owned children/process groups before joining threads.
    """

    def __init__(self, retention_seconds: float = _RETENTION_SECONDS, max_jobs: int = _MAX_JOBS):
        if max_jobs <= 0:
            raise ValueError("max_jobs must be greater than zero")
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()
        self._lock = threading.Lock()
        self._retention = retention_seconds
        self._max_jobs = max_jobs
        self._shutting_down = False

    def create(self, op: str, model_id: str) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex,
            op=op,
            model_id=model_id,
            state="pending",
            started_at=_now_iso(),
        )
        with self._lock:
            if self._shutting_down:
                raise JobStoreShuttingDownError("admin job store is shutting down")
            self._reap_expired()
            self._make_room_for_job()
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
        logger.info("job %s created (op=%s, model=%s)", job.job_id, op, model_id)
        return job

    def update(self, job_id: str, **fields: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for k, v in fields.items():
                setattr(job, k, v)
            if job.state in ("done", "failed") and job.finished_at_monotonic is None:
                job.finished_at = _now_iso()
                job.finished_at_monotonic = time.monotonic()
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._reap_expired()
            return self._jobs.get(job_id)

    def list_recent(self) -> list[Job]:
        """Return jobs newest-first, capped at `max_jobs`."""
        with self._lock:
            self._reap_expired()
            return [self._jobs[jid] for jid in reversed(self._order) if jid in self._jobs]

    def start_thread(self, job_id: str, thread: threading.Thread) -> None:
        """Atomically attach and start a job thread unless shutdown won.

        Starting while holding the store lock closes the create/attach race:
        shutdown either sees the registered thread, or marks the store closed
        first and this method refuses to start it.
        """
        with self._lock:
            if self._shutting_down:
                raise JobStoreShuttingDownError("admin job store is shutting down")
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"unknown admin job {job_id!r}")
            if job.thread is not None:
                raise RuntimeError(f"admin job {job_id!r} already owns a thread")
            job.thread = thread
            try:
                thread.start()
            except Exception:
                job.thread = None
                raise

    def spawn_process(
        self,
        job_id: str,
        cmd: list[str],
        **popen_kwargs: Any,
    ) -> subprocess.Popen:
        """Spawn and atomically register one subprocess owned by `job_id`.

        POSIX children start a new session, making the child PID the process
        group ID. The group is recorded only after validating that concrete
        identity and ensuring it is not this process's own group. On other
        platforms the Popen object remains the positive ownership token and
        shutdown falls back to signalling that child directly.
        """
        with self._lock:
            if self._shutting_down:
                raise JobStoreShuttingDownError("admin job store is shutting down")
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"unknown admin job {job_id!r}")
            if job.process is not None:
                raise RuntimeError(f"admin job {job_id!r} already owns a subprocess")

            supplied_env = popen_kwargs.pop("env", None)
            child_env = os.environ.copy()
            if supplied_env is not None:
                child_env.update(supplied_env)
            # Inner Muse helpers inherit this marker and stay inside this one
            # authoritative outer process group instead of escaping into a
            # nested session that JobStore cannot drain.
            child_env[_MANAGED_JOB_GROUP_ENV] = "1"
            popen_kwargs["env"] = child_env

            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            elif os.name == "nt":
                create_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                popen_kwargs["creationflags"] = (
                    int(popen_kwargs.get("creationflags", 0)) | create_group
                )

            process = subprocess.Popen(cmd, **popen_kwargs)
            pid = _concrete_safe_id(getattr(process, "pid", None))
            pgid = _validated_isolated_group(pid)
            job.process = process
            job.process_pid = pid
            job.process_group_id = pgid
            job.process_group_released = False
            registration_error: Exception | None = None
            if pid is None:
                registration_error = ResourceRegistryError(
                    "child returned an unsafe process identity"
                )
            else:
                try:
                    resource_id = register_process(
                        kind="admin_job",
                        pid=pid,
                        owner_pid=os.getpid(),
                        models=[job.model_id],
                    )
                    if not isinstance(resource_id, str) or not resource_id:
                        raise ResourceRegistryError(
                            "resource registration returned an invalid identifier"
                        )
                    job.resource_id = resource_id
                except (ResourceRegistryError, TypeError, ValueError) as exc:
                    registration_error = exc

            if registration_error is not None:
                # Returning an unregistered child would recreate the exact
                # post-crash leak this registry exists to prevent. Roll back
                # through the concrete Popen ownership token. If exit cannot
                # be proven, retain the target on the Job so supervisor
                # shutdown can make another bounded cleanup attempt.
                target = _ProcessTarget(
                    process=process,
                    pid=pid,
                    pgid=pgid,
                    process_lock=job.process_lock,
                    job=job,
                )
                _terminate_targets(
                    [target], timeout=_REGISTRATION_ROLLBACK_SECONDS,
                )
                if _target_cleanup_complete(target):
                    job.process = None
                    job.process_pid = None
                    job.process_group_id = None
                    job.process_group_released = False
                    detail = "the child was rolled back"
                else:
                    detail = "the child remains retained for shutdown cleanup"
                raise ResourceRegistryError(
                    f"admin job {job_id} cannot run without a persisted "
                    f"process identity; {detail}: {registration_error}"
                ) from registration_error
            return process

    def release_process(self, job_id: str, process: subprocess.Popen) -> None:
        """Release an exact child only after its full group and registry settle."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.process is not process:
                return
            target = _ProcessTarget(
                process=process,
                pid=job.process_pid,
                pgid=job.process_group_id,
                process_lock=job.process_lock,
                job=job,
            )
        state = _target_state(target)
        if state is False and not _target_cleanup_complete(target):
            _terminate_targets([target], timeout=0.0)
        if not _target_cleanup_complete(target):
            logger.warning(
                "admin job %s child/group is not positively released; "
                "retaining ownership record",
                job_id,
            )
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.process is not process:
                return
            resource_id = job.resource_id
            if resource_id is not None:
                try:
                    unregister_process(resource_id)
                except ResourceRegistryError as e:
                    logger.warning(
                        "admin job %s child resource record could not be removed: %s",
                        job_id,
                        e,
                    )
                    return
            job.process = None
            job.process_pid = None
            job.process_group_id = None
            job.process_group_released = False
            job.resource_id = None

    def process_cancel_requested(
        self, job_id: str, process: subprocess.Popen,
    ) -> bool:
        """Return whether shutdown requested cancellation of this child."""
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(
                self._shutting_down
                or (
                    job is not None
                    and job.process is process
                    and job.process_cancel_requested
                )
            )

    def wait_process(
        self,
        job_id: str,
        process: subprocess.Popen,
        *,
        timeout: float,
    ) -> int:
        """Wait for an exact child without racing shutdown's signal path.

        Short non-reaping observations release the per-job identity lock so
        shutdown can signal the still-pinned leader. Once exit is observable,
        the stored group receives its final signal before this method reaps
        and returns the exact leader status.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.process is not process:
                raise RuntimeError(
                    f"admin job {job_id!r} does not own the requested subprocess"
                )
            target = _ProcessTarget(
                process=process,
                pid=job.process_pid,
                pgid=job.process_group_id,
                process_lock=job.process_lock,
                job=job,
            )

        timeout = max(0.0, timeout)
        deadline = time.monotonic() + timeout
        while True:
            state = _target_state(target)
            if state is False:
                _terminate_targets(
                    [target], timeout=max(0.0, deadline - time.monotonic()),
                )
                if not _target_cleanup_complete(target):
                    raise RuntimeError(
                        f"admin job {job_id!r} subprocess group cleanup "
                        "could not be proven"
                    )
                returncode = getattr(process, "returncode", None)
                if type(returncode) is not int:
                    raise RuntimeError(
                        f"admin job {job_id!r} subprocess was not exactly reaped"
                    )
                return returncode
            if state is None:
                raise RuntimeError(
                    f"admin job {job_id!r} subprocess identity is ambiguous"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    cmd=getattr(process, "args", "admin subprocess"),
                    timeout=timeout,
                )
            time.sleep(min(_PROCESS_WAIT_SLICE_SECONDS, remaining))

    def terminate_process(
        self,
        job_id: str,
        process: subprocess.Popen,
        *,
        timeout: float = 5.0,
    ) -> None:
        """TERM, bounded-wait, then KILL one positively-owned child."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.process is not process:
                return
            target = _ProcessTarget(
                process=process,
                pid=job.process_pid,
                pgid=job.process_group_id,
                process_lock=job.process_lock,
                job=job,
            )
        _terminate_targets([target], timeout=max(0.0, timeout))

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Terminate owned subprocesses and join live worker threads.

        ``timeout`` is the TOTAL budget shared across all threads (a single
        deadline), not a per-child or per-thread timeout. The store is marked
        closed while holding its lock, so a child cannot be spawned after
        shutdown snapshots the ownership set.
        """
        timeout = max(0.0, timeout)
        started = time.monotonic()
        deadline = started + timeout
        with self._lock:
            self._shutting_down = True
            targets: list[_ProcessTarget] = []
            for job in self._jobs.values():
                if job.process is None:
                    continue
                job.process_cancel_requested = True
                targets.append(_ProcessTarget(
                    process=job.process,
                    pid=job.process_pid,
                    pgid=job.process_group_id,
                    process_lock=job.process_lock,
                    job=job,
                ))
            threads = [j.thread for j in self._jobs.values() if j.thread is not None]

        # Reserve some of the total budget for worker threads to consume the
        # child's exit and publish a terminal Job state.
        process_budget = max(0.0, (deadline - time.monotonic()) * 0.8)
        _terminate_targets(targets, timeout=process_budget)

        for i, t in enumerate(threads):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "job-thread join budget (%.1fs) exhausted; %d of %d "
                    "thread(s) left unjoined (daemon, die with process)",
                    timeout, len(threads) - i, len(threads),
                )
                break
            try:
                t.join(timeout=remaining)
            except Exception as e:  # noqa: BLE001
                logger.warning("error joining job thread: %s", e)

        # Job threads normally release their exact child in ``finally``. If a
        # thread exited early, make the same retry here; registry failures
        # deliberately keep the Job non-evictable and make shutdown nonzero.
        for target in targets:
            if target.job is not None:
                self.release_process(target.job.job_id, target.process)

        with self._lock:
            return all(
                job.process is None
                and job.process_pid is None
                and job.process_group_id is None
                and job.resource_id is None
                and (
                    job.thread is None
                    or _thread_stopped(job.thread)
                )
                for job in self._jobs.values()
            )

    def _make_room_for_job(self) -> None:
        """Evict terminal history or reject when all bounded slots are live.

        Caller must hold `self._lock`.
        """
        while len(self._jobs) >= self._max_jobs:
            evict_id = next(
                (
                    jid for jid in self._order
                    if jid in self._jobs
                    and _job_is_evictable(self._jobs[jid])
                ),
                None,
            )
            if evict_id is None:
                raise JobStoreFullError(
                    f"admin job capacity reached ({self._max_jobs} active jobs)"
                )
            self._jobs.pop(evict_id, None)
            try:
                self._order.remove(evict_id)
            except ValueError:
                pass

    def _reap_expired(self) -> None:
        """Drop jobs whose finished_at_monotonic is older than retention.

        Caller must hold `self._lock`.
        """
        if self._retention <= 0:
            return
        cutoff = time.monotonic() - self._retention
        expired = [
            jid for jid, j in self._jobs.items()
            if j.finished_at_monotonic is not None and j.finished_at_monotonic < cutoff
            and _job_is_evictable(j)
        ]
        for jid in expired:
            self._jobs.pop(jid, None)
            try:
                self._order.remove(jid)
            except ValueError:
                pass


@dataclass
class _ProcessTarget:
    """Validated identity metadata for one owned Popen object."""

    process: subprocess.Popen
    pid: int | None
    pgid: int | None
    process_lock: Any
    job: Job | None = None
    final_signal_delivered: bool = False
    leader_reaped: bool = False
    group_released: bool = False

    def __post_init__(self) -> None:
        if self.job is not None and self.job.process_group_released:
            self.group_released = True
            self.final_signal_delivered = True
        if type(getattr(self.process, "returncode", None)) is int:
            self.leader_reaped = True


def _job_is_evictable(job: Job) -> bool:
    """Return whether forgetting this terminal job cannot lose ownership."""
    return bool(
        job.state in ("done", "failed")
        and job.process is None
        and job.resource_id is None
        and (job.thread is None or _thread_stopped(job.thread))
    )


def _thread_stopped(thread: Any) -> bool:
    try:
        return thread.is_alive() is not True
    except RuntimeError:
        return True
    except Exception:  # noqa: BLE001
        return False


def _concrete_safe_id(value: Any) -> int | None:
    """Return a signal-safe concrete PID/PGID, rejecting mocks and PID 1."""
    if type(value) is not int or value <= 1:
        return None
    return value


def _validated_isolated_group(pid: int | None) -> int | None:
    """Return the child's isolated PGID, or None when it cannot be proven."""
    if os.name != "posix" or pid is None:
        return None
    try:
        pgid = _concrete_safe_id(os.getpgid(pid))
        own_pgid = _concrete_safe_id(os.getpgrp())
    except OSError:
        return None
    if own_pgid is None or pgid != pid or pgid == own_pgid:
        logger.error(
            "refusing process-group ownership for admin child pid=%r pgid=%r own_pgid=%r",
            pid, pgid, own_pgid,
        )
        return None
    return pgid


def _supports_pinned_target(target: _ProcessTarget) -> bool:
    """Whether WNOWAIT can pin this exact real POSIX PID==PGID leader."""
    return bool(
        os.name == "posix"
        and target.pid is not None
        and target.pgid == target.pid
        and isinstance(target.process, _REAL_POPEN_TYPE)
        and callable(getattr(os, "waitid", None))
        and getattr(os, "P_PID", None) is not None
        and getattr(os, "WEXITED", None) is not None
        and getattr(os, "WNOHANG", None) is not None
        and getattr(os, "WNOWAIT", None) is not None
    )


def _target_state_locked(target: _ProcessTarget) -> bool | None:
    """Return True for alive, False for exited, or None when ambiguous."""
    if target.job is not None and target.job.process is not target.process:
        return None
    if _concrete_safe_id(getattr(target.process, "pid", None)) != target.pid:
        logger.error(
            "refusing admin child operation after process identity changed: %r",
            target.pid,
        )
        return None
    if type(getattr(target.process, "returncode", None)) is int:
        return False
    if _supports_pinned_target(target):
        flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
        try:
            status = os.waitid(os.P_PID, target.pid, flags)
        except InterruptedError:
            return True
        except (ChildProcessError, OSError) as e:
            logger.warning(
                "could not observe admin child %r without reaping: %s",
                target.pid,
                e,
            )
            return None
        return not (
            status is not None
            and getattr(status, "si_pid", target.pid) == target.pid
        )
    if os.name == "posix" and isinstance(target.process, _REAL_POPEN_TYPE):
        return None
    try:
        observed = target.process.poll()
        if type(observed) is int and type(
            getattr(target.process, "returncode", None)
        ) is not int:
            # Lightweight test/fallback process doubles need not implement
            # Popen's side effect of storing poll() on returncode; mirror it
            # so the exact-reap contract remains coherent.
            target.process.returncode = observed
        return observed is None
    except Exception as e:  # noqa: BLE001
        # Signalling a numeric identity after an ambiguous poll is unsafe:
        # fail closed and leave the registry record for explicit repair.
        logger.warning("could not inspect owned admin child %r: %s", target.pid, e)
        return None


def _target_state(target: _ProcessTarget) -> bool | None:
    with target.process_lock:
        return _target_state_locked(target)


def _target_alive_locked(target: _ProcessTarget) -> bool:
    """Return whether the exact, unreaped Popen leader is positively alive."""
    return _target_state_locked(target) is True


def _target_alive(target: _ProcessTarget) -> bool:
    """Check one owned target while serializing against wait/reap."""
    with target.process_lock:
        return _target_alive_locked(target)


def _target_exited(target: _ProcessTarget) -> bool:
    """Return whether the exact Popen leader is positively observed exited."""
    with target.process_lock:
        return _target_state_locked(target) is False


def _target_cleanup_complete(target: _ProcessTarget) -> bool:
    return bool(target.leader_reaped and target.group_released)


def _signal_target(
    target: _ProcessTarget,
    sig: signal.Signals,
    *,
    force: bool = False,
) -> bool:
    """Signal one exact target while retaining its pinned group identity."""
    with target.process_lock:
        if target.job is not None and target.job.process is not target.process:
            return False
        if target.group_released or (
            target.job is not None and target.job.process_group_released
        ):
            target.group_released = True
            target.final_signal_delivered = True
            return True

        state = _target_state_locked(target)

        if _supports_pinned_target(target):
            if type(getattr(target.process, "returncode", None)) is int:
                logger.error(
                    "refusing admin process-group signal after leader %r "
                    "was already reaped",
                    target.pid,
                )
                return False
            if state is None:
                return False
            if not force and state is not True:
                return state is False
            try:
                os.killpg(target.pgid, sig)
            except ProcessLookupError:
                if force:
                    target.final_signal_delivered = True
                return True
            except (OSError, TypeError, ValueError) as e:
                logger.warning(
                    "could not signal admin process group %s: %s", target.pgid, e,
                )
                return False
            if force:
                target.final_signal_delivered = True
            return True

        if state is False:
            if force:
                target.final_signal_delivered = True
            return True
        if state is not True:
            return False

        if target.pgid is not None and os.name == "posix":
            try:
                current_pgid = _concrete_safe_id(os.getpgid(target.pid))
                own_pgid = _concrete_safe_id(os.getpgrp())
            except OSError as e:
                logger.warning(
                    "could not revalidate admin process group %r: %s",
                    target.pgid, e,
                )
                return False
            if (
                own_pgid is None
                or target.pgid <= 1
                or target.pgid == own_pgid
                or target.pgid != target.pid
                or current_pgid != target.pgid
            ):
                logger.error(
                    "refusing unsafe admin process-group signal "
                    "pid=%r pgid=%r current_pgid=%r own_pgid=%r",
                    target.pid, target.pgid, current_pgid, own_pgid,
                )
                return False
            try:
                os.killpg(target.pgid, sig)
            except ProcessLookupError:
                if force:
                    target.final_signal_delivered = True
                return True
            except OSError as e:
                logger.warning(
                    "could not signal admin process group %s: %s",
                    target.pgid, e,
                )
                return False
            if force:
                target.final_signal_delivered = True
            return True

        # A Popen object is itself a positive ownership token, but reject an
        # unsafe/non-concrete pid before calling its signalling methods. This
        # is especially important in tests: MagicMock pid values must never
        # coerce to PID 1 or otherwise reach the OS.
        if target.pid is None or target.pid <= 1:
            logger.error("refusing admin child signal with unsafe pid=%r", target.pid)
            return False
        try:
            if force:
                target.process.kill()
                target.final_signal_delivered = True
            else:
                target.process.terminate()
        except ProcessLookupError:
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("could not signal admin child %s: %s", target.pid, e)
            return False
        return True


def _wait_for_targets(targets: list[_ProcessTarget], deadline: float) -> list[_ProcessTarget]:
    """Return targets not positively exited at one shared deadline."""
    remaining = list(targets)
    while remaining:
        remaining = [
            target for target in remaining
            if _target_state(target) is not False
        ]
        if not remaining:
            break
        delay = min(0.05, deadline - time.monotonic())
        if delay <= 0:
            break
        time.sleep(delay)
    return remaining


def _reap_target(target: _ProcessTarget) -> bool:
    """Reap the exact leader only after final group cleanup was delivered."""
    with target.process_lock:
        if target.job is not None and target.job.process is not target.process:
            return False
        returncode = getattr(target.process, "returncode", None)
        if type(returncode) is int:
            target.leader_reaped = True
            if target.final_signal_delivered:
                target.group_released = True
                if target.job is not None:
                    target.job.process_group_released = True
            return _target_cleanup_complete(target)
        if not target.final_signal_delivered:
            return False
        if _target_state_locked(target) is not False:
            return False
        try:
            returncode = target.process.wait(timeout=0.0)
        except (subprocess.TimeoutExpired, ChildProcessError, OSError) as e:
            logger.warning("could not reap admin child %r: %s", target.pid, e)
            return False
        except Exception as e:  # noqa: BLE001
            logger.warning("could not reap admin child %r: %s", target.pid, e)
            return False
        if type(returncode) is not int:
            return False
        if type(getattr(target.process, "returncode", None)) is not int:
            target.process.returncode = returncode
        target.leader_reaped = True
        target.group_released = True
        if target.job is not None:
            target.job.process_group_released = True
        return True


def _terminate_targets(targets: list[_ProcessTarget], *, timeout: float) -> bool:
    """TERM targets, final-KILL every pinned group, then reap exact leaders."""
    if not targets:
        return True
    started = time.monotonic()
    deadline = started + max(0.0, timeout)
    term_deadline = started + max(0.0, timeout) * 0.6

    live = [target for target in targets if _target_alive(target)]
    for target in live:
        _signal_target(target, signal.SIGTERM)
    survivors = _wait_for_targets(live, term_deadline)
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    survivor_ids = {id(target) for target in survivors}
    for target in targets:
        # Pinned groups are KILLed even when their leader honored TERM; exact
        # child fallbacks simply mark an already-exited leader as releasable.
        _signal_target(target, kill_signal, force=True)
        if id(target) in survivor_ids:
            logger.warning(
                "admin child %r did not exit after TERM; sent final kill",
                target.pid,
            )
    remaining = _wait_for_targets(targets, deadline)
    remaining_ids = {id(target) for target in remaining}
    for target in targets:
        if id(target) not in remaining_ids:
            _reap_target(target)
    return all(_target_cleanup_complete(target) for target in targets)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Module-level default. Tests can build their own JobStore without
# touching this; production code reaches it through get_default_store.
_default_store: JobStore | None = None


def get_default_store() -> JobStore:
    global _default_store
    if _default_store is None:
        _default_store = JobStore()
    return _default_store


def reset_default_store() -> None:
    """Test hook: drop the singleton so next get_default_store rebuilds it."""
    global _default_store
    _default_store = None
