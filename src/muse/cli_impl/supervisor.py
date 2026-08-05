"""`muse serve` supervisor: orchestrate workers + run gateway.

Responsibilities (v0.40.0+, lazy load):
  1. Read catalog at boot (only for validation, not for eager spawning).
  2. Construct a LoadDirector and hang it off SupervisorState.
  3. Stamp `unservable_reasons` for enabled catalog rows that lack memory
     data or whose declared memory_gb exceeds device capacity at boot.
  4. Start gateway immediately (zero workers initially). First request
     to a model triggers `LoadDirector.acquire`, which calls back into
     this module's `load_model_into_worker` to spawn the worker.
  5. On shutdown: SIGTERM workers (whatever was loaded by then), wait for
     exit.

A module-level SupervisorState singleton holds the worker list and shared
metadata. Admin endpoints (muse.admin.*) read and mutate the state under
its RLock; the auto-restart monitor reads `state.workers` directly. The
state is registered by `run_supervisor` and cleared on its way out.
"""
from __future__ import annotations

import logging
import math
import os
import re
import secrets
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from muse.cli_impl.serve_util import run_uvicorn

from muse.cli_impl.idle_sweeper import IdleSweeper
from muse.core import config
from muse.core.catalog import CatalogError, _read_catalog, get_manifest

from muse.core.memory_probe import (
    available_capacity_gb,
    declared_device,
    resolve_memory_pool,
)
from muse.core.resource_registry import (
    ResourceRegistryError,
    register_process,
    unregister_process,
)

from muse.observability.store import TelemetryStore
from muse.observability.recorder import init_recorder, reset_recorder
from muse.observability.sampler import Sampler, VramTracker
from muse.observability.logs import LogHub

logger = logging.getLogger(__name__)

_SUPERVISOR_PID_ENV = "MUSE_SUPERVISOR_PID"
_WORKER_NONCE_ENV = "MUSE_WORKER_NONCE"
_WORKER_NONCE_HEADER = "X-Muse-Worker-Nonce"
_WORKER_LOG_READ_BYTES = 16 * 1024
_WORKER_LOG_LINE_BYTES = 64 * 1024
_WORKER_LOG_TRUNCATION = b"...[truncated]\n"
_OS_NAME = os.name
_REAL_POPEN_TYPE = subprocess.Popen


@dataclass
class WorkerSpec:
    """Everything needed to spawn and supervise one worker subprocess.

    Fields mutated by the monitor thread (after startup):
      - process: replaced on restart
      - restart_count: consecutive unsuccessful restart attempts
        (caps at _MAX_RESTARTS); a successful ready replacement resets it
        (see _attempt_restart)
      - failure_count: consecutive unhealthy polls
      - last_spawn_at: time.monotonic() of most recent spawn (for backoff)
      - status: pending -> running -> unhealthy -> dead
    """
    models: list[str]
    python_path: str
    port: int
    device: str = "auto"
    process: object = field(default=None)
    restart_count: int = 0
    failure_count: int = 0
    last_spawn_at: float = 0.0
    status: str = "pending"
    # Opaque token for the generation that owns an in-flight transition.
    # The waitable ownership record lives in SupervisorState; this marker
    # lets routing/monitor code cheaply exclude the spec during slow I/O.
    job_id: str | None = None
    # Opaque persistent ownership record used by `muse doctor resources`.
    # Normal shutdown still uses the in-memory Popen handle.
    resource_id: str | None = None
    # Older resource identities whose unregister step failed during an atomic
    # spawn replacement. Kept explicitly until bounded cleanup retries them.
    retained_resource_ids: list[str] = field(default_factory=list)
    # Exact POSIX process-group identity captured immediately after spawn.
    # The leader remains unreaped until a final group KILL has been delivered,
    # pinning this numeric identity against PID/PGID reuse.
    process_group_id: int | None = None
    process_group_released: bool = False
    # Per-generation readiness identity. A different service already bound to
    # the numeric port cannot satisfy this worker's health check.
    worker_nonce: str | None = None
    # Reader thread paired with exactly ``process`` when worker stdout is
    # piped into the observability LogHub. Kept so teardown can join it after
    # that exact child reaches EOF instead of abandoning a daemon thread.
    log_thread: threading.Thread | None = None
    # Serializes every observation, wait/reap, signal, replacement, and
    # cleanup of the exact Popen generation. Without this lock, the monitor
    # can reap a leader between shutdown's identity check and killpg(), making
    # the stored numeric PGID eligible for unrelated-process reuse.
    process_lock: Any = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class WorkerShutdownResult:
    """Exact ownership outcome from one bounded bulk-worker teardown."""

    released: tuple[WorkerSpec, ...]
    retained: tuple[WorkerSpec, ...]

    @property
    def complete(self) -> bool:
        return not self.retained

    def retained_spec(self, spec: WorkerSpec) -> bool:
        return any(candidate is spec for candidate in self.retained)


@dataclass
class _WorkerShutdownTarget:
    spec: WorkerSpec
    process: Any
    pid: int | None
    process_group_id: int | None
    resource_id: str | None
    log_thread: threading.Thread | None
    retained_resource_ids: tuple[str, ...] = ()
    final_signal_delivered: bool = False
    leader_reaped: bool = False


@dataclass(frozen=True)
class WorkerOperation:
    """One exclusive worker transition for a shared Python environment.

    Workers that share ``python_path`` are restarted as a unit, so model
    list mutations for that environment must have exactly one owner.  The
    monotonically increasing generation makes ownership unambiguous even
    when an owner label (for example, an admin job id) is reused in tests.
    Waiters block on ``done`` outside ``SupervisorState.lock`` and then
    re-read catalog and worker state; they never modify an in-flight plan.
    """

    python_path: str
    generation: int
    owner: str
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def token(self) -> str:
        """Opaque marker stored on WorkerSpec for monitor exclusion."""
        return f"{self.owner}@{self.generation}"


def _new_gate():
    """Default factory for SupervisorState.concurrency_gate.

    Lazy import keeps the dataclass default construction free of any module
    the field type names (queueing.py imports only stdlib + config, so this
    is cheap; the lazy form simply keeps the import off supervisor's top).
    """
    from muse.cli_impl.queueing import ConcurrencyGate
    return ConcurrencyGate()


def _new_notifier():
    """Default factory for SupervisorState.capacity_notifier."""
    from muse.cli_impl.queueing import CapacityNotifier
    return CapacityNotifier()


@dataclass
class SupervisorState:
    """Runtime state shared across the supervisor and admin endpoints.

    `workers` is the live list of spawned WorkerSpec records. Admin
    operations (enable/disable) mutate it; the monitor thread reads it.
    Under lazy load, the LoadDirector also adds and removes WorkerSpec
    records via `load_model_into_worker` and `unload_model_from_worker`
    in `muse.admin.operations`.

    `device` is the supervisor-wide device flag (cuda/cpu/auto/mps).
    Admin-spawned workers inherit it unless their MANIFEST capability
    pins a specific device.

    `started_at` is monotonic seconds at supervisor boot; admin uptime
    queries can subtract this to report worker uptimes.

    `director` is the LoadDirector singleton. Populated by
    `run_supervisor` after construction; admin endpoints reach the
    director through `state.director`. None outside of a running
    supervisor (tests building bare states get a coherent default).

    `unservable_reasons` is a per-model-id map populated at boot by
    `validate_catalog_at_boot`. Maps model_id to a string explaining
    why the model cannot be served (no memory data, exceeds device
    capacity, etc). The gateway short-circuits 503 for these models
    before calling `director.acquire`; `/v1/models` surfaces the
    reason to clients.

    `lock` is a reentrant lock guarding all mutations of `workers`,
    `unservable_reasons`, and the worker-operation registry. Slow worker I/O
    runs outside it while per-venv operation Events provide serialization.

    `stop_event` is the supervisor-wide shutdown signal. Set as soon as
    Uvicorn receives SIGINT / SIGTERM (and again in `run_supervisor`'s
    cleanup); consumed by readiness waits, the auto-restart monitor, and
    the idle sweeper so a single Ctrl+C unblocks every supervisor-owned
    thread at once.
    A bare default state (e.g. one returned by `get_supervisor_state`
    when nothing is registered) gets a fresh Event so admin or test
    code that touches `state.stop_event` doesn't crash on None.

    `idle_sweeper` and `idle_sweeper_thread` hold the v0.40.1 idle-
    timeout sweeper after `run_supervisor` boots it. Exposed on the
    state so tests can introspect the sweeper and so future admin
    endpoints can read its tick metadata without a module-level
    singleton lookup.
    """
    workers: list[WorkerSpec] = field(default_factory=list)
    device: str = "auto"
    started_at: float = field(default_factory=time.monotonic)
    director: "Any | None" = None
    unservable_reasons: dict[str, str] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    monitor_thread: "threading.Thread | None" = None
    idle_sweeper: "IdleSweeper | None" = None
    idle_sweeper_thread: "threading.Thread | None" = None
    # Observability (Task 11). Both are None unless `telemetry.enabled` is
    # true, in which case `_init_telemetry` populates them during
    # `run_supervisor` boot, before the gateway is built, so the mounted
    # dashboard router can read them. `telemetry_store` is the sqlite-backed
    # TelemetryStore; `log_hub` is the per-model ring-buffer log fan-out that
    # `spawn_worker` pipes each worker's stdout into.
    telemetry_store: "Any | None" = None
    log_hub: "Any | None" = None
    telemetry_recorder: "Any | None" = None
    telemetry_sampler: "Any | None" = None
    telemetry_vram_tracker: "Any | None" = None
    telemetry_prune_thread: "threading.Thread | None" = None
    # Persistent identity of this exact supervisor process. Retained along
    # with the singleton when any subordinate cleanup fails, so a later
    # repair/retry path never loses the ownership record.
    supervisor_resource_id: str | None = None
    # #319 same-model cold-load coalescing (v0.51.0). model_id -> asyncio.Future
    # gate. The FIRST request for a cold model becomes the loader (dispatches
    # one off-loop director.acquire); concurrent requests for the SAME model
    # await the gate on the event loop (no thread), so only one thread parks
    # per model-load instead of N-1. Touched ONLY from the gateway's single
    # event loop, so a plain dict is safe (the loader election is await-free).
    cold_load_gates: dict = field(default_factory=dict)
    # Request queueing (spec 2026-07-08). `concurrency_gate` enforces the
    # per-model concurrency cap with FIFO waiters kept ON the event loop;
    # `capacity_notifier` broadcasts "capacity may have freed" from the
    # LoadDirector's release/eviction paths so gateway capacity-waiters wake
    # and re-decide. Both are default-constructed so a bare test state gets
    # real, usable primitives; run_supervisor wires
    # director.capacity_listener -> capacity_notifier.notify.
    concurrency_gate: "ConcurrencyGate" = field(default_factory=_new_gate)
    capacity_notifier: "CapacityNotifier" = field(default_factory=_new_notifier)
    # Exclusive worker transitions keyed by per-model venv Python path.
    # Admin enable/disable, director load/unload, removal, and monitor
    # restart all share this registry.  Entries are waitable and removed by
    # their owner in a finally block; generations remain so stale completion
    # cannot be mistaken for a later transition with the same owner label.
    worker_operations: dict[str, WorkerOperation] = field(default_factory=dict)
    worker_operation_generations: dict[str, int] = field(default_factory=dict)


def claim_worker_operation(
    state: SupervisorState, *, python_path: str, owner: str,
) -> tuple[WorkerOperation, bool]:
    """Return the current operation or atomically create a new generation.

    The boolean is true only for the caller that owns the new record.  This
    function acquires ``state.lock`` itself; callers already holding the
    state's reentrant lock may use it without opening a check/claim window.
    """
    with state.lock:
        current = state.worker_operations.get(python_path)
        if current is not None:
            return current, False
        generation = state.worker_operation_generations.get(python_path, 0) + 1
        state.worker_operation_generations[python_path] = generation
        operation = WorkerOperation(
            python_path=python_path,
            generation=generation,
            owner=owner,
        )
        state.worker_operations[python_path] = operation
        return operation, True


def finish_worker_operation(
    state: SupervisorState, operation: WorkerOperation,
) -> None:
    """Release exactly ``operation`` and wake every waiter.

    Identity, rather than owner text or generation alone, prevents a stale
    owner's cleanup from deleting a later operation record.
    """
    with state.lock:
        if state.worker_operations.get(operation.python_path) is operation:
            state.worker_operations.pop(operation.python_path, None)
        operation.done.set()


# Module-level singleton; admin routes reach this through
# get_supervisor_state. Tests build their own SupervisorState instances
# and either set it via set_supervisor_state or pass it directly.
_state: "SupervisorState | None" = None


def get_supervisor_state() -> SupervisorState:
    """Return the active SupervisorState, or an empty sentinel.

    The sentinel is fresh on every call when nothing is set; this means
    admin endpoints loaded outside a running supervisor (e.g. unit tests
    spinning up the gateway in isolation) get a coherent empty state
    instead of a None that crashes the routes.
    """
    return _state if _state is not None else SupervisorState()


def set_supervisor_state(state: SupervisorState) -> None:
    """Register a SupervisorState as the active singleton."""
    global _state
    _state = state


def clear_supervisor_state() -> None:
    """Test hook + supervisor shutdown: drop the active singleton."""
    global _state
    _state = None


def _concrete_worker_id(value: Any) -> int | None:
    """Return a signal-safe concrete PID/PGID, rejecting mocks and PID 1."""
    if type(value) is not int or value <= 1:
        return None
    return value


def _validated_worker_process_group(proc: Any) -> int | None:
    """Capture the child's isolated POSIX group immediately after spawn."""
    if _OS_NAME != "posix":
        return None
    pid = _concrete_worker_id(getattr(proc, "pid", None))
    if pid is None:
        return None
    try:
        process_group = _concrete_worker_id(os.getpgid(pid))
        own_group = _concrete_worker_id(os.getpgrp())
    except OSError as exc:
        logger.warning("could not validate worker process group for %s: %s", pid, exc)
        return None
    if process_group != pid or own_group is None or process_group == own_group:
        logger.error(
            "refusing worker process-group ownership pid=%r pgid=%r own_pgid=%r",
            pid, process_group, own_group,
        )
        return None
    return process_group


def _publish_worker_log(hub: "Any", model_id: str, raw: bytes) -> None:
    """Decode and publish one already-bounded worker log record."""
    line = raw.decode("utf-8", errors="replace")
    hub.append(model_id, line)
    print(f"[{model_id}] {line}", end="", flush=True)


def _pump_worker_logs(proc: "Any", model_id: str, hub: "Any") -> None:
    """Drain one worker pipe without ever retaining an unbounded line.

    The reader thread exclusively owns ``proc.stdout`` including its close.
    Shutdown never closes a buffered/text wrapper from another thread (which
    can block on the reader's internal lock). Fixed-size binary reads plus a
    fixed-size partial-line buffer bound memory before LogHub sees the text.
    """
    stream = getattr(proc, "stdout", None)
    if stream is None:
        return
    pending = bytearray()
    discarding_long_line = False
    try:
        while True:
            chunk = stream.read(_WORKER_LOG_READ_BYTES)
            if not chunk:
                break
            data = (
                chunk.encode("utf-8", errors="replace")
                if isinstance(chunk, str)
                else bytes(chunk)
            )
            cursor = 0
            while cursor < len(data):
                if discarding_long_line:
                    newline = data.find(b"\n", cursor)
                    if newline < 0:
                        break
                    cursor = newline + 1
                    discarding_long_line = False
                    continue

                newline = data.find(b"\n", cursor)
                end = newline + 1 if newline >= 0 else len(data)
                segment = data[cursor:end]
                if len(pending) + len(segment) <= _WORKER_LOG_LINE_BYTES:
                    pending.extend(segment)
                else:
                    prefix_limit = max(
                        0,
                        _WORKER_LOG_LINE_BYTES - len(_WORKER_LOG_TRUNCATION),
                    )
                    if len(pending) > prefix_limit:
                        del pending[prefix_limit:]
                    take = max(0, prefix_limit - len(pending))
                    pending.extend(segment[:take])
                    pending.extend(
                        _WORKER_LOG_TRUNCATION[
                            : _WORKER_LOG_LINE_BYTES - len(pending)
                        ]
                    )
                    _publish_worker_log(hub, model_id, bytes(pending))
                    pending.clear()
                    if newline < 0:
                        discarding_long_line = True

                cursor = end
                if newline >= 0 and not discarding_long_line:
                    if pending:
                        _publish_worker_log(hub, model_id, bytes(pending))
                        pending.clear()
            # A discarded overlong record remains discarded until a later
            # fixed-size chunk contains its newline.
        if pending and not discarding_long_line:
            _publish_worker_log(hub, model_id, bytes(pending))
    except Exception:
        logger.warning("log pump for %r stopped", model_id, exc_info=True)
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            logger.debug("could not close log pipe for %r", model_id, exc_info=True)


def spawn_worker(spec: WorkerSpec, *, device: str, log_hub: "Any | None" = None) -> None:
    """Start a worker subprocess using its venv's Python.

    Persists `device` onto the spec so the monitor thread can respawn
    with the same settings on restart. Records last_spawn_at for the
    backoff timer in _attempt_restart.

    When `log_hub` is given (telemetry enabled), the worker's stdout is
    piped and a daemon thread pumps each line into the hub via
    `_pump_worker_logs`, keyed by the worker's first model id (lazy-load
    spawns one model per worker, so this is the common case; a
    multi-model worker's logs are attributed to `spec.models[0]` only).
    Every worker receives the supervisor PID for its parent-death watchdog.
    On POSIX it also gets a new session, keeping terminal signals directed at
    the supervisor; the supervisor remains responsible for orderly worker
    teardown and can signal the whole worker process group if needed.
    """
    with spec.process_lock:
        if spec.process is not None:
            raise RuntimeError(
                f"worker on port {spec.port} still owns a prior process generation"
            )
        if spec.log_thread is not None:
            try:
                spec.log_thread.join(timeout=0)
            except RuntimeError:
                pass
            if spec.log_thread.is_alive() is True:
                raise RuntimeError(
                    f"worker on port {spec.port} still owns a prior log reader"
                )
            spec.log_thread = None
        if spec.retained_resource_ids:
            raise RuntimeError(
                f"worker on port {spec.port} still owns superseded "
                "resource records"
            )

        spec.device = device
        cmd = [
            spec.python_path, "-m", "muse.cli", "_worker",
            "--host", "127.0.0.1",
            "--port", str(spec.port),
            "--device", device,
        ]
        for m in spec.models:
            cmd.extend(["--model", m])
        logger.info("spawning worker: %s", " ".join(cmd))
        child_env = os.environ.copy()
        child_env[_SUPERVISOR_PID_ENV] = str(os.getpid())
        worker_nonce = secrets.token_urlsafe(32)
        child_env[_WORKER_NONCE_ENV] = worker_nonce
        popen_kwargs: dict[str, Any] = {
            "env": child_env,
            "start_new_session": os.name == "posix",
        }
        if log_hub is not None:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=0, **popen_kwargs,
            )
            spec.process = proc
            spec.worker_nonce = worker_nonce
            spec.process_group_released = False
            spec.process_group_id = _validated_worker_process_group(proc)
            model_id = spec.models[0] if spec.models else "worker"
            log_thread = threading.Thread(
                target=_pump_worker_logs, args=(proc, model_id, log_hub),
                daemon=True, name=f"muse-logpump-{spec.port}",
            )
            spec.log_thread = log_thread
            try:
                log_thread.start()
            except BaseException:
                # The RLock permits the exact cleanup helper to reuse the same
                # identity boundary without exposing a half-started child.
                _shutdown_workers([spec])
                raise
        else:
            proc = subprocess.Popen(cmd, **popen_kwargs)
            spec.process = proc
            spec.worker_nonce = worker_nonce
            spec.process_group_released = False
            spec.process_group_id = _validated_worker_process_group(proc)
            spec.log_thread = None
        previous_resource_id = spec.resource_id
        try:
            next_resource_id = register_process(
                kind="worker",
                pid=int(proc.pid),
                owner_pid=os.getpid(),
                port=spec.port,
                models=spec.models,
            )
        except (ResourceRegistryError, TypeError, ValueError) as exc:
            # Never return a child that post-crash repair cannot identify.
            # Detach the previous generation's record before reusing the
            # exact-Popen shutdown helper so rollback cannot unregister it.
            spec.resource_id = None
            _shutdown_workers([spec])
            # The old persistent record remains owned regardless of whether
            # the unregistered new generation exited. Reattach it so a
            # retained new Popen never causes the prior registry identity to
            # disappear from in-memory cleanup state.
            if spec.resource_id is None:
                spec.resource_id = previous_resource_id
            if spec.process is None:
                detail = "the new process was rolled back"
            else:
                detail = "the new process remains retained for supervisor cleanup"
            raise ResourceRegistryError(
                f"worker on port {spec.port} could not be persisted; "
                f"{detail}"
            ) from exc
        spec.resource_id = next_resource_id
        if (
            next_resource_id is not None
            and previous_resource_id is not None
            and next_resource_id != previous_resource_id
        ):
            try:
                unregister_process(previous_resource_id)
            except ResourceRegistryError:
                logger.warning("could not remove superseded worker resource record")
                if previous_resource_id not in spec.retained_resource_ids:
                    spec.retained_resource_ids.append(previous_resource_id)
        spec.last_spawn_at = time.monotonic()


class WorkerIdentityError(RuntimeError):
    """A listener on the assigned port is not the expected worker generation."""


def _supports_pinned_worker_leader(
    proc: Any, pid: int | None, process_group_id: int | None,
) -> bool:
    """Whether WNOWAIT can preserve this exact real POSIX group identity."""
    return bool(
        _OS_NAME == "posix"
        and pid is not None
        and process_group_id == pid
        and isinstance(proc, _REAL_POPEN_TYPE)
        and callable(getattr(os, "waitid", None))
        and getattr(os, "P_PID", None) is not None
        and getattr(os, "WEXITED", None) is not None
        and getattr(os, "WNOHANG", None) is not None
        and getattr(os, "WNOWAIT", None) is not None
    )


def _worker_process_state_locked(spec: WorkerSpec, proc: Any) -> bool | None:
    """Return True alive, False exited, or None when identity is ambiguous.

    Real isolated POSIX leaders are observed with WNOWAIT, never ``poll()``;
    retaining the zombie pins PID==PGID until teardown sends its final group
    signal. The caller holds ``spec.process_lock``.
    """
    if spec.process is not proc:
        return None
    pid = _concrete_worker_id(getattr(proc, "pid", None))
    if pid is None:
        return None
    if type(getattr(proc, "returncode", None)) is int:
        return False
    if _supports_pinned_worker_leader(proc, pid, spec.process_group_id):
        flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
        try:
            status = os.waitid(os.P_PID, pid, flags)
        except InterruptedError:
            return True
        except (ChildProcessError, OSError) as exc:
            logger.warning("could not observe worker leader %s safely: %s", pid, exc)
            return None
        return not (
            status is not None
            and getattr(status, "si_pid", pid) == pid
        )
    if _OS_NAME == "posix" and isinstance(proc, _REAL_POPEN_TYPE):
        # A real worker is spawned into a new session. If its PID==PGID could
        # not be validated and stored at spawn, polling could reap the leader
        # and make that unknown numeric group reusable before descendants are
        # drained. Retain ownership and fail closed instead.
        return None
    try:
        observed = proc.poll()
        if type(observed) is int and type(
            getattr(proc, "returncode", None)
        ) is not int:
            proc.returncode = observed
        return observed is None
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not inspect worker leader %s: %s", pid, exc)
        return None


def _require_worker_generation_alive(spec: WorkerSpec, proc: Any) -> None:
    with spec.process_lock:
        if spec.process is not proc:
            raise WorkerIdentityError(
                f"worker on port {spec.port} changed process generation during startup"
            )
        state = _worker_process_state_locked(spec, proc)
    if state is False:
        raise RuntimeError(
            f"worker process on port {spec.port} exited before readiness"
        )
    if state is not True:
        raise WorkerIdentityError(
            f"could not verify worker process identity on port {spec.port}"
        )


def wait_for_ready(
    *, port: int, timeout: float = 60.0, poll_interval: float = 0.5,
    stop_event: "threading.Event | None" = None,
    expected_nonce: str | None = None,
    worker: WorkerSpec | None = None,
) -> None:
    """Block until http://127.0.0.1:<port>/health returns 200, or timeout.

    Raises TimeoutError if the worker never becomes ready. If `stop_event`
    is supplied, shutdown aborts the wait promptly instead of leaving an
    executor thread parked for the full model-load timeout.
    """
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last_err: Exception | None = None
    expected_process = None
    if worker is not None:
        with worker.process_lock:
            expected_process = worker.process
        if expected_process is None:
            raise WorkerIdentityError(
                f"worker on port {port} has no process generation to verify"
            )
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("worker startup cancelled: supervisor shutdown requested")
        if worker is not None:
            _require_worker_generation_alive(worker, expected_process)
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                if worker is not None:
                    _require_worker_generation_alive(worker, expected_process)
                if expected_nonce is not None:
                    observed_nonce = r.headers.get(_WORKER_NONCE_HEADER)
                    if observed_nonce != expected_nonce:
                        raise WorkerIdentityError(
                            f"port {port} is occupied by a stale or unrelated "
                            "service (worker readiness nonce mismatch)"
                        )
                return
        except httpx.HTTPError as e:
            last_err = e
        if stop_event is not None:
            if stop_event.wait(poll_interval):
                raise RuntimeError(
                    "worker startup cancelled: supervisor shutdown requested"
                )
        else:
            time.sleep(poll_interval)
    raise TimeoutError(
        f"worker on port {port} did not become ready within {timeout}s "
        f"(last error: {last_err})"
    )


def check_worker_health(
    *, port: int, timeout: float = 2.0,
    expected_nonce: str | None = None,
) -> bool:
    """Single /health poll. Returns True iff the worker responds 200.

    Swallows all httpx errors; they indicate "unhealthy" for our purposes.
    Used by the monitor thread's periodic liveness check.
    """
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
        return bool(
            r.status_code == 200
            and (
                expected_nonce is None
                or r.headers.get(_WORKER_NONCE_HEADER) == expected_nonce
            )
        )
    except httpx.HTTPError:
        return False


# Monitor defaults (module constants; not CLI-configurable in this iteration)
_MONITOR_INTERVAL = 5.0
_FAILURE_THRESHOLD = 3
_MAX_RESTARTS = 10
_BACKOFF_CAP = 30.0  # seconds
_BACKOFF_BASE = 1.0


def _attempt_restart(
    spec: WorkerSpec,
    *,
    stop_event: "threading.Event",
    max_restarts: int = _MAX_RESTARTS,
    backoff_base: float = _BACKOFF_BASE,
    backoff_cap: float = _BACKOFF_CAP,
    ready_timeout: float = 60.0,
    log_hub: "Any | None" = None,
) -> None:
    """Terminate existing process if alive, wait backoff, respawn.

    Mutates spec.process, spec.restart_count, spec.failure_count, spec.status.
    Marks spec.status = "dead" if restart_count reaches max_restarts.
    Returns early if stop_event fires during backoff.

    restart_count counts consecutive unsuccessful restart attempts, matching
    the documented budget: the failure branch increments it, while a
    replacement that reaches readiness resets it to zero.

    `log_hub` is forwarded to `spawn_worker` so a respawned worker keeps
    piping its stdout into the LogHub when telemetry is enabled (mirrors
    the admin `_restart_worker_inplace` path in `muse.admin.operations`).
    """
    if spec.restart_count >= max_restarts:
        logger.error(
            "worker on port %d: exhausted %d restart attempts; marking dead",
            spec.port, max_restarts,
        )
        spec.status = "dead"
        return

    # Exponential backoff, capped
    backoff = min(backoff_base * (2 ** spec.restart_count), backoff_cap)
    logger.warning(
        "worker on port %d: restart attempt %d after %.1fs backoff",
        spec.port, spec.restart_count + 1, backoff,
    )
    # wait() returns True if event was set during the wait (skip restart)
    if stop_event.wait(backoff):
        return

    # Retire the exact previous generation before spawning its replacement.
    # This also reaps an already-exited child and drains its log reader.  A
    # teardown that cannot prove ownership/exited state deliberately retains
    # the handle, causing spawn_worker to fail closed rather than overwrite a
    # process identity that another thread could later signal by stale PID.
    with spec.process_lock:
        has_previous_generation = spec.process is not None
    if has_previous_generation:
        _shutdown_workers([spec], grace=3.0)

    # Respawn. restart_count bumps ONLY on failure below (see docstring):
    # a successful respawn must not count toward the unsuccessful-attempts
    # budget, else a worker that flaps and cleanly recovers many times
    # over its lifetime would eventually be marked dead despite never
    # having a run of consecutive failures.
    try:
        spawn_worker(spec, device=spec.device, log_hub=log_hub)
        wait_for_ready(
            port=spec.port,
            timeout=ready_timeout,
            stop_event=stop_event,
            expected_nonce=spec.worker_nonce,
            worker=spec,
        )
        spec.failure_count = 0
        spec.restart_count = 0
        spec.status = "running"
        logger.info("worker on port %d: successfully restarted", spec.port)
    except (subprocess.SubprocessError, TimeoutError, OSError, RuntimeError) as e:
        # OSError covers FileNotFoundError / PermissionError from Popen when
        # the venv python is missing or non-executable (e.g. the venv was
        # deleted, or its python symlink broke on a system upgrade). Without
        # catching it, the exception escapes _monitor_workers and kills the
        # monitor daemon thread, silently disabling health-monitoring and
        # auto-restart for ALL workers (M10).
        logger.error("worker on port %d: restart failed: %s", spec.port, e)
        _shutdown_workers([spec])
        spec.restart_count += 1
        spec.status = "unhealthy"


def _monitor_workers(
    specs: list[WorkerSpec],
    stop_event: "threading.Event",
    *,
    interval: float = _MONITOR_INTERVAL,
    failure_threshold: int = _FAILURE_THRESHOLD,
    max_restarts: int = _MAX_RESTARTS,
    state: "SupervisorState | None" = None,
) -> None:
    """Poll each worker; restart after `failure_threshold` consecutive failures.

    Exits when stop_event is set. Called from the monitor daemon thread
    started by run_supervisor (Task B4).

    Concurrency: `specs` is the live `state.workers` list shared with
    admin operations (enable/disable) that may call `state.workers.remove`
    under `state.lock` while the monitor is iterating. To avoid
    `RuntimeError: list changed size during iteration`, we snapshot the
    list at the top of each poll tick with `list(specs)`. The snapshot
    holds a reference to each WorkerSpec (not a copy), so in-place
    mutations to spec fields (status, failure_count, etc.) are visible
    to both the monitor and admin operations without extra coordination.
    A spec removed from `state.workers` during the tick may still be
    referenced by the snapshot. Immediately before restart, the monitor
    atomically rechecks identity membership and claims the same per-venv
    operation registry used by admin/director transitions, so it never
    restarts a removed spec or overlaps another owner.

    `state`, when given, is read for its `log_hub` attribute at EACH
    restart (not captured once at thread-start time), so a restart still
    forwards the live LogHub even though the monitor thread is started
    before `_init_telemetry` populates `state.log_hub` during supervisor
    boot. Optional (defaults to None) so existing callers that invoke
    this with just `(specs, stop_event)` keep today's behavior (no log
    piping on restart) unchanged.
    """
    while not stop_event.is_set():
        for spec in list(specs):  # snapshot: safe against concurrent remove()
            if stop_event.is_set():
                return
            if spec.status == "dead":
                continue

            # Skip specs in the middle of an admin- or director-driven
            # transition. job_id is set when an in-flight operation has
            # claimed the spec (enable_model, load_model_into_worker,
            # restart-in-place). The owning operation is responsible for
            # the spawn / readiness wait; the monitor must not race it
            # by polling /health (which fails until the worker binds the
            # port) and triggering a duplicate restart. The owning op
            # clears job_id on success or marks the spec dead on failure.
            if spec.job_id is not None:
                continue

            # Poll/reap the exact process generation under its identity lock.
            # Teardown and admin signaling use the same lock, so no thread can
            # retain a numeric PID/PGID after this poll makes it reusable.
            with spec.process_lock:
                observed_process = spec.process
                if observed_process is None:
                    process_exited = True
                    returncode: Any = None
                else:
                    process_state = _worker_process_state_locked(
                        spec, observed_process,
                    )
                    process_exited = process_state is not True
                    observed_returncode = getattr(
                        observed_process, "returncode", None,
                    )
                    returncode = (
                        observed_returncode
                        if type(observed_returncode) is int
                        else None
                    )

            if process_exited:
                logger.warning(
                    "worker on port %d: process exited with code %s",
                    spec.port, returncode,
                )
                with spec.process_lock:
                    if spec.process is not observed_process:
                        continue
                spec.failure_count = failure_threshold
            else:
                healthy = check_worker_health(
                    port=spec.port,
                    expected_nonce=spec.worker_nonce,
                )
                with spec.process_lock:
                    if spec.process is not observed_process:
                        continue
                if healthy:
                    spec.failure_count = 0
                    spec.restart_count = 0
                    spec.status = "running"
                    continue
                spec.failure_count += 1
                if spec.status == "running":
                    spec.status = "unhealthy"
                logger.info(
                    "worker on port %d: unhealthy (%d/%d consecutive failures)",
                    spec.port, spec.failure_count, failure_threshold,
                )

            if spec.failure_count >= failure_threshold:
                operation: WorkerOperation | None = None
                if state is not None:
                    # Re-check membership + ownership and claim the venv in
                    # one lock hold.  An admin operation may have started
                    # after the earlier job_id check; without this atomic
                    # claim the monitor could restart from a stale model
                    # list while that operation restarts or removes it.
                    with state.lock:
                        still_tracked = any(item is spec for item in specs)
                        if not still_tracked or spec.job_id is not None:
                            continue
                        operation, claimed = claim_worker_operation(
                            state,
                            python_path=spec.python_path,
                            owner=f"monitor-restart-{spec.port}",
                        )
                        if not claimed:
                            continue
                        spec.job_id = operation.token
                try:
                    _attempt_restart(
                        spec, stop_event=stop_event,
                        max_restarts=max_restarts,
                        log_hub=getattr(state, "log_hub", None),
                    )
                finally:
                    if operation is not None:
                        with state.lock:
                            if spec.job_id == operation.token:
                                spec.job_id = None
                        finish_worker_operation(state, operation)

        # Sleep with early-exit if stop_event fires
        if stop_event.wait(interval):
            return


def _signal_worker_process(
    proc: Any, sig: signal.Signals, *, force: bool = False,
) -> bool:
    """Signal one proven-live worker generation.

    The caller must hold the owning ``WorkerSpec.process_lock``.  The single
    initial poll is both the liveness check and the no-PID-reuse boundary: if
    the leader is alive it cannot release its PID until a later reap, and the
    shared lock prevents any other Muse thread from performing that reap
    before this function signals.  Ambiguous or exited processes fail closed.

    Returns whether a signal method was invoked.
    """
    try:
        returncode = proc.poll()
    except Exception:  # noqa: BLE001
        return False
    if returncode is not None:
        return False

    if os.name == "posix":
        try:
            pid = proc.pid
            # Never allow invalid/mock metadata to become killpg(1), which is
            # kill(-1) at the syscall layer and broadcasts to every process
            # the caller may signal. Also refuse our own process group: real
            # workers are spawned with start_new_session=True, so neither
            # guard excludes a legitimate managed worker.
            if type(pid) is not int or pid <= 1:
                raise ValueError(f"unsafe worker pid {pid}")
            pgid = os.getpgid(pid)
            if (
                type(pgid) is not int
                or pgid <= 1
                or pgid != pid
                or pgid == os.getpgrp()
            ):
                raise ValueError(f"unsafe worker process group {pgid}")
            os.killpg(pgid, sig)
            return True
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    if force:
        proc.kill()
    else:
        proc.terminate()
    return True


def _signal_worker_target(
    target: _WorkerShutdownTarget,
    sig: signal.Signals,
    *,
    force: bool,
) -> bool:
    """Signal one still-owned target without surrendering its group identity."""
    spec = target.spec
    proc = target.process
    with spec.process_lock:
        if spec.process is not proc:
            return False
        if spec.process_group_released:
            target.final_signal_delivered = True
            return True

        state = _worker_process_state_locked(spec, proc)
        pinned_group = _supports_pinned_worker_leader(
            proc, target.pid, target.process_group_id,
        )
        if pinned_group:
            if type(getattr(proc, "returncode", None)) is int:
                logger.error(
                    "refusing worker process-group signal after leader %r "
                    "was already reaped",
                    target.pid,
                )
                return False
            if state is None:
                return False
            # Once the leader is observable as a zombie, PID==PGID remains
            # pinned but the group can still contain live descendants. TERM
            # is unnecessary at that point, while the final KILL is mandatory
            # before the exact leader is reaped and the numeric PGID can be
            # reused by the OS.
            if not force and state is not True:
                return state is False
            try:
                os.killpg(target.process_group_id, sig)
            except ProcessLookupError:
                # With the unreaped PID==PGID leader still pinning identity,
                # ESRCH proves there are no signalable members left.
                if force:
                    target.final_signal_delivered = True
                return True
            except (OSError, TypeError, ValueError) as exc:
                logger.warning(
                    "could not signal worker process group %s: %s",
                    target.process_group_id,
                    exc,
                )
                return False
            if force:
                target.final_signal_delivered = True
            return True

        if state is False:
            # Exact-child fallback has no separately owned group to drain.
            if force:
                target.final_signal_delivered = True
            return True
        if state is not True:
            return False
        try:
            if force:
                proc.kill()
                target.final_signal_delivered = True
            else:
                proc.terminate()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "could not signal exact worker process %s: %s", target.pid, exc,
            )
            return False
        return True


def _wait_for_worker_targets(
    targets: list[_WorkerShutdownTarget], *, deadline: float,
) -> tuple[list[_WorkerShutdownTarget], list[_WorkerShutdownTarget]]:
    """Observe exact targets within one shared deadline without early reap."""
    exited: list[_WorkerShutdownTarget] = []
    pending = list(targets)
    while pending:
        next_pending: list[_WorkerShutdownTarget] = []
        for target in pending:
            spec = target.spec
            with spec.process_lock:
                if spec.process is not target.process:
                    next_pending.append(target)
                    continue
                state = _worker_process_state_locked(spec, target.process)
            if state is False:
                exited.append(target)
            else:
                next_pending.append(target)
        if not next_pending or time.monotonic() >= deadline:
            return exited, next_pending
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        pending = next_pending
    return exited, []


def _reap_worker_target(target: _WorkerShutdownTarget) -> bool:
    """Reap an exact leader only after its owned group was fully released."""
    spec = target.spec
    proc = target.process
    with spec.process_lock:
        if spec.process is not proc:
            return False
        if type(getattr(proc, "returncode", None)) is int:
            target.leader_reaped = True
            if target.final_signal_delivered:
                spec.process_group_released = True
            return bool(spec.process_group_released)
        if not target.final_signal_delivered:
            return False
        state = _worker_process_state_locked(spec, proc)
        if state is not False:
            return False
        try:
            returncode = proc.wait(timeout=0.0)
        except (subprocess.TimeoutExpired, ChildProcessError, OSError) as exc:
            logger.warning(
                "could not reap worker leader %s safely: %s", target.pid, exc,
            )
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "could not reap worker leader %s: %s", target.pid, exc,
            )
            return False
        if type(returncode) is not int:
            return False
        if type(getattr(proc, "returncode", None)) is not int:
            proc.returncode = returncode
        target.leader_reaped = True
        spec.process_group_released = True
        return True


def _shutdown_workers(
    specs: list[WorkerSpec], grace: float = 5.0,
) -> WorkerShutdownResult:
    """Boundedly TERM/KILL exact workers and report retained ownership.

    TERM, KILL, and log-reader cleanup each use one shared phase deadline,
    so shutdown latency is bounded independently of worker count. Every
    process observation, signal, reap, and handle mutation remains serialized
    by the per-spec identity lock. A process or reader whose exit cannot be
    established stays attached to its ``WorkerSpec`` and is reported in
    ``retained`` rather than being silently forgotten.
    """
    if (
        isinstance(grace, bool)
        or not isinstance(grace, (int, float))
        or not math.isfinite(float(grace))
        or grace < 0
    ):
        raise ValueError("worker shutdown grace must be a finite non-negative number")

    requested: list[WorkerSpec] = []
    seen_specs: set[int] = set()
    targets: list[_WorkerShutdownTarget] = []
    for spec in list(specs):
        identity = id(spec)
        if identity in seen_specs:
            continue
        seen_specs.add(identity)
        requested.append(spec)
        with spec.process_lock:
            if spec.process is not None:
                proc = spec.process
                targets.append(_WorkerShutdownTarget(
                    spec=spec,
                    process=proc,
                    pid=_concrete_worker_id(getattr(proc, "pid", None)),
                    process_group_id=spec.process_group_id,
                    resource_id=spec.resource_id,
                    log_thread=spec.log_thread,
                    retained_resource_ids=tuple(spec.retained_resource_ids),
                    final_signal_delivered=spec.process_group_released,
                    leader_reaped=(
                        type(getattr(proc, "returncode", None)) is int
                    ),
                ))

    # Phase 1: every target receives TERM before any target consumes grace.
    for target in targets:
        try:
            _signal_worker_target(target, signal.SIGTERM, force=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "failed to SIGTERM worker on port %d: %s",
                target.spec.port,
                exc,
            )

    term_deadline = time.monotonic() + float(grace)
    _, survivors = _wait_for_worker_targets(
        targets, deadline=term_deadline,
    )

    # Phase 2: every pinned POSIX group receives one final KILL before its
    # leader is reaped, including groups whose leader already honored TERM.
    # Exact-child fallbacks receive KILL only while still alive.
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    survivor_ids = {id(target) for target in survivors}
    for target in targets:
        if id(target) in survivor_ids:
            logger.warning(
                "worker on port %d did not exit in %.1fs; killing",
                target.spec.port,
                grace,
            )
        try:
            _signal_worker_target(target, kill_signal, force=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "failed to kill worker on port %d: %s",
                target.spec.port,
                exc,
            )

    kill_grace = max(1.0, min(float(grace), 2.0))
    kill_deadline = time.monotonic() + kill_grace
    exited, survivors = _wait_for_worker_targets(
        targets, deadline=kill_deadline,
    )
    exited_ids = {id(target) for target in exited}
    for target in targets:
        if id(target) in exited_ids:
            _reap_worker_target(target)
    for target in survivors:
        logger.warning(
            "worker on port %d remained alive after bounded teardown",
            target.spec.port,
        )

    # The pump thread is the sole owner allowed to read and close stdout.
    # Once the full group is gone, pipe EOF lets every reader settle in
    # parallel without a potentially blocking cross-thread TextIO close.
    reader_deadline = time.monotonic() + _BACKGROUND_JOIN_TIMEOUT_SECONDS
    reader_stopped: dict[int, bool] = {}
    for target in targets:
        log_thread = target.log_thread
        stopped = log_thread is None
        if log_thread is not None:
            try:
                log_thread.join(
                    timeout=max(0.0, reader_deadline - time.monotonic()),
                )
            except RuntimeError:
                # A thread whose start failed is already inert.
                pass
            except Exception:  # noqa: BLE001
                logger.warning(
                    "could not join log pump for worker on port %d",
                    target.spec.port,
                    exc_info=True,
                )
            try:
                stopped = log_thread.is_alive() is not True
            except RuntimeError:
                stopped = True
            if not stopped:
                logger.warning(
                    "log pump for worker on port %d did not stop",
                    target.spec.port,
                )
        reader_stopped[id(target)] = stopped

    # Finalize only exact generations whose group, leader, reader, and
    # persistent identity all settled. Any failed step deliberately keeps
    # every remaining handle attached for a later retry/repair path.
    for target in targets:
        spec = target.spec
        with spec.process_lock:
            if (
                spec.process is not target.process
                or not target.leader_reaped
                or not spec.process_group_released
                or not reader_stopped[id(target)]
                or spec.resource_id != target.resource_id
                or tuple(spec.retained_resource_ids) != target.retained_resource_ids
            ):
                continue
            registry_released = True
            registry_ids = tuple(dict.fromkeys((
                *((target.resource_id,) if target.resource_id is not None else ()),
                *target.retained_resource_ids,
            )))
            for resource_id in registry_ids:
                try:
                    unregister_process(resource_id)
                except ResourceRegistryError as exc:
                    registry_released = False
                    logger.warning(
                        "could not unregister worker on port %d: %s",
                        spec.port,
                        exc,
                    )
                else:
                    if spec.resource_id == resource_id:
                        spec.resource_id = None
                    try:
                        spec.retained_resource_ids.remove(resource_id)
                    except ValueError:
                        pass
            if not registry_released:
                continue
            spec.resource_id = None
            spec.retained_resource_ids.clear()
            if spec.log_thread is target.log_thread:
                spec.log_thread = None
            spec.process = None
            spec.process_group_id = None
            spec.process_group_released = False
            spec.worker_nonce = None

    # A previous partial cleanup can leave only an inert reader or registry
    # record. Retry those exact non-process owners too.
    orphan_deadline = time.monotonic() + _BACKGROUND_JOIN_TIMEOUT_SECONDS
    for spec in requested:
        with spec.process_lock:
            if spec.process is not None:
                continue
            log_thread = spec.log_thread
            resource_id = spec.resource_id
            retained_resource_ids = tuple(spec.retained_resource_ids)
        reader_done = log_thread is None
        if log_thread is not None:
            try:
                log_thread.join(
                    timeout=max(0.0, orphan_deadline - time.monotonic()),
                )
                reader_done = log_thread.is_alive() is not True
            except RuntimeError:
                reader_done = True
            except Exception:  # noqa: BLE001
                reader_done = False
        if not reader_done:
            continue
        registry_done = True
        released_registry_ids: list[str] = []
        registry_ids = tuple(dict.fromkeys((
            *((resource_id,) if resource_id is not None else ()),
            *retained_resource_ids,
        )))
        for owned_resource_id in registry_ids:
            try:
                unregister_process(owned_resource_id)
            except ResourceRegistryError as exc:
                registry_done = False
                logger.warning(
                    "could not unregister worker on port %d: %s", spec.port, exc,
                )
            else:
                released_registry_ids.append(owned_resource_id)
        with spec.process_lock:
            if spec.process is None:
                if spec.log_thread is log_thread:
                    spec.log_thread = None
                if (
                    spec.resource_id == resource_id
                    and resource_id in released_registry_ids
                ):
                    spec.resource_id = None
                for released_resource_id in released_registry_ids:
                    try:
                        spec.retained_resource_ids.remove(released_resource_id)
                    except ValueError:
                        pass
                if not registry_done:
                    continue
                spec.process_group_id = None
                spec.process_group_released = False
                spec.worker_nonce = None

    released: list[WorkerSpec] = []
    retained: list[WorkerSpec] = []
    for spec in requested:
        with spec.process_lock:
            destination = (
                released
                if (
                    spec.process is None
                    and spec.log_thread is None
                    and spec.resource_id is None
                    and not spec.retained_resource_ids
                )
                else retained
            )
            destination.append(spec)
    return WorkerShutdownResult(tuple(released), tuple(retained))


class _MemoryProbeAdapter:
    """Thin adapter wrapping `muse.core.memory_probe` module functions
    as bound methods on an object.

    LoadDirector accepts an object with live GPU/CPU memory methods plus an
    optional CUDA-runtime detector. The memory_probe module exposes these as
    free functions; this adapter satisfies the duck-typed composition seam.
    """

    def gpu_free_gb(self, device_id: int = 0) -> float | None:
        from muse.core import memory_probe
        return memory_probe.gpu_free_gb(device_id)

    def cpu_free_gb(self) -> float:
        from muse.core import memory_probe
        return memory_probe.cpu_free_gb()

    def gpu_total_gb(self, device_id: int = 0) -> float | None:
        from muse.core import memory_probe
        return memory_probe.gpu_total_gb(device_id)

    def cpu_total_gb(self) -> float:
        from muse.core import memory_probe
        return memory_probe.cpu_total_gb()

    def cuda_available(self) -> bool:
        from muse.core import memory_probe
        return memory_probe.cuda_runtime_available()


def _cuda_checker_for_probe(memory_probe: "Any") -> "Callable[[], bool] | None":
    """Return an explicitly implemented CUDA checker, never a loose mock child."""
    descriptor = getattr(type(memory_probe), "cuda_available", None)
    if not callable(descriptor):
        return None
    checker = getattr(memory_probe, "cuda_available", None)
    return checker if callable(checker) else None


def _optional_probe_gb(memory_probe: "Any", method_name: str) -> float | None:
    """Read an optional numeric probe method without trusting loose mocks.

    Older injected probes implement only ``*_free_gb``.  Looking up a
    missing method on ``MagicMock`` fabricates another mock, so descriptor
    inspection is required before calling it.  Invalid or failed optional
    readings fall back to the established free-memory behavior.
    """
    descriptor = getattr(type(memory_probe), method_name, None)
    if not callable(descriptor):
        return None
    method = getattr(memory_probe, method_name, None)
    if not callable(method):
        return None
    try:
        value = method()
        if isinstance(value, bool) or value is None:
            return None
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            return None
        return result
    except Exception as exc:  # noqa: BLE001
        logger.debug("optional memory probe %s failed: %s", method_name, exc)
        return None


def _weights_size_gb(catalog_entry: dict) -> float:
    """On-disk size of a model's downloaded weights, in GB (0.0 if unknown).

    Last resort in the sizing ladder: a model that was never probed and
    declares no `memory_gb` can still be sized from the bytes already on
    disk, so it loads on demand (evicting LRU as needed) instead of being
    503'd "no memory estimate". Sums recognized weight files under the
    entry's `local_dir` (an HF snapshot dir whose weight files are symlinks
    into the blob store; `os.path.getsize` follows them). Returns 0.0 when
    `local_dir` is absent, missing, or unreadable.

    This UNDERestimates live runtime (no activations / KV cache); the
    LoadDirector's observed-peak writeback self-heals the estimate upward
    after the first real load, and the auto-restart monitor recovers a
    worker that an initial under-estimate happens to OOM.

    GGUF exception: a GGUF snapshot dir routinely holds several quant
    variants of one model (q3/q4/q5/q8/f16), but only the declared
    `capabilities.gguf_file` actually loads. Summing the whole tree would
    OVERestimate wildly (a 4B q4 whose repo ships six quants sums to ~15 GB
    vs its ~2.6 GB weight), and overestimation is the dangerous direction:
    it 503s a servable model as "exceeds device capacity". So when a
    specific `gguf_file` is declared, size from that one file, falling back
    to the tree walk only when it is absent on disk (stale path).
    """
    local_dir = catalog_entry.get("local_dir")
    if not isinstance(local_dir, (str, os.PathLike)) or not local_dir:
        return 0.0
    manifest = catalog_entry.get("manifest") or {}
    if not isinstance(manifest, dict):
        manifest = {}
    capabilities = manifest.get("capabilities") or {}
    if not isinstance(capabilities, dict):
        capabilities = {}
    gguf_file = capabilities.get("gguf_file")
    if isinstance(gguf_file, (str, os.PathLike)) and gguf_file:
        try:
            return os.path.getsize(os.path.join(local_dir, gguf_file)) / (1024 ** 3)
        except OSError:
            # Declared file missing/unreadable: fall through to the tree walk
            # rather than returning 0.0 and stamping the model unservable.
            pass
    # Diffusers / transformers snapshot: sum only the WEIGHT files, and
    # de-dup redundant format/dtype variants of the same component -- a repo
    # routinely ships both fp16 and fp32 weights AND both .safetensors and
    # .ckpt/.bin of the same tensor, plus large non-weight blobs (dataset
    # attribution CSVs, READMEs). Summing the whole tree OVERestimates
    # wildly (Stable Audio Open summed 14.6 GB vs its ~2.6 GB real load;
    # SD-1.5 double-counts fp16+fp32), which 503s a servable model as
    # "exceeds device capacity". This is the general form of the GGUF fix
    # above. Per component (same dir + same canonical stem) we count only
    # ONE variant -- the one that actually loads: fp16 over fp32,
    # safetensors over .bin/.ckpt. Under-counting is the SAFE direction
    # (the observed-peak writeback self-heals upward and the monitor
    # recovers a rare OOM); over-counting is the dangerous one.
    try:
        # (root, canonical_stem) -> (preference_rank, size_bytes); lower rank
        # = the variant we count. os.path.getsize follows the HF blob symlinks.
        groups: dict[tuple, tuple[int, int]] = {}
        for root, _dirs, files in os.walk(local_dir):
            for name in files:
                key_rank = _weight_key(name)
                if key_rank is None:
                    continue  # not a weight file (config/README/CSV/image/...)
                canon, rank = key_rank
                try:
                    size = os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
                key = (root, canon)
                if key not in groups or rank < groups[key][0]:
                    groups[key] = (rank, size)
    except OSError:
        return 0.0
    return sum(size for _rank, size in groups.values()) / (1024 ** 3)


# Weight-bearing file extensions the sizer counts; everything else (configs,
# READMEs, dataset-attribution CSVs, preview images) is skipped.
_WEIGHT_EXTS = (
    ".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".onnx", ".msgpack", ".h5",
    ".gguf",
)
# 16-bit dtype tags the sizer PREFERS to count: smaller than fp32 and closer
# to what muse loads by default. _DTYPE_TAGS is the full set stripped from a
# stem so every dtype variant of one component groups together; _HALF_TAGS is
# the subset that earns the preference bonus. Keeping them derived from one
# base (rather than two hand-maintained lists) is what stops the strip list
# and the rank list from silently drifting apart.
_HALF_TAGS = (".fp16", ".f16", ".float16", ".bf16")
_DTYPE_TAGS = _HALF_TAGS + (".fp32", ".float32", ".f32")
_TRANSFORMERS_CHECKPOINT_RE = re.compile(
    r"^(model|pytorch_model|tf_model|flax_model)(-\d{5}-of-\d{5})?$"
)


def _weight_key(name: str) -> tuple[str, int] | None:
    """Canonical component key + preference rank for a weight file, or None
    if `name` is not a recognized weight file.

    The canonical key strips the weight extension and any dtype tag so that
    `diffusion_pytorch_model.safetensors`,
    `diffusion_pytorch_model.fp16.safetensors`, and
    `diffusion_pytorch_model.bin` all map to one stem
    (`diffusion_pytorch_model`); common Transformers checkpoint prefixes
    (`model` / `pytorch_model` / `tf_model` / `flax_model`, with an optional
    shard suffix) normalize together so `model-00001-of-00002.safetensors`
    and `pytorch_model-00001-of-00002.bin` are the same shard.

    Lower rank = the variant the sizer counts (the one that actually loads):
    a 16-bit dtype over fp32, safetensors over pickle (.bin/.ckpt/.pt).
    Counting the loaded variant, not the largest, leans the estimate DOWN --
    the safe direction against spurious "exceeds device capacity" 503s.
    """
    lower = name.lower()
    ext = next((e for e in _WEIGHT_EXTS if lower.endswith(e)), None)
    if ext is None:
        return None
    stem = name[: -len(ext)]
    stem_l = stem.lower()
    rank = -1 if ext == ".safetensors" else 0
    for tag in _DTYPE_TAGS:
        if stem_l.endswith(tag):
            stem, stem_l = stem[: -len(tag)], stem_l[: -len(tag)]
            if tag in _HALF_TAGS:
                rank -= 2
            break
    m = _TRANSFORMERS_CHECKPOINT_RE.match(stem_l)
    if m is not None:
        stem = f"model{m.group(2) or ''}"
    return stem, rank


def _entry_with_live_manifest(model_id: str, catalog_entry: dict) -> dict:
    """Return an entry carrying the manifest used by the worker runtime.

    Bundled/discovered models historically omit ``manifest`` from
    ``catalog.json``.  Capacity validation still needs their author-declared
    device pin, otherwise a probe measurement's device can be mistaken for
    current placement.  Synthetic/legacy rows that discovery cannot resolve
    retain their catalog-only fallback behavior.
    """
    try:
        manifest = get_manifest(model_id)
    except (CatalogError, KeyError):
        return catalog_entry
    if catalog_entry.get("manifest") == manifest:
        return catalog_entry
    enriched = dict(catalog_entry)
    enriched["manifest"] = manifest
    return enriched


def _effective_model_device(
    catalog_entry: dict,
    *,
    measured_device: str,
    supervisor_device: str = "auto",
) -> str:
    """Resolve placement using the same precedence as ``load_backend``."""
    override = str(catalog_entry.get("device_override") or "").lower()
    if override:
        return override

    manifest = catalog_entry.get("manifest") or {}
    if not isinstance(manifest, dict):
        manifest = {}
    capabilities = manifest.get("capabilities") or {}
    if not isinstance(capabilities, dict):
        capabilities = {}
    manifest_device = str(capabilities.get("device") or "auto").lower()
    if manifest_device not in ("", "auto"):
        return manifest_device

    requested = str(supervisor_device or "auto").lower()
    if requested not in ("", "auto"):
        return requested

    # A real live manifest explicitly says auto, so placement must follow
    # current runtime detection rather than a stale probe device.  The
    # measured fallback remains for old/synthetic catalog rows with no
    # discoverable manifest at all.
    if catalog_entry.get("manifest") is not None:
        return "auto"
    return measured_device


def _has_memory_data(catalog_entry: dict) -> tuple[bool, float, str]:
    """Return (has_data, memory_gb, device).

    Sizing ladder, in order of preference:
      1. `manifest.capabilities.memory_gb` annotation (hand-set or
         from a script's MANIFEST).
      2. `measurements.<device>.peak_bytes` from a probe run / self-healed
         lazy-load observation.
      3. on-disk weights size summed from the entry's `local_dir`.

    `device` is read from `manifest.capabilities.device` and lowercased.
    Falls back to "auto" when absent, matching the worker's own default
    (see muse.core.memory_probe.declared_device).

    `has_data` is True when ANY source is present; False only when the
    model declares nothing, was never probed, AND has no weights on disk.
    The boot validation flags False entries as unservable with the
    probe-prompt reason. Because pulled models always have weights on
    disk, that 503 path is effectively reserved for pre-worker / removed
    entries.
    """
    manifest = catalog_entry.get("manifest", {}) or {}
    if not isinstance(manifest, dict):
        manifest = {}
    capabilities = manifest.get("capabilities", {}) or {}
    if not isinstance(capabilities, dict):
        capabilities = {}
    device = declared_device(capabilities)
    declared = capabilities.get("memory_gb")

    measurements = catalog_entry.get("measurements", {}) or {}
    if not isinstance(measurements, dict):
        measurements = {}
    # Probe records key by the resolved device (e.g. "cpu" / "cuda")
    # so we look up by the same key. "gpu" alias normalizes to "cuda"
    # to match what the probe writes.
    measurement_key = "cuda" if device == "gpu" else device
    measurement = measurements.get(measurement_key) or {}
    if not isinstance(measurement, dict):
        measurement = {}
    measured = measurement.get("peak_bytes")

    # Bundled models have no persisted manifest in catalog.json, so `device`
    # falls back to "cpu" above even when the probe ran on cuda. If the
    # manifest-derived device has no measurement but the catalog has one for
    # another device, use it and adopt the measurement's own recorded device
    # so the capacity check below picks the right memory pool. Without this,
    # `muse models probe` never clears a bundled GPU model's "no memory
    # estimate" flag (the probe writes measurements.cuda; the lookup reads
    # measurements.cpu).
    #
    # Gate on `declared is None`: this recovery only matters when there is no
    # declared memory_gb (the bundled-model case). A model that DOES declare
    # memory_gb already trusts its manifest device below, so we must not let a
    # stale cross-device measurement (e.g. a CPU probe of a declared-cuda model
    # pulled with --no-probe) overwrite that device and mis-size the GPU model
    # against the CPU pool.
    if measured is None and declared is None:
        for dev_key, rec in measurements.items():
            if not isinstance(rec, dict):
                continue
            peak = rec.get("peak_bytes")
            if peak:
                measured = peak
                device = str(rec.get("device") or dev_key).lower() or device
                break

    if declared is not None:
        if not isinstance(declared, bool):
            try:
                declared_gb = float(declared)
            except (TypeError, ValueError):
                declared_gb = math.nan
            if math.isfinite(declared_gb) and declared_gb >= 0.0:
                return True, declared_gb, device
        # An explicit malformed declaration must not be silently replaced by
        # a smaller on-disk fallback while the unchanged manifest continues
        # to hand the bad value to the director.
        return False, 0.0, device
    if measured is not None and not isinstance(measured, bool):
        try:
            measured_gb = float(measured) / (1024 ** 3)
        except (TypeError, ValueError):
            measured_gb = math.nan
        if math.isfinite(measured_gb) and measured_gb > 0.0:
            return True, measured_gb, device

    # Last resort: size the model from the bytes already on disk so a
    # never-probed model still loads on demand instead of 503'ing.
    weights_gb = _weights_size_gb(catalog_entry)
    if weights_gb > 0:
        return True, weights_gb, device

    return False, 0.0, device


def _has_invalid_declared_memory(catalog_entry: dict) -> bool:
    """Whether capabilities.memory_gb is present but unusable."""
    manifest = catalog_entry.get("manifest") or {}
    if not isinstance(manifest, dict):
        return False
    capabilities = manifest.get("capabilities") or {}
    if not isinstance(capabilities, dict):
        return False
    declared = capabilities.get("memory_gb")
    if declared is None:
        return False
    if isinstance(declared, bool):
        return True
    try:
        numeric = float(declared)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(numeric) or numeric < 0.0


def _capacity_pools(
    memory_probe: "Any",
    *,
    gpu_headroom_gb: float,
    cpu_headroom_gb: float,
    gpu_budget_gb: "float | None" = None,
    cpu_budget_gb: "float | None" = None,
) -> tuple[float, "float | None", bool]:
    """Return hard CPU/GPU capacity and CUDA-runtime availability.

    ``model_unservable`` is a permanent-fit verdict, so production probes
    use physical total RAM/VRAM here.  Current free memory is transient and
    is enforced later by :class:`LoadDirector`, which may reclaim
    Muse-owned workers.  Older injected probes that expose only free memory
    retain their historical behavior for compatibility.  A configured
    budget caps the physical ceiling (and is the GPU fallback when total
    VRAM cannot be measured).

    The third value is deliberately separate from GPU capacity: an explicit
    CUDA model may use an operator-supplied budget, while an ``auto`` model
    selects CUDA only when NVML or torch reports a CUDA-compatible runtime.
    """
    cpu_free_gb = float(memory_probe.cpu_free_gb())
    cpu_total_gb = _optional_probe_gb(memory_probe, "cpu_total_gb")
    cpu_available = available_capacity_gb(
        live_free_gb=(cpu_total_gb if cpu_total_gb is not None else cpu_free_gb),
        budget_gb=cpu_budget_gb,
        headroom_gb=cpu_headroom_gb,
    )
    cpu_available_gb = float(cpu_available or 0.0)
    gpu_free = memory_probe.gpu_free_gb()
    gpu_total_gb = _optional_probe_gb(memory_probe, "gpu_total_gb")
    gpu_available_gb = available_capacity_gb(
        live_free_gb=(gpu_total_gb if gpu_total_gb is not None else gpu_free),
        budget_gb=gpu_budget_gb,
        headroom_gb=gpu_headroom_gb,
    )
    checker = _cuda_checker_for_probe(memory_probe)
    cuda_available = gpu_free is not None
    if not cuda_available and checker is not None:
        try:
            cuda_available = bool(checker())
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA runtime availability check failed: %s", exc)
    return cpu_available_gb, gpu_available_gb, cuda_available


def _available_pools(
    memory_probe: "Any",
    *,
    gpu_headroom_gb: float,
    cpu_headroom_gb: float,
    gpu_budget_gb: "float | None" = None,
    cpu_budget_gb: "float | None" = None,
) -> tuple[float, "float | None"]:
    """Compatibility view of the CPU/GPU available-capacity pair."""
    cpu_available_gb, gpu_available_gb, _cuda_available = _capacity_pools(
        memory_probe,
        gpu_headroom_gb=gpu_headroom_gb,
        cpu_headroom_gb=cpu_headroom_gb,
        gpu_budget_gb=gpu_budget_gb,
        cpu_budget_gb=cpu_budget_gb,
    )
    return cpu_available_gb, gpu_available_gb


def _servability_reason(
    entry: dict,
    *,
    cpu_available_gb: float,
    gpu_available_gb: "float | None",
    auto_cuda_available: "bool | None" = None,
    supervisor_device: str = "auto",
) -> "str | None":
    """The unservable reason for one catalog entry, or None if servable.

    Single source of truth for boot validation AND the live request-path
    re-check (`revalidate_servability`), so the two verdicts never drift.
    Applies the sizing ladder (`_has_memory_data`) then a device-capacity
    check against the caller-supplied hard-capacity pools.

    Returns:
      - "no memory estimate ..." when the model is not sizable at all
        (no annotation, no probe, no weights on disk).
      - "exceeds device capacity ..." when sized but it does not fit the
        device's capacity pool (or a cuda model on a host with no capacity
        info). This is a HARD stop: the gateway 503s without deferring
        to the director, because a model that does not fit even an empty
        working set can only make the director evict everything and 503.
      - None when sizable AND it fits.
    """
    has_data, sized_gb, measured_device = _has_memory_data(entry)
    if not has_data:
        if _has_invalid_declared_memory(entry):
            return (
                "invalid memory estimate; capabilities.memory_gb must be a "
                "finite non-negative number"
            )
        return "no memory estimate; run `muse models probe` to populate"
    device = _effective_model_device(
        entry,
        measured_device=measured_device,
        supervisor_device=supervisor_device,
    )
    # Preserve the private helper's historical standalone behavior when a
    # caller does not provide the explicit detector verdict.
    if auto_cuda_available is None:
        auto_cuda_available = gpu_available_gb is not None
    pool = resolve_memory_pool(
        device,
        gpu_free_gb=None,
        cuda_available=auto_cuda_available,
    )
    resolved_device = (
        pool if device in ("auto", "", "cuda", "gpu") else device
    )
    if pool == "cuda":
        if gpu_available_gb is None:
            return (
                "exceeds device capacity (no GPU capacity info available; "
                "install nvidia-ml-py / pynvml or set memory budget)"
            )
        available_gb = gpu_available_gb
    else:
        available_gb = cpu_available_gb
    if sized_gb > available_gb:
        return (
            f"exceeds device capacity ({sized_gb:.1f} GB > "
            f"{available_gb:.1f} GB capacity on {resolved_device})"
        )
    return None


def validate_catalog_at_boot(
    state: SupervisorState,
    *,
    memory_probe: "Any | None" = None,
    gpu_budget_gb: "float | None" = None,
    cpu_budget_gb: "float | None" = None,
    gpu_headroom_gb: float = 1.0,
    cpu_headroom_gb: float = 2.0,
) -> None:
    """Walk the enabled catalog and stamp unservable_reasons.

    For each `enabled: true` row in the catalog:
      - If the row has neither `manifest.capabilities.memory_gb` nor
        `measurements.<device>.peak_bytes`, mark it
        "no memory estimate; run muse models probe".
      - If the row's declared memory_gb exceeds physical device capacity
        (total minus headroom), mark it "exceeds device capacity".
      - GPU rows with no physical VRAM info use a configured GPU budget as a
        static fallback; without either source they exceed capacity.
      - Auto rows use the GPU pool when NVML or torch reports a CUDA-
        compatible runtime, including ROCm.

    The result is stored in `state.unservable_reasons`; the gateway and
    `/v1/models` consult this dict to short-circuit 503 before calling
    the director.

    A corrupt catalog.json with no last-known-good cache makes
    `_read_catalog` raise `CatalogError` (see its corrupt-guard
    docstring). Boot must not crash with a raw traceback over this: log
    one clear, actionable line naming the catalog path (no exc_info; the
    underlying corruption was already logged by `_read_catalog`) and
    degrade gracefully by returning early with no stamps, exactly like
    the pre-existing missing/empty-catalog case. The gateway still 503s
    `catalog_unavailable` per request via its own CatalogError handling
    around `get_manifest`, so operators get a live, actionable signal on
    every request rather than a supervisor that refuses to boot.

    `memory_probe` defaults to the production adapter; tests inject a
    MagicMock with the desired return values.
    """
    if memory_probe is None:
        memory_probe = _MemoryProbeAdapter()

    try:
        catalog = _read_catalog()
    except CatalogError as exc:
        logger.error(
            "muse serve: catalog is corrupt; boot continues with no "
            "models validated until it is fixed: %s", exc,
        )
        return

    cpu_available_gb, gpu_available_gb, cuda_available = _capacity_pools(
        memory_probe,
        gpu_headroom_gb=gpu_headroom_gb,
        cpu_headroom_gb=cpu_headroom_gb,
        gpu_budget_gb=gpu_budget_gb,
        cpu_budget_gb=cpu_budget_gb,
    )

    for model_id, entry in catalog.items():
        if not entry.get("enabled", True):
            continue
        # Skip pre-worker entries; they cannot load anyway.
        if not entry.get("python_path"):
            continue

        reason = _servability_reason(
            _entry_with_live_manifest(model_id, entry),
            cpu_available_gb=cpu_available_gb,
            gpu_available_gb=gpu_available_gb,
            auto_cuda_available=cuda_available,
            supervisor_device=state.device,
        )
        if reason is not None:
            state.unservable_reasons[model_id] = reason


def revalidate_servability(
    state: SupervisorState,
    model_id: str,
    *,
    memory_probe: "Any | None" = None,
    gpu_budget_gb: "float | None" = None,
    cpu_budget_gb: "float | None" = None,
    gpu_headroom_gb: float = 1.0,
    cpu_headroom_gb: float = 2.0,
) -> str | None:
    """Re-derive one model's unservable verdict against the LIVE catalog.

    `validate_catalog_at_boot` stamps `state.unservable_reasons` once at
    boot. That snapshot goes stale when a `muse models probe`, manifest edit,
    weights landing on disk, placement change, or budget change makes a
    previously unservable model valid. This re-reads the (mtime-cached)
    catalog for ONE model and re-runs the SAME `_servability_reason` boot
    uses -- the estimate and the permanent capacity ceiling -- then updates
    the stamp, so the gateway reflects reality WITHOUT a supervisor restart.

    Crucially this preserves a genuine "exceeds device capacity" stamp: a
    model that cannot fit even an empty working set is NOT cleared just
    because it became sizable. Clearing it would route an impossible request
    into the director, whose eviction loop would tear down the whole idle
    working set before 503'ing. The gateway 503s such a model directly.

    Scoped to one model: no full-catalog walk. Reads live memory via the
    probe (defaults to the production adapter; tests inject a MagicMock).
    Returns the current reason (None when now servable). Mutations to
    `state.unservable_reasons` are made under `state.lock`; the probe read
    and `_servability_reason` run outside the lock.
    """
    if memory_probe is None:
        memory_probe = _MemoryProbeAdapter()
    # Gateway callers historically pass only headroom. Recover numeric
    # budgets from the same director instance so revalidation cannot drift
    # from its subsequent admission decision. Avoid loose MagicMock attrs.
    if state.director is not None:
        if gpu_budget_gb is None:
            configured = getattr(state.director, "gpu_budget_gb", None)
            if isinstance(configured, (int, float)) and not isinstance(configured, bool):
                gpu_budget_gb = float(configured)
        if cpu_budget_gb is None:
            configured = getattr(state.director, "cpu_budget_gb", None)
            if isinstance(configured, (int, float)) and not isinstance(configured, bool):
                cpu_budget_gb = float(configured)
    catalog = _read_catalog()
    entry = catalog.get(model_id)
    if entry is None:
        # Removed from the catalog since boot. Clear the now-stale stamp and
        # return None so the gateway falls through to get_manifest, which
        # 404s `model_not_found` if truly gone (or serves a bundled fallback)
        # -- rather than 503'ing with a reason that names a model that no
        # longer exists.
        with state.lock:
            state.unservable_reasons.pop(model_id, None)
        return None
    cpu_available_gb, gpu_available_gb, cuda_available = _capacity_pools(
        memory_probe,
        gpu_headroom_gb=gpu_headroom_gb,
        cpu_headroom_gb=cpu_headroom_gb,
        gpu_budget_gb=gpu_budget_gb,
        cpu_budget_gb=cpu_budget_gb,
    )
    reason = _servability_reason(
        _entry_with_live_manifest(model_id, entry),
        cpu_available_gb=cpu_available_gb,
        gpu_available_gb=gpu_available_gb,
        auto_cuda_available=cuda_available,
        supervisor_device=state.device,
    )
    with state.lock:
        if reason is None:
            state.unservable_reasons.pop(model_id, None)
        else:
            state.unservable_reasons[model_id] = reason
        return reason


def backfill_manifest_memory(
    manifest: dict,
    model_id: str,
    *,
    supervisor_device: str = "auto",
) -> dict:
    """Return a copy of `manifest` sized (and device-pinned) from the catalog.

    Two backfills, both drawn from the model's catalog entry:

    1. **memory_gb** -- The LoadDirector sizes loads (and drives LRU eviction)
       from `capabilities.memory_gb`. A probed-only or never-probed model
       declares none, so without this the director would treat it as 0 GB
       ("fits anywhere", never evicting). We fill it from the sizing ladder
       (`_has_memory_data`: probe measurement, else on-disk weights size). An
       explicit declared `memory_gb` always wins.

    2. **device** -- An operator `set-device` pin (catalog `device_override`)
       decides where the worker actually loads, mirroring load_backend's
       tier-1 precedence. We fold it into `capabilities.device` so the
       director sizes, admits, and evicts against the pool the worker will
       load on. Without this a cuda model pinned to cpu makes the director
       needlessly evict GPU models to make room for a host-RAM load, and the
       inverse pin over-commits VRAM. The override fires regardless of the
       memory backfill (a model may declare memory_gb yet still be pinned).

    The input is never mutated; a copy is made lazily only when a backfill
    actually changes something.
    """
    catalog = _read_catalog()
    entry = catalog.get(model_id)
    out = manifest
    caps = manifest.get("capabilities", {}) or {}

    if entry is not None and caps.get("memory_gb") is None:
        gb: float | None = None
        # A LoRA entry's own dir holds only the adapter (tens of MB), so
        # the weights-on-disk fallback would grossly undersize the load.
        # When it has no probe measurement of its own, size it from its
        # muse-id base entry instead. A probed LoRA entry measured the
        # real base+adapter peak; prefer that.
        if caps.get("lora_adapter") and not (entry.get("measurements") or {}):
            base = caps.get("base_model")
            base_entry = (
                catalog.get(base) if base and "/" not in base else None
            )
            if base_entry is not None:
                has_b, gb_b, _d = _has_memory_data(base_entry)
                if has_b and gb_b > 0:
                    gb = gb_b
        if gb is None:
            has_data, gb_own, _device = _has_memory_data(entry)
            if has_data and gb_own > 0:
                gb = gb_own
        if gb is not None:
            out = dict(out)
            out_caps = dict(caps)
            out_caps["memory_gb"] = gb
            out["capabilities"] = out_caps

    override = (entry or {}).get("device_override")
    if override:
        if out is manifest:
            out = dict(out)
        out_caps = dict(out.get("capabilities", {}) or {})
        out_caps["device"] = override
        out["capabilities"] = out_caps
    else:
        manifest_device = str(
            (out.get("capabilities", {}) or {}).get("device") or "auto"
        ).lower()
        requested = str(supervisor_device or "auto").lower()
        if manifest_device in ("", "auto") and requested not in ("", "auto"):
            if out is manifest:
                out = dict(out)
            out_caps = dict(out.get("capabilities", {}) or {})
            out_caps["device"] = requested
            out["capabilities"] = out_caps

    return out


def build_load_director(
    *,
    enable_fn: Callable[[str], int],
    disable_fn: Callable[[str], None],
    memory_probe: Any,
) -> "Any":
    """Construct a LoadDirector with config-derived budgets/headroom.

    This is the v0.5x doc-drift fix: `MUSE_GPU_BUDGET_GB`,
    `MUSE_CPU_BUDGET_GB`, `MUSE_GPU_HEADROOM_GB`, `MUSE_CPU_HEADROOM_GB`
    were documented as active env knobs but `LoadDirector.__init__`
    only ever saw its own hardcoded defaults (None, None, 1.0, 2.0)
    because nothing passed them in. Extracted as a standalone factory
    (rather than inlined at the one call site) so the config wiring is
    independently unit-testable without spinning up a full
    SupervisorState.

    Defaults match today's hardcoded LoadDirector.__init__ values, so a
    deployment that sets nothing sees identical behavior; the knobs
    simply start working for operators who do set them.
    """
    from muse.cli_impl.load_director import LoadDirector

    return LoadDirector(
        enable_fn=enable_fn,
        disable_fn=disable_fn,
        memory_probe=memory_probe,
        gpu_budget_gb=config.get("server.gpu_budget_gb"),
        cpu_budget_gb=config.get("server.cpu_budget_gb"),
        gpu_headroom_gb=config.get("server.gpu_headroom_gb"),
        cpu_headroom_gb=config.get("server.cpu_headroom_gb"),
        cuda_available_fn=_cuda_checker_for_probe(memory_probe),
    )


def _build_load_director(state: SupervisorState) -> "Any":
    """Construct a LoadDirector wired to the supervisor's enable/disable.

    `enable_fn` and `disable_fn` are thin wrappers around the new
    `load_model_into_worker` and `unload_model_from_worker` operations
    in `muse.admin.operations`. Those operations spawn / terminate
    workers WITHOUT touching the catalog's `enabled` flag - lazy load
    is "is there a worker for this model right now?", orthogonal to
    the catalog's "is this model in service?" flag. Reusing the
    existing `enable_model` / `disable_model` ops would re-couple the
    two states, defeating the v0.40.0 design.

    Imported lazily to break the cycle: supervisor.py is imported by
    admin.operations on its way up, so an unconditional top-level
    import would loop.
    """
    from muse.admin.operations import (
        load_model_into_worker,
        unload_model_from_worker,
    )

    def enable_fn(model_id: str) -> int:
        return load_model_into_worker(model_id, state=state)

    def disable_fn(model_id: str) -> None:
        unload_model_from_worker(model_id, state=state)

    return build_load_director(
        enable_fn=enable_fn,
        disable_fn=disable_fn,
        memory_probe=_MemoryProbeAdapter(),
    )


# Fallback when the configured idle-sweep interval is not usable (see
# _resolve_idle_sweep_interval). Matches the documented / registry default
# for server.idle_sweep_interval_seconds.
_DEFAULT_IDLE_SWEEP_INTERVAL_SECONDS = 30.0
_BACKGROUND_JOIN_TIMEOUT_SECONDS = 5.0


def _resolve_idle_sweep_interval() -> float:
    """Resolve the idle-sweep tick interval, clamped to a safe value.

    `IdleSweeper._run` sleeps via `stop_event.wait(interval_seconds)`
    between ticks. A 0, negative, or non-finite (NaN/inf) interval makes
    `wait` return (almost) immediately, busy-looping `tick()` against the
    director lock on every iteration. The adjacent default_idle_timeout
    resolution already guards its own <= 0 case; this mirrors that guard
    (and `serve_util.shutdown_grace_seconds`'s analogous guard for the
    graceful-shutdown timeout) for the sweep interval.
    """
    value = config.get("server.idle_sweep_interval_seconds")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return _DEFAULT_IDLE_SWEEP_INTERVAL_SECONDS
    return float(value)


def _join_background_thread(
    thread: threading.Thread | None,
    *,
    name: str,
    timeout: float = _BACKGROUND_JOIN_TIMEOUT_SECONDS,
) -> bool:
    """Join one owned background thread without losing a live handle."""
    if thread is None:
        return True
    if thread is threading.current_thread():
        logger.warning("cannot join %s from itself", name)
        return False
    try:
        thread.join(timeout=timeout)
    except RuntimeError:
        # join() rejects a thread whose start() failed. Such a thread is
        # already inert; a genuinely live thread remains a cleanup failure.
        if thread.is_alive() is not True:
            return True
        logger.warning("could not join %s", name, exc_info=True)
        return False
    if thread.is_alive() is True:
        logger.warning("%s did not stop within %.1fs", name, timeout)
        return False
    return True


def _shutdown_telemetry(state: SupervisorState) -> bool:
    """Stop every telemetry owner in dependency order.

    Producers are joined before the recorder is drained, and the recorder is
    stopped before SQLite closes. Handles are cleared only after confirmed
    teardown so a failed bounded join remains visible and retryable.
    """
    state.stop_event.set()

    sampler = state.telemetry_sampler
    sampler_stopped = True
    if sampler is not None:
        try:
            sampler_stopped = sampler.stop() is not False
        except Exception:
            sampler_stopped = False
            logger.warning("telemetry sampler stop failed", exc_info=True)
        if sampler_stopped:
            state.telemetry_sampler = None

    prune_thread = state.telemetry_prune_thread
    prune_stopped = _join_background_thread(
        prune_thread,
        name="telemetry prune thread",
    )
    if prune_stopped:
        state.telemetry_prune_thread = None

    # A producer that is still live may touch both the recorder and store.
    # Leave their handles open/retryable rather than racing close against it.
    if not (sampler_stopped and prune_stopped):
        return False

    recorder = state.telemetry_recorder
    recorder_stopped = True
    if recorder is not None:
        try:
            recorder_stopped = reset_recorder(expected=recorder)
            if not recorder_stopped:
                # A newer supervisor may own the global slot. Stopping this
                # exact stale instance is safe; resetting the newer one is not.
                recorder_stopped = recorder.stop() is not False
        except Exception:
            recorder_stopped = False
            logger.warning("telemetry recorder stop failed", exc_info=True)
        if recorder_stopped:
            state.telemetry_recorder = None
    if not recorder_stopped:
        return False

    store = state.telemetry_store
    if store is not None:
        try:
            store.close()
        except Exception:
            logger.warning("telemetry store close failed", exc_info=True)
            return False
        state.telemetry_store = None
    state.log_hub = None
    state.telemetry_vram_tracker = None
    return True


def _init_telemetry(state: SupervisorState) -> None:
    """Boot-time telemetry wiring: store + recorder + log hub + sampler + prune.

    Called from `run_supervisor` when `telemetry.enabled` is true, after
    `state.stop_event` is installed but before the gateway is built, so the
    mounted dashboard router can read `state.telemetry_store` /
    `state.log_hub` from its very first request.

    Factored out (rather than inlined in `run_supervisor`) so it is
    unit-testable without spinning up uvicorn: tests build a minimal
    SupervisorState (a `director` stub with `.loaded` / `.in_flight_loads`
    and a real `stop_event`), call this directly, and assert on the
    resulting state + recorder.

    Wires four pieces:
      - `TelemetryStore` at `<catalog_dir>/telemetry.db`, plus
        `init_recorder(store, enabled=True)` so `muse.observability
        .recorder.record(...)` calls from request-handling code actually
        persist instead of hitting the shared no-op recorder.
      - `LogHub` sized from `telemetry.log_buffer_kb`, attached to
        `state.log_hub` so `spawn_worker(..., log_hub=state.log_hub)`
        callers pipe worker stdout into it.
      - A periodic `Sampler` recording free VRAM/RAM + loaded/in-flight
        counts, reading the live director state via closures (so it
        always reflects the current loaded set, not a snapshot). Shares
        `state.stop_event` (same pattern as `IdleSweeper`) so a single
        Ctrl+C/SIGTERM unblocks the sampler's loop along with the other
        supervisor-owned daemon threads; `run_supervisor`'s shutdown
        `finally` block also calls `sampler.stop()` to join the thread
        and `state.telemetry_store.close()` to release the sqlite handle.
      - A retention-prune daemon that shares `state.stop_event` with the
        rest of the supervisor's background threads, deleting events
        older than `telemetry.retention_days` once an hour.
    """
    if any(
        owner is not None
        for owner in (
            state.telemetry_store,
            state.telemetry_recorder,
            state.telemetry_sampler,
            state.telemetry_vram_tracker,
            state.telemetry_prune_thread,
        )
    ):
        raise RuntimeError("telemetry is already initialized for this supervisor")
    if state.stop_event.is_set():
        raise RuntimeError("supervisor shutdown requested during telemetry startup")
    if not config.get("telemetry.require_auth"):
        logger.warning(
            "telemetry authentication is disabled; dashboard data and worker "
            "logs are available without an admin token"
        )

    store_path = Path(config.get("paths.catalog_dir")).expanduser() / "telemetry.db"
    store = TelemetryStore(store_path)
    state.telemetry_store = store
    try:
        log_buffer_kb = config.get("telemetry.log_buffer_kb")
        state.log_hub = LogHub(buffer_bytes=int(log_buffer_kb) * 1024)

        vram_tracker = VramTracker()
        state.telemetry_vram_tracker = vram_tracker
        sampler = Sampler(
            interval=float(config.get("telemetry.sample_interval_seconds")),
            loaded_fn=lambda: state.director.loaded,
            inflight_fn=lambda: len(
                getattr(state.director, "in_flight_loads", {}) or {}
            ),
            stop_event=state.stop_event,
            vram_tracker=vram_tracker,
            active_interval=float(
                config.get("telemetry.trace_sample_interval_seconds")
            ),
        )
        state.telemetry_sampler = sampler
        state.telemetry_recorder = init_recorder(store, enabled=True)
        try:
            sampler.sample_once()
        except Exception:
            logger.warning("initial telemetry sample failed", exc_info=True)
        if not sampler.start():
            raise RuntimeError(
                "supervisor shutdown requested during telemetry startup"
            )

        retention_days = config.get("telemetry.retention_days")

        def _prune_loop() -> None:
            while not state.stop_event.wait(3600):
                try:
                    store.prune(time.time() - float(retention_days) * 86400)
                except Exception:
                    logger.warning("telemetry prune failed", exc_info=True)

        prune_thread = threading.Thread(
            target=_prune_loop,
            daemon=True,
            name="muse-telemetry-prune",
        )
        state.telemetry_prune_thread = prune_thread
        prune_thread.start()
        if state.stop_event.is_set():
            raise RuntimeError(
                "supervisor shutdown requested during telemetry startup"
            )
    except BaseException:
        _shutdown_telemetry(state)
        raise


def run_supervisor(*, host: str, port: int, device: str) -> int:
    """Entry point for `muse serve` (v0.40.0+: lazy load).

    Boot sequence:
      1. Construct a SupervisorState with an empty worker list.
      2. Construct a LoadDirector and hang it off state.director.
      3. Run validate_catalog_at_boot to stamp unservable_reasons.
      4. Start the auto-restart monitor (it watches state.workers, which
         is empty at boot but will fill via director.acquire).
      5. Start the gateway. First request per model triggers the
         director's enable_fn, which spawns the worker.
      6. On shutdown: SIGTERM whatever workers are loaded (could be 0).

    No worker spawn at boot. No first-ready wait. The gateway is
    reachable instantly. Cold-start latency moves from boot to first
    request per model.

    Registers a SupervisorState singleton so admin endpoints under
    `/v1/admin/*` can inspect and mutate the worker list. The monitor
    thread reads `state.workers` directly, so director-triggered worker
    spawns + admin-triggered enable/disable + auto-restart all show up
    in one consistent live list.
    """
    from muse.admin.jobs import get_default_store, reset_default_store
    from muse.cli_impl.gateway import build_gateway

    # Validate the complete configuration before publishing state, replacing
    # the JobStore singleton, registering resources, or starting any thread.
    # Lenient point reads below are therefore operating on an already-proven
    # configuration snapshot for this supervisor boot.
    config.validated_config()

    state = SupervisorState(workers=[], device=device)
    set_supervisor_state(state)
    stop_event = state.stop_event
    monitor_thread: threading.Thread | None = None
    sweeper_thread: threading.Thread | None = None
    supervisor_resource_id: str | None = None
    job_store: Any | None = None
    cleanup_complete = True

    try:
        # Gateway lifespan permanently closes its JobStore. A same-interpreter
        # supervisor re-entry gets a fresh store, and this exact instance is
        # retained so every startup failure can close it in the outer finally.
        reset_default_store()
        job_store = get_default_store()

        state.director = _build_load_director(state)
        # A release-to-zero or completed eviction wakes queued capacity
        # waiters so they can re-run admission promptly.
        state.director.capacity_listener = state.capacity_notifier.notify

        # Permanent servability validation belongs inside the transaction:
        # malformed catalogs or capacity probes must not strand the singleton
        # or any owners initialized before gateway startup.
        validate_catalog_at_boot(
            state,
            gpu_budget_gb=state.director.gpu_budget_gb,
            cpu_budget_gb=state.director.cpu_budget_gb,
            gpu_headroom_gb=state.director.gpu_headroom_gb,
            cpu_headroom_gb=state.director.cpu_headroom_gb,
        )
        if state.unservable_reasons:
            for mid, reason in sorted(state.unservable_reasons.items()):
                logger.warning("unservable model %r: %s", mid, reason)

        # Publish each owner handle before calling start(). If thread startup
        # itself fails, the outer cleanup can distinguish an inert unstarted
        # object from one that actually began running.
        monitor_thread = threading.Thread(
            target=_monitor_workers,
            args=(state.workers, stop_event),
            kwargs={"state": state},
            daemon=True,
            name="muse-monitor",
        )
        state.monitor_thread = monitor_thread
        monitor_thread.start()
        logger.info(
            "auto-restart monitor running "
            "(interval=%.1fs, threshold=%d, budget=%d)",
            _MONITOR_INTERVAL,
            _FAILURE_THRESHOLD,
            _MAX_RESTARTS,
        )

        sweep_interval = _resolve_idle_sweep_interval()
        raw_default_idle = config.get("server.idle_timeout_seconds")
        default_idle_timeout: float | None = (
            raw_default_idle
            if raw_default_idle is not None and raw_default_idle > 0
            else None
        )
        sweeper = IdleSweeper(
            director=state.director,
            catalog_lookup=get_manifest,
            interval_seconds=sweep_interval,
            default_idle_timeout_seconds=default_idle_timeout,
            stop_event=stop_event,
        )
        state.idle_sweeper = sweeper
        sweeper_thread = sweeper.start()
        state.idle_sweeper_thread = sweeper_thread
        logger.info(
            "idle sweeper running (interval=%.1fs, default_idle_timeout=%s)",
            sweep_interval,
            f"{default_idle_timeout:.0f}s" if default_idle_timeout else "off",
        )

        # Telemetry is initialized before gateway construction so dashboard
        # routes observe a complete state from their first request.
        if config.get("telemetry.enabled"):
            _init_telemetry(state)
            logger.info(
                "telemetry enabled (db=%s)",
                Path(config.get("paths.catalog_dir")).expanduser()
                / "telemetry.db",
            )

        parent_pid = os.getppid()
        owner_pid = parent_pid if type(parent_pid) is int and parent_pid > 1 else None
        try:
            supervisor_resource_id = register_process(
                kind="supervisor",
                pid=os.getpid(),
                owner_pid=owner_pid,
                port=port,
            )
            state.supervisor_resource_id = supervisor_resource_id
        except ResourceRegistryError as exc:
            raise ResourceRegistryError(
                "cannot start an untracked Muse supervisor: "
                f"{exc}"
            ) from exc
        # Build gateway with a live SupervisorState reference. Routes
        # are derived per-request from state.workers (running-only) so
        # director-spawned workers join the routing table without an
        # app rebuild.
        app = build_gateway(state=state)

        logger.info(
            "starting gateway on %s:%d (lazy load: %d unservable model(s))",
            host, port, len(state.unservable_reasons),
        )
        # run_uvicorn sets a BOUNDED timeout_graceful_shutdown so the first
        # Ctrl-C exits within a fixed window even when a connection lingers
        # (SSE stream / long inference / idle keep-alive). uvicorn.run's
        # default (None) waits forever, stranding port 8000 and forcing the
        # operator to kill the process before restarting.
        run_uvicorn(
            app, host=host, port=port, shutdown_event=stop_event,
        )
    except KeyboardInterrupt:
        logger.info("shutting down (SIGINT)")
    finally:
        # Every startup and runtime path converges here. Stop background
        # producers before workers so none can restart/evict during teardown.
        stop_event.set()
        try:
            monitor_stopped = _join_background_thread(
                monitor_thread, name="worker monitor",
            )
            if monitor_stopped:
                state.monitor_thread = None
            else:
                cleanup_complete = False
        except Exception as e:  # noqa: BLE001
            cleanup_complete = False
            logger.warning("worker monitor cleanup failed: %s", e)
        try:
            sweeper_stopped = _join_background_thread(
                sweeper_thread, name="idle sweeper",
            )
            if sweeper_stopped:
                state.idle_sweeper_thread = None
                state.idle_sweeper = None
            else:
                cleanup_complete = False
        except Exception as e:  # noqa: BLE001
            cleanup_complete = False
            logger.warning("idle sweeper cleanup failed: %s", e)
        try:
            if _shutdown_telemetry(state) is False:
                cleanup_complete = False
        except Exception as e:  # noqa: BLE001
            cleanup_complete = False
            logger.warning("telemetry cleanup failed: %s", e)
        try:
            worker_result = _shutdown_workers(state.workers)
            if (
                isinstance(worker_result, WorkerShutdownResult)
                and not worker_result.complete
            ):
                cleanup_complete = False
        except Exception as e:  # noqa: BLE001
            cleanup_complete = False
            logger.warning("worker cleanup failed: %s", e)
        if job_store is not None:
            try:
                if job_store.shutdown() is False:
                    cleanup_complete = False
            except Exception as e:  # noqa: BLE001
                cleanup_complete = False
                logger.warning("admin job-store shutdown failed: %s", e)
        if cleanup_complete and supervisor_resource_id is not None:
            try:
                unregister_process(supervisor_resource_id)
            except ResourceRegistryError as e:
                cleanup_complete = False
                logger.warning("could not unregister supervisor resource: %s", e)
            else:
                supervisor_resource_id = None
                state.supervisor_resource_id = None
        if cleanup_complete:
            clear_supervisor_state()
        else:
            logger.error(
                "supervisor cleanup incomplete; retained ownership state "
                "and resource records for retry/repair"
            )
    return 0 if cleanup_complete else 1
