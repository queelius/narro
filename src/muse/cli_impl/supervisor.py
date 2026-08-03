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

from muse.core.memory_probe import declared_device

from muse.observability.store import TelemetryStore
from muse.observability.recorder import init_recorder, reset_recorder
from muse.observability.sampler import Sampler
from muse.observability.logs import LogHub

logger = logging.getLogger(__name__)

_SUPERVISOR_PID_ENV = "MUSE_SUPERVISOR_PID"


@dataclass
class WorkerSpec:
    """Everything needed to spawn and supervise one worker subprocess.

    Fields mutated by the monitor thread (after startup):
      - process: replaced on restart
      - restart_count: consecutive/cumulative UNSUCCESSFUL restart attempts
        (caps at _MAX_RESTARTS); a restart that succeeds does not bump it
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
    # Job that owns the in-flight transition (if any). Set when the
    # admin enable / restart-in-place op claims the spec; cleared
    # when the op finishes. Other concurrent enable requests for the
    # same model coalesce onto this job_id rather than launching a
    # duplicate spawn.
    job_id: str | None = None


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

    `lock` is a reentrant lock guarding all mutations of `workers` +
    `unservable_reasons`. v1 uses one global lock; per-model locks
    are deferred until contention becomes measurable.

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


def _pump_worker_logs(proc: "Any", model_id: str, hub: "Any") -> None:
    """Daemon-thread reader loop: pipe one worker's stdout into a LogHub.

    Reads `proc.stdout` line by line, appending each line to the hub
    (for `/v1/admin/*` log tailing and the dashboard) and re-emitting it
    to the aggregate supervisor log so `muse serve`'s own stdout is
    unchanged. `line` from a text-mode pipe already carries its trailing
    newline, hence `end=""` on the re-emit.

    Runs until `proc.stdout` hits EOF (the worker process exited) or
    raises; either way the thread must exit quietly rather than crash
    the supervisor process it runs alongside.
    """
    try:
        for line in proc.stdout:
            hub.append(model_id, line)
            print(f"[{model_id}] {line}", end="", flush=True)
    except Exception:
        logger.warning("log pump for %r stopped", model_id, exc_info=True)


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
    popen_kwargs: dict[str, Any] = {
        "env": child_env,
        "start_new_session": os.name == "posix",
    }
    if log_hub is not None:
        spec.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, **popen_kwargs,
        )
        model_id = spec.models[0] if spec.models else "worker"
        t = threading.Thread(
            target=_pump_worker_logs, args=(spec.process, model_id, log_hub),
            daemon=True, name=f"muse-logpump-{spec.port}",
        )
        t.start()
    else:
        spec.process = subprocess.Popen(cmd, **popen_kwargs)
    spec.last_spawn_at = time.monotonic()


def wait_for_ready(
    *, port: int, timeout: float = 60.0, poll_interval: float = 0.5,
    stop_event: "threading.Event | None" = None,
) -> None:
    """Block until http://127.0.0.1:<port>/health returns 200, or timeout.

    Raises TimeoutError if the worker never becomes ready. If `stop_event`
    is supplied, shutdown aborts the wait promptly instead of leaving an
    executor thread parked for the full model-load timeout.
    """
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("worker startup cancelled: supervisor shutdown requested")
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
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


def check_worker_health(*, port: int, timeout: float = 2.0) -> bool:
    """Single /health poll. Returns True iff the worker responds 200.

    Swallows all httpx errors; they indicate "unhealthy" for our purposes.
    Used by the monitor thread's periodic liveness check.
    """
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
        return r.status_code == 200
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

    restart_count counts consecutive/cumulative UNSUCCESSFUL restart
    attempts, matching the documented "10 unsuccessful restart attempts"
    cap: it is bumped only in the except branch below (a spawn or
    readiness failure), never on a successful respawn. A worker that
    flaps and recovers cleanly any number of times over its lifetime
    therefore never exhausts the budget; only a run of failures does.

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

    # Terminate existing process if still alive (best-effort).
    if spec.process is not None and spec.process.poll() is None:
        _shutdown_workers([spec], grace=3.0)
    spec.process = None

    # Respawn. restart_count bumps ONLY on failure below (see docstring):
    # a successful respawn must not count toward the unsuccessful-attempts
    # budget, else a worker that flaps and cleanly recovers many times
    # over its lifetime would eventually be marked dead despite never
    # having a run of consecutive failures.
    try:
        spawn_worker(spec, device=spec.device, log_hub=log_hub)
        wait_for_ready(
            port=spec.port, timeout=ready_timeout, stop_event=stop_event,
        )
        spec.failure_count = 0
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
    iterated in that tick; its process is already being torn down by the
    operation that removed it, so any restart the monitor would trigger
    is harmless (the spec will not be re-added to state.workers).

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

            # Process-death detection is unambiguous; short-circuit
            if spec.process is not None and spec.process.poll() is not None:
                logger.warning(
                    "worker on port %d: process exited with code %s",
                    spec.port, spec.process.returncode,
                )
                spec.failure_count = failure_threshold
            else:
                healthy = check_worker_health(port=spec.port)
                if healthy:
                    spec.failure_count = 0
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
                _attempt_restart(
                    spec, stop_event=stop_event, max_restarts=max_restarts,
                    log_hub=getattr(state, "log_hub", None),
                )

        # Sleep with early-exit if stop_event fires
        if stop_event.wait(interval):
            return


def _signal_worker_process(
    proc: Any, sig: signal.Signals, *, force: bool = False,
) -> None:
    """Signal a worker and, on POSIX, any descendants in its own session."""
    if os.name == "posix":
        try:
            pid = int(proc.pid)
            # Never allow invalid/mock metadata to become killpg(1), which is
            # kill(-1) at the syscall layer and broadcasts to every process
            # the caller may signal. Also refuse our own process group: real
            # workers are spawned with start_new_session=True, so neither
            # guard excludes a legitimate managed worker.
            if pid <= 1:
                raise ValueError(f"unsafe worker pid {pid}")
            pgid = os.getpgid(pid)
            if pgid <= 1 or pgid != pid or pgid == os.getpgrp():
                raise ValueError(f"unsafe worker process group {pgid}")
            os.killpg(pgid, sig)
            return
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    if force:
        proc.kill()
    else:
        proc.terminate()


def _shutdown_workers(specs: list[WorkerSpec], grace: float = 5.0) -> None:
    """SIGTERM worker groups, then SIGKILL and reap processes that linger."""
    # Snapshot both spec and process. Concurrent restart paths may replace
    # spec.process while we wait; shutdown must keep targeting the process it
    # originally claimed rather than a newly spawned replacement.
    targets = [(spec, spec.process) for spec in list(specs) if spec.process is not None]

    for spec, proc in targets:
        try:
            if proc.poll() is None:
                _signal_worker_process(proc, signal.SIGTERM)
        except Exception as e:
            logger.warning("failed to SIGTERM worker on port %d: %s", spec.port, e)

    for spec, proc in targets:
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            logger.warning(
                "worker on port %d did not exit in %.1fs; killing",
                spec.port, grace,
            )
            try:
                _signal_worker_process(
                    proc,
                    getattr(signal, "SIGKILL", signal.SIGTERM),
                    force=True,
                )
                # Reap after SIGKILL so no zombie remains owned by Muse.
                proc.wait(timeout=max(1.0, min(grace, 2.0)))
            except Exception as e:
                logger.warning(
                    "failed to kill/reap worker on port %d: %s", spec.port, e,
                )
        except Exception as e:
            logger.warning("error waiting for worker on port %d: %s", spec.port, e)


class _MemoryProbeAdapter:
    """Thin adapter wrapping `muse.core.memory_probe` module functions
    as bound methods on an object.

    LoadDirector accepts any object with `.gpu_free_gb()` and
    `.cpu_free_gb()`. The memory_probe module exposes those as
    free functions; this adapter satisfies the duck-typed contract
    without forcing a class definition into memory_probe.py
    (deferred to a future refactor; for now an adapter at the
    composition seam is the smaller change).
    """

    def gpu_free_gb(self, device_id: int = 0) -> float | None:
        from muse.core import memory_probe
        return memory_probe.gpu_free_gb(device_id)

    def cpu_free_gb(self) -> float:
        from muse.core import memory_probe
        return memory_probe.cpu_free_gb()


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
    if not local_dir:
        return 0.0
    capabilities = (catalog_entry.get("manifest") or {}).get("capabilities") or {}
    gguf_file = capabilities.get("gguf_file")
    if gguf_file:
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
    capabilities = manifest.get("capabilities", {}) or {}
    device = declared_device(capabilities)
    declared = capabilities.get("memory_gb")

    measurements = catalog_entry.get("measurements", {}) or {}
    # Probe records key by the resolved device (e.g. "cpu" / "cuda")
    # so we look up by the same key. "gpu" alias normalizes to "cuda"
    # to match what the probe writes.
    measurement_key = "cuda" if device == "gpu" else device
    measured = (measurements.get(measurement_key) or {}).get("peak_bytes")

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
            rec = rec or {}
            peak = rec.get("peak_bytes")
            if peak:
                measured = peak
                device = str(rec.get("device") or dev_key).lower() or device
                break

    if declared is not None:
        try:
            declared_gb = float(declared)
        except (TypeError, ValueError):
            declared_gb = 0.0
        return True, declared_gb, device
    if measured is not None and measured > 0:
        try:
            measured_gb = float(measured) / (1024 ** 3)
        except (TypeError, ValueError):
            measured_gb = 0.0
        return True, measured_gb, device

    # Last resort: size the model from the bytes already on disk so a
    # never-probed model still loads on demand instead of 503'ing.
    weights_gb = _weights_size_gb(catalog_entry)
    if weights_gb > 0:
        return True, weights_gb, device

    return False, 0.0, device


def _available_pools(
    memory_probe: "Any",
    *,
    gpu_headroom_gb: float,
    cpu_headroom_gb: float,
) -> tuple[float, "float | None"]:
    """Live (CPU, GPU) available pools in GB, each minus its headroom.

    GPU is None when no live VRAM info is available (pynvml absent / AMD /
    driver mismatch). Shared by boot validation and the request-path
    re-check so both size capacity from the SAME live readings.
    """
    cpu_free_gb = float(memory_probe.cpu_free_gb())
    cpu_available_gb = max(0.0, cpu_free_gb - cpu_headroom_gb)
    gpu_free = memory_probe.gpu_free_gb()
    if gpu_free is None:
        gpu_available_gb = None
    else:
        gpu_available_gb = max(0.0, float(gpu_free) - gpu_headroom_gb)
    return cpu_available_gb, gpu_available_gb


def _servability_reason(
    entry: dict,
    *,
    cpu_available_gb: float,
    gpu_available_gb: "float | None",
) -> "str | None":
    """The unservable reason for one catalog entry, or None if servable.

    Single source of truth for boot validation AND the live request-path
    re-check (`revalidate_servability`), so the two verdicts never drift.
    Applies the sizing ladder (`_has_memory_data`) then a device-capacity
    check against the caller-supplied available pools.

    Returns:
      - "no memory estimate ..." when the model is not sizable at all
        (no annotation, no probe, no weights on disk).
      - "exceeds device capacity ..." when sized but it does not fit the
        device's available pool (or a cuda model on a host with no live
        VRAM info). This is a HARD stop: the gateway 503s without deferring
        to the director, because a model that does not fit even an empty
        working set can only make the director evict everything and 503.
      - None when sizable AND it fits.
    """
    has_data, sized_gb, device = _has_memory_data(entry)
    if not has_data:
        return "no memory estimate; run `muse models probe` to populate"
    # Resolve the manifest "auto"/"" convention to the concrete pool this
    # model loads on, mirroring the runtime select_device + the
    # LoadDirector: a GPU is present iff we have live VRAM info
    # (gpu_available_gb is not None). Without this, an auto-device model on
    # a GPU host would be sized against the (large) CPU pool and never
    # 503 even when it cannot fit VRAM -- deferring an impossible model
    # into the director's evict-everything-then-fail path.
    if device in ("auto", ""):
        device = "cuda" if gpu_available_gb is not None else "cpu"
    if device in ("cuda", "gpu"):
        if gpu_available_gb is None:
            return (
                "exceeds device capacity (no GPU info available; "
                "install nvidia-ml-py / pynvml or set memory budget)"
            )
        available_gb = gpu_available_gb
    else:
        available_gb = cpu_available_gb
    if sized_gb > available_gb:
        return (
            f"exceeds device capacity ({sized_gb:.1f} GB > "
            f"{available_gb:.1f} GB available on {device})"
        )
    return None


def validate_catalog_at_boot(
    state: SupervisorState,
    *,
    memory_probe: "Any | None" = None,
    gpu_headroom_gb: float = 1.0,
    cpu_headroom_gb: float = 2.0,
) -> None:
    """Walk the enabled catalog and stamp unservable_reasons.

    For each `enabled: true` row in the catalog:
      - If the row has neither `manifest.capabilities.memory_gb` nor
        `measurements.<device>.peak_bytes`, mark it
        "no memory estimate; run muse models probe".
      - If the row's declared memory_gb exceeds free at boot
        (live free minus headroom), mark it "exceeds device capacity".
      - GPU rows with no live VRAM info (pynvml unavailable) are
        treated as exceeding capacity until probe data lands.

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

    cpu_available_gb, gpu_available_gb = _available_pools(
        memory_probe,
        gpu_headroom_gb=gpu_headroom_gb,
        cpu_headroom_gb=cpu_headroom_gb,
    )

    for model_id, entry in catalog.items():
        if not entry.get("enabled", True):
            continue
        # Skip pre-worker entries; they cannot load anyway.
        if not entry.get("python_path"):
            continue

        reason = _servability_reason(
            entry,
            cpu_available_gb=cpu_available_gb,
            gpu_available_gb=gpu_available_gb,
        )
        if reason is not None:
            state.unservable_reasons[model_id] = reason


def revalidate_servability(
    state: SupervisorState,
    model_id: str,
    *,
    memory_probe: "Any | None" = None,
    gpu_headroom_gb: float = 1.0,
    cpu_headroom_gb: float = 2.0,
) -> str | None:
    """Re-derive one model's unservable verdict against the LIVE catalog.

    `validate_catalog_at_boot` stamps `state.unservable_reasons` once at
    boot. That snapshot goes stale two ways: a `muse models probe` (or a
    manifest edit, or weights simply landing on disk) makes a previously
    unsizable model sizable; or memory frees up so a model that did not fit
    at boot now does. This re-reads the (mtime-cached) catalog for ONE model
    and re-runs the SAME `_servability_reason` boot uses -- the estimate AND
    the device-capacity check against LIVE free memory -- then updates the
    stamp, so the gateway reflects reality WITHOUT a supervisor restart.

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
    cpu_available_gb, gpu_available_gb = _available_pools(
        memory_probe,
        gpu_headroom_gb=gpu_headroom_gb,
        cpu_headroom_gb=cpu_headroom_gb,
    )
    reason = _servability_reason(
        entry,
        cpu_available_gb=cpu_available_gb,
        gpu_available_gb=gpu_available_gb,
    )
    with state.lock:
        if reason is None:
            state.unservable_reasons.pop(model_id, None)
        else:
            state.unservable_reasons[model_id] = reason
        return reason


def backfill_manifest_memory(manifest: dict, model_id: str) -> dict:
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


def _init_telemetry(state: SupervisorState) -> None:
    """Boot-time telemetry wiring: store + recorder + log hub + sampler + prune.

    Called from `run_supervisor` when `telemetry.enabled` is true, after
    `state.stop_event` is set but before the gateway is built, so the
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
    store_path = Path(config.get("paths.catalog_dir")).expanduser() / "telemetry.db"
    store = TelemetryStore(store_path)
    init_recorder(store, enabled=True)

    log_buffer_kb = config.get("telemetry.log_buffer_kb")
    hub = LogHub(buffer_bytes=int(log_buffer_kb) * 1024)

    state.telemetry_store = store
    state.log_hub = hub

    sampler = Sampler(
        interval=float(config.get("telemetry.sample_interval_seconds")),
        loaded_fn=lambda: state.director.loaded,
        inflight_fn=lambda: len(getattr(state.director, "in_flight_loads", {}) or {}),
        stop_event=state.stop_event,
    )
    sampler.start()
    state.telemetry_sampler = sampler

    retention_days = config.get("telemetry.retention_days")

    def _prune_loop() -> None:
        while not state.stop_event.wait(3600):
            try:
                store.prune(time.time() - float(retention_days) * 86400)
            except Exception:
                logger.warning("telemetry prune failed", exc_info=True)

    t = threading.Thread(target=_prune_loop, daemon=True, name="muse-telemetry-prune")
    t.start()


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
    from muse.cli_impl.gateway import build_gateway

    state = SupervisorState(workers=[], device=device)
    state.director = _build_load_director(state)
    # Wire the director's capacity signal to the gateway capacity notifier
    # (spec 2026-07-08): a release-to-zero or a completed eviction fires
    # capacity_listener, waking queued capacity-waiters so they re-decide.
    state.director.capacity_listener = state.capacity_notifier.notify
    set_supervisor_state(state)

    # Validate the catalog. Models with no memory data or memory > free
    # at boot get a 503 reason stamped on state.unservable_reasons.
    validate_catalog_at_boot(state)
    if state.unservable_reasons:
        for mid, reason in sorted(state.unservable_reasons.items()):
            logger.warning("unservable model %r: %s", mid, reason)

    # The auto-restart monitor and the idle sweeper share state.stop_event
    # so a single Ctrl+C / SIGTERM unblocks both at once. Allocate a
    # fresh Event for this supervisor lifecycle (replacing the dataclass
    # default) so a re-entered run_supervisor in the same process always
    # gets a clean unset-state event.
    stop_event = threading.Event()
    state.stop_event = stop_event

    # The monitor thread reads `state.workers` (a live reference), so
    # workers spawned later via the director's enable_fn show up on the
    # next polling tick without extra coordination. Started always (not
    # gated on a non-empty worker list, since lazy load means workers
    # arrive later).
    # `state` is passed (not a captured `state.log_hub` value) because the
    # monitor thread starts before `_init_telemetry` (below) populates
    # `state.log_hub`; `_monitor_workers` reads `state.log_hub` live, at
    # restart time, so the restart path still forwards the hub once
    # telemetry finishes wiring up (see `_monitor_workers` docstring).
    monitor_thread = threading.Thread(
        target=_monitor_workers,
        args=(state.workers, stop_event),
        kwargs={"state": state},
        daemon=True,
        name="muse-monitor",
    )
    monitor_thread.start()
    logger.info(
        "auto-restart monitor running (interval=%.1fs, threshold=%d, budget=%d)",
        _MONITOR_INTERVAL, _FAILURE_THRESHOLD, _MAX_RESTARTS,
    )

    # Idle-timeout sweeper (v0.40.1). Per-model idle eviction runs on a
    # background thread that shares stop_event with the monitor; the
    # sweeper reads loaded-set entries via the director's public surface
    # and unloads anything past its `capabilities.idle_timeout_seconds`.
    sweep_interval = _resolve_idle_sweep_interval()
    # Global default idle timeout, applied to models that declare no
    # per-model capabilities.idle_timeout_seconds. The registry default
    # is 600.0s (v0.5x); an operator who wants the old "never idle-evict"
    # behavior sets MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS=0 (or negative)
    # explicitly -- the "<=0 disables" guard below still applies. A bad
    # / unparseable env value can no longer crash boot: the registry
    # itself warns and falls back to its default (see Config.get).
    _raw_default_idle = config.get("server.idle_timeout_seconds")
    default_idle_timeout: float | None = (
        _raw_default_idle if _raw_default_idle is not None and _raw_default_idle > 0 else None
    )
    sweeper = IdleSweeper(
        director=state.director,
        catalog_lookup=get_manifest,
        interval_seconds=sweep_interval,
        default_idle_timeout_seconds=default_idle_timeout,
        stop_event=stop_event,
    )
    sweeper_thread = sweeper.start()
    state.idle_sweeper = sweeper
    state.idle_sweeper_thread = sweeper_thread
    logger.info(
        "idle sweeper running (interval=%.1fs, default_idle_timeout=%s)",
        sweep_interval,
        f"{default_idle_timeout:.0f}s" if default_idle_timeout else "off",
    )

    # Telemetry (observability dashboard). Opt-out via
    # MUSE_TELEMETRY_ENABLED=false / telemetry.enabled: false. Wired
    # before build_gateway so the mounted dashboard router sees a
    # populated state.telemetry_store / state.log_hub on its first
    # request.
    if config.get("telemetry.enabled"):
        _init_telemetry(state)
        logger.info(
            "telemetry enabled (db=%s)",
            Path(config.get("paths.catalog_dir")).expanduser() / "telemetry.db",
        )

    try:
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
        # Tell the monitor + idle sweeper to stop BEFORE killing workers.
        # Otherwise the monitor could spawn a restart while we're
        # terminating processes, and the sweeper could try to evict a
        # model whose worker is mid-shutdown.
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
        if sweeper_thread is not None:
            sweeper_thread.join(timeout=5.0)
        # Telemetry teardown (symmetric with the store/sampler wiring in
        # _init_telemetry). Both attributes are None when telemetry.enabled
        # is False, so this is a no-op in that case. state.stop_event is
        # already shared with the sampler's loop (set above), so stop()
        # here is belt-and-suspenders + joins the sampler thread.
        sampler = getattr(state, "telemetry_sampler", None)
        if sampler is not None:
            try:
                sampler.stop()
            except Exception:
                logger.warning("telemetry sampler stop failed", exc_info=True)
        store = getattr(state, "telemetry_store", None)
        if store is not None:
            # Stop the recorder's flush thread (and drain its final batch
            # into the store) BEFORE closing the store. reset_recorder()
            # -> TelemetryRecorder.stop() joins the flush thread and then
            # does one last flush() against the still-open store; closing
            # the store first would let a subsequent periodic flush tick
            # call insert_many on a closed sqlite connection, logging
            # "flush failed" noise on every shutdown.
            try:
                reset_recorder()
            except Exception:
                logger.warning("telemetry recorder stop failed", exc_info=True)
            try:
                store.close()
            except Exception:
                logger.warning("telemetry store close failed", exc_info=True)
        # Whatever workers were loaded by the director get torn down here.
        # Empty list is a no-op.
        _shutdown_workers(state.workers)
        clear_supervisor_state()
        # Best-effort: join admin job threads so a Ctrl+C during a pull
        # doesn't leave dangling daemons. Import lazily to keep the
        # supervisor's startup path free of admin concerns.
        try:
            from muse.admin.jobs import get_default_store
            get_default_store().shutdown()
        except Exception as e:  # noqa: BLE001
            logger.warning("admin job-store shutdown failed: %s", e)
    return 0
