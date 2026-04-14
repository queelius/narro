"""`muse serve` supervisor: orchestrate workers + run gateway.

Responsibilities (across E1-E4):
  1. Read catalog (E1)
  2. Group models by venv (same python_path = same worker) (E1)
  3. Allocate a local port per worker (E1)
  4. Spawn worker subprocesses (E2)
  5. Wait for each worker's /health to become responsive (E2)
  6. Build gateway routes + run gateway uvicorn (E3)
  7. On shutdown: SIGTERM workers, wait for exit (E3)

This task (E1) implements steps 1-3 only; the rest land in E2-E3.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field

import httpx
import uvicorn

from muse.core.catalog import _read_catalog
from muse.core.venv import find_free_port

logger = logging.getLogger(__name__)


@dataclass
class WorkerSpec:
    """Everything needed to spawn and supervise one worker subprocess.

    Fields mutated by the monitor thread (after startup):
      - process: replaced on restart
      - restart_count: total restart attempts (caps at _MAX_RESTARTS)
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


def plan_workers(port_start: int = 9001, port_end: int = 9999) -> list[WorkerSpec]:
    """Read catalog, group by venv, allocate ports.

    Returns one WorkerSpec per unique venv (identified by python_path).
    Pre-worker catalog entries (missing python_path) are logged + skipped.
    """
    catalog = _read_catalog()

    # Group by python_path. Preserve insertion order for determinism.
    groups: dict[str, list[str]] = {}
    for model_id, entry in catalog.items():
        python = entry.get("python_path")
        if not python:
            logger.warning(
                "skipping pre-worker catalog entry %r - no python_path; "
                "re-run `muse pull %s` to create its venv",
                model_id, model_id,
            )
            continue
        # Default True covers legacy entries without the field
        # (also backfilled by _read_catalog's setdefault)
        if not entry.get("enabled", True):
            logger.info(
                "skipping disabled model %r (use `muse models enable %s` to re-enable)",
                model_id, model_id,
            )
            continue
        groups.setdefault(python, []).append(model_id)

    specs: list[WorkerSpec] = []
    used_ports: set[int] = set()
    for python_path, models in groups.items():
        # Allocate a free port, avoiding collisions with ports already
        # assigned to earlier specs in this planning pass.
        while True:
            port = find_free_port(start=port_start, end=port_end)
            if port not in used_ports:
                used_ports.add(port)
                break
            port_start = port + 1
        specs.append(WorkerSpec(
            models=sorted(models),
            python_path=python_path,
            port=port,
        ))
    return specs


def spawn_worker(spec: WorkerSpec, *, device: str) -> None:
    """Start a worker subprocess using its venv's Python.

    Persists `device` onto the spec so the monitor thread can respawn
    with the same settings on restart. Records last_spawn_at for the
    backoff timer in _attempt_restart.
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
    spec.process = subprocess.Popen(cmd)
    spec.last_spawn_at = time.monotonic()


def wait_for_ready(
    *, port: int, timeout: float = 60.0, poll_interval: float = 0.5,
) -> None:
    """Block until http://127.0.0.1:<port>/health returns 200, or timeout.

    Raises TimeoutError if the worker never becomes ready. Polls with
    short sleeps so slow workers (big model loads) still get through.
    """
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError as e:
            last_err = e
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
) -> None:
    """Terminate existing process if alive, wait backoff, respawn.

    Mutates spec.process, spec.restart_count, spec.failure_count, spec.status.
    Marks spec.status = "dead" if restart_count reaches max_restarts.
    Returns early if stop_event fires during backoff.
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

    # Terminate existing process if still alive (best-effort)
    if spec.process is not None and spec.process.poll() is None:
        try:
            spec.process.terminate()
            try:
                spec.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                spec.process.kill()
        except Exception as e:
            logger.warning("worker on port %d: terminate failed: %s", spec.port, e)

    # Respawn. Always bump restart_count so we can't loop forever.
    spec.restart_count += 1
    try:
        spawn_worker(spec, device=spec.device)
        wait_for_ready(port=spec.port, timeout=ready_timeout)
        spec.failure_count = 0
        spec.status = "running"
        logger.info("worker on port %d: successfully restarted", spec.port)
    except (subprocess.SubprocessError, TimeoutError) as e:
        logger.error("worker on port %d: restart failed: %s", spec.port, e)
        spec.status = "unhealthy"


def _monitor_workers(
    specs: list[WorkerSpec],
    stop_event: "threading.Event",
    *,
    interval: float = _MONITOR_INTERVAL,
    failure_threshold: int = _FAILURE_THRESHOLD,
    max_restarts: int = _MAX_RESTARTS,
) -> None:
    """Poll each worker; restart after `failure_threshold` consecutive failures.

    Exits when stop_event is set. Called from the monitor daemon thread
    started by run_supervisor (Task B4).
    """
    while not stop_event.is_set():
        for spec in specs:
            if stop_event.is_set():
                return
            if spec.status == "dead":
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
                _attempt_restart(spec, stop_event=stop_event, max_restarts=max_restarts)

        # Sleep with early-exit if stop_event fires
        if stop_event.wait(interval):
            return


def _shutdown_workers(specs: list[WorkerSpec], grace: float = 5.0) -> None:
    """SIGTERM all workers; SIGKILL any that don't exit within `grace` seconds."""
    for spec in specs:
        if spec.process is None:
            continue
        try:
            spec.process.terminate()
        except Exception as e:
            logger.warning("failed to SIGTERM worker on port %d: %s", spec.port, e)

    for spec in specs:
        if spec.process is None:
            continue
        try:
            spec.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            logger.warning("worker on port %d did not exit in %ds; killing", spec.port, grace)
            spec.process.kill()
        except Exception as e:
            logger.warning("error waiting for worker on port %d: %s", spec.port, e)


def run_supervisor(*, host: str, port: int, device: str) -> int:
    """Entry point for `muse serve`.

    Plans workers from catalog, spawns them, waits for ready, then starts
    the auto-restart monitor thread + gateway. Guarantees clean shutdown
    of workers and monitor on exit.
    """
    from muse.cli_impl.gateway import WorkerRoute, build_gateway

    specs = plan_workers()
    if not specs:
        logger.warning(
            "no pulled models with a venv - server will start empty. "
            "Pull a model first: `muse pull <model-id>`"
        )

    stop_event = threading.Event()
    monitor_thread: threading.Thread | None = None

    try:
        for spec in specs:
            spawn_worker(spec, device=device)

        for spec in specs:
            logger.info("waiting for worker on port %d (%s)", spec.port, spec.models)
            wait_for_ready(port=spec.port)
            spec.status = "running"

        # Start the auto-restart monitor AFTER all workers are ready so
        # the initial wait_for_ready isn't racing with the monitor's own
        # readiness tracking.
        if specs:
            monitor_thread = threading.Thread(
                target=_monitor_workers,
                args=(specs, stop_event),
                daemon=True,
                name="muse-monitor",
            )
            monitor_thread.start()
            logger.info(
                "auto-restart monitor running (interval=%.1fs, threshold=%d, budget=%d)",
                _MONITOR_INTERVAL, _FAILURE_THRESHOLD, _MAX_RESTARTS,
            )

        routes: list[WorkerRoute] = []
        for spec in specs:
            worker_url = f"http://127.0.0.1:{spec.port}"
            for m in spec.models:
                routes.append(WorkerRoute(model_id=m, worker_url=worker_url))
        app = build_gateway(routes)

        logger.info("starting gateway on %s:%d", host, port)
        uvicorn.run(app, host=host, port=port, log_config=None)
    except KeyboardInterrupt:
        logger.info("shutting down (SIGINT)")
    finally:
        # Tell the monitor to stop BEFORE killing workers. Otherwise the
        # monitor could spawn a restart while we're terminating processes.
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
        _shutdown_workers(specs)
    return 0
