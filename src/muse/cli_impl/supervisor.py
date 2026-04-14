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
import time
from dataclasses import dataclass, field

import httpx
import uvicorn

from muse.core.catalog import _read_catalog
from muse.core.venv import find_free_port

logger = logging.getLogger(__name__)


@dataclass
class WorkerSpec:
    """Everything needed to spawn one worker subprocess."""
    models: list[str]
    python_path: str
    port: int
    # Populated after subprocess.Popen in Task E2
    process: object = field(default=None)


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

    Mutates spec.process with the Popen handle so the supervisor can
    manage the subprocess later (wait_for_ready, shutdown).
    """
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

    Plans workers from catalog, spawns them, waits for ready, then runs
    the gateway on (host, port). Guarantees worker cleanup on exit.
    """
    from muse.cli_impl.gateway import WorkerRoute, build_gateway

    specs = plan_workers()
    if not specs:
        logger.warning(
            "no pulled models with a venv - server will start empty. "
            "Pull a model first: `muse pull <model-id>`"
        )

    try:
        for spec in specs:
            spawn_worker(spec, device=device)

        for spec in specs:
            logger.info("waiting for worker on port %d (%s)", spec.port, spec.models)
            wait_for_ready(port=spec.port)

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
        _shutdown_workers(specs)
    return 0
