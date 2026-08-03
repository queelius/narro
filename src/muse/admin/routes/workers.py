"""Worker introspection and control routes.

Endpoints (under /v1/admin):

  GET  /workers                  list all workers + their state
  POST /workers/{port}/restart   SIGTERM by port; monitor handles bringup

Worker fields surfaced: port, models, pid, uptime_seconds, restart_count,
status. PID is read from spec.process.pid; uptime from monotonic delta
since last_spawn_at. Both are `null` when the process hasn't been spawned
yet (rare; the restart path may briefly pass through a process=None state).
"""
from __future__ import annotations

import signal
import time

from fastapi import APIRouter

from muse.cli_impl.supervisor import (
    _signal_worker_process,
    get_supervisor_state,
)
from muse.core.errors import error_response


def build_workers_router() -> APIRouter:
    router = APIRouter()

    @router.get("/workers")
    def list_workers():
        state = get_supervisor_state()
        with state.lock:
            now = time.monotonic()
            workers = []
            for spec in state.workers:
                with spec.process_lock:
                    process = spec.process
                    pid = getattr(process, "pid", None) if process else None
                uptime = (
                    max(0.0, now - spec.last_spawn_at)
                    if spec.last_spawn_at else None
                )
                workers.append({
                    "port": spec.port,
                    "models": list(spec.models),
                    "pid": pid if isinstance(pid, int) else None,
                    "uptime_seconds": uptime,
                    "restart_count": spec.restart_count,
                    "status": spec.status,
                })
        return {"workers": workers}

    @router.post("/workers/{port}/restart")
    def restart_worker(port: int):
        state = get_supervisor_state()
        with state.lock:
            spec = next((w for w in state.workers if w.port == port), None)
            if spec is None:
                return error_response(
                    404, "worker_not_found", f"no worker on port {port}",
                )
            with spec.process_lock:
                proc = spec.process
                if proc is None:
                    return error_response(
                        409, "worker_not_running",
                        f"worker on port {port} is not currently running",
                    )
                try:
                    signalled = _signal_worker_process(proc, signal.SIGTERM)
                except Exception as e:  # noqa: BLE001
                    return error_response(
                        500, "terminate_failed",
                        f"failed to SIGTERM worker on port {port}: {e}",
                    )
                if not signalled:
                    return error_response(
                        409, "worker_not_running",
                        f"worker on port {port} is not currently running",
                    )
                # Reset the restart budget only after the exact live process
                # generation was signalled. This is the operator escape hatch
                # for a worker that exhausted the automatic restart budget.
                spec.restart_count = 0
                spec.failure_count = 0
        return {
            "port": port,
            "signal": "SIGTERM",
            "message": "auto-restart monitor will respawn the worker",
            "restart_count_reset": True,
        }

    return router
