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
from dataclasses import dataclass, field

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
