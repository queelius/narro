"""`muse _worker` implementation: runs ONE worker (optionally hosting
multiple models from the same venv) and starts uvicorn.

Invoked by the supervisor (`muse serve`) via subprocess:
    <venv>/bin/python -m muse.cli _worker --port 9001 --model soprano-80m

Can also be run standalone for debugging. Not advertised in top-level help.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
from contextlib import ExitStack
from pathlib import Path

from muse.cli_impl.serve_util import run_uvicorn
from muse.core import config
from muse.core.catalog import (
    _model_resource_lease,
    get_manifest,
    is_pulled,
    known_models,
    load_backend,
)
from muse.core.discovery import discover_modalities
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app

log = logging.getLogger(__name__)

_SUPERVISOR_PID_ENV = "MUSE_SUPERVISOR_PID"
_WORKER_NONCE_ENV = "MUSE_WORKER_NONCE"
_WORKER_NONCE_HEADER = "X-Muse-Worker-Nonce"


def _watch_parent(
    *, expected_parent_pid: int, poll_interval: float = 1.0,
    stop_event: threading.Event | None = None,
) -> None:
    """Terminate this worker if it is reparented away from its supervisor."""
    stopper = stop_event or threading.Event()
    while not stopper.is_set():
        if os.getppid() != expected_parent_pid:
            log.warning(
                "supervisor pid %d disappeared; terminating orphan worker",
                expected_parent_pid,
            )
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except OSError:
                pass
            return
        stopper.wait(poll_interval)


def _start_parent_watchdog(expected_parent_pid: int) -> None:
    """Start the supervisor parent-death guard for a managed worker."""
    threading.Thread(
        target=_watch_parent,
        kwargs={"expected_parent_pid": expected_parent_pid},
        daemon=True,
        name="muse-parent-watchdog",
    ).start()


def _bundled_modalities_dir() -> Path:
    """Directory containing muse's built-in modality packages."""
    # worker.py sits at src/muse/cli_impl/worker.py; parents[1] is src/muse/.
    return Path(__file__).resolve().parents[1] / "modalities"


def _env_modalities_dir() -> Path | None:
    """Optional extra modalities dir from `$MUSE_MODALITIES_DIR` env var.

    Intended as an escape hatch for power users experimenting with new
    modality contracts, not a normal extension surface. Most users
    should extend via model scripts instead (see $MUSE_MODELS_DIR).
    """
    env = config.get("paths.modalities_dir")
    return Path(env) if env else None


def _modality_dirs() -> list[Path]:
    """Scan order for modality discovery: bundled first, then env override.

    First-found-wins on MODALITY tag collision, so bundled modalities
    shadow env-dir entries that declare the same MIME tag.
    """
    dirs = [_bundled_modalities_dir()]
    env = _env_modalities_dir()
    if env is not None:
        dirs.append(env)
    return dirs


def run_worker(*, host: str, port: int, models: list[str], device: str) -> int:
    """Load the specified models into a registry and run uvicorn.

    `models` is the exact set of model-ids to load into this process.
    The supervisor decides which models share a worker; the worker just
    loads what it's told.

    Fail-fast contract: if any assigned model fails to load for any
    reason (unknown id, not pulled, backend import error), the worker
    returns exit code 2 BEFORE starting uvicorn. A partial worker
    masquerading as healthy is worse than a crashed one: the
    supervisor's restart-then-mark-dead machinery only engages when
    the worker process actually exits, and /health only reports
    'degraded' when the supervisor sees a worker unreachable or dead.

    `models == []` is a valid test configuration (empty-registry
    router mounting smoke test); it does not trigger the fail-fast.
    """
    with ExitStack() as leases:
        # Sorted acquisition prevents two multi-model workers from deadlocking
        # if a future placement strategy gives them overlapping assignments.
        for model_id in sorted(set(models)):
            leases.enter_context(_model_resource_lease(model_id, wait=True))
        return _run_worker_with_leases(
            host=host,
            port=port,
            models=models,
            device=device,
        )


def _run_worker_with_leases(
    *, host: str, port: int, models: list[str], device: str,
) -> int:
    """Load and serve after acquiring every assigned model resource lease."""
    raw_supervisor_pid = os.environ.get(_SUPERVISOR_PID_ENV)
    if raw_supervisor_pid:
        try:
            supervisor_pid = int(raw_supervisor_pid)
        except ValueError:
            log.warning("ignoring invalid %s=%r", _SUPERVISOR_PID_ENV, raw_supervisor_pid)
        else:
            if supervisor_pid > 0:
                _start_parent_watchdog(supervisor_pid)

    registry = ModalityRegistry()
    routers: dict = {}
    failures: list[str] = []

    catalog = known_models()
    to_load = [m for m in models if m in catalog]
    unknown = [m for m in models if m not in catalog]
    if unknown:
        log.warning("ignoring unknown models: %s", unknown)
        failures.extend(unknown)

    for model_id in to_load:
        if not is_pulled(model_id):
            log.error("model %s not pulled; worker cannot host it", model_id)
            failures.append(model_id)
            continue
        entry = catalog[model_id]
        log.info("loading %s (%s)", model_id, entry.modality)
        try:
            backend = load_backend(model_id, device=device)
        except Exception as e:
            log.error("failed to load %s: %s", model_id, e)
            failures.append(model_id)
            continue
        manifest = get_manifest(model_id)
        registry.register(entry.modality, backend, manifest=manifest)

    if failures:
        log.error(
            "worker exiting (exit 2): %d/%d assigned models failed to load: %s",
            len(failures), len(models), failures,
        )
        return 2

    # Always mount every discovered modality router so empty-registry
    # requests get the OpenAI envelope rather than FastAPI's default
    # {"detail": "Not Found"}. Adding a new modality requires zero
    # changes here: drop a subpackage under src/muse/modalities/ that
    # exports MODALITY + build_router, and discovery picks it up.
    for tag, build_router in discover_modalities(_modality_dirs()).items():
        log.info("mounting modality router for %s", tag)
        routers[tag] = build_router(registry)

    app = create_app(registry=registry, routers=routers)
    worker_nonce = os.environ.get(_WORKER_NONCE_ENV)
    if worker_nonce:
        @app.middleware("http")
        async def _worker_identity_header(request, call_next):
            response = await call_next(request)
            response.headers[_WORKER_NONCE_HEADER] = worker_nonce
            return response
    # run_uvicorn sets a bounded timeout_graceful_shutdown so a standalone
    # `muse _worker` process (or one whose supervisor SIGTERMs it) exits
    # promptly on Ctrl-C instead of hanging on an in-flight connection.
    run_uvicorn(app, host=host, port=port)
    return 0
