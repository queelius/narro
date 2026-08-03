"""Orchestrate admin operations against the supervisor.

Each operation reads/mutates SupervisorState (workers, device) under the
state's RLock. Async operations (enable, pull, probe) spawn a daemon
thread tracked by the JobStore; the thread updates the Job as it
progresses. Sync operations (disable, remove) return their result
directly and raise OperationError on user-facing failures.

Subprocess-based ops (pull, probe) shell out to `muse pull <id>` and
`muse models probe <id>` respectively. This keeps clean isolation: the
gateway never imports torch / diffusers / llama-cpp; per-model venvs do.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
from typing import Any, Callable

from muse.admin.jobs import Job, JobStore
from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    _shutdown_workers,
    backfill_manifest_memory,
    spawn_worker,
    wait_for_ready,
)
from muse.core.catalog import (
    _read_catalog,
    get_manifest,
    is_pulled,
    known_models,
    remove as catalog_remove,
    set_enabled,
)
from muse.core.venv import find_free_port

logger = logging.getLogger(__name__)


# Worker statuses that cannot serve traffic and will not recover on their
# own: a "dead" worker exhausted its restart budget (or failed its initial
# spawn); an "unhealthy" one is failing health checks. Both linger in
# state.workers with job_id=None (the monitor never removes dead specs), so
# the "already running / already loaded" fast paths in enable_model and
# load_model_into_worker must NOT treat them as serviceable: they are
# dropped so the caller falls through to a fresh spawn (H1).
_UNSERVICEABLE_STATUSES = ("dead", "unhealthy")


def _drop_unserviceable(
    state: SupervisorState, model_id: str,
) -> tuple[WorkerSpec | None, WorkerSpec | None]:
    """Resolve the spec listing model_id into (serviceable, dropped).

    If the matching spec is dead/unhealthy, remove it from state.workers and
    return it as `dropped` (with `serviceable=None`) so the caller respawns
    instead of claiming a stale port. A dropped spec may still own a LIVE
    subprocess (an 'unhealthy' spec is one whose spawn succeeded but whose
    wait_for_ready timed out, so its worker is alive and holding VRAM); the
    caller MUST _shutdown_workers([dropped]) OUTSIDE state.lock to reap it,
    else it orphans and leaks memory. Shutdown is not done here because this
    runs under state.lock and SIGTERM+grace would block the lock for up to
    5s. Must be called while holding state.lock.
    """
    existing = next(
        (s for s in state.workers if model_id in s.models), None,
    )
    if existing is not None and existing.status in _UNSERVICEABLE_STATUSES:
        state.workers.remove(existing)
        return None, existing
    return existing, None


def _pick_free_port(
    state: SupervisorState, *, start: int = 9001, end: int = 9999,
) -> int:
    """find_free_port that also skips ports already reserved by specs in
    state.workers (M1).

    find_free_port probes the OS: it returns a port that is unbound *right
    now*, but a pending spec's worker may not have called bind() yet, so its
    reserved port still looks free. Two concurrent cold loads of different
    models would then both pick it; the loser fails to bind and
    wait_for_ready times out (~120s) despite ~999 free ports. Excluding the
    ports already held by pending/live specs closes that window. MUST be
    called while holding state.lock so the reserved-port snapshot is
    consistent with the append that follows.
    """
    used = {s.port for s in state.workers if s.port}
    while True:
        port = find_free_port(start=start, end=end)
        if port not in used:
            return port
        start = port + 1


class OperationError(Exception):
    """Raised by sync operations on user-facing failures.

    Routes catch this and translate it into an HTTP envelope with the
    bound (status, code, message) without leaking internals. Async
    operations write the same fields into the Job's `error` instead.
    """

    def __init__(self, code: str, message: str, status: int = 400,
                 retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        # Spec 2026-07-08: True marks a TRANSIENT capacity failure (in-use
        # models will release memory); the gateway parks and retries these
        # instead of surfacing the 503. False (default) = surface as today.
        self.retryable = retryable


def _ensure_server_running(state: SupervisorState) -> None:
    """Reject new worker starts once gateway shutdown has begun."""
    if state.stop_event.is_set():
        raise OperationError(
            "server_shutting_down",
            "server shutdown is in progress; worker startup cancelled",
            status=503,
        )


def find_worker_for_model(state: SupervisorState, model_id: str) -> WorkerSpec | None:
    """Return the WorkerSpec hosting `model_id`, or None if no worker hosts it.

    Acquires the state's RLock for the duration of the iteration so
    concurrent enable/disable can't slip a model in or out mid-search.
    """
    with state.lock:
        for spec in state.workers:
            if model_id in spec.models:
                return spec
    return None


def enable_model(
    model_id: str,
    *,
    state: SupervisorState,
    store: JobStore,
    job: Job,
) -> None:
    """Async operation: ensure `model_id` is loaded in some worker.

    Plan-then-execute: state.lock is held only across state.workers
    mutations + planning. The slow steps (spawn_worker + wait_for_ready,
    or _restart_worker_inplace) run outside the lock so other admin
    endpoints don't block for the full 120s readiness window.

    Concurrent enables for the same model coalesce onto the first
    caller's job_id: the second caller observes the existing pending
    spec and returns its job_id rather than spawning a duplicate
    worker. Both poll the same JobStore entry as the first caller's
    spawn drives.

    Terminal paths:
      1. already_running: model is hosted in a worker with status="running"
         (or status="pending" but no job_id, the legacy / test shape).
         Result: loaded=True, spawned_new=False.
      2. coalesce: model is in an in-flight pending spec with a job_id.
         Result: loaded=False, spawned_new=False, coalesced_job_id set.
      3. restart_sibling: a venv-group sibling exists and is running;
         join it via restart-in-place. Slow step runs outside the lock.
      4. spawn_new: brand-new worker for this model's python_path.
         Slow step runs outside the lock.
    """
    store.update(job.job_id, state="running")
    try:
        _ensure_server_running(state)
        catalog_known = known_models()
        if model_id not in catalog_known:
            raise OperationError(
                "model_not_found", f"unknown model {model_id!r}", status=404,
            )
        if not is_pulled(model_id):
            raise OperationError(
                "model_not_pulled",
                f"model {model_id!r} not pulled; run pull first",
                status=409,
            )

        catalog = _read_catalog()
        python_path = catalog[model_id].get("python_path")
        if not python_path:
            raise OperationError(
                "missing_venv",
                f"model {model_id!r} has no per-model venv on record",
                status=409,
            )

        # Phase 1: plan + claim under a brief lock.
        plan: str | None = None
        spec_ref: WorkerSpec | None = None
        coalesced_job_id: str | None = None
        dropped: WorkerSpec | None = None

        with state.lock:
            set_enabled(model_id, True)

            existing, dropped = _drop_unserviceable(state, model_id)
            if existing is not None:
                if existing.status == "running" or existing.job_id is None:
                    # Either truly running, or a legacy / test-built spec
                    # without a job_id. Both treated as "already loaded".
                    plan = "already_running"
                    spec_ref = existing
                else:
                    # In-flight on someone else's job: coalesce.
                    plan = "coalesce"
                    spec_ref = existing
                    coalesced_job_id = existing.job_id
            else:
                sibling = next(
                    (s for s in state.workers if s.python_path == python_path),
                    None,
                )
                if sibling is not None and (
                    sibling.status == "running" or sibling.job_id is None
                ):
                    # Restart-in-place candidate: either truly running,
                    # or a legacy / test-built spec with no in-flight
                    # job. Claim it.
                    sibling.models = sorted(set(sibling.models) | {model_id})
                    sibling.status = "restarting"
                    sibling.job_id = job.job_id
                    plan = "restart_sibling"
                    spec_ref = sibling
                elif sibling is not None and sibling.job_id is not None:
                    # Sibling already mid-restart for someone else: append
                    # our model so the in-flight restart picks it up,
                    # then coalesce onto that job.
                    sibling.models = sorted(set(sibling.models) | {model_id})
                    plan = "coalesce"
                    spec_ref = sibling
                    coalesced_job_id = sibling.job_id
                else:
                    new_port = _pick_free_port(state)
                    new_spec = WorkerSpec(
                        models=[model_id],
                        python_path=python_path,
                        port=new_port,
                        device=state.device,
                    )
                    new_spec.status = "pending"
                    new_spec.job_id = job.job_id
                    state.workers.append(new_spec)
                    plan = "spawn_new"
                    spec_ref = new_spec

        # Phase 2: execute outside the lock. The slow steps (spawn,
        # wait_for_ready, restart-in-place) run here so other admin
        # endpoints don't block. Status flips happen under brief
        # reacquisitions of state.lock.
        #
        # Reap a dropped dead/unhealthy worker first (outside the lock): it
        # may still own a live subprocess holding VRAM, so terminate it
        # before spawning the replacement (H1 follow-up). No-op when its
        # process already exited or was never set.
        if dropped is not None:
            _shutdown_workers([dropped])

        assert spec_ref is not None and plan is not None  # for type checker

        if plan == "already_running":
            store.update(
                job.job_id, state="done",
                result={
                    "model_id": model_id,
                    "worker_port": spec_ref.port,
                    "loaded": True,
                    "spawned_new": False,
                },
            )
            return

        if plan == "coalesce":
            store.update(
                job.job_id, state="done",
                result={
                    "model_id": model_id,
                    "worker_port": spec_ref.port,
                    "loaded": False,
                    "spawned_new": False,
                    "coalesced_job_id": coalesced_job_id,
                },
            )
            return

        if plan == "restart_sibling":
            try:
                _restart_worker_inplace(
                    spec_ref,
                    device=state.device,
                    log_hub=getattr(state, "log_hub", None),
                    stop_event=state.stop_event,
                )
            except Exception:
                with state.lock:
                    spec_ref.status = "dead"
                    spec_ref.job_id = None
                raise
            with state.lock:
                spec_ref.job_id = None
            store.update(
                job.job_id, state="done",
                result={
                    "model_id": model_id,
                    "worker_port": spec_ref.port,
                    "loaded": True,
                    "spawned_new": False,
                },
            )
            return

        if plan == "spawn_new":
            try:
                _ensure_server_running(state)
                spawn_worker(
                    spec_ref,
                    device=state.device,
                    log_hub=getattr(state, "log_hub", None),
                )
                wait_for_ready(
                    port=spec_ref.port,
                    timeout=120.0,
                    stop_event=state.stop_event,
                )
            except Exception:
                _shutdown_workers([spec_ref])
                with state.lock:
                    spec_ref.status = "dead"
                    spec_ref.job_id = None
                _ensure_server_running(state)
                raise
            with state.lock:
                spec_ref.status = "running"
                spec_ref.job_id = None
            store.update(
                job.job_id, state="done",
                result={
                    "model_id": model_id,
                    "worker_port": spec_ref.port,
                    "loaded": True,
                    "spawned_new": True,
                },
            )
            return
    except OperationError as e:
        store.update(job.job_id, state="failed", error=e.message)
    except Exception as e:  # noqa: BLE001
        logger.exception("enable_model failed")
        store.update(job.job_id, state="failed", error=str(e))


def load_model_into_worker(model_id: str, *, state: SupervisorState) -> int:
    """Sync operation: spawn a worker for `model_id` and return its port.

    Lazy-load companion to `enable_model`. The crucial difference: this
    does NOT call `set_enabled(model_id, True)`. Lazy load is
    "give me a worker for this model right now"; the catalog's
    `enabled` flag is the persistent "this model is in service" state,
    which is orthogonal. v0.40.0 decouples those two concepts so a
    user can have 20 enabled models in their catalog and only the few
    actually under traffic occupy worker memory.

    The LoadDirector's `enable_fn` callable is bound to this operation.
    It serves as the "load this model now" hook the director invokes
    during the load phase of `acquire`.

    Three terminal paths:
      1. already_running: model already hosted by a running worker;
         return that worker's port (the director's hot-acquire path
         normally short-circuits this, but races during boot or after
         admin-triggered enable can land here).
      2. restart_sibling: a venv-group sibling exists and is running;
         the existing `_restart_worker_inplace` path joins it.
      3. spawn_new: brand-new worker for this model's python_path.

    Concurrency contract (mirrors `enable_model`): the in-flight
    pending spec is stamped with `spec.job_id = "director-load-<id>"`
    before the slow spawn / restart phase, and cleared on success.
    The auto-restart monitor skips specs whose `job_id` is non-None,
    so it cannot race the director-driven cold load (which can take
    10-60s for real models, far longer than the monitor's 5s tick).

    Raises OperationError on user-facing failures (model not found,
    not pulled, no venv on record). Other exceptions propagate to the
    director, which cleans up its in-flight Event and re-raises.
    """
    _ensure_server_running(state)
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise OperationError(
            "model_not_found", f"unknown model {model_id!r}", status=404,
        )
    if not is_pulled(model_id):
        raise OperationError(
            "model_not_pulled",
            f"model {model_id!r} not pulled; run pull first",
            status=409,
        )

    catalog = _read_catalog()
    python_path = catalog[model_id].get("python_path")
    if not python_path:
        raise OperationError(
            "missing_venv",
            f"model {model_id!r} has no per-model venv on record",
            status=409,
        )

    # Sentinel that marks the spec as "owned by a director-driven load
    # in flight." The monitor skips specs with non-None job_id, so this
    # protects the slow spawn window from a duplicate restart attempt.
    job_sentinel = f"director-load-{model_id}"

    # Phase 1: plan + claim under a brief lock.
    plan: str | None = None
    spec_ref: WorkerSpec | None = None
    dropped: WorkerSpec | None = None

    with state.lock:
        existing, dropped = _drop_unserviceable(state, model_id)
        if existing is not None and (
            existing.status == "running" or existing.job_id is None
        ):
            # Already loaded; return the existing port.
            plan = "already_running"
            spec_ref = existing
        else:
            sibling = next(
                (s for s in state.workers
                 if s.python_path == python_path and s.status == "running"),
                None,
            )
            if sibling is not None:
                # Join the sibling's venv group via restart-in-place.
                sibling.models = sorted(set(sibling.models) | {model_id})
                sibling.status = "restarting"
                sibling.job_id = job_sentinel
                plan = "restart_sibling"
                spec_ref = sibling
            else:
                new_port = _pick_free_port(state)
                new_spec = WorkerSpec(
                    models=[model_id],
                    python_path=python_path,
                    port=new_port,
                    device=state.device,
                )
                new_spec.status = "pending"
                new_spec.job_id = job_sentinel
                state.workers.append(new_spec)
                plan = "spawn_new"
                spec_ref = new_spec

    # Phase 2: execute outside the lock (the slow path).
    # Reap a dropped dead/unhealthy worker first: it may still own a live
    # subprocess holding VRAM, so terminate it before spawning the
    # replacement (H1 follow-up). No-op if its process already exited.
    if dropped is not None:
        _shutdown_workers([dropped])

    assert spec_ref is not None and plan is not None  # type checker

    if plan == "already_running":
        return spec_ref.port

    if plan == "restart_sibling":
        try:
            _restart_worker_inplace(
                spec_ref,
                device=state.device,
                log_hub=getattr(state, "log_hub", None),
                stop_event=state.stop_event,
            )
        except Exception:
            with state.lock:
                spec_ref.status = "dead"
                spec_ref.job_id = None
            raise
        with state.lock:
            spec_ref.job_id = None
        return spec_ref.port

    if plan == "spawn_new":
        try:
            _ensure_server_running(state)
            spawn_worker(
                spec_ref,
                device=state.device,
                log_hub=getattr(state, "log_hub", None),
            )
            wait_for_ready(
                port=spec_ref.port,
                timeout=120.0,
                stop_event=state.stop_event,
            )
        except Exception:
            _shutdown_workers([spec_ref])
            with state.lock:
                spec_ref.status = "dead"
                spec_ref.job_id = None
            _ensure_server_running(state)
            raise
        with state.lock:
            spec_ref.status = "running"
            spec_ref.job_id = None
        return spec_ref.port

    # Unreachable; the assert above guarantees plan is set.
    raise OperationError("unreachable", "load_model_into_worker fell through")


def unload_model_from_worker(model_id: str, *, state: SupervisorState) -> None:
    """Sync operation: drop `model_id` from its worker without disabling.

    Lazy-load companion to `disable_model`. Crucial difference: does
    NOT call `set_enabled(model_id, False)`. The catalog stays "in
    service"; this just frees the memory slot so the director can
    load another model.

    Plan-then-execute (mirrors `enable_model`): under state.lock we
    pop the spec / mutate the model list and stamp `job_id` so the
    monitor leaves the spec alone during the slow phase. The slow
    steps (`_shutdown_workers` or `_restart_worker_inplace`, each a
    multi-second subprocess wait or spawn cycle) run OUTSIDE the lock
    so concurrent admin reads / hot-acquires never block on us.

    Three paths:
      1. model_id not loaded in any worker: no-op.
      2. model_id is the only model in a worker: pop the spec, then
         terminate the worker (lock released for SIGTERM + grace).
      3. model_id is one of several in a worker (venv-group sibling):
         claim the spec via job_id, then restart-in-place with the
         reduced model list (lock released for spawn + readiness wait).

    On path (2), state.workers is mutated in place via
    `state.workers.remove(spec)`. Rebinding the attribute would
    desynchronize the auto-restart monitor thread, which captured the
    original list reference at supervisor boot.
    """
    spec_to_shutdown: WorkerSpec | None = None
    spec_to_restart: WorkerSpec | None = None

    # Phase 1: plan + claim under a brief lock.
    with state.lock:
        spec = find_worker_for_model(state, model_id)
        if spec is None:
            return

        spec.models = [m for m in spec.models if m != model_id]
        if not spec.models:
            # Stamp job_id BEFORE popping so a monitor tick that
            # snapshotted this spec earlier (see _monitor_workers'
            # `list(specs)` snapshot) skips it via the `job_id is not
            # None` guard instead of racing the outside-lock shutdown
            # below: without this, the monitor could see the SIGTERMed
            # process exit, ratchet failure_count to threshold, and
            # _attempt_restart would spawn a brand-new subprocess on
            # this port that is never tracked in state.workers again
            # (orphan, leaked VRAM).
            spec.job_id = f"director-unload-{model_id}"
            # Pop the spec NOW so concurrent admin reads see the
            # eviction commitment immediately. In-place mutation
            # keeps the monitor's captured list reference live.
            state.workers.remove(spec)
            spec_to_shutdown = spec
        else:
            # Sibling models still live; claim the spec for restart-in-place
            # via job_id so the monitor doesn't race us during the
            # _restart_worker_inplace window.
            spec.job_id = f"director-unload-{model_id}"
            spec_to_restart = spec

    # Phase 2: slow steps run OUTSIDE the lock.
    if spec_to_shutdown is not None:
        _shutdown_workers([spec_to_shutdown])
        return

    assert spec_to_restart is not None
    try:
        _restart_worker_inplace(
            spec_to_restart,
            device=state.device,
            log_hub=getattr(state, "log_hub", None),
            stop_event=state.stop_event,
        )
    except Exception:
        with state.lock:
            spec_to_restart.status = "dead"
            spec_to_restart.job_id = None
        raise
    with state.lock:
        spec_to_restart.job_id = None


def disable_model(model_id: str, *, state: SupervisorState) -> dict:
    """Sync operation: catalog flip + worker unload.

    Plan-then-execute (mirrors `enable_model` and
    `unload_model_from_worker`): the catalog flip + state.workers
    mutation happen under state.lock; the slow shutdown / restart-
    in-place phase runs OUTSIDE the lock. The auto-restart monitor
    skips the spec while we own it via `job_id`.

    Three paths:
      1. model unknown -> OperationError(404).
      2. model not loaded in any worker -> catalog flip only.
      3. model is loaded -> drop it from the worker; if it was the only
         model in that worker, pop the spec then terminate it (slow
         step outside the lock); else restart-in-place with the reduced
         load list (slow step outside the lock).

    On the sole-tenant path, state.workers is mutated in place via
    `state.workers.remove(spec)`. Rebinding the attribute would
    desynchronize the auto-restart monitor thread, which captured the
    original list reference at supervisor boot.
    """
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise OperationError(
            "model_not_found", f"unknown model {model_id!r}", status=404,
        )

    spec_to_shutdown: WorkerSpec | None = None
    spec_to_restart: WorkerSpec | None = None
    result_unloaded: dict | None = None

    # Phase 1: plan + claim under a brief lock.
    with state.lock:
        spec = find_worker_for_model(state, model_id)
        try:
            set_enabled(model_id, False)
        except KeyError:
            # Model is in known_models() but not in catalog.json (e.g.
            # bundled-but-not-pulled). Still report a coherent shape.
            pass

        if spec is None:
            result_unloaded = {
                "model_id": model_id,
                "loaded": False,
                "worker_terminated": False,
                "remaining_models_in_worker": [],
            }
        else:
            spec.models = [m for m in spec.models if m != model_id]
            if not spec.models:
                # Stamp job_id BEFORE popping so a monitor tick that
                # snapshotted this spec earlier skips it via the
                # `job_id is not None` guard rather than racing the
                # outside-lock shutdown below and respawning an orphan
                # worker on the freed port (see unload_model_from_worker
                # for the full race description).
                spec.job_id = f"admin-disable-{model_id}"
                # Pop the spec NOW. In-place mutation keeps the monitor's
                # captured list reference live.
                state.workers.remove(spec)
                spec_to_shutdown = spec
            else:
                # Claim the spec for restart-in-place via job_id so the
                # monitor doesn't race us during _restart_worker_inplace.
                spec.job_id = f"admin-disable-{model_id}"
                spec_to_restart = spec

    # Early-out path: no worker to touch.
    if result_unloaded is not None:
        return result_unloaded

    # Phase 2: slow steps run OUTSIDE the lock.
    if spec_to_shutdown is not None:
        _shutdown_workers([spec_to_shutdown])
        return {
            "model_id": model_id,
            "loaded": False,
            "worker_terminated": True,
            "worker_port": spec_to_shutdown.port,
            "remaining_models_in_worker": [],
        }

    assert spec_to_restart is not None
    try:
        _restart_worker_inplace(
            spec_to_restart,
            device=state.device,
            log_hub=getattr(state, "log_hub", None),
            stop_event=state.stop_event,
        )
    except Exception:
        # M12: invariant on restart failure in the disable path.
        #
        # At this point:
        #   - model_id has already been removed from spec_to_restart.models
        #     (done in Phase 1, under the lock).
        #   - The catalog `enabled` flag for model_id is already False
        #     (set_enabled called in Phase 1).
        #   - _restart_worker_inplace tried to SIGTERM the worker process
        #     and respawn it. If the restart failed, the worker may be in
        #     an unknown state: it could be dead, partially started, or
        #     (if SIGTERM was not delivered) still serving the OLD model
        #     list that includes model_id.
        #
        # INTENT of disable is "stop serving model_id." On restart failure
        # we cannot guarantee the new process will not serve the stale
        # model, so we adopt the defensive invariant:
        #   REMOVE the worker from state.workers and terminate the process.
        # This ensures model_id is NOT reachable via any gateway route
        # (state.workers is the routing source of truth) even if the OS
        # process is still alive for a moment.
        #
        # After this, the operator must re-enable the remaining sibling
        # models explicitly if they want them served again. The dead spec
        # is removed from state.workers (no zombie entry); the monitor
        # will not try to restart it because it's gone from the list.
        with state.lock:
            # Remove from the live worker list so the gateway cannot route
            # to this worker. The spec may or may not still be in
            # state.workers (a concurrent eviction could have removed it);
            # discard() is safe either way.
            try:
                state.workers.remove(spec_to_restart)
            except ValueError:
                pass  # already removed by a concurrent operation
            spec_to_restart.status = "dead"
            spec_to_restart.job_id = None
        # Best-effort SIGTERM: the process may already be dead from the
        # failed restart attempt, but terminate it to be safe. Errors
        # are swallowed; the process state is "best effort" at this point.
        try:
            _shutdown_workers([spec_to_restart])
        except Exception:  # noqa: BLE001
            pass
        raise
    with state.lock:
        spec_to_restart.job_id = None
    return {
        "model_id": model_id,
        "loaded": False,
        "worker_terminated": False,
        "worker_port": spec_to_restart.port,
        "remaining_models_in_worker": list(spec_to_restart.models),
    }


def warmup_model(model_id: str, *, state: SupervisorState) -> dict:
    """Sync operation: pre-load `model_id` via the LoadDirector without
    serving a request.

    Lazy-load companion to `enable_model`. Differs from
    `load_model_into_worker` (the director's enable_fn) in that it goes
    through the director's full warmup pathway: decide / load / commit
    with on-demand LRU eviction, but with the loaded LoadEntry's
    refcount=0 so the model is immediately eligible for eviction if
    pressure arrives before any request lands.

    The route handler returns this dict inline (no JobStore wrapping)
    because warmup is a simple synchronous operation from the caller's
    perspective: either it succeeds (returns a port) or it raises an
    OperationError that the route maps to an HTTP status. The director
    internally may take 10-60 seconds during a cold load, but that's
    just the duration of one HTTP request.

    Returns: {"model_id": ..., "worker_port": int}.

    Raises:
      OperationError("model_not_found", status=404): unknown model id.
      OperationError("model_not_pulled", status=409): the model is in
        the catalog but its weights/venv haven't been pulled yet.
        Validated upfront, before involving the director, to mirror
        `enable_model`'s preflight behavior.
      OperationError("director_unavailable", status=503): supervisor
        state has no director (supervisor not booted).
      OperationError("model_too_large_for_device", status=503): from
        the director when on-demand LRU eviction can't free enough.
    """
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise OperationError(
            "model_not_found", f"unknown model {model_id!r}", status=404,
        )
    if not is_pulled(model_id):
        raise OperationError(
            "model_not_pulled",
            f"model {model_id!r} not pulled; run pull first",
            status=409,
        )

    if state.director is None:
        raise OperationError(
            "director_unavailable",
            "supervisor director is not initialized; warmup requires a running `muse serve`",
            status=503,
        )

    manifest = get_manifest(model_id)
    # Backfill capabilities.memory_gb (and any device_override) from the
    # catalog sizing ladder, exactly like the gateway request path does
    # before calling director.acquire. Without this, a never-probed
    # model reads memory_gb as the fallback 0.0, so the director thinks
    # it "fits" for free, reserves 0 memory against concurrent loads,
    # and can over-admit -> OOM.
    manifest = backfill_manifest_memory(manifest, model_id)
    worker_port = state.director.warmup(model_id, manifest=manifest)
    return {"model_id": model_id, "worker_port": worker_port}


def remove_model(model_id: str, *, state: SupervisorState, purge: bool) -> dict:
    """Sync operation: drop the catalog entry. Refuses if currently loaded.

    Caller must `disable` first when the model is hosted by a worker;
    otherwise the running process holds open file descriptors against
    the venv we're about to delete.
    """
    catalog_known = known_models()
    if model_id not in catalog_known and not is_pulled(model_id):
        raise OperationError(
            "model_not_found", f"unknown model {model_id!r}", status=404,
        )
    # A "dead" spec (exhausted its restart budget or failed initial spawn)
    # lingers in state.workers but its process is gone, so it holds no FDs
    # against the venv we're about to delete: it must NOT block removal.
    # Every other status (running / pending / restarting / unhealthy) may
    # still own a live subprocess, so those still 409 until the operator
    # disables first (which reaps the process). See _drop_unserviceable for
    # why "unhealthy" is treated as possibly-live. We check EVERY hosting
    # spec, not just the first found, so a dead spec listed ahead of a live
    # one can't mask the live one.
    with state.lock:
        live_host = any(
            model_id in s.models and s.status != "dead" for s in state.workers
        )
    if live_host:
        raise OperationError(
            "model_loaded",
            f"model {model_id!r} is currently loaded; disable it first",
            status=409,
        )
    catalog_remove(model_id, purge=purge)
    return {"model_id": model_id, "removed": True, "purged": bool(purge)}


def probe_model(
    model_id: str,
    *,
    no_inference: bool,
    device: str | None,
    store: JobStore,
    job: Job,
) -> None:
    """Async wrapper around `muse models probe <id>`.

    Spawns a subprocess in the supervisor's interpreter (which dispatches
    into the model's per-model venv via the existing probe machinery).
    Captures stdout/stderr into job.log_lines.
    """
    store.update(job.job_id, state="running")
    # "--" terminates option parsing so a caller-influenced model_id that
    # begins with "-" (e.g. "--evil-id") is always treated as the
    # positional identifier, never mis-parsed as a click option. Verified
    # against the real CLI: `muse models probe --no-inference -- --x`
    # correctly reports "unknown model '--x'" rather than choking on an
    # unrecognized option. Options must precede "--" so they still parse
    # as options; only the identifier goes after it.
    cmd = [sys.executable, "-m", "muse.cli", "models", "probe"]
    if no_inference:
        cmd.append("--no-inference")
    if device is not None:
        cmd.extend(["--device", device])
    cmd.append("--json")
    cmd.extend(["--", model_id])
    _run_subprocess_into_job(cmd, store=store, job=job, success_op="probe")


def pull_model(identifier: str, *, store: JobStore, job: Job) -> None:
    """Async wrapper around `muse pull <identifier>`.

    `identifier` may be a curated alias, a bundled model id, or a
    resolver URI. The subprocess persists the resulting catalog entry
    directly (catalog.json is written under MUSE_CATALOG_DIR); the next
    `enable` call sees it because `known_models()` re-merges whenever
    the catalog file's mtime changes. (It did NOT before that mtime
    keying existed: this process's known_models cache froze at first
    call, the subprocess's own cache resets were invisible here, and
    enable 404'd "unknown model" for anything pulled after the freeze.)
    """
    store.update(job.job_id, state="running")
    # "--" terminates option parsing (see probe_model's comment) so an
    # identifier beginning with "-" is always treated as positional.
    cmd = [sys.executable, "-m", "muse.cli", "pull", "--", identifier]
    _run_subprocess_into_job(cmd, store=store, job=job, success_op="pull")


def _run_subprocess_into_job(
    cmd: list[str],
    *,
    store: JobStore,
    job: Job,
    success_op: str,
) -> None:
    """Run `cmd` to completion; capture stdout/stderr into the Job.

    On success: state=done, log_lines populated, result contains the op
    name + return code + stdout. On failure: state=failed, error has the
    return code + stderr.
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        log_lines = (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
        if proc.returncode == 0:
            store.update(
                job.job_id, state="done",
                log_lines=log_lines,
                result={
                    "op": success_op,
                    "returncode": 0,
                    "stdout": proc.stdout,
                },
            )
        else:
            store.update(
                job.job_id, state="failed",
                log_lines=log_lines,
                error=f"exit {proc.returncode}: {proc.stderr.strip() or 'subprocess failed'}",
            )
    except subprocess.TimeoutExpired:
        store.update(job.job_id, state="failed", error="subprocess timed out")
    except Exception as e:  # noqa: BLE001
        logger.exception("subprocess job failed")
        store.update(job.job_id, state="failed", error=str(e))


def _restart_worker_inplace(
    spec: WorkerSpec,
    *,
    device: str,
    log_hub: "Any | None" = None,
    stop_event: "threading.Event | None" = None,
) -> None:
    """Terminate + respawn one worker with its current `spec.models`.

    Used by enable_model (joining a venv group) and disable_model
    (dropping a model from a multi-model worker). Reuses the spec's
    port, python_path, and updated models list. restart_count bumps so
    the auto-restart monitor sees this in its bookkeeping; failure_count
    resets so we don't carry over stale unhealthy state.

    `log_hub` is forwarded to `spawn_worker` so a restart-in-place keeps
    piping the worker's stdout into the LogHub when telemetry is enabled
    (callers pass `state.log_hub`).
    """
    _shutdown_workers([spec])
    spec.process = None
    if stop_event is not None and stop_event.is_set():
        raise OperationError(
            "server_shutting_down",
            "server shutdown is in progress; worker restart cancelled",
            status=503,
        )
    spec.failure_count = 0
    spec.restart_count += 1
    try:
        spawn_worker(spec, device=device, log_hub=log_hub)
        wait_for_ready(
            port=spec.port, timeout=120.0, stop_event=stop_event,
        )
    except Exception:
        _shutdown_workers([spec])
        if stop_event is not None and stop_event.is_set():
            raise OperationError(
                "server_shutting_down",
                "server shutdown is in progress; worker restart cancelled",
                status=503,
            ) from None
        raise
    spec.status = "running"


def launch_async(
    op_fn: Callable[..., None],
    *,
    op_name: str,
    model_id: str,
    store: JobStore,
    op_args: tuple = (),
    **kwargs: Any,
) -> Job:
    """Create a Job + spawn a daemon thread that runs op_fn(...).

    `op_fn` must accept (positional args from `op_args`, keyword args
    `job=Job`, `store=JobStore`, **kwargs). The thread is daemonized so
    a Ctrl+C on the supervisor takes it down with the process;
    JobStore.shutdown joins them with a timeout on graceful exit.

    `model_id` is the JobStore label (for /v1/admin/jobs/{id} display);
    if `op_args` is empty, model_id is ALSO passed as the first
    positional argument (the common case for enable_model / probe_model
    / pull_model whose signature starts with `model_id`).
    """
    job = store.create(op_name, model_id)
    if not op_args:
        op_args = (model_id,)
    thread = threading.Thread(
        target=op_fn,
        args=op_args,
        kwargs={"job": job, "store": store, **kwargs},
        daemon=True,
        name=f"muse-admin-{op_name}-{job.job_id}",
    )
    job.thread = thread
    thread.start()
    return job
