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
import time
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

from muse.admin.jobs import (
    Job,
    JobStore,
    JobStoreFullError,
    JobStoreShuttingDownError,
)
from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerOperation,
    WorkerShutdownResult,
    WorkerSpec,
    _shutdown_workers,
    backfill_manifest_memory,
    claim_worker_operation,
    finish_worker_operation,
    spawn_worker,
    wait_for_ready,
)
from muse.core.catalog import (
    _read_catalog,
    get_manifest,
    is_enabled,
    is_pulled,
    known_models,
    remove as catalog_remove,
    set_enabled,
)
from muse.core.venv import find_free_port

logger = logging.getLogger(__name__)


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


def _director_method(state: SupervisorState, name: str) -> Callable[..., Any] | None:
    """Return one real director API without triggering MagicMock attributes."""
    director = getattr(state, "director", None)
    descriptor = getattr(type(director), name, None)
    if director is None or not callable(descriptor):
        return None

    def call(*args: Any, **kwargs: Any) -> Any:
        return descriptor(director, *args, **kwargs)

    return call


def _allow_director_model(state: SupervisorState, model_id: str) -> None:
    allow = _director_method(state, "allow_model")
    if allow is not None:
        allow(model_id)


def _require_model_python_path(model_id: str) -> str:
    """Return a pulled model's interpreter path or a user-facing error."""
    if model_id not in known_models():
        raise OperationError(
            "model_not_found", f"unknown model {model_id!r}", status=404,
        )
    if not is_pulled(model_id):
        raise OperationError(
            "model_not_pulled",
            f"model {model_id!r} not pulled; run pull first",
            status=409,
        )
    entry = _read_catalog().get(model_id)
    python_path = entry.get("python_path") if isinstance(entry, dict) else None
    if not python_path:
        raise OperationError(
            "missing_venv",
            f"model {model_id!r} has no per-model venv on record",
            status=409,
        )
    return str(python_path)


def _wait_for_worker_operation(
    state: SupervisorState, operation: WorkerOperation,
) -> None:
    """Wait outside the state lock, remaining interruptible by shutdown."""
    while not operation.done.wait(0.1):
        _ensure_server_running(state)
    _ensure_server_running(state)


def _reinsert_retained_workers(
    state: SupervisorState,
    positioned_specs: list[tuple[int, WorkerSpec]],
) -> None:
    """Keep incompletely released process generations supervisor-owned."""
    with state.lock:
        for former_index, spec in sorted(positioned_specs, key=lambda item: item[0]):
            spec.status = "dead"
            spec.job_id = None
            if any(candidate is spec for candidate in state.workers):
                continue
            state.workers.insert(
                min(max(0, former_index), len(state.workers)),
                spec,
            )


@dataclass(frozen=True)
class _LoadOutcome:
    port: int
    spawned_new: bool
    coalesced_owner: str | None = None


def _load_model_with_ownership(
    model_id: str,
    *,
    state: SupervisorState,
    owner: str,
    enable_in_catalog: bool,
) -> _LoadOutcome:
    """Load a model while exclusively owning its shared venv transition.

    A waiter never edits the current owner's WorkerSpec.  It waits for the
    generation's completion, re-reads catalog + worker state, and either
    observes the requested model running or owns a later generation.  This
    intentionally permits two serialized restarts for concurrent requests
    for different models in one venv; mutating the first restart's model
    list after it constructed its command could otherwise report a model as
    loaded even though that command never included it.
    """
    coalesced_owner: str | None = None

    while True:
        _ensure_server_running(state)
        python_path = _require_model_python_path(model_id)
        operation, claimed = claim_worker_operation(
            state, python_path=python_path, owner=owner,
        )
        if not claimed:
            if coalesced_owner is None:
                coalesced_owner = operation.owner
            _wait_for_worker_operation(state, operation)
            continue

        try:
            # Removal or repull may have won immediately before our claim.
            # Revalidate only after ownership; never spawn from the stale
            # catalog snapshot used to discover the coordination key.
            current_python_path = _require_model_python_path(model_id)
            if current_python_path != python_path:
                continue

            plan: str
            spec_ref: WorkerSpec
            planned_models: tuple[str, ...] | None = None
            dropped: list[WorkerSpec] = []
            dropped_positions: dict[int, int] = {}
            sibling_rollback: tuple[
                WorkerSpec, list[str], str, str | None,
            ] | None = None
            new_spec: WorkerSpec | None = None

            if enable_in_catalog:
                # Catalog flock/fsync may block. The per-venv operation claim
                # already serializes this model transition, so do not hold the
                # central routing lock across filesystem I/O.
                set_enabled(model_id, True)
                # Explicit enable is the only transition that clears a
                # persistent director block installed by admin disable.
                _allow_director_model(state, model_id)
            else:
                # Re-check after owning the venv generation. A catalog disable
                # can race the gateway's initial check; a stale request must
                # not resurrect the worker after that disable committed.
                current_entry = _read_catalog().get(model_id)
                if (
                    isinstance(current_entry, dict)
                    and not bool(current_entry.get("enabled", True))
                ):
                    raise OperationError(
                        "model_disabled",
                        f"model {model_id!r} is disabled",
                        status=409,
                    )

            with state.lock:
                worker_positions = {
                    id(worker): index
                    for index, worker in enumerate(state.workers)
                }
                existing = next(
                    (s for s in state.workers if model_id in s.models), None,
                )
                if existing is not None and existing.status == "running":
                    plan = "already_running"
                    spec_ref = existing
                else:
                    # A pending/restarting spec without the matching active
                    # operation is stale.  Remove it from routing ownership
                    # and reap its possible process before replacement.
                    if existing is not None:
                        dropped_positions[id(existing)] = worker_positions[id(existing)]
                        state.workers.remove(existing)
                        dropped.append(existing)

                    sibling = next(
                        (
                            s for s in state.workers
                            if s.python_path == python_path
                            and s.status == "running"
                        ),
                        None,
                    )
                    if sibling is not None:
                        sibling_rollback = (
                            sibling,
                            list(sibling.models),
                            sibling.status,
                            sibling.job_id,
                        )
                        planned_models = tuple(
                            sorted(set(sibling.models) | {model_id}),
                        )
                        sibling.models = list(planned_models)
                        sibling.status = "restarting"
                        sibling.job_id = operation.token
                        plan = "restart_sibling"
                        spec_ref = sibling
                    else:
                        # Reap any stale same-venv records before adding the
                        # sole replacement spec. This also heals historical
                        # duplicate pending records deterministically.
                        stale_siblings = [
                            s for s in state.workers
                            if s.python_path == python_path
                        ]
                        for stale in stale_siblings:
                            dropped_positions[id(stale)] = worker_positions[id(stale)]
                            state.workers.remove(stale)
                            dropped.append(stale)
                        new_spec = WorkerSpec(
                            models=[model_id],
                            python_path=python_path,
                            port=_pick_free_port(state),
                            device=state.device,
                        )
                        new_spec.status = "pending"
                        new_spec.job_id = operation.token
                        state.workers.append(new_spec)
                        planned_models = (model_id,)
                        plan = "spawn_new"
                        spec_ref = new_spec

            if dropped:
                shutdown_result = _shutdown_workers(dropped)
                retained = (
                    list(shutdown_result.retained)
                    if isinstance(shutdown_result, WorkerShutdownResult)
                    else []
                )
                if retained:
                    with state.lock:
                        if sibling_rollback is not None:
                            sibling, models, status, job_id = sibling_rollback
                            if any(candidate is sibling for candidate in state.workers):
                                sibling.models = models
                                sibling.status = status
                                sibling.job_id = job_id
                        if new_spec is not None:
                            try:
                                state.workers.remove(new_spec)
                            except ValueError:
                                pass
                    _reinsert_retained_workers(
                        state,
                        [
                            (dropped_positions[id(spec)], spec)
                            for spec in retained
                        ],
                    )
                    raise OperationError(
                        "worker_shutdown_incomplete",
                        "a stale worker process could not be fully released",
                        status=503,
                    )

            if plan == "already_running":
                return _LoadOutcome(
                    port=spec_ref.port,
                    spawned_new=False,
                    coalesced_owner=coalesced_owner,
                )

            if plan == "restart_sibling":
                assert planned_models is not None
                try:
                    _restart_worker_inplace(
                        spec_ref,
                        models=planned_models,
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
                return _LoadOutcome(
                    port=spec_ref.port,
                    spawned_new=False,
                    coalesced_owner=coalesced_owner,
                )

            assert plan == "spawn_new"
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
                    expected_nonce=spec_ref.worker_nonce,
                    worker=spec_ref,
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
            return _LoadOutcome(
                port=spec_ref.port,
                spawned_new=True,
                coalesced_owner=coalesced_owner,
            )
        finally:
            finish_worker_operation(state, operation)


def enable_model(
    model_id: str,
    *,
    state: SupervisorState,
    store: JobStore,
    job: Job,
) -> None:
    """Async operation: load ``model_id`` or await its current owner."""
    store.update(job.job_id, state="running")
    try:
        outcome = _load_model_with_ownership(
            model_id,
            state=state,
            owner=job.job_id,
            enable_in_catalog=True,
        )
        result: dict[str, Any] = {
            "model_id": model_id,
            "worker_port": outcome.port,
            "loaded": True,
            "spawned_new": outcome.spawned_new,
        }
        if outcome.coalesced_owner is not None:
            result["coalesced_job_id"] = outcome.coalesced_owner
        store.update(job.job_id, state="done", result=result)
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

    Concurrency contract (mirrors `enable_model`): one generation-numbered
    operation owns each shared ``python_path``. A concurrent admin/director
    caller waits for that generation, then revalidates the catalog and
    running model list. The owner also stamps its opaque token on the spec,
    so the auto-restart monitor cannot race the slow transition.

    Raises OperationError on user-facing failures (model not found,
    not pulled, no venv on record). Other exceptions propagate to the
    director, which cleans up its in-flight Event and re-raises.
    """
    outcome = _load_model_with_ownership(
        model_id,
        state=state,
        owner=f"director-load-{model_id}",
        enable_in_catalog=False,
    )
    return outcome.port


def unload_model_from_worker(model_id: str, *, state: SupervisorState) -> None:
    """Sync operation: drop `model_id` from its worker without disabling.

    Lazy-load companion to `disable_model`. Crucial difference: does
    NOT call `set_enabled(model_id, False)`. The catalog stays "in
    service"; this just frees the memory slot so the director can
    load another model.

    Plan-then-execute (mirrors `enable_model`): first own the shared venv
    generation, then under state.lock pop the spec or install an immutable
    reduced-model plan and stamp the operation token. The slow steps
    (`_shutdown_workers` or `_restart_worker_inplace`) run outside the lock,
    while a competing worker transition waits on the generation event.

    Three paths:
      1. model_id not loaded in any worker: no-op.
      2. model_id is the only model in a worker: pop the spec, then
         terminate the worker (lock released for SIGTERM + grace).
      3. model_id is one of several in a worker (venv-group sibling):
         claim the spec via the operation token, then restart-in-place with the
         reduced model list (lock released for spawn + readiness wait).

    On path (2), state.workers is mutated in place via
    `state.workers.remove(spec)`. Rebinding the attribute would
    desynchronize the auto-restart monitor thread, which captured the
    original list reference at supervisor boot.
    """
    while True:
        with state.lock:
            initial = next(
                (s for s in state.workers if model_id in s.models), None,
            )
            if initial is None:
                return
            python_path = initial.python_path

        operation, claimed = claim_worker_operation(
            state,
            python_path=python_path,
            owner=f"director-unload-{model_id}",
        )
        if not claimed:
            _wait_for_worker_operation(state, operation)
            continue

        try:
            spec_to_shutdown: WorkerSpec | None = None
            shutdown_index: int | None = None
            spec_to_restart: WorkerSpec | None = None
            planned_models: tuple[str, ...] = ()
            retry = False

            with state.lock:
                spec = next(
                    (s for s in state.workers if model_id in s.models), None,
                )
                if spec is None:
                    return
                if spec.python_path != python_path:
                    retry = True
                else:
                    planned_models = tuple(
                        m for m in spec.models if m != model_id
                    )
                    spec.job_id = operation.token
                    if not planned_models:
                        # Remove before shutdown so routing no longer sees
                        # the eviction target. The token remains on stale
                        # monitor snapshots until that process is reaped.
                        shutdown_index = state.workers.index(spec)
                        state.workers.remove(spec)
                        spec_to_shutdown = spec
                    else:
                        spec.models = list(planned_models)
                        spec.status = "restarting"
                        spec_to_restart = spec

            if retry:
                continue
            if spec_to_shutdown is not None:
                shutdown_result = _shutdown_workers([spec_to_shutdown])
                if (
                    isinstance(shutdown_result, WorkerShutdownResult)
                    and shutdown_result.retained_spec(spec_to_shutdown)
                ):
                    with state.lock:
                        spec_to_shutdown.status = "dead"
                        spec_to_shutdown.job_id = None
                        if not any(
                            candidate is spec_to_shutdown
                            for candidate in state.workers
                        ):
                            index = min(
                                shutdown_index
                                if shutdown_index is not None
                                else len(state.workers),
                                len(state.workers),
                            )
                            state.workers.insert(index, spec_to_shutdown)
                    raise OperationError(
                        "worker_shutdown_incomplete",
                        f"worker on port {spec_to_shutdown.port} could not "
                        "be fully released",
                        status=503,
                    )
                return

            assert spec_to_restart is not None
            try:
                _restart_worker_inplace(
                    spec_to_restart,
                    models=planned_models,
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
            return
        finally:
            finish_worker_operation(state, operation)


def disable_model(model_id: str, *, state: SupervisorState) -> dict:
    """Disable through one director-owned teardown generation when available."""
    if model_id not in known_models():
        raise OperationError(
            "model_not_found", f"unknown model {model_id!r}", status=404,
        )

    begin = _director_method(state, "begin_model_disable")
    finish = _director_method(state, "finish_eviction")
    claim = begin(model_id) if begin is not None else None
    try:
        result = _disable_model_worker(model_id, state=state)
    except BaseException as exc:
        if claim is not None and finish is not None:
            finish(claim, success=False, error=exc)
            # A failure before catalog commit restores the prior allow state.
            # Once the catalog is disabled, remain fail-closed until explicit
            # enable even if the worker teardown itself was incomplete.
            try:
                catalog_disabled = not is_enabled(model_id)
            except Exception:  # noqa: BLE001
                catalog_disabled = True
            if not catalog_disabled and not bool(getattr(claim, "was_blocked", False)):
                _allow_director_model(state, model_id)
        raise

    if claim is not None and finish is not None:
        if finish(claim, success=True) is not True:
            raise OperationError(
                "director_state_changed",
                f"disable ownership for model {model_id!r} changed unexpectedly",
                status=503,
            )
    return result


def _disable_model_worker(model_id: str, *, state: SupervisorState) -> dict:
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

    while True:
        catalog_entry = _read_catalog().get(model_id)
        with state.lock:
            initial = next(
                (s for s in state.workers if model_id in s.models), None,
            )
        python_path = (
            initial.python_path
            if initial is not None
            else (
                str(catalog_entry.get("python_path"))
                if isinstance(catalog_entry, dict)
                and catalog_entry.get("python_path")
                else None
            )
        )

        # Bundled-but-unpulled models have no venv and therefore no worker
        # transition with which to race. Preserve the coherent no-op shape.
        if python_path is None:
            try:
                set_enabled(model_id, False)
            except KeyError:
                pass
            return {
                "model_id": model_id,
                "loaded": False,
                "worker_terminated": False,
                "remaining_models_in_worker": [],
            }

        operation, claimed = claim_worker_operation(
            state,
            python_path=python_path,
            owner=f"admin-disable-{model_id}",
        )
        if not claimed:
            _wait_for_worker_operation(state, operation)
            continue

        try:
            spec_to_shutdown: WorkerSpec | None = None
            shutdown_index: int | None = None
            spec_to_restart: WorkerSpec | None = None
            result_unloaded: dict | None = None
            planned_models: tuple[str, ...] = ()
            retry = False

            current_entry = _read_catalog().get(model_id)
            current_python_path = (
                str(current_entry.get("python_path"))
                if isinstance(current_entry, dict)
                and current_entry.get("python_path")
                else None
            )
            if current_python_path is not None and current_python_path != python_path:
                continue

            try:
                set_enabled(model_id, False)
            except KeyError:
                pass

            with state.lock:
                spec = next(
                    (s for s in state.workers if model_id in s.models), None,
                )
                if spec is not None and spec.python_path != python_path:
                    retry = True
                else:
                    if spec is None:
                        result_unloaded = {
                            "model_id": model_id,
                            "loaded": False,
                            "worker_terminated": False,
                            "remaining_models_in_worker": [],
                        }
                    else:
                        planned_models = tuple(
                            m for m in spec.models if m != model_id
                        )
                        spec.job_id = operation.token
                        if not planned_models:
                            shutdown_index = state.workers.index(spec)
                            state.workers.remove(spec)
                            spec_to_shutdown = spec
                        else:
                            spec.models = list(planned_models)
                            spec.status = "restarting"
                            spec_to_restart = spec

            if retry:
                continue
            if result_unloaded is not None:
                return result_unloaded
            if spec_to_shutdown is not None:
                shutdown_result = _shutdown_workers([spec_to_shutdown])
                if (
                    isinstance(shutdown_result, WorkerShutdownResult)
                    and shutdown_result.retained_spec(spec_to_shutdown)
                ):
                    with state.lock:
                        spec_to_shutdown.status = "dead"
                        spec_to_shutdown.job_id = None
                        if not any(
                            candidate is spec_to_shutdown
                            for candidate in state.workers
                        ):
                            index = min(
                                shutdown_index
                                if shutdown_index is not None
                                else len(state.workers),
                                len(state.workers),
                            )
                            state.workers.insert(index, spec_to_shutdown)
                    raise OperationError(
                        "worker_shutdown_incomplete",
                        f"worker on port {spec_to_shutdown.port} could not "
                        "be fully released",
                        status=503,
                    )
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
                    models=planned_models,
                    device=state.device,
                    log_hub=getattr(state, "log_hub", None),
                    stop_event=state.stop_event,
                )
            except Exception:
                # A failed replacement could still serve its old command.
                # Remove it from routing and reap it before relinquishing
                # ownership so the disabled model cannot reappear.
                with state.lock:
                    former_index = (
                        state.workers.index(spec_to_restart)
                        if any(
                            candidate is spec_to_restart
                            for candidate in state.workers
                        )
                        else len(state.workers)
                    )
                    try:
                        state.workers.remove(spec_to_restart)
                    except ValueError:
                        pass
                    spec_to_restart.status = "dead"
                    spec_to_restart.job_id = None
                try:
                    shutdown_result = _shutdown_workers([spec_to_restart])
                except Exception:  # noqa: BLE001
                    shutdown_result = None
                retained = (
                    isinstance(shutdown_result, WorkerShutdownResult)
                    and shutdown_result.retained_spec(spec_to_restart)
                )
                if shutdown_result is None:
                    with spec_to_restart.process_lock:
                        retained = (
                            spec_to_restart.process is not None
                            or spec_to_restart.log_thread is not None
                        )
                if retained:
                    _reinsert_retained_workers(
                        state, [(former_index, spec_to_restart)],
                    )
                raise
            with state.lock:
                spec_to_restart.job_id = None
            return {
                "model_id": model_id,
                "loaded": False,
                "worker_terminated": False,
                "worker_port": spec_to_restart.port,
                "remaining_models_in_worker": list(planned_models),
            }
        finally:
            finish_worker_operation(state, operation)


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
      OperationError("model_disabled", status=409): the model was disabled
        by the operator and cannot be warmed until it is enabled again.
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
    if not is_enabled(model_id):
        raise OperationError(
            "model_disabled",
            f"model {model_id!r} is disabled; enable it before warmup",
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
    manifest = backfill_manifest_memory(
        manifest, model_id, supervisor_device=state.device,
    )
    worker_port = state.director.warmup(model_id, manifest=manifest)
    return {"model_id": model_id, "worker_port": worker_port}


def _spec_may_hold_worker_resources(spec: WorkerSpec) -> bool:
    """Return whether a spec can still own a process or its open files."""
    if spec.status != "dead":
        return True
    with spec.process_lock:
        # Even an exited leader remains an ownership token until its full
        # process group, reader, and persistent registry record are released.
        # Never poll here: reaping a pinned PID==PGID would make the stored
        # numeric group identity unsafe before supervisor cleanup drains it.
        return bool(
            spec.process is not None
            or spec.log_thread is not None
            or spec.resource_id is not None
        )


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
    catalog_entry = _read_catalog().get(model_id)
    with state.lock:
        initial = next(
            (s for s in state.workers if model_id in s.models), None,
        )
    python_path = (
        initial.python_path
        if initial is not None
        else (
            str(catalog_entry.get("python_path"))
            if isinstance(catalog_entry, dict) and catalog_entry.get("python_path")
            else None
        )
    )

    # A malformed pulled entry with no interpreter cannot be loaded, so it
    # has no worker-operation key. Catalog removal still applies its own
    # filesystem confinement and transaction lock.
    if python_path is None:
        catalog_remove(model_id, purge=purge)
        _allow_director_model(state, model_id)
        return {"model_id": model_id, "removed": True, "purged": bool(purge)}

    while True:
        operation, claimed = claim_worker_operation(
            state,
            python_path=python_path,
            owner=f"admin-remove-{model_id}",
        )
        if not claimed:
            _wait_for_worker_operation(state, operation)
            # The completed operation may have replaced the catalog entry
            # with a different venv. Recompute before claiming a generation.
            refreshed = _read_catalog().get(model_id)
            if isinstance(refreshed, dict) and refreshed.get("python_path"):
                python_path = str(refreshed["python_path"])
            continue

        try:
            # Recheck only after owning the venv. A concurrent load that won
            # first is now visible and blocks deletion; a load that arrives
            # later waits until removal completes and then fails its catalog
            # revalidation rather than spawning from a deleted venv.
            refreshed = _read_catalog().get(model_id)
            refreshed_python_path = (
                str(refreshed.get("python_path"))
                if isinstance(refreshed, dict) and refreshed.get("python_path")
                else None
            )
            if (
                refreshed_python_path is not None
                and refreshed_python_path != python_path
            ):
                python_path = refreshed_python_path
                continue
            with state.lock:
                live_host = next(
                    (
                        s for s in state.workers
                        if model_id in s.models
                        and _spec_may_hold_worker_resources(s)
                    ),
                    None,
                )
                live_shared_venv = next(
                    (
                        s for s in state.workers
                        if purge
                        and s.python_path == python_path
                        and _spec_may_hold_worker_resources(s)
                    ),
                    None,
                )
            if live_host is not None or live_shared_venv is not None:
                shared_detail = (
                    " or its shared environment is in use"
                    if live_host is None else ""
                )
                raise OperationError(
                    "model_loaded",
                    f"model {model_id!r} is currently loaded{shared_detail}; "
                    "disable the affected worker first",
                    status=409,
                )
            if not is_pulled(model_id):
                raise OperationError(
                    "model_not_found", f"unknown model {model_id!r}", status=404,
                )
            catalog_remove(model_id, purge=purge)
            _allow_director_model(state, model_id)
            return {
                "model_id": model_id,
                "removed": True,
                "purged": bool(purge),
            }
        finally:
            finish_worker_operation(state, operation)


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


_SUBPROCESS_TIMEOUT_SECONDS = 1800
_SUBPROCESS_STREAM_TAIL_BYTES = 256 * 1024
_SUBPROCESS_READ_CHUNK_BYTES = 64 * 1024
_SUBPROCESS_OUTPUT_DRAIN_SECONDS = 5.0
_SUBPROCESS_FINAL_DRAIN_SECONDS = 1.0
_SUBPROCESS_LOG_MAX_LINES = 2000


class _BoundedByteTail:
    """Thread-safe fixed-size tail of one subprocess byte stream."""

    def __init__(self, *, limit: int, label: str) -> None:
        self._limit = limit
        self._label = label
        self._data = bytearray()
        self._total = 0
        self._read_error: str | None = None
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._total += len(chunk)
            if len(chunk) >= self._limit:
                self._data[:] = chunk[-self._limit:]
                return
            overflow = len(self._data) + len(chunk) - self._limit
            if overflow > 0:
                del self._data[:overflow]
            self._data.extend(chunk)

    def record_read_error(self, error: BaseException) -> None:
        with self._lock:
            self._read_error = type(error).__name__

    def render(self) -> str:
        with self._lock:
            data = bytes(self._data)
            omitted = self._total - len(data)
            read_error = self._read_error

        parts: list[str] = []
        if omitted:
            parts.append(
                f"[... {self._label} truncated; {omitted} byte(s) omitted ...]\n"
            )
        parts.append(data.decode("utf-8", errors="replace"))
        if read_error:
            if parts[-1] and not parts[-1].endswith("\n"):
                parts.append("\n")
            parts.append(
                f"[... {self._label} capture failed: {read_error} ...]"
            )
        return "".join(parts)


class _ProcessOutputCapture:
    """Continuously drain two child pipes while retaining bounded tails.

    `Popen.communicate()` accumulates each complete stream in memory. Pulls
    and probes can be noisy and long-running, so two daemon readers drain the
    pipes concurrently (avoiding a full-stderr/full-stdout deadlock) while
    `_BoundedByteTail` keeps memory independent of total child output.
    """

    def __init__(self, stdout: BinaryIO | None, stderr: BinaryIO | None) -> None:
        self._streams = {"stdout": stdout, "stderr": stderr}
        self._tails = {
            label: _BoundedByteTail(
                limit=_SUBPROCESS_STREAM_TAIL_BYTES,
                label=label,
            )
            for label in self._streams
        }
        self._threads = {
            label: threading.Thread(
                target=self._drain,
                args=(label, stream),
                name=f"muse-admin-{label}-drain",
                daemon=True,
            )
            for label, stream in self._streams.items()
        }
        self._started: set[str] = set()

    def start(self) -> None:
        for label, thread in self._threads.items():
            thread.start()
            self._started.add(label)

    def finish(self, timeout: float) -> bool:
        """Boundedly drain, then close owned pipes and join their readers."""
        timeout = max(0.0, timeout)
        started_at = time.monotonic()
        deadline = started_at + timeout
        # Preserve most of the budget for natural EOF, while reserving time
        # to close our read descriptors and let blocked readers unwind when a
        # descendant inherited a write descriptor after the leader exited.
        natural_deadline = started_at + timeout * 0.8
        for label in self._started:
            thread = self._threads[label]
            thread.join(timeout=max(0.0, natural_deadline - time.monotonic()))

        complete = len(self._started) == len(self._threads)
        for label, thread in self._threads.items():
            if label in self._started and thread.is_alive():
                complete = False
            stream = self._streams[label]
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    logger.debug("could not close owned %s capture pipe", label)

        for label in self._started:
            thread = self._threads[label]
            if thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return complete

    def snapshot(self) -> tuple[str, str]:
        return self._tails["stdout"].render(), self._tails["stderr"].render()

    def _drain(self, label: str, stream: BinaryIO | None) -> None:
        if stream is None:
            self._tails[label].record_read_error(RuntimeError("missing pipe"))
            return
        try:
            while True:
                chunk = stream.read(_SUBPROCESS_READ_CHUNK_BYTES)
                if not chunk:
                    return
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", errors="replace")
                self._tails[label].append(bytes(chunk))
        except Exception as e:  # noqa: BLE001
            logger.warning("could not capture admin subprocess %s: %s", label, e)
            self._tails[label].record_read_error(e)


def _bounded_subprocess_log_lines(stdout: str, stderr: str) -> list[str]:
    """Combine stream tails while bounding pathological one-byte lines."""
    lines = stdout.splitlines() + stderr.splitlines()
    if len(lines) <= _SUBPROCESS_LOG_MAX_LINES:
        return lines
    retained = _SUBPROCESS_LOG_MAX_LINES - 1
    omitted = len(lines) - retained
    return [
        f"[... {omitted} earlier combined log line(s) omitted ...]",
        *lines[-retained:],
    ]


def _run_subprocess_into_job(
    cmd: list[str],
    *,
    store: JobStore,
    job: Job,
    success_op: str,
) -> None:
    """Run `cmd` to completion; retain bounded stdout/stderr tails.

    On success: state=done, log_lines populated, result contains the op
    name + return code + stdout. On failure: state=failed, error has the
    return code + stderr. Pipes are drained concurrently throughout the run,
    so neither a noisy child nor the serialized Job payload can grow memory
    without bound.
    """
    proc: subprocess.Popen | None = None
    capture: _ProcessOutputCapture | None = None
    try:
        proc = store.spawn_process(
            job.job_id,
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        capture = _ProcessOutputCapture(proc.stdout, proc.stderr)
        capture.start()
        store.wait_process(
            job.job_id,
            proc,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )

        output_closed = capture.finish(_SUBPROCESS_OUTPUT_DRAIN_SECONDS)
        if not output_closed:
            # The direct child exited but a descendant may still hold an
            # inherited pipe FD. The exact leader has already been reaped, so
            # its numeric PGID is no longer a safe identity. `finish` closes
            # Muse's owned read descriptors and boundedly joins the readers.
            capture.finish(_SUBPROCESS_FINAL_DRAIN_SECONDS)

        stdout, stderr = capture.snapshot()
        log_lines = _bounded_subprocess_log_lines(stdout, stderr)
        if store.process_cancel_requested(job.job_id, proc):
            store.update(
                job.job_id,
                state="failed",
                log_lines=log_lines,
                error="subprocess cancelled during server shutdown",
            )
            return
        if not output_closed:
            store.update(
                job.job_id,
                state="failed",
                log_lines=log_lines,
                error="subprocess output streams did not close",
            )
            return
        if proc.returncode == 0:
            store.update(
                job.job_id, state="done",
                log_lines=log_lines,
                result={
                    "op": success_op,
                    "returncode": 0,
                    "stdout": stdout,
                },
            )
        else:
            store.update(
                job.job_id, state="failed",
                log_lines=log_lines,
                error=f"exit {proc.returncode}: {stderr.strip() or 'subprocess failed'}",
            )
    except subprocess.TimeoutExpired:
        if proc is not None:
            store.terminate_process(job.job_id, proc, timeout=5.0)
        if capture is not None:
            capture.finish(_SUBPROCESS_FINAL_DRAIN_SECONDS)
            timeout_stdout, timeout_stderr = capture.snapshot()
        else:
            timeout_stdout, timeout_stderr = "", ""
        cancelled = bool(
            proc is not None
            and store.process_cancel_requested(job.job_id, proc)
        )
        store.update(
            job.job_id,
            state="failed",
            error=(
                "subprocess cancelled during server shutdown"
                if cancelled else "subprocess timed out"
            ),
            log_lines=_bounded_subprocess_log_lines(
                timeout_stdout, timeout_stderr,
            ),
        )
    except JobStoreShuttingDownError:
        store.update(
            job.job_id,
            state="failed",
            error="subprocess cancelled during server shutdown",
        )
    except Exception as e:  # noqa: BLE001
        if proc is not None:
            store.terminate_process(job.job_id, proc, timeout=5.0)
        if capture is not None:
            capture.finish(_SUBPROCESS_FINAL_DRAIN_SECONDS)
        logger.exception("subprocess job failed")
        store.update(job.job_id, state="failed", error=str(e))
    finally:
        if proc is not None:
            store.release_process(job.job_id, proc)


def _restart_worker_inplace(
    spec: WorkerSpec,
    *,
    models: tuple[str, ...] | None = None,
    device: str,
    log_hub: "Any | None" = None,
    stop_event: "threading.Event | None" = None,
) -> None:
    """Terminate + respawn one worker from an immutable model snapshot.

    Used by enable_model (joining a venv group) and disable_model
    (dropping a model from a multi-model worker). ``models`` is captured by
    the operation owner before the slow phase; a concurrent caller waits
    for that operation rather than changing this command plan. Direct
    legacy callers may omit it to snapshot the current ``spec.models`` at
    function entry. Reuses the spec's port and python_path. A replacement that
    reaches readiness resets the consecutive auto-restart failure budget.

    `log_hub` is forwarded to `spawn_worker` so a restart-in-place keeps
    piping the worker's stdout into the LogHub when telemetry is enabled
    (callers pass `state.log_hub`).
    """
    planned_models = tuple(spec.models) if models is None else tuple(models)
    _shutdown_workers([spec])
    with spec.process_lock:
        if spec.process is not None:
            raise OperationError(
                "worker_shutdown_incomplete",
                f"worker on port {spec.port} could not be fully released",
                status=503,
            )
    if stop_event is not None and stop_event.is_set():
        raise OperationError(
            "server_shutting_down",
            "server shutdown is in progress; worker restart cancelled",
            status=503,
        )
    spec.failure_count = 0
    spec.models = list(planned_models)
    try:
        spawn_worker(spec, device=device, log_hub=log_hub)
        wait_for_ready(
            port=spec.port,
            timeout=120.0,
            stop_event=stop_event,
            expected_nonce=spec.worker_nonce,
            worker=spec,
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
    spec.restart_count = 0
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
    try:
        job = store.create(op_name, model_id)
    except JobStoreFullError as e:
        raise OperationError(
            "admin_job_capacity", str(e), status=503, retryable=True,
        ) from e
    except JobStoreShuttingDownError as e:
        raise OperationError(
            "server_shutting_down", str(e), status=503, retryable=True,
        ) from e
    if not op_args:
        op_args = (model_id,)
    thread = threading.Thread(
        target=op_fn,
        args=op_args,
        kwargs={"job": job, "store": store, **kwargs},
        daemon=True,
        name=f"muse-admin-{op_name}-{job.job_id}",
    )
    try:
        store.start_thread(job.job_id, thread)
    except JobStoreShuttingDownError as e:
        store.update(
            job.job_id,
            state="failed",
            error="admin job cancelled during server shutdown",
        )
        raise OperationError(
            "server_shutting_down", str(e), status=503, retryable=True,
        ) from e
    except Exception as e:  # noqa: BLE001
        store.update(
            job.job_id,
            state="failed",
            error="admin job thread could not be started",
        )
        raise OperationError(
            "admin_job_start_failed",
            "admin job thread could not be started",
            status=503,
            retryable=True,
        ) from e
    return job
