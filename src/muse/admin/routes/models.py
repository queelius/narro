"""Per-model admin routes.

Endpoints (all under /v1/admin):

  POST  /models/{id}/enable      async; spawns or joins worker
  POST  /models/{id}/disable     sync; unloads from worker
  POST  /models/{id}/probe       async; runs measure
  POST  /models/_/pull           async; runs `muse pull <identifier>`
                                 (the `_` placeholder is a documented stub
                                 because resolver URIs do not survive
                                 path encoding well; identifier is in body)
  DELETE /models/{id}            sync; catalog removal (?purge=bool)
  GET   /models/{id}/status      sync; merged catalog + worker view

All routes return JSON with the OpenAI envelope on errors. Async ops
return 202 + {job_id, status}; clients poll /v1/admin/jobs/{job_id}.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse

from muse.admin.jobs import get_default_store
from muse.admin.operations import (
    OperationError,
    disable_model,
    enable_model,
    find_worker_for_model,
    launch_async,
    probe_model,
    pull_model,
    remove_model,
    warmup_model,
)
from muse.cli_impl.supervisor import get_supervisor_state
from muse.core.catalog import _read_catalog, known_models
from muse.core.errors import error_response


def _operation_error_to_response(e: OperationError) -> JSONResponse:
    return error_response(e.status, e.code, e.message)


def build_models_router() -> APIRouter:
    router = APIRouter()

    @router.post("/models/{model_id}/enable")
    def enable(model_id: str, _body: dict | None = Body(default=None)):
        state = get_supervisor_state()
        store = get_default_store()
        # Quick existence check up-front so unknown ids return 404
        # synchronously rather than via job.error.
        if model_id not in known_models():
            return error_response(
                404, "model_not_found", f"unknown model {model_id!r}",
            )
        try:
            job = launch_async(
                enable_model,
                op_name="enable",
                model_id=model_id,
                store=store,
                state=state,
            )
        except OperationError as e:
            return _operation_error_to_response(e)
        return JSONResponse(
            status_code=202,
            content={"job_id": job.job_id, "status": job.state},
        )

    @router.post("/models/{model_id}/disable")
    def disable(model_id: str, _body: dict | None = Body(default=None)):
        state = get_supervisor_state()
        try:
            return disable_model(model_id, state=state)
        except OperationError as e:
            return _operation_error_to_response(e)

    @router.post("/models/{model_id}/warmup")
    def warmup(model_id: str, _body: dict | None = Body(default=None)):
        """Pre-load `model_id` via the LoadDirector without serving traffic.

        Synchronous: the route returns the {"model_id", "worker_port"}
        dict inline once the director's warmup completes. Cold loads
        can take 10-60 seconds; the request blocks for that duration.
        For async warmup with progress polling, the future async-job
        wrapper (deferred to v1.next) would mirror the enable route's
        202 + job_id pattern.
        """
        state = get_supervisor_state()
        try:
            return warmup_model(model_id, state=state)
        except OperationError as e:
            return _operation_error_to_response(e)

    @router.post("/models/{model_id}/probe")
    def probe(model_id: str, body: dict | None = Body(default=None)):
        state = get_supervisor_state()
        store = get_default_store()
        if model_id not in known_models():
            return error_response(
                404, "model_not_found", f"unknown model {model_id!r}",
            )
        body = body or {}
        no_inference = bool(body.get("no_inference", False))
        device = body.get("device")
        try:
            job = launch_async(
                probe_model,
                op_name="probe",
                model_id=model_id,
                store=store,
                no_inference=no_inference,
                device=device,
            )
        except OperationError as e:
            return _operation_error_to_response(e)
        return JSONResponse(
            status_code=202,
            content={"job_id": job.job_id, "status": job.state},
        )

    @router.post("/models/{model_id}/pull")
    def pull(model_id: str, body: dict | None = Body(default=None)):
        """Pull `identifier` from the body, OR fall back to the path id.

        Documented usage: POST /v1/admin/models/_/pull with
        {"identifier": "hf://..."} so resolver URIs don't have to be
        path-encoded. If the body is missing/empty, the path id is
        used directly (works for bundled-script ids and curated aliases
        without `://`).
        """
        store = get_default_store()
        body = body or {}
        identifier = body.get("identifier") or model_id
        if not identifier or identifier == "_":
            return error_response(
                400,
                "missing_identifier",
                "pull requires an `identifier` in the request body or a non-`_` path",
            )
        # The job tracks the operand under model_id even when the path
        # was the `_` placeholder; that way job.model_id is the meaningful
        # identifier the user pulled. op_args supplies the positional
        # identifier argument that pull_model expects.
        try:
            job = launch_async(
                pull_model,
                op_name="pull",
                model_id=identifier,
                store=store,
                op_args=(identifier,),
            )
        except OperationError as e:
            return _operation_error_to_response(e)
        return JSONResponse(
            status_code=202,
            content={"job_id": job.job_id, "status": job.state},
        )

    @router.delete("/models/{model_id}")
    def delete(
        model_id: str,
        purge: bool = Query(default=False),
    ):
        state = get_supervisor_state()
        try:
            return remove_model(model_id, state=state, purge=purge)
        except OperationError as e:
            return _operation_error_to_response(e)

    @router.get("/models/{model_id}/status")
    def status(model_id: str):
        state = get_supervisor_state()
        catalog_known = known_models()
        catalog = _read_catalog()

        if model_id not in catalog_known and model_id not in catalog:
            return error_response(
                404, "model_not_found", f"unknown model {model_id!r}",
            )

        entry = catalog.get(model_id, {})
        modality = (
            catalog_known[model_id].modality if model_id in catalog_known else None
        )
        spec = find_worker_for_model(state, model_id)

        out: dict[str, Any] = {
            "model_id": model_id,
            "modality": modality,
            "enabled": entry.get("enabled", True) if entry else False,
            "loaded": spec is not None,
            "worker_port": spec.port if spec is not None else None,
            "worker_pid": _get_pid(spec),
            "worker_uptime_seconds": _get_uptime(spec),
            "worker_status": spec.status if spec is not None else None,
            "restart_count": spec.restart_count if spec is not None else 0,
            "last_error": None,
            "measurements": entry.get("measurements") or {},
        }
        return out

    return router


def _get_pid(spec) -> int | None:
    if spec is None or spec.process is None:
        return None
    pid = getattr(spec.process, "pid", None)
    return pid if isinstance(pid, int) else None


def _get_uptime(spec) -> float | None:
    if spec is None or not spec.last_spawn_at:
        return None
    return max(0.0, time.monotonic() - spec.last_spawn_at)
