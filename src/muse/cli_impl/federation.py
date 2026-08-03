"""FastAPI coordinator app for muse federation (v1: model-locality routing).

`build_coordinator` assembles a small FastAPI app that fronts a fixed set
of remote muse `serve` nodes (each itself a gateway). Unlike the
single-host gateway (`muse.cli_impl.gateway`), the coordinator does not
own workers or a LoadDirector; it only reads a `NodeRegistry` snapshot
(polled on a background interval) and forwards each request to whichever
node currently serves the requested model.

Three explicit routes are registered BEFORE the catch-all proxy so they
are not shadowed by it (FastAPI dispatches by first-match registration
order, same discipline as `muse.cli_impl.gateway.build_gateway`):

  - GET /health: aggregate reachability across the node snapshot.
  - GET /v1/models: OpenAI-shape union of model ids across reachable
    nodes, plus muse-specific `loaded` / `nodes` extras.
  - GET /v1/federation/nodes: operator-facing per-node detail (loaded
    model list, in_flight, last-poll age).

The catch-all extracts `model` from the request, asks `select_node` for
the best node, and forwards. `_forward` is imported from the existing
gateway module and re-exported as a MODULE-LEVEL name here so tests can
`monkeypatch.setattr(fed, "_forward", fake_forward)` -- re-binding the
import target has no effect on the gateway module's own callers, and
this module always calls its OWN `_forward` name (not
`gateway._forward`), so the patch takes effect.

Failover: a `_forward` call that fails at CONNECT time (the node is
unreachable/down -- never a mid-stream failure, and never a request the
node already accepted; `_forward` only raises those exceptions
synchronously before it returns a Response; once it returns a
StreamingResponse the relay runs independently and this function is not
on the hook for it) retries ONCE against a different candidate for the
same model. If no other candidate exists, or the retry also fails, the
coordinator returns 502 `no_node_available`. A `httpx.ReadTimeout` (the
node accepted the request and is processing or stuck) is deliberately
NOT a failover trigger: retrying would re-execute a full, possibly
non-idempotent generation on a second node. A ReadTimeout is mapped to a
504 response without retrying.

`run_coordinator` is this module's other export: it resolves the node
list (CLI `--node` entries merged with an optional `federation.yaml`),
builds a `NodeRegistry` + the coordinator app, and blocks serving it via
uvicorn. It is the implementation behind the `muse federate` CLI command
in `muse.cli`.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from muse.cli_impl.gateway import _forward, _openai_error, extract_model_from_request
from muse.cli_impl.serve_util import run_uvicorn
from muse.federation.nodes import load_nodes
from muse.federation.registry import NodeRegistry
from muse.federation.router import select_node
from muse.federation.state import NodeState

# Re-bound as a module-level name (not read through `gateway._forward`)
# specifically so `monkeypatch.setattr(fed, "_forward", fake_forward)`
# takes effect: this module's code paths all call the bare `_forward`
# name below, which Python resolves against THIS module's globals.
_forward = _forward

# Connect-phase failures that make a node worth failing over from.
# Deliberately NOT httpx.HTTPError broadly: a worker that connects and
# then returns a 4xx/5xx response is not a transport failure, it is a
# real answer that should be relayed to the client, not retried against
# a different node. Also deliberately NOT httpx.ReadTimeout: a read
# timeout means the node ACCEPTED the request and is processing (or
# stuck) -- failing over would re-execute a full (possibly
# non-idempotent) generation on a second node. Only connect-phase
# failures (the node is unreachable/down) trigger failover; a request
# the node has already accepted is never retried elsewhere and any
# ReadTimeout maps to 504 without failover.
_FAILOVER_EXCEPTIONS = (httpx.ConnectError, httpx.ConnectTimeout)


def build_coordinator(
    registry: Any,
    *,
    timeout: float,
    rr_counter: dict | None = None,
) -> FastAPI:
    """Build the federation coordinator FastAPI app.

    `registry` is a `muse.federation.registry.NodeRegistry` (or anything
    exposing `.snapshot() -> list[NodeState]`; tests pass a plain fake).
    `timeout` bounds each forwarded request. `rr_counter` is an optional
    shared dict passed through to `select_node` for round-robin
    tie-breaking across requests. When the caller passes `None` (the
    default -- this is what `run_coordinator` does for the deployed
    coordinator), a fresh `{}` is created and stored on `app.state`
    instead, so a real deployment actually rotates same-model traffic
    across tied candidates rather than pinning every request to the
    alphabetically-first node. Pass an explicit dict to inject a shared
    counter (or to observe/reset it) from a test.

    Registry lifecycle is wired via a FastAPI lifespan context manager:
    the coordinator warms the snapshot with one `await
    registry.refresh_once()` BEFORE serving (best-effort: a slow or
    failing node at boot is caught and logged rather than crashing
    startup, since the background loop will retry it), then
    `registry.start()` runs (launching the background refresh task on
    the server's OWN event loop, which is what `NodeRegistry.start`
    needs -- `asyncio.ensure_future` requires a running loop), and
    `registry.aclose()` runs on shutdown. This only fires when the ASGI
    lifespan protocol actually runs (real `run_uvicorn` serving, or a
    `with TestClient(app) as c:` block); a bare `TestClient(app).get(...)`
    (as the existing route tests in this module use) never triggers
    lifespan, so those tests' plain-`.snapshot()`-only fakes (which don't
    implement `refresh_once`/`start`/`aclose`) are unaffected by this
    wiring.
    """
    rr_counter = {} if rr_counter is None else rr_counter

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            await registry.refresh_once()
        except Exception:
            logging.getLogger(__name__).warning(
                "federation: initial refresh_once failed at boot; serving "
                "with an empty snapshot until the background loop's first "
                "poll lands",
                exc_info=True,
            )
        registry.start()
        try:
            yield
        finally:
            await registry.aclose()

    app = FastAPI(title="Muse Federation Coordinator", lifespan=_lifespan)
    app.state.registry = registry
    app.state.timeout = timeout
    app.state.rr_counter = rr_counter

    @app.get("/health")
    async def health():
        states: list[NodeState] = registry.snapshot()
        nodes = [
            {
                "name": s.spec.name,
                "url": s.spec.url,
                "reachable": s.reachable,
                "model_count": len(s.models),
                "in_flight": s.in_flight,
            }
            for s in states
        ]
        status = "ok" if any(s.reachable for s in states) else "degraded"
        return {"status": status, "nodes": nodes}

    @app.get("/v1/models")
    async def list_models():
        states: list[NodeState] = registry.snapshot()
        # model_id -> (loaded_anywhere, [node names that have it])
        union: dict[str, tuple[bool, list[str]]] = {}
        for s in states:
            if not s.reachable:
                continue
            for model_id, avail in s.models.items():
                loaded_anywhere, names = union.get(model_id, (False, []))
                names = names + [s.spec.name]
                union[model_id] = (loaded_anywhere or avail.loaded, names)

        data = [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "muse",
                "loaded": loaded_anywhere,
                "nodes": names,
            }
            for model_id, (loaded_anywhere, names) in sorted(union.items())
        ]
        return {"object": "list", "data": data}

    @app.get("/v1/federation/nodes")
    async def federation_nodes():
        states: list[NodeState] = registry.snapshot()
        now = time.monotonic()
        nodes = [
            {
                "name": s.spec.name,
                "url": s.spec.url,
                "reachable": s.reachable,
                "loaded": sorted(
                    model_id for model_id, avail in s.models.items() if avail.loaded
                ),
                "in_flight": s.in_flight,
                "last_poll_age_seconds": max(0.0, now - s.last_poll_ts),
            }
            for s in states
        ]
        return {"nodes": nodes}

    @app.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def proxy(request: Request, full_path: str) -> JSONResponse:
        model_id = await extract_model_from_request(request)
        if model_id is None:
            return _openai_error(
                400, "model_required",
                "request is missing a `model` field (required for "
                "federation routing)",
            )

        states: list[NodeState] = registry.snapshot()
        node = select_node(model_id, states, rr_counter=app.state.rr_counter)
        if node is None:
            return _openai_error(
                404, "model_not_available",
                f"no node serves model {model_id!r}",
            )

        target_url = f"{node.spec.url}/{full_path}"
        try:
            return await _forward(request, target_url, app.state.timeout)
        except _FAILOVER_EXCEPTIONS:
            # Retry ONCE against another candidate for the same model,
            # excluding the failed node's url. Re-running select_node
            # against the filtered snapshot re-derives loaded-preference
            # and in-flight tie-break rather than blindly grabbing
            # "the other one" (there may be more than two candidates).
            remaining = [s for s in states if s.spec.url != node.spec.url]
            fallback = select_node(model_id, remaining, rr_counter=app.state.rr_counter)
            if fallback is None:
                return _openai_error(
                    502, "no_node_available",
                    f"node {node.spec.url!r} failed and no other node "
                    f"serves model {model_id!r}",
                    error_type="server_error",
                )
            fallback_url = f"{fallback.spec.url}/{full_path}"
            try:
                return await _forward(request, fallback_url, app.state.timeout)
            except _FAILOVER_EXCEPTIONS:
                return _openai_error(
                    502, "no_node_available",
                    f"both {node.spec.url!r} and {fallback.spec.url!r} "
                    f"failed for model {model_id!r}",
                    error_type="server_error",
                )
            except httpx.TimeoutException:
                return _openai_error(
                    504, "node_timeout",
                    f"node {fallback.spec.url!r} timed out while serving "
                    f"model {model_id!r}",
                    error_type="server_error",
                )
            except httpx.TransportError:
                return _openai_error(
                    502, "node_forward_failed",
                    f"node {fallback.spec.url!r} failed while serving "
                    f"model {model_id!r}",
                    error_type="server_error",
                )
        except httpx.TimeoutException:
            return _openai_error(
                504, "node_timeout",
                f"node {node.spec.url!r} timed out while serving "
                f"model {model_id!r}",
                error_type="server_error",
            )
        except httpx.TransportError:
            return _openai_error(
                502, "node_forward_failed",
                f"node {node.spec.url!r} failed while serving "
                f"model {model_id!r}",
                error_type="server_error",
            )

    return app


def run_coordinator(
    *,
    host: str,
    port: int,
    cli_nodes: list[str] | None,
    config_path: str | None,
) -> int:
    """Resolve federation nodes, build the coordinator app, and serve it.

    Node-list resolution order:
      1. `config_path` (the CLI `--config` flag) if given.
      2. Else `federation.config_file` (`MUSE_FEDERATION_CONFIG` / config.yaml).
      3. Else `<catalog_dir>/federation.yaml`, ONLY if that file actually
         exists (so a fresh install with no federation.yaml does not
         error trying to read a missing file -- `load_nodes` treats
         `config_path=None` as "no yaml source" and merges CLI nodes only).

    `load_nodes` then merges CLI `--node` entries with the resolved yaml
    file (CLI wins on url collision, per `muse.federation.nodes`).

    Returns 2 (and starts nothing) when the merged node list is empty --
    a coordinator with zero nodes would boot healthy and 404 every
    request, which is a worse failure mode than refusing to start.
    Otherwise builds the `NodeRegistry` + coordinator app (whose lifespan
    hook starts/stops the registry's background refresh loop, see
    `build_coordinator`) and blocks on uvicorn; returns 0 after a clean
    shutdown.
    """
    from muse.core import config

    resolved_config_path = config_path or config.get("federation.config_file")
    if resolved_config_path is None:
        catalog_dir = Path(config.get("paths.catalog_dir")).expanduser()
        candidate = catalog_dir / "federation.yaml"
        if candidate.exists():
            resolved_config_path = str(candidate)

    nodes = load_nodes(cli_nodes=cli_nodes, config_path=resolved_config_path)
    if not nodes:
        print(
            "no federation nodes: pass --node URL or create a federation.yaml",
            file=sys.stderr,
        )
        return 2

    registry = NodeRegistry(
        nodes,
        refresh_interval=config.get("federation.refresh_interval_seconds"),
        poll_timeout=config.get("federation.poll_timeout_seconds"),
    )
    app = build_coordinator(
        registry, timeout=config.get("federation.forward_timeout_seconds")
    )
    run_uvicorn(app, host=host, port=port)
    return 0
