"""FastAPI gateway: proxy requests by model-id to the right worker.

The gateway is the user-facing process (port 8000 by default). Workers
live on internal ports (9001+). The gateway:
  1. (v0.40.0+) Routes requests via state.director.acquire(model_id, manifest=...).
     The director may load the model on demand if it isn't currently
     loaded (lazy load), evict an LRU model under memory pressure, or
     short-circuit 503 if the requested model exceeds device capacity.
  2. Extracts `model` from each request (body for POST, query for GET)
  3. Forwards the request to the hosting worker, streaming the response (D4)
  4. Aggregates /v1/models and /health across all workers (D3)
  5. Mounts /v1/admin/* with bearer-token auth (v0.28.0+)

The legacy static-routes path (build_gateway(routes=...)) survives for
tests that pre-date the lazy-load wiring. When `state.director` is set,
the proxy skips the static map entirely and calls director.acquire on
every request; release fires on completion in a finally clause that
spans both buffered and streaming responses.

Proxy routing is modality-agnostic: any request with a `model` field
routes to the worker hosting that model, regardless of URL path. This
means future modalities (/v1/embeddings, /v1/audio/transcriptions, ...)
work without gateway changes.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Imported at module-top so tests can patch
# `muse.cli_impl.gateway.get_manifest` / `._read_catalog` to inject
# manifests or a catalog snapshot without touching state on disk.
# `_read_catalog` backs the /v1/models listing of enabled-but-unloaded
# models (v0.47.3); it is mtime-cached so per-request reads are cheap.
from muse.core import config
from muse.core.catalog import CatalogError, _read_catalog, get_manifest, is_enabled, known_models
from muse.core.discovery import model_optional_paths
from muse.core.errors import error_type_for_status
from muse.core.server import _format_loaded_at, build_model_entry

# Bare name (not `from muse.observability import recorder`) so tests can
# monkeypatch `muse.cli_impl.gateway.record` directly. `record` is itself
# fire-and-forget (see muse.observability.recorder), but the call site in
# `_route_via_director` still wraps it in try/except: telemetry must never
# break request forwarding, even if a future change to `record` regresses
# that guarantee.
from muse.observability.dashboard import build_dashboard_router
from muse.observability.recorder import record

# Module-top (no import cycle: supervisor imports gateway only lazily).
# `revalidate_servability` re-checks a stale boot unservable stamp against
# the live catalog; `backfill_manifest_memory` sizes the load from the
# catalog when the manifest declares no memory_gb. Both are on the request
# hot path, so importing per-request would pay a needless cost.
from muse.cli_impl.supervisor import (
    backfill_manifest_memory,
    revalidate_servability,
)

# OperationError lives at module-top because `_route_via_director` must
# trap director.acquire() failures; importing inside a hot-path function
# pays a re-import cost on every request even though Python caches the
# module object.
from muse.admin.operations import OperationError

# Request-queueing primitives (spec 2026-07-08). Module-top because the
# request path constructs a deadline + gate slot on every request; queueing.py
# imports only stdlib + config (no ML deps), so this is safe on the CLI import
# path.
from muse.cli_impl.queueing import QueueFull, QueueTimeout

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
_MAX_BUFFERED_WORKER_RESPONSE_BYTES = 1 * _MB
_WORKER_RESPONSE_CHUNK_BYTES = 64 * 1024
_MAX_MODEL_ID_CHARS = 512
_HOP_BY_HOP_HEADERS = frozenset({
    "connection",
    "keep-alive",
    "proxy-connection",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
})


class RequestBodyTooLarge(ValueError):
    """An inbound gateway request exceeded its configured byte budget."""

    def __init__(self, *, limit: int, observed: int) -> None:
        self.limit = limit
        self.observed = observed
        super().__init__(f"request body exceeds {limit} byte limit")


class RequestBodyLimitMiddleware:
    """Bound and replay every inbound HTTP body before route dispatch.

    Enforcement at the ASGI boundary covers explicit admin, aggregation,
    dashboard, and future routes as well as the catch-all inference proxy.
    The downstream app receives one replayable request message, so FastAPI
    body parsing and the proxy's own cache can consume it normally.
    """

    def __init__(self, app: Any, *, limit: int) -> None:
        if isinstance(limit, bool) or type(limit) is not int or limit <= 0:
            raise ValueError("request body limit must be a positive integer")
        self.app = app
        self.limit = limit

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        declared: int | None = None
        for raw_name, raw_value in scope.get("headers", []):
            if raw_name.lower() != b"content-length":
                continue
            try:
                parsed = int(raw_value)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                declared = parsed if declared is None else max(declared, parsed)
        if declared is not None and declared > self.limit:
            await self._reject(scope, receive, send)
            return

        chunks: list[bytes] = []
        total = 0
        terminal_message: dict | None = None
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                terminal_message = message
                break
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes):
                chunk = bytes(chunk)
            total += len(chunk)
            if total > self.limit:
                await self._reject(scope, receive, send)
                return
            if chunk:
                chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        replayed = False

        async def replay_receive() -> dict:
            nonlocal replayed
            if not replayed:
                replayed = True
                if terminal_message is not None:
                    return terminal_message
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _reject(self, scope: dict, receive: Any, send: Any) -> None:
        response = _openai_error(
            413,
            "request_too_large",
            f"request body exceeds the configured {self.limit} byte limit",
        )
        await response(scope, receive, send)


class ShutdownCancellationMiddleware:
    """Suppress only shutdown-driven HTTP task cancellation.

    Uvicorn may cancel handlers after its graceful-shutdown deadline.  Once
    the supervisor's stop event is set, that cancellation is expected control
    flow rather than an ASGI application failure.  Before response start we
    make a best-effort OpenAI-shaped 503; after start we simply let the
    connection close.  Client-disconnect cancellation while the server is
    otherwise running still propagates unchanged.
    """

    def __init__(self, app: Any, *, stop_event: Any) -> None:
        self.app = app
        self.stop_event = stop_event

    def _server_is_stopping(self) -> bool:
        is_set = getattr(self.stop_event, "is_set", None)
        return bool(callable(is_set) and is_set() is True)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def tracking_send(message: dict) -> None:
            nonlocal response_started
            await send(message)
            if message.get("type") == "http.response.start":
                response_started = True

        try:
            await self.app(scope, receive, tracking_send)
        except asyncio.CancelledError:
            if not self._server_is_stopping():
                raise
            if response_started:
                return
            response = _openai_error(
                503,
                "server_shutting_down",
                "request cancelled because the server is shutting down",
                error_type="server_error",
            )
            try:
                await response(scope, receive, send)
            except asyncio.CancelledError:
                # The transport may already be gone, or Uvicorn may issue a
                # second cancellation. Shutdown remains clean either way.
                return
            except Exception:  # noqa: BLE001
                return


class _OwnedStreamingResponse(StreamingResponse):
    """StreamingResponse whose ASGI call owns an idempotent async cleanup.

    A normal generator ``finally`` is insufficient when sending
    ``http.response.start`` fails before the first ``__anext__``.  The outer
    response lifecycle always runs this callback; the relay generator invokes
    the same callback for direct-iteration tests and early iterator close.
    """

    def __init__(
        self,
        *args: Any,
        cleanup: Callable[[], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        self._cleanup = cleanup
        super().__init__(*args, **kwargs)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._cleanup()


def _valid_model_id(value: Any) -> bool:
    """Return whether a routing key is a bounded, printable string."""
    return bool(
        isinstance(value, str)
        and value.strip()
        and len(value) <= _MAX_MODEL_ID_CHARS
        and not any(ord(char) < 32 or ord(char) == 127 for char in value)
    )


async def _prefetch_worker_response(
    response: httpx.Response,
    *,
    limit: int = _MAX_BUFFERED_WORKER_RESPONSE_BYTES,
) -> tuple[bytes | None, list[bytes], Any | None]:
    """Buffer a small worker response, or hand off a bounded prefix.

    ``httpx.Response.aread()`` has no byte ceiling.  A worker returning a
    large image, audio file, or malformed endless response could therefore
    make the gateway retain the entire body for every concurrent request.
    Read raw bytes in fixed-size chunks instead.  Bodies at or below ``limit``
    preserve the existing buffered ``Response`` path; larger bodies return a
    prefix plus the still-open iterator so the caller can relay the remainder
    with ``StreamingResponse``.

    Raw bytes are intentional: content-encoding headers are forwarded, so the
    gateway must not transparently decompress and then forward a stale header.
    """
    if limit <= 0:
        raise ValueError("worker response buffer limit must be positive")

    iterator = response.aiter_raw(
        chunk_size=_WORKER_RESPONSE_CHUNK_BYTES,
    ).__aiter__()
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = await iterator.__anext__()
        except StopAsyncIteration:
            return b"".join(chunks), [], None
        if not chunk:
            continue
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            return None, chunks, iterator


def _proxy_headers(
    headers: Any,
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[tuple[bytes, bytes]]:
    """Copy end-to-end headers, removing RFC hop-by-hop fields.

    In addition to the standard fixed names, HTTP/1.1 lets a sender nominate
    extension hop-by-hop headers through ``Connection``.  Forwarding either
    class through the gateway can leak connection-local state or confuse the
    next server/client in the chain. Raw pairs are retained so repeatable
    end-to-end fields (especially multiple ``Set-Cookie`` values) never collapse
    through a dict/``Headers.items()`` conversion.
    """
    raw = getattr(headers, "raw", None)
    if isinstance(raw, (list, tuple)):
        pairs = [(bytes(key), bytes(value)) for key, value in raw]
    else:
        pairs = [
            (
                str(key).encode("latin-1"),
                str(value).encode("latin-1"),
            )
            for key, value in headers.items()
        ]

    blocked = {
        name.lower().encode("ascii")
        for name in (_HOP_BY_HOP_HEADERS | exclude)
    }
    for key, value in pairs:
        if key.lower() != b"connection":
            continue
        blocked.update(
            token.strip().lower()
            for token in value.split(b",")
            if token.strip()
        )
    return [
        (key, value)
        for key, value in pairs
        if key.lower() not in blocked
    ]


def _install_response_headers(
    response: Response,
    headers: list[tuple[bytes, bytes]],
) -> Response:
    """Install raw forwarded pairs without losing generated safe headers."""
    forwarded_names = {key.lower() for key, _value in headers}
    generated = [
        (key, value)
        for key, value in response.raw_headers
        if key.lower() not in forwarded_names
    ]
    response.raw_headers = [*generated, *headers]
    return response


@dataclass(frozen=True)
class WorkerRoute:
    """One entry in the gateway's routing table.

    A worker may host multiple models; each gets its own WorkerRoute
    pointing at the same worker_url.
    """
    model_id: str
    worker_url: str


async def cache_bounded_request_body(request: Request, *, limit: int) -> bytes:
    """Read and cache at most ``limit`` bytes from an ASGI request stream.

    ``Content-Length`` is only an early rejection optimization; chunked or
    dishonest clients are still bounded while streaming.  Starlette normally
    sets ``request._body`` in ``Request.body()``.  We set the same cache after
    our bounded read so model extraction and the eventual worker forward see
    exactly one body without consuming the receive channel twice.
    """
    if limit <= 0:
        raise ValueError("request body limit must be positive")
    cached = getattr(request, "_body", None)
    if isinstance(cached, bytes):
        if len(cached) > limit:
            raise RequestBodyTooLarge(limit=limit, observed=len(cached))
        return cached

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except (TypeError, ValueError):
            declared = None
        if declared is not None and declared > limit:
            raise RequestBodyTooLarge(limit=limit, observed=declared)

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            raise RequestBodyTooLarge(limit=limit, observed=total)
        chunks.append(chunk)
    body = b"".join(chunks)
    request._body = body
    return body


async def extract_model_from_request(request: Any) -> Any:
    """Extract the `model` field from a request.

    - POST with JSON body: body["model"]
    - POST with multipart/form-data body: form["model"]
      (used by /v1/audio/transcriptions, /v1/audio/translations, and
      future multipart endpoints like /v1/images/edits.)
    - GET: query_params["model"]
    - Anything else: None

    Returns None (not raises) on missing/invalid. The caller decides
    what "no model specified" means (400, or fall back to default).
    """
    if request.method == "GET":
        return request.query_params.get("model")

    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body_bytes = await request.body()
                body = json.loads(body_bytes)
                if not isinstance(body, dict):
                    return None
                return body.get("model")
            except (json.JSONDecodeError, ValueError):
                return None
        if "multipart/form-data" in content_type:
            form = None
            try:
                # Read body FIRST so Starlette caches it on _body.
                # request.form() consumes the receive stream; without
                # caching, _forward's later request.body() raises
                # "Stream consumed". With _body set, stream() yields
                # the cached bytes and form() parses from those.
                await request.body()
                form = await request.form()
                model = form.get("model")
                # form values are strings or UploadFile; only the
                # string value is a valid model id.
                return model if isinstance(model, str) else None
            except Exception:  # noqa: BLE001
                return None
            finally:
                if form is not None:
                    try:
                        await form.close()
                    except Exception:  # noqa: BLE001
                        logger.debug("failed to close parsed multipart form", exc_info=True)

    return None


def _openai_error(
    status: int,
    code: str,
    message: str,
    *,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    """OpenAI-compatible error envelope.

    `error_type` defaults to `invalid_request_error` (the OpenAI shape for
    400 / 404). 5xx paths (lazy-load 503 "model_unservable" and
    "model_too_large_for_device", or any director-raised OperationError
    with status >= 500) pass `server_error` so SDK clients that branch on
    type can distinguish "this request is malformed" from "this server is
    full." This matches the chat_completion router's `server_error` choice
    for backend-stream failures (`muse.modalities.chat_completion.routes`).
    """
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "type": error_type}},
    )


def _resolve_default_model(modality: str) -> str | None:
    """First ENABLED catalog model whose modality matches `modality`.

    Backs the gateway's MODEL_OPTIONAL_PATHS seam (see `proxy` below):
    a request to a path like /v1/translate that omits `model` resolves
    here instead of 400ing model_required. Iterates `known_models()` in
    its discovery order (bundled scripts first, in glob-sorted order,
    then persisted resolver entries), so a bundled default (e.g. m2m100
    for text/translation) wins on a fresh install. `is_enabled` returns
    False both for a genuinely catalog-disabled model AND for a bundled
    script that was never pulled (no catalog.json entry at all), so an
    unpulled bundled model is correctly skipped rather than treated as
    servable.
    """
    for candidate_id, entry in known_models().items():
        if entry.modality == modality and is_enabled(candidate_id):
            return candidate_id
    return None


def build_gateway(
    routes: list[WorkerRoute] | None = None,
    timeout: float = 300.0,
    *,
    state: "object | None" = None,
    aggregation_timeout: float | None = None,
    max_request_body_bytes: int | None = None,
) -> FastAPI:
    """Build the gateway FastAPI app.

    Two routing modes:

      - state-driven (preferred for `muse serve`): pass `state`, a
        SupervisorState. Routes are re-derived per request from
        state.workers, filtered to status=='running'. Late-promoting
        workers join the routing table without an app rebuild and
        pending workers don't surface in /v1/models. The
        SupervisorState's lock guards each snapshot.
      - static: pass `routes`, a fixed list of WorkerRoute. Used by
        tests that want a frozen routing table.

    `aggregation_timeout` bounds the per-worker httpx client used by the
    /v1/models and /health fan-out (NOT the per-request forward timeout,
    which stays `timeout`). Defaults to `None`, which resolves through
    `muse.core.config` (`server.aggregation_timeout_seconds` /
    `MUSE_AGGREGATION_TIMEOUT_SECONDS`, itself defaulting to 5.0) so an
    explicit caller-supplied value always wins and an unconfigured
    deployment sees the exact same 5.0s behavior as before this knob
    existed.

    The app exposes:

      - aggregated /v1/models, /health (registered first so the catch-all
        proxy below doesn't shadow them)
      - admin /v1/admin/* (registered before the proxy so admin paths win;
        each admin route requires Authorization: Bearer <MUSE_ADMIN_TOKEN>)
      - catch-all /{full_path:path} that forwards by `model` field to the
        worker hosting that model.
    """
    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # Lifecycle: when uvicorn shuts the gateway down, drain any
        # outstanding admin job threads (pull subprocesses are SIGTERM'd
        # via JobStore.shutdown's join loop).
        try:
            yield
        finally:
            # Signal shutdown while Uvicorn's event loop is still alive.
            # Cold loads run in asyncio's default executor; their readiness
            # waits consume this event and unwind before asyncio joins that
            # executor. Without this early signal, Ctrl+C can appear to finish
            # Uvicorn while the supervisor remains hidden for up to 120s.
            if state is not None:
                state.stop_event.set()
            try:
                from muse.admin.jobs import get_default_store
                get_default_store().shutdown(timeout=5.0)
            except Exception as e:  # noqa: BLE001
                logger.warning("admin job-store shutdown failed: %s", e)

    app = FastAPI(title="Muse Gateway", lifespan=_lifespan)
    app.state.routes_state = state
    app.state.static_routes = (
        {r.model_id: r for r in routes} if routes is not None else {}
    )
    app.state.timeout = timeout
    app.state.max_request_body_bytes = (
        max_request_body_bytes
        if max_request_body_bytes is not None
        else int(config.get("server.max_request_body_mb")) * _MB
    )
    if (
        isinstance(app.state.max_request_body_bytes, bool)
        or type(app.state.max_request_body_bytes) is not int
        or app.state.max_request_body_bytes <= 0
    ):
        raise ValueError("max_request_body_bytes must be a positive integer")
    app.add_middleware(
        RequestBodyLimitMiddleware,
        limit=app.state.max_request_body_bytes,
    )
    stop_event = getattr(state, "stop_event", None) if state is not None else None
    if callable(getattr(stop_event, "is_set", None)):
        # Added after the body limiter so it is the outermost user middleware
        # and covers body reads, queue waits, route dispatch, and response-body
        # iteration rather than only selected awaits inside the proxy route.
        app.add_middleware(
            ShutdownCancellationMiddleware,
            stop_event=stop_event,
        )
    app.state.aggregation_timeout = (
        aggregation_timeout if aggregation_timeout is not None
        else config.get("server.aggregation_timeout_seconds")
    )

    def _routes_now() -> dict[str, WorkerRoute]:
        """Return the current routing table.

        State-driven: filter state.workers to status=='running' and build
        the table fresh; this means a worker that promotes after gateway
        boot becomes routable on the next request.
        Static: return the frozen dict captured at build_gateway time.
        """
        if app.state.routes_state is not None:
            out: dict[str, WorkerRoute] = {}
            s = app.state.routes_state
            with s.lock:
                for spec in s.workers:
                    if getattr(spec, "status", None) != "running":
                        continue
                    url = f"http://127.0.0.1:{spec.port}"
                    for m in spec.models:
                        out[m] = WorkerRoute(model_id=m, worker_url=url)
            return out
        return dict(app.state.static_routes)

    app.state.routes_now = _routes_now

    @app.get("/_gateway-info")
    def info():
        cur = app.state.routes_now()
        return {
            "routes": [
                {"model_id": r.model_id, "worker_url": r.worker_url}
                for r in cur.values()
            ],
        }

    @app.get("/v1/models")
    async def list_models():
        cur = app.state.routes_now()
        worker_urls = {r.worker_url for r in cur.values()}
        aggregated: list[dict] = []
        async with httpx.AsyncClient(timeout=app.state.aggregation_timeout) as client:
            async def _one(url: str) -> list[dict]:
                try:
                    r = await client.get(f"{url}/v1/models")
                    if r.status_code != 200:
                        return []
                    body = r.json()
                    if not isinstance(body, dict):
                        return []
                    data = body.get("data", [])
                    if not isinstance(data, list):
                        return []
                    return [item for item in data if isinstance(item, dict)]
                except Exception as e:  # noqa: BLE001
                    # Broad on purpose: a worker is untrusted input once it
                    # answers at all. httpx.HTTPError covers transport
                    # failures (down / timeout); a 200 response with a
                    # non-JSON or non-dict body raises ValueError /
                    # AttributeError out of r.json() / .get(), which must
                    # degrade this ONE worker's contribution to "nothing"
                    # rather than propagate through gather() and 500 the
                    # whole aggregated endpoint for every client.
                    logger.warning(
                        "worker %s unreachable or returned invalid data: %s",
                        url, e,
                    )
                    return []
            results = await asyncio.gather(
                *[_one(u) for u in worker_urls], return_exceptions=True,
            )
        for items in results:
            # _one() already catches everything it can; return_exceptions=True
            # is belt-and-suspenders so a future gap in _one degrades this one
            # worker instead of 500ing the endpoint.
            if isinstance(items, BaseException):
                logger.warning("worker aggregation task failed: %s", items)
                continue
            aggregated.extend(items)

        # v0.47.4: fill last_loaded_at on resident entries. Workers
        # self-report it as null (they run outside a supervisor and have
        # no director); the gateway owns the load timestamps via
        # state.director.loaded[id].loaded_at, so we join them in here
        # before appending the unloaded rows.
        state = app.state.routes_state
        if state is not None:
            _enrich_loaded_at(state, aggregated)
        # v0.47.3: also advertise enabled-but-unloaded catalog models so
        # /v1/models reflects the catalog, not just resident workers. The
        # loaded workers above are authoritative for loaded=True; here we
        # append every enabled catalog row not already present, with
        # loaded=False + its boot unservable_reason (if any). Skipped in
        # legacy static-routes mode (no SupervisorState to read the
        # unservable map from).
        if state is not None:
            aggregated.extend(_unloaded_catalog_entries(state, aggregated))
        return {"object": "list", "data": aggregated}

    @app.get("/health")
    async def health():
        cur = app.state.routes_now()
        worker_urls = {r.worker_url for r in cur.values()}
        modalities: set[str] = set()
        models: set[str] = set()
        any_down = False
        async with httpx.AsyncClient(timeout=app.state.aggregation_timeout) as client:
            async def _one(url: str) -> dict | None:
                try:
                    r = await client.get(f"{url}/health")
                    if r.status_code != 200:
                        return None
                    body = r.json()
                    return body if isinstance(body, dict) else None
                except Exception as e:  # noqa: BLE001
                    # See the matching comment in list_models: a 200 with a
                    # non-JSON body must degrade this worker to "down", not
                    # 500 the whole /health aggregation.
                    logger.warning(
                        "worker %s unreachable or returned invalid data: %s",
                        url, e,
                    )
                    return None
            results = await asyncio.gather(
                *[_one(u) for u in worker_urls], return_exceptions=True,
            )
        for body in results:
            if body is None or isinstance(body, BaseException):
                any_down = True
                continue
            raw_modalities = body.get("modalities", [])
            raw_models = body.get("models", [])
            if isinstance(raw_modalities, list):
                modalities.update(
                    value for value in raw_modalities if isinstance(value, str)
                )
            if isinstance(raw_models, list):
                models.update(
                    value for value in raw_models if isinstance(value, str)
                )
        # v0.47.4: also report enabled-but-unloaded catalog models so the
        # serviceable surface matches /v1/models. A request naming one of
        # these triggers an on-demand load; reporting only resident
        # workers under-stated what the gateway can actually serve.
        state = app.state.routes_state
        if state is not None:
            already = [{"id": m} for m in models]
            for entry in _unloaded_catalog_entries(state, already):
                models.add(entry["id"])
                if entry.get("modality"):
                    modalities.add(entry["modality"])
        return {
            "status": "degraded" if any_down else "ok",
            "modalities": sorted(modalities),
            "models": sorted(models),
        }

    # Admin routes mounted BEFORE the catch-all so /v1/admin/* paths
    # win the dispatch even though the proxy below would also match
    # them. Auth is enforced on every admin route via the Depends in
    # build_admin_router, so an anonymous /v1/admin/* request returns
    # 503 (no token configured) or 401/403 (token mismatch) before
    # touching any worker. Importing inside build_gateway keeps the
    # admin module out of the import path of muse.cli_impl.worker
    # (which loads inside per-model venvs that may not have fastapi
    # extras installed).
    from muse.admin.errors import install_admin_error_handler
    from muse.admin.routes import build_admin_router
    app.include_router(build_admin_router())
    # Unwrap the admin auth dependency's OpenAI-shaped HTTPException
    # details so /v1/admin/* auth errors emit a bare {"error": {...}}
    # envelope, matching the route-level admin errors instead of the
    # default {"detail": {"error": {...}}} double-wrap.
    install_admin_error_handler(app)

    # Dashboard router (/dashboard shell + gated /v1/telemetry/* JSON+SSE
    # endpoints). Gated on BOTH: telemetry.enabled (a global off-switch)
    # AND state is not None (the legacy static-routes test path passes no
    # SupervisorState, so there is no state.telemetry_store/log_hub for
    # the data endpoints to read). Mounting here is safe even before Task
    # 11 wires telemetry_store/log_hub onto SupervisorState: the /dashboard
    # HTML shell itself never touches state, only the gated data endpoints
    # do, and those aren't hit until a client actually requests them.
    if config.get("telemetry.enabled") and state is not None:
        app.include_router(build_dashboard_router(state))

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(request: Request, full_path: str):
        # NOTE: aggregated endpoints (/v1/models, /health) land in D3 as
        # explicit routes registered BEFORE this catch-all, so they win
        # via FastAPI's registration order.

        try:
            await cache_bounded_request_body(
                request, limit=app.state.max_request_body_bytes,
            )
        except RequestBodyTooLarge as exc:
            return _openai_error(
                413,
                "request_too_large",
                f"request body exceeds the configured {exc.limit} byte limit",
            )

        model_id = await extract_model_from_request(request)
        if model_id is not None and not _valid_model_id(model_id):
            return _openai_error(
                400,
                "invalid_model",
                "`model` must be a non-empty string of at most "
                f"{_MAX_MODEL_ID_CHARS} characters without control characters",
            )
        if model_id is None:
            # MODEL_OPTIONAL_PATHS seam (spec 2026-07-09): a modality
            # package may declare paths where `model` is optional (e.g.
            # text_translation's /v1/translate, /translate, /languages --
            # LibreTranslate clients never send a model field). Any OTHER
            # path with no model keeps today's 400 model_required exactly.
            tag = model_optional_paths().get(request.url.path)
            if tag is None:
                return _openai_error(
                    400, "model_required",
                    "request is missing a `model` field (required for gateway routing)",
                )
            default_id = _resolve_default_model(tag)
            if default_id is None:
                return _openai_error(
                    503, "no_default_model",
                    f"no enabled model of modality {tag!r} is available to "
                    f"serve {request.url.path!r}; pull and enable one, or "
                    f"pass an explicit `model` field",
                    error_type="server_error",
                )
            logger.debug(
                "gateway: resolved default model %r (modality %r) for %r",
                default_id, tag, request.url.path,
            )
            model_id = default_id

        # Director-driven path (v0.40.0+ lazy load). When the SupervisorState
        # carries a LoadDirector, every request acquires the worker port
        # via director.acquire and releases on completion. The director
        # transparently handles cold load, LRU eviction, and singleton-
        # load coordination. Static-routes mode (build_gateway(routes=...)
        # with no state.director) still works for the legacy test path.
        s = app.state.routes_state
        if s is not None and getattr(s, "director", None) is not None:
            return await _route_via_director(
                request, full_path, model_id, s, app.state.timeout,
            )

        # Legacy static-routes path: look up the worker URL in a frozen
        # dict and forward. No acquire/release. Used by tests that
        # pre-date Task F.
        cur = app.state.routes_now()
        route = cur.get(model_id)
        if route is None:
            return _openai_error(
                404, "model_not_found",
                f"model {model_id!r} is not registered with any worker; "
                f"known: {sorted(cur)}",
            )

        target_url = f"{route.worker_url}/{full_path}"
        try:
            return await _forward(request, target_url, app.state.timeout)
        except httpx.TimeoutException:
            return _openai_error(
                504, "worker_timeout",
                f"worker for model {model_id!r} timed out",
                error_type="server_error",
            )
        except httpx.TransportError:
            return _openai_error(
                502, "worker_unavailable",
                f"worker for model {model_id!r} is unavailable",
                error_type="server_error",
            )

    return app


def _enrich_loaded_at(state: Any, aggregated: list[dict]) -> None:
    """Fill last_loaded_at on resident /v1/models entries, in place.

    Per-model workers self-report last_loaded_at=None because they run
    outside a supervisor and own no LoadDirector. The gateway holds the
    load timestamps in state.director.loaded[id].loaded_at (monotonic
    seconds), so it joins them onto the worker-reported rows here.

    Best-effort and non-fatal: a missing director, a non-dict `loaded`
    (e.g. a bare MagicMock in tests), or any read failure leaves the
    entries untouched. Entries that already carry a non-null
    last_loaded_at are never overwritten.
    """
    director = getattr(state, "director", None)
    if director is None:
        return
    loaded = getattr(director, "loaded", None)
    if not isinstance(loaded, dict):
        return
    # Snapshot id -> loaded_at under the director lock (held briefly); the
    # monotonic->wall-clock render happens outside the lock.
    lock = getattr(director, "lock", None)
    try:
        if lock is not None:
            with lock:
                snap = {
                    mid: getattr(e, "loaded_at", None)
                    for mid, e in loaded.items()
                }
        else:
            snap = {
                mid: getattr(e, "loaded_at", None)
                for mid, e in loaded.items()
            }
    except Exception as e:  # noqa: BLE001
        logger.warning("/v1/models: loaded_at join failed: %s", e)
        return
    for entry in aggregated:
        if entry.get("last_loaded_at") is not None:
            continue
        mono = snap.get(entry.get("id"))
        if mono is None:
            continue
        entry["last_loaded_at"] = _format_loaded_at({"loaded_at": mono})


def _unloaded_catalog_entries(state: Any, loaded: list[dict]) -> list[dict]:
    """Build /v1/models entries for enabled-but-unloaded catalog models.

    `loaded` is the entries already aggregated from resident workers; their
    ids are skipped. Returns one entry per enabled catalog row with a
    `python_path` that isn't already loaded, marked loaded=False with its
    boot `unservable_reason` (if any). Best-effort: a catalog read failure
    yields no extra entries rather than failing the endpoint.
    """
    loaded_ids = {e.get("id") for e in loaded}
    try:
        catalog = _read_catalog()
    except Exception as e:  # noqa: BLE001
        logger.warning("/v1/models: catalog read failed: %s", e)
        return []
    with state.lock:
        reasons = dict(getattr(state, "unservable_reasons", {}) or {})
    out: list[dict] = []
    for model_id, entry in catalog.items():
        if model_id in loaded_ids:
            continue
        if not entry.get("enabled", True):
            continue
        if not entry.get("python_path"):
            continue  # pre-worker entry; cannot load
        try:
            manifest = get_manifest(model_id)
        except KeyError:
            manifest = entry.get("manifest", {}) or {}
        modality = manifest.get("modality") or entry.get("modality", "")
        out.append(build_model_entry(
            model_id, modality, manifest,
            loaded=False, last_loaded_at=None,
            unservable_reason=reasons.get(model_id),
        ))
    return out


def _director_headroom(director: Any) -> dict[str, float]:
    """Extract `{gpu_headroom_gb, cpu_headroom_gb}` kwargs from `director`.

    Used to thread the LoadDirector's OWN configured headroom into
    `revalidate_servability` so the gateway's servability gate and the
    director's admit/evict fit check never diverge (see
    `_route_via_director` step 1). Only includes a key when the attribute
    is present AND numeric: a bare `MagicMock()` director (as several
    tests construct) auto-creates a `MagicMock` attribute for any name
    accessed, which is not a float and would blow up the headroom
    subtraction in `_available_pools`. Omitting the key in that case falls
    through to `revalidate_servability`'s own hardcoded defaults, matching
    pre-fix behavior for tests that don't care about headroom.
    """
    kwargs: dict[str, float] = {}
    gpu = getattr(director, "gpu_headroom_gb", None)
    if isinstance(gpu, (int, float)):
        kwargs["gpu_headroom_gb"] = float(gpu)
    cpu = getattr(director, "cpu_headroom_gb", None)
    if isinstance(cpu, (int, float)):
        kwargs["cpu_headroom_gb"] = float(cpu)
    return kwargs


def _cold_model_needs_servability_check(director: Any, model_id: str) -> bool:
    """Whether a real director has no resident entry for ``model_id``.

    Boot validation cannot see models pulled or materially edited after the
    supervisor starts.  Cold requests therefore re-check permanent fit even
    when no old stamp exists.  Loose test doubles expose a fabricated
    ``loaded`` mock; only a concrete dict opts into this additional path.
    """
    loaded = getattr(director, "loaded", None)
    if not isinstance(loaded, dict):
        return False
    lock = getattr(director, "lock", None)
    if lock is None:
        return model_id not in loaded
    with lock:
        return model_id not in loaded


async def _route_via_director(
    request: Request,
    full_path: str,
    model_id: str,
    state: Any,
    timeout: float,
) -> Response:
    """Director-driven request routing for v0.40.0 lazy load.

    Order of operations:

      1. Re-check permanent servability when the boot snapshot carries an
         unservable stamp OR the model is cold (it may have been pulled or
         edited after boot). If unservable, return 503 without calling the
         director; transient pressure is not stamped and proceeds to the
         director's admission/eviction path.
      2. Resolve the manifest via `muse.core.catalog.get_manifest`. The
         director's load + eviction decisions read `capabilities.memory_gb`
         and `capabilities.device` from this dict. KeyError (model not in
         catalog) -> 404 model_not_found. CatalogError (catalog.json is
         corrupt and no last-known-good cache exists) -> 503
         catalog_unavailable, so a corrupt catalog degrades to a clean
         OpenAI-shaped error instead of an uncaught 500.
      3. Call `state.director.acquire(model_id, manifest=...)`. This may
         block on cold load + eviction. On `OperationError` (the director's
         user-facing failure type, e.g. model_too_large_for_device), map
         to the corresponding HTTP status. Other exceptions propagate to
         FastAPI's default 500 path.
      4. On a successful acquire, forward the request to the returned port.
         The release call MUST fire on completion regardless of outcome.
         Buffered responses release before returning. Streaming responses
         release at end-of-iteration (NOT at first-chunk dispatch), with an
         outer response-lifecycle callback covering a downstream send failure
         before the relay generator is first advanced.
    """
    # Explicit model ids obey the same catalog activation flag as default
    # resolution.  Historically only optional-model routing called
    # is_enabled(), so a disabled model could still cold-load by name.
    try:
        catalog_entry = _read_catalog().get(model_id)
    except CatalogError as exc:
        return _openai_error(
            503,
            "catalog_unavailable",
            f"model catalog is temporarily unavailable: {exc}",
            error_type="server_error",
        )
    if (
        isinstance(catalog_entry, dict)
        and not bool(catalog_entry.get("enabled", True))
    ):
        return _openai_error(
            409,
            "model_disabled",
            f"model {model_id!r} is disabled; enable it before requesting it",
        )

    # 1. Permanent-servability short-circuit (BEFORE acquire). The boot-time
    # validate_catalog_at_boot snapshot can go stale: a `muse models probe`
    # (or weights landing on disk) that arrives after boot makes a
    # previously-unsizable model sizable, while manifest, placement, or
    # budget changes can alter its permanent fit. When a stamp exists,
    # re-derive the full verdict for THIS one model against the LIVE catalog
    # and current capacity ceiling, so those changes take effect without a
    # supervisor restart. revalidate_servability clears the stamp (returns
    # None) only when the model is sizable AND fits; a genuine "exceeds
    # device capacity" stamp is RETAINED, so an impossible-to-fit model 503s
    # here and never reaches the director's eviction loop (which would
    # otherwise evict the whole idle working set before 503'ing). Reading +
    # mutating the dict happens under state.lock inside revalidate_servability.
    # A cold model with no stamp is checked too: it may have been pulled or
    # grown after boot and could otherwise wipe the idle working set before
    # the director discovered that it exceeds physical capacity.
    with state.lock:
        unservable_reason = state.unservable_reasons.get(model_id)
    needs_servability_check = (
        unservable_reason is not None
        or _cold_model_needs_servability_check(state.director, model_id)
    )
    if needs_servability_check:
        # Thread the SAME headroom state.director was built with, not
        # revalidate_servability's hardcoded 1.0/2.0 defaults. Otherwise an
        # operator-configured server.gpu_headroom_gb / cpu_headroom_gb
        # (MUSE_GPU_HEADROOM_GB / MUSE_CPU_HEADROOM_GB) makes this gate use
        # a DIFFERENT headroom than the director's own admit/evict fit
        # check, so the gate can falsely 503 a model the director would
        # actually admit (or vice versa). state.director is guaranteed
        # non-None here (checked by the caller before dispatching to this
        # function).
        try:
            unservable_reason = revalidate_servability(
                state, model_id, **_director_headroom(state.director),
            )
        except CatalogError as exc:
            logger.error(
                "servability check for %r failed: catalog is unavailable: %s",
                model_id, exc,
            )
            return _openai_error(
                503,
                "catalog_unavailable",
                f"model catalog is temporarily unavailable: {exc}",
                error_type="server_error",
            )
    if unservable_reason is not None:
        return _openai_error(
            503,
            "model_unservable",
            f"model {model_id!r} cannot be served: {unservable_reason}",
            error_type="server_error",
        )

    # 2. Resolve manifest. Per-request lookup is cheap: `known_models()`
    # memoizes its merge against catalog.json's (path, mtime_ns) and
    # `_read_catalog()` is mtime-cached the same way, so unchanged
    # catalogs return immediately. Any catalog write -- including the
    # admin pull endpoint's `muse pull` subprocess or an operator's CLI
    # pull beside this supervisor -- bumps the mtime and both re-read, so
    # models pulled after boot route without a restart. KeyError -> 404
    # model_not_found.
    try:
        manifest = get_manifest(model_id)
    except KeyError:
        return _openai_error(
            404, "model_not_found",
            f"model {model_id!r} is not in the catalog",
        )
    except CatalogError as exc:
        # catalog.json is corrupt and no last-known-good cache exists
        # (muse.core.catalog._read_catalog's corrupt-guard). Pre-fix this
        # propagated uncaught to FastAPI's default handler, surfacing a
        # bare {"detail": "Internal Server Error"} instead of muse's
        # OpenAI-shaped envelope. Log once here (no traceback needed;
        # _read_catalog already logged the underlying corruption) and
        # degrade to a clean 503, matching the model_unservable shape
        # used elsewhere in this function.
        logger.error(
            "get_manifest(%r) failed: catalog is unavailable: %s",
            model_id, exc,
        )
        return _openai_error(
            503,
            "catalog_unavailable",
            f"model catalog is temporarily unavailable: {exc}",
            error_type="server_error",
        )

    # 2b. Size the load. The director sizes loads + drives LRU eviction
    # from capabilities.memory_gb; a probed-only or never-probed model
    # declares none, so backfill it from the catalog (probe measurement,
    # else on-disk weights) -- otherwise the director treats the model as
    # 0 GB and never evicts to make room for it.
    try:
        manifest = backfill_manifest_memory(
            manifest, model_id, supervisor_device=state.device,
        )
    except CatalogError as exc:
        # The catalog can change between get_manifest() and the sizing
        # backfill. Preserve the same clean failure contract at both reads.
        logger.error(
            "capacity backfill for %r failed: catalog is unavailable: %s",
            model_id, exc,
        )
        return _openai_error(
            503,
            "catalog_unavailable",
            f"model catalog is temporarily unavailable: {exc}",
            error_type="server_error",
        )

    # 3. Queueing (spec 2026-07-08). ONE deadline covers the concurrency-gate
    # wait + the capacity wait + all retries. queue_timeout_seconds == 0
    # degrades to today's no-wait behavior: an occupied slot / capacity 503
    # surfaces immediately. `queued_ms` is initialized here so it is always
    # defined by the time the telemetry record() call runs below. Per the
    # spec, queued_ms measures ONLY time parked in the gate + capacity
    # waits -- NOT the successful director.acquire cold-load span that
    # follows, which can run tens of seconds and would otherwise conflate
    # queue delay with load time. gate_wait_seconds and
    # capacity_wait_seconds are each measured tightly around their own
    # await below, not via a single wall-clock span across this whole block.
    queued_ms = 0.0
    queue_budget = config.get("server.queue_timeout_seconds") or 0.0
    queue_t0 = time.monotonic()
    deadline = queue_t0 + max(0.0, float(queue_budget))
    gate = state.concurrency_gate
    cap = _effective_max_concurrency(manifest)

    # 3a. Concurrency gate: take one per-model slot (no-op when cap is
    # None/<=0). QueueFull when the per-model queue is already at
    # server.max_queue_depth; QueueTimeout when no slot frees within budget.
    # gate_wait_seconds is measured tightly around this one await so it
    # never picks up any of the cold-load time that follows (see the
    # capacity-wait measurement below for the matching discipline).
    gate_wait_t0 = time.monotonic()
    try:
        slot_taken = await gate.acquire_slot(model_id, cap, deadline=deadline)
    except QueueTimeout:
        # Report the ACTUAL elapsed wait, not the configured budget: an
        # early fast-fail would otherwise claim "waited 300s" (v0.57.3).
        return _openai_error(
            503, "queue_timeout",
            f"waited {time.monotonic() - queue_t0:.0f}s for a slot on "
            f"model {model_id!r} (queue depth {gate.depth(model_id)})",
            error_type="server_error",
        )
    except QueueFull as exc:
        return _openai_error(
            503, "queue_full",
            f"queue for model {model_id!r} is full (depth {exc.depth})",
            error_type="server_error",
        )
    gate_wait_seconds = time.monotonic() - gate_wait_t0

    slot_released = False

    def _release_slot() -> None:
        # Exactly-once slot release: the flag guards against the forward leg's
        # multiple release sites (buffered finally, stream-relay finally,
        # early-failure except) all calling this. release_slot is itself
        # threadsafe (schedules on the gate's captured loop when off-loop) and
        # over-release-safe, so this is belt-and-suspenders. Release only
        # when acquire_slot actually took a slot (#332): re-deriving the cap
        # here would desync if server.default_max_concurrency changed live.
        nonlocal slot_released
        if slot_released or not slot_taken:
            return
        slot_released = True
        gate.release_slot(model_id)

    # 3b. Acquire, OFF the event loop and COALESCED per model, wrapped in a
    # bounded capacity wait. director.acquire may block for tens of seconds on
    # a cold load; running it synchronously would freeze the single gateway
    # event loop (v0.50.3). _acquire_coalesced elects ONE loader per cold model
    # (#319). _acquire_with_capacity_wait parks on a retryable capacity 503
    # until a release/eviction fires capacity_notifier or the deadline lapses.
    try:
        worker_port, capacity_wait_seconds = await _acquire_with_capacity_wait(
            lambda: _acquire_coalesced(state, model_id, manifest),
            state.capacity_notifier, deadline=deadline, model_id=model_id,
        )
    except QueueTimeout:
        _release_slot()
        return _openai_error(
            503, "queue_timeout",
            f"waited {time.monotonic() - queue_t0:.0f}s for capacity for "
            f"model {model_id!r} (queue depth {gate.depth(model_id)})",
            error_type="server_error",
        )
    except OperationError as exc:
        _release_slot()
        return _openai_error(
            exc.status, exc.code, exc.message,
            error_type=error_type_for_status(exc.status),
        )
    except Exception:  # noqa: BLE001
        # A non-OperationError out of director.acquire (e.g. a pynvml
        # hiccup inside _decide) must not surface as a raw 500 to the
        # request that happened to elect the LOADER role, while same-model
        # WAITERS on the identical failure get a mapped 503 via
        # _acquire_coalesced's mapless-failure branch. Release the slot,
        # log the real exception, and return the SAME status + code waiters
        # see so loader and waiters degrade identically.
        _release_slot()
        logger.error(
            "director.acquire(%r) raised an unexpected error",
            model_id, exc_info=True,
        )
        return _openai_error(
            503, "model_load_failed",
            f"load of {model_id!r} failed",
            error_type="server_error",
        )
    except asyncio.CancelledError:
        # The off-loop callback releases any refcount acquired after this
        # coroutine is cancelled. During normal client disconnects preserve
        # cancellation semantics. During supervisor shutdown, Uvicorn sets
        # stop_event before cancelling overdue handlers; return a clean 503
        # so shutdown does not log an ASGI exception and emit a bare 500.
        _release_slot()
        stop_event = getattr(state, "stop_event", None)
        is_set = getattr(stop_event, "is_set", None)
        if callable(is_set) and is_set() is True:
            return _openai_error(
                503, "server_shutting_down",
                "request cancelled because the server is shutting down",
                error_type="server_error",
            )
        raise
    except BaseException:
        _release_slot()
        raise
    queued_ms = (gate_wait_seconds + capacity_wait_seconds) * 1000.0

    # 4. Forward. The release calls (director.release + the gate slot release)
    # are wired into the response shape: for buffered responses, fire in a
    # finally clause once the body is read; for SSE streams, fire from inside
    # the relay generator's finally clause when the upstream iteration ends (or
    # raises). Both paths converge in `_forward_with_release`.
    target_url = f"http://127.0.0.1:{worker_port}/{full_path}"
    t0 = time.monotonic()
    try:
        response = await _forward_with_release(
            request, target_url, timeout,
            director=state.director, model_id=model_id,
            extra_release=_release_slot,
        )
    except asyncio.CancelledError:
        stop_event = getattr(state, "stop_event", None)
        is_set = getattr(stop_event, "is_set", None)
        if callable(is_set) and is_set() is True:
            return _openai_error(
                503, "server_shutting_down",
                "request cancelled because the server is shutting down",
                error_type="server_error",
            )
        raise
    except httpx.TimeoutException:
        return _openai_error(
            504, "worker_timeout",
            f"worker for model {model_id!r} timed out",
            error_type="server_error",
        )
    except httpx.TransportError:
        return _openai_error(
            502, "worker_unavailable",
            f"worker for model {model_id!r} is unavailable",
            error_type="server_error",
        )
    latency_ms = (time.monotonic() - t0) * 1000.0
    stream = isinstance(response, StreamingResponse)
    # Fire-and-forget: telemetry must NEVER break request forwarding, even
    # if `record` or `_modality_from_path` regresses. Latency semantic:
    # for buffered responses this is the full request duration; for
    # streams it is time-to-response-object only, since
    # StreamingResponse returns before the body actually streams (the
    # `stream` flag records which one this event measured).
    try:
        record(
            "request",
            model_id=model_id,
            modality=_modality_from_path(full_path),
            latency_ms=latency_ms,
            queued_ms=queued_ms,
            status=getattr(response, "status_code", None),
            stream=stream,
        )
    except Exception:  # noqa: BLE001
        logger.warning("telemetry record('request') failed", exc_info=True)
    return response


def _modality_from_path(full_path: str) -> str:
    """Derive a modality label from the request path, structurally.

    `full_path` has no leading slash (e.g. "v1/chat/completions",
    "v1/images/generations", "v1/embeddings"). Strips a leading "v1/" if
    present, then keeps the first two remaining segments (or the one
    segment if that's all there is) as the label, e.g. "chat/completions",
    "images/generations", "embeddings". Deliberately NOT a hardcoded
    per-route lookup table: new routes get a sensible label for free.
    """
    path = full_path.split("?", 1)[0].strip("/")
    parts = path.split("/") if path else []
    if parts and parts[0] == "v1":
        parts = parts[1:]
    return "/".join(parts[:2]) if parts else "unknown"


def _effective_max_concurrency(manifest: dict) -> int | None:
    """Per-model concurrency cap: capabilities.max_concurrency, else the
    config default, else None (unlimited). Lenient: junk values -> next tier.

    Resolution (spec 2026-07-08):
      1. manifest capabilities.max_concurrency (model author's declared cap)
      2. server.default_max_concurrency (env MUSE_DEFAULT_MAX_CONCURRENCY)
      3. None -> unlimited (the gate no-ops, today's behavior)
    A non-positive or non-int value at any tier falls through to the next.
    """
    caps = (manifest or {}).get("capabilities") or {}
    declared = caps.get("max_concurrency")
    if declared is not None:
        try:
            n = int(declared)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    default = config.get("server.default_max_concurrency") or 0
    try:
        return int(default) if int(default) > 0 else None
    except (TypeError, ValueError):
        return None


async def _acquire_with_capacity_wait(acquire_once, notifier, *, deadline,
                                      model_id: str) -> tuple[int, float]:
    """Bounded retry around one acquire attempt (spec 2026-07-08).

    `acquire_once` is a zero-arg async callable (the coalesced acquire).
    On a retryable capacity OperationError: park on the notifier's
    generation event (armed BEFORE the attempt, so a release that lands
    mid-attempt still wakes us), bounded by the shared deadline, then
    retry. Non-retryable errors and every other exception propagate
    unchanged. Deadline exhaustion raises QueueTimeout.

    Returns `(worker_port, waited_seconds)`, where `waited_seconds` is the
    total time spent parked in `event.wait()` across all retries (0.0 if
    the first attempt succeeds). This excludes the time spent inside
    `acquire_once()` itself (e.g. a successful cold load), so callers can
    report queue delay separately from load time.
    """
    waited_seconds = 0.0
    while True:
        event = notifier.snapshot()  # arm first: no missed wakeup
        try:
            port = await acquire_once()
            return port, waited_seconds
        except OperationError as exc:
            if not getattr(exc, "retryable", False):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise  # zero-budget: surface today's immediate 503
            wait_t0 = time.monotonic()
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                raise QueueTimeout(model_id) from None
            finally:
                waited_seconds += time.monotonic() - wait_t0


async def _acquire_off_loop(state, model_id, manifest, *, on_settle=None) -> int:
    """Run director.acquire OFF the event loop, cancellation-safe.

    Shared by the coalescing loader and by each waiter's own re-acquire.
    Returns the worker port; propagates OperationError on a director failure.
    `on_settle`, if given, is attached as a done-callback to the acquire
    future so the loader can settle its coalescing gate ALWAYS -- even if the
    loader coroutine is cancelled, shield keeps the acquire (and thus the
    callback) alive, so waiters never hang.
    """
    director = state.director
    acquire_future = asyncio.ensure_future(
        asyncio.to_thread(director.acquire, model_id, manifest=manifest)
    )
    if on_settle is not None:
        acquire_future.add_done_callback(on_settle)
    try:
        return await asyncio.shield(acquire_future)
    except asyncio.CancelledError:
        # Cancelled (client disconnect / timeout) while the load ran in its
        # detached thread. shield kept it running; if it SUCCEEDS it bumped
        # refcount, so release once it settles or the model is pinned forever.
        def _release_if_acquired(fut: "asyncio.Future") -> None:
            if not fut.cancelled() and fut.exception() is None:
                try:
                    director.release(model_id)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "release after cancelled acquire failed for %r",
                        model_id, exc_info=True,
                    )
        acquire_future.add_done_callback(_release_if_acquired)
        raise


async def _acquire_coalesced(state, model_id, manifest) -> int:
    """Coalesce concurrent cold acquires of the SAME model (#319).

    The loader election is await-free (a dict get + set on the one event-loop
    thread), so a plain dict needs no lock and there is no two-loader TOCTOU.
    At most ONE thread parks per model-load; same-model waiters park on the
    loop (await a shared Future), not in the ThreadPoolExecutor.
    """
    gates = state.cold_load_gates
    gate = gates.get(model_id)
    if gate is None:
        # LOADER: create + store the gate BEFORE any await (atomic election),
        # then run the one off-loop load. _settle owns the gate lifecycle.
        loop = asyncio.get_running_loop()
        gate = loop.create_future()
        gates[model_id] = gate

        def _settle(acq_fut: "asyncio.Future") -> None:
            # Runs on the loop thread when the load settles. Compare-and-remove
            # (only this Future's owner deletes its entry, so a re-elected
            # loader's gate is never clobbered). NEVER set_exception -- a pure
            # ("ok"|"fail", info) signal avoids "exception never retrieved"
            # noise when there are zero waiters.
            if gates.get(model_id) is gate:
                del gates[model_id]
            if gate.done():
                return
            if acq_fut.cancelled():
                gate.set_result(("fail", None))
            elif acq_fut.exception() is not None:
                exc = acq_fut.exception()
                # v0.57.3 (review finding): carry `retryable` through the
                # tuple. Without it a waiter rebuilds an OperationError
                # whose retryable defaults to False and fast-fails a
                # transient capacity 503 the loader itself parks on --
                # defeating the capacity-wait for same-model cold bursts.
                info = (
                    (exc.status, exc.code, exc.message,
                     getattr(exc, "retryable", False))
                    if isinstance(exc, OperationError) else None
                )
                gate.set_result(("fail", info))
            else:
                gate.set_result(("ok", None))

        return await _acquire_off_loop(state, model_id, manifest, on_settle=_settle)

    # WAITER: await the shared gate WITHOUT a thread. shield so cancelling THIS
    # waiter cannot cancel the shared gate and poison the whole group.
    await asyncio.shield(gate)
    status, info = gate.result()
    if status == "fail":
        # Propagate the loader's failure; do NOT re-acquire. A retry herd would
        # re-park N-1 threads -- the exact #319 pathology. A mapped failure
        # carries the loader's own (status, code, message); a rare UNMAPPED one
        # (director.acquire raised a non-OperationError, e.g. a pynvml hiccup in
        # _decide) becomes a generic 503 so waiters still fail-fast without
        # re-dispatching a thundering herd of cold acquires.
        if info is not None:
            s, code, msg, retryable = info
            raise OperationError(code, msg, status=s, retryable=retryable)
        raise OperationError(
            "model_load_failed",
            f"load of {model_id!r} failed",
            status=503,
        )
    # "ok": take our OWN refcount via a full acquire. This re-decides
    # hot-or-cold -- the model may have been evicted between the loader's
    # commit and now; the director's own singleton collapse handles a genuine
    # reload. The hot case is a sub-ms refcount bump.
    return await _acquire_off_loop(state, model_id, manifest)


async def _forward_with_release(
    request: Request,
    target_url: str,
    timeout: float,
    *,
    director: Any,
    model_id: str,
    extra_release: "Callable[[], None] | None" = None,
) -> Response:
    """Forward + release variant: same shape as `_forward`, but wires the
    director.release call into the response lifecycle.

    `extra_release`, when given, is invoked immediately after each
    director.release site (spec 2026-07-08: the concurrency-gate slot
    release). It is called at every point director.release fires -- buffered
    finally, stream-relay finally, body-read except, stream-open except -- so
    the gate slot is returned on exactly the paths the director refcount is,
    and never stranded. Its own exceptions are logged and swallowed so a gate
    hiccup can never break the refcount release or the response.

    Small non-stream response: release runs after bounded raw prefetch and
    before the Response is returned. Large non-stream responses transition
    to the same relay lifecycle as SSE after a bounded in-memory prefix.

    Streaming response: release runs from one idempotent callback owned by
    both the relay generator and the outer ASGI response lifecycle. It runs
    after full iteration, iteration failure, or a downstream send failure
    before iteration starts. This matches the spec's "release on stream-close"
    requirement: a release on first-chunk dispatch would decrement refcount
    before the request actually finished, opening a window where the model
    could be evicted mid-response.

    Both paths also call director.release on the early-failure branches
    (request-body read, stream-open raise, response-prefetch raise), so refcount
    is never stranded.
    """
    def _fire_extra_release() -> None:
        # Guarded gate-slot release, paired with each director.release site.
        if extra_release is None:
            return
        try:
            extra_release()
        except Exception:  # noqa: BLE001
            logger.warning("gate release failed", exc_info=True)

    # The body read happens AFTER the caller's director.acquire() bumped
    # the refcount. A ClientDisconnect (client vanished mid-body) here --
    # reachable for a body-bearing GET, whose body is unread during model
    # extraction -- would otherwise skip release and strand the refcount,
    # wedging eviction. Release the slot first, then re-raise (L16).
    try:
        body = await request.body()
    except BaseException:
        # BaseException, not just Exception: a CancelledError here (client
        # disconnect / request cancellation mid body-read) is a BaseException,
        # so `except Exception` would skip the release and strand the refcount
        # -- pinning the model non-evictable. We re-raise, so cancellation
        # still propagates.
        try:
            director.release(model_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "director.release(%r) raised during body-read cleanup",
                model_id, exc_info=True,
            )
        _fire_extra_release()
        raise
    client = None
    stream_ctx = None
    try:
        fwd_headers = _proxy_headers(
            request.headers,
            exclude=frozenset({"host", "content-length"}),
        )
        client = httpx.AsyncClient(timeout=timeout)
        stream_ctx = client.stream(
            method=request.method,
            url=target_url,
            headers=fwd_headers,
            content=body,
            params=dict(request.query_params),
        )
        response = await stream_ctx.__aenter__()
    except BaseException:
        # BaseException, not just Exception: a CancelledError here (the client
        # disconnected while we were opening the worker connection) must
        # release the refcount too, or the hot model is pinned non-evictable
        # forever. We re-raise below, so cancellation still propagates.
        # __aenter__ raised: release the refcount FIRST, then aclose
        # the AsyncClient. Order matters for cascading failure: if we
        # awaited aclose() first and IT raised, control would exit via
        # the new exception and director.release(model_id) would never
        # run. Refcount leaks, eviction wedges. By releasing the
        # director slot first we guarantee the refcount returns to
        # baseline regardless of what aclose() does. The aclose() call
        # below is wrapped so any failure there is logged but does not
        # mask the original stream-open exception we re-raise.
        # Regression watchdog: tests/cli_impl/test_gateway.py
        # ::TestAsyncClientLifecycle::test_stream_open_failure_aclose_client
        # (aclose runs); tests/cli_impl/test_gateway_lazy.py
        # ::TestAcquireRelease::test_release_runs_even_when_aclose_raises_during_stream_open_failure
        # (release runs even when aclose explodes).
        try:
            director.release(model_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "director.release(%r) raised during stream-open cleanup",
                model_id, exc_info=True,
            )
        _fire_extra_release()
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "AsyncClient.aclose() raised during stream-open cleanup",
                    exc_info=True,
                )
        raise

    content_type = str(response.headers.get("content-type", "") or "")
    is_stream = "text/event-stream" in content_type.lower()

    resp_headers = _proxy_headers(
        response.headers,
        exclude=frozenset({"content-length"}),
    )

    stream_cleanup_started = False

    async def _cleanup_owned_stream() -> None:
        """Release director/gate/HTTP ownership exactly once."""
        nonlocal stream_cleanup_started
        if stream_cleanup_started:
            return
        stream_cleanup_started = True
        try:
            director.release(model_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "director.release(%r) raised at stream-close",
                model_id, exc_info=True,
            )
        _fire_extra_release()
        try:
            await stream_ctx.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.warning(
                "stream_ctx.__aexit__ raised at stream-close",
                exc_info=True,
            )
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            logger.warning(
                "AsyncClient.aclose() raised at stream-close",
                exc_info=True,
            )

    if is_stream:
        async def relay():
            # The relay generator owns the release call: it fires when
            # the upstream iteration completes (full body sent) OR when
            # it raises (worker died, client disconnected). Either way,
            # the model's refcount drops back to baseline at end-of-life.
            #
            # Order: release FIRST, then close the stream and client.
            # If aexit / aclose raised before release, the refcount would
            # leak. Putting release at the top of the finally chain
            # decouples it from cascading failures in the cleanup path.
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await _cleanup_owned_stream()

        return _install_response_headers(
            _OwnedStreamingResponse(
                relay(),
                cleanup=_cleanup_owned_stream,
                status_code=response.status_code,
                media_type=content_type,
            ),
            resp_headers,
        )

    # Preserve the buffered path for small bodies, but never let a worker
    # response grow gateway memory without bound.  Once the bounded prefix
    # crosses the cap, ownership of the open iterator/stream/client moves to
    # the relay generator and release fires when downstream consumption ends.
    try:
        content, prefix, iterator = await _prefetch_worker_response(response)
    except BaseException:
        try:
            director.release(model_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "director.release(%r) raised during response-prefetch cleanup",
                model_id, exc_info=True,
            )
        _fire_extra_release()
        try:
            await stream_ctx.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.warning(
                "stream_ctx.__aexit__ raised during response-prefetch cleanup",
                exc_info=True,
            )
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            logger.warning(
                "AsyncClient.aclose() raised during response-prefetch cleanup",
                exc_info=True,
            )
        raise

    if content is None:
        async def relay_large_response():
            try:
                for chunk in prefix:
                    yield chunk
                async for chunk in iterator:
                    yield chunk
            finally:
                await _cleanup_owned_stream()

        return _install_response_headers(
            _OwnedStreamingResponse(
                relay_large_response(),
                cleanup=_cleanup_owned_stream,
                status_code=response.status_code,
                media_type=content_type,
            ),
            resp_headers,
        )

    # Small buffered response: release before fallible HTTP cleanup so a
    # cleanup error can never strand the director refcount.
    try:
        director.release(model_id)
    except Exception:  # noqa: BLE001
        logger.warning(
            "director.release(%r) raised during buffered-response cleanup",
            model_id, exc_info=True,
        )
    _fire_extra_release()
    try:
        await stream_ctx.__aexit__(None, None, None)
    except Exception:  # noqa: BLE001
        logger.warning(
            "stream_ctx.__aexit__ raised during buffered-response cleanup",
            exc_info=True,
        )
    try:
        await client.aclose()
    except Exception:  # noqa: BLE001
        logger.warning(
            "AsyncClient.aclose() raised during buffered-response cleanup",
            exc_info=True,
        )

    return _install_response_headers(
        Response(
            content=content,
            status_code=response.status_code,
        ),
        resp_headers,
    )


async def _forward(request: Request, target_url: str, timeout: float) -> Response:
    """Forward a request to target_url.

    Relays SSE immediately. Other response bodies are prefetched only up to a
    fixed memory ceiling: small bodies preserve the buffered Response shape,
    while larger bodies continue through StreamingResponse.

    The httpx client and stream context are held open for the duration of
    a streaming response so chunks dispatch as they arrive from the worker
    (not after full synthesis completes). Same producer-consumer shape as
    the audio/speech router's internal streaming.
    """
    body = await request.body()
    fwd_headers = _proxy_headers(
        request.headers,
        exclude=frozenset({"host", "content-length"}),
    )

    client = httpx.AsyncClient(timeout=timeout)
    stream_ctx = None
    try:
        stream_ctx = client.stream(
            method=request.method,
            url=target_url,
            headers=fwd_headers,
            content=body,
            params=dict(request.query_params),
        )
        response = await stream_ctx.__aenter__()
    except BaseException:
        # __aenter__ raised: stream_ctx is not entered, so __aexit__
        # is not appropriate. Close the client to release the
        # underlying connection pool slot without masking the original.
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            logger.warning(
                "AsyncClient.aclose() raised during direct stream-open cleanup",
                exc_info=True,
            )
        raise

    content_type = str(response.headers.get("content-type", "") or "")
    is_stream = "text/event-stream" in content_type.lower()

    resp_headers = _proxy_headers(
        response.headers,
        exclude=frozenset({"content-length"}),
    )

    stream_cleanup_started = False

    async def _cleanup_direct_stream() -> None:
        nonlocal stream_cleanup_started
        if stream_cleanup_started:
            return
        stream_cleanup_started = True
        try:
            await stream_ctx.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.warning(
                "stream_ctx.__aexit__ raised during direct stream cleanup",
                exc_info=True,
            )
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            logger.warning(
                "AsyncClient.aclose() raised during direct stream cleanup",
                exc_info=True,
            )

    if is_stream:
        async def relay():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await _cleanup_direct_stream()

        return _install_response_headers(
            _OwnedStreamingResponse(
                relay(),
                cleanup=_cleanup_direct_stream,
                status_code=response.status_code,
                media_type=content_type,
            ),
            resp_headers,
        )

    # Buffer only a bounded prefix.  Large binary/non-SSE responses relay
    # without forcing the gateway to hold the complete body in RAM.
    try:
        content, prefix, iterator = await _prefetch_worker_response(response)
    except BaseException:
        try:
            await stream_ctx.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.warning(
                "stream_ctx.__aexit__ raised during direct prefetch cleanup",
                exc_info=True,
            )
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            logger.warning(
                "AsyncClient.aclose() raised during direct prefetch cleanup",
                exc_info=True,
            )
        raise

    if content is None:
        async def relay_large_response():
            try:
                for chunk in prefix:
                    yield chunk
                async for chunk in iterator:
                    yield chunk
            finally:
                await _cleanup_direct_stream()

        return _install_response_headers(
            _OwnedStreamingResponse(
                relay_large_response(),
                cleanup=_cleanup_direct_stream,
                status_code=response.status_code,
                media_type=content_type,
            ),
            resp_headers,
        )

    try:
        await stream_ctx.__aexit__(None, None, None)
    except Exception:  # noqa: BLE001
        logger.warning(
            "stream_ctx.__aexit__ raised during direct buffered cleanup",
            exc_info=True,
        )
    try:
        await client.aclose()
    except Exception:  # noqa: BLE001
        logger.warning(
            "AsyncClient.aclose() raised during direct buffered cleanup",
            exc_info=True,
        )

    return _install_response_headers(
        Response(
            content=content,
            status_code=response.status_code,
        ),
        resp_headers,
    )
