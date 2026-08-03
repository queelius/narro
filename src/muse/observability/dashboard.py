"""Dashboard router: telemetry JSON endpoints + SSE log stream + HTML shell.

Wire surface:
  GET /dashboard                         -- HTMLResponse, UN-GATED shell.
  GET /v1/telemetry/summary              -- gated JSON snapshot.
  GET /v1/telemetry/series               -- gated JSON time series.
  POST /v1/telemetry/logs-ticket         -- gated (header) ticket mint.
  GET /v1/telemetry/logs/{model_id}      -- gated (ticket OR header) SSE log tail.

`/dashboard` is intentionally un-gated: it is a static shell that prompts
the browser for a token and stores it in sessionStorage before hitting
any of the gated data endpoints. Gating those endpoints (not the shell)
means the page always loads, even with no token configured yet, so the
operator can see the prompt instead of a blank 503.

The SSE logs endpoint is a special case: `EventSource` cannot set a
custom Authorization header, so it cannot use `require_dashboard_auth`
as a dependency. Instead the dashboard JS first mints a short-lived
ticket via the header-gated `POST /v1/telemetry/logs-ticket`, then opens
the `EventSource` with that ticket in the query string. The logs route
itself checks the ticket (or, for curl-style clients, an Authorization
header) inline before opening the stream. The admin token itself never
rides a URL.

`state` is a duck-typed namespace (see muse.cli_impl.supervisor's
SupervisorState in production, or a SimpleNamespace in tests) exposing:
  state.telemetry_store  -- a TelemetryStore
  state.log_hub          -- a LogHub
  state.director.loaded  -- dict[str, LoadEntry-like]
  state.node_url / state.node_id (optional) -- a stable node identifier
"""
from __future__ import annotations

import asyncio
import contextlib
import math
import queue
import socket
import time

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from muse.admin.auth import _err
from muse.core import config
from muse.observability.dashboard_auth import check_dashboard_token, require_dashboard_auth
from muse.observability.log_tickets import LogTicketStore
from muse.observability.recorder import get_recorder

# Roughly 60 buckets across any requested window.
_TARGET_BUCKET_COUNT = 60
_SSE_POLL_INTERVAL_SECONDS = 0.25


def _node_id(state) -> str:
    node = getattr(state, "node_url", None)
    if node:
        return node
    node = getattr(state, "node_id", None)
    if node:
        return node
    return socket.gethostname()


async def _stream_model_logs(hub, model_id: str, request: Request):
    """Yield past + live log lines for one model as SSE-shaped dicts.

    First drains `hub.snapshot(model_id)` (the buffered history), then
    subscribes and polls for new lines until the client disconnects.
    Polling (rather than a blocking `queue.Queue.get`) keeps this on the
    event loop with no extra thread; the `asyncio.sleep` between drains
    is the only per-iteration cost. `hub.unsubscribe` runs in a `finally`
    so a disconnect, a cancellation, or an exception in the loop body all
    still release the subscription -- no subscriber leak.

    Extracted to module scope (rather than nested in the route) so it is
    directly unit-testable with a fake `request.is_disconnected()`,
    without going through the ASGI transport (which fully drains an
    unbounded async generator before returning, i.e. would hang forever
    on a stream with no natural end).
    """
    history, q = hub.subscribe_with_snapshot(model_id)
    try:
        for line in history:
            yield {"data": line}
        while True:
            if await request.is_disconnected():
                break
            try:
                while True:
                    line = q.get_nowait()
                    yield {"data": line}
            except queue.Empty:
                pass
            await asyncio.sleep(_SSE_POLL_INTERVAL_SECONDS)
    finally:
        hub.unsubscribe(model_id, q)


def _loaded_entry_dict(model_id: str, entry, queue_depth: int | None = None) -> dict:
    """Defensively project a LoadEntry-like object into the wire shape.

    Only fields verified on muse.cli_impl.load_director.LoadEntry are
    read directly (memory_gb, last_touched_at); everything else (namely
    "pool", which is not a LoadEntry attribute) falls back to None via
    getattr rather than assuming a name that might not exist.

    `queue_depth` (spec 2026-07-08 Task 4) is passed in by the caller
    (already resolved from the gateway's ConcurrencyGate, or 0 when no
    gate is bound) rather than read off `entry`, since queue depth is a
    property of the model's concurrency gate, not of the LoadEntry.
    """
    return {
        "model_id": model_id,
        "pool": getattr(entry, "pool", None),
        "gb": getattr(entry, "memory_gb", None),
        "last_used": getattr(entry, "last_touched_at", None),
        "queue_depth": queue_depth,
    }


def build_dashboard_router(state) -> APIRouter:
    router = APIRouter()
    tickets = LogTicketStore(config.get("telemetry.log_ticket_ttl_seconds"))

    @router.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    @router.get(
        "/v1/telemetry/summary",
        dependencies=[Depends(require_dashboard_auth)],
    )
    def summary() -> dict:
        director = state.director
        gate = getattr(state, "concurrency_gate", None)
        depths = gate.depths() if gate is not None else {}
        director_lock = getattr(director, "lock", None)
        lock_context = (
            director_lock
            if hasattr(director_lock, "__enter__")
            else contextlib.nullcontext()
        )
        with lock_context:
            loaded = [
                _loaded_entry_dict(
                    model_id,
                    entry,
                    queue_depth=depths.get(model_id, 0),
                )
                for model_id, entry in director.loaded.items()
            ]
            in_flight = len(getattr(director, "in_flight_loads", {}) or {})
        loaded_ids = {entry["model_id"] for entry in loaded}
        # #331: waiters parked on a model during its own cold start are in
        # the gate but NOT in director.loaded -- exactly when the queue is
        # deepest. Surface them separately so the pressure is visible.
        queued = [
            {"model_id": model_id, "queue_depth": depth}
            for model_id, depth in sorted(depths.items())
            if depth > 0 and model_id not in loaded_ids
        ]
        return {
            "node": _node_id(state),
            "loaded": loaded,
            "queued": queued,
            "in_flight": in_flight,
            "dropped_events": get_recorder().dropped,
        }

    @router.get(
        "/v1/telemetry/series",
        dependencies=[Depends(require_dashboard_auth)],
    )
    def series(metric: str = Query(...), window: float = Query(3600)) -> dict:
        if not math.isfinite(window) or window <= 0:
            raise _err(
                400,
                "invalid_window",
                "Telemetry window must be a positive finite number of seconds",
            )
        since_ts = time.time() - window
        bucket_seconds = max(window / _TARGET_BUCKET_COUNT, 1)
        try:
            return state.telemetry_store.series(metric, since_ts, bucket_seconds)
        except ValueError:
            raise _err(400, "invalid_metric", "Unknown telemetry metric")

    @router.post(
        "/v1/telemetry/logs-ticket",
        dependencies=[Depends(require_dashboard_auth)],
    )
    def mint_logs_ticket() -> dict:
        ticket, expires_in = tickets.mint()
        return {"ticket": ticket, "expires_in": expires_in}

    @router.get("/v1/telemetry/logs/{model_id}")
    async def logs(
        model_id: str,
        request: Request,
        ticket: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> EventSourceResponse:
        # This route cannot use require_dashboard_auth as a dependency:
        # EventSource clients cannot set an Authorization header, so they
        # authenticate via a short-lived ticket instead. A valid ticket is
        # accepted outright; otherwise fall back to the header check (for
        # curl-style clients), which raises the usual 503/401/403.
        # ?access_token=<admin-token> is intentionally NOT accepted here.
        if not (ticket and tickets.validate(ticket)):
            check_dashboard_token(authorization)
        return EventSourceResponse(_stream_model_logs(state.log_hub, model_id, request))

    return router


DASHBOARD_HTML = """<title>muse dashboard</title>
<style>
  body { font-family: monospace; background: #111; color: #ddd; margin: 0; padding: 1em; }
  h1 { font-size: 1.2em; }
  #token-bar { margin-bottom: 1em; }
  #token-bar input { width: 24em; padding: 0.3em; }
  #token-bar button { padding: 0.3em 0.8em; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
  th, td { border: 1px solid #444; padding: 0.3em 0.6em; text-align: left; }
  th { background: #222; }
  .panel { border: 1px solid #333; padding: 0.6em; margin-bottom: 1em; }
  svg { background: #1a1a1a; }
  #log-panel { height: 16em; overflow-y: auto; background: #000; padding: 0.5em; white-space: pre-wrap; }
  select, input[type=text] { background: #222; color: #ddd; border: 1px solid #444; }
</style>
<h1>muse observability dashboard</h1>
<div id="token-bar">
  <label>Token: <input type="text" id="token-input" placeholder="admin token"></label>
  <button id="connect-btn">Connect</button>
  <span id="status"></span>
</div>
<div class="panel">
  <h2>Loaded models</h2>
  <table id="loaded-table">
    <thead><tr><th>model_id</th><th>pool</th><th>gb</th><th>last_used</th></tr></thead>
    <tbody></tbody>
  </table>
  <div id="counters"></div>
</div>
<div class="panel">
  <h2>request_rate</h2>
  <svg id="chart-request-rate" width="600" height="120"></svg>
</div>
<div class="panel">
  <h2>latency</h2>
  <svg id="chart-latency" width="600" height="120"></svg>
</div>
<div class="panel">
  <h2>Logs</h2>
  <label>model_id: <input type="text" id="log-model-id" value="default"></label>
  <button id="log-connect-btn">Tail logs</button>
  <div id="log-panel"></div>
</div>
<script>
(function () {
  "use strict";

  var TOKEN_KEY = "muse_dashboard_token";
  var POLL_MS = 2000;
  var pollTimer = null;
  var logSource = null;

  function getToken() {
    return sessionStorage.getItem(TOKEN_KEY) || "";
  }

  function setToken(t) {
    sessionStorage.setItem(TOKEN_KEY, t);
  }

  function setStatus(msg) {
    document.getElementById("status").textContent = msg;
  }

  function authHeaders() {
    return { "Authorization": "Bearer " + getToken() };
  }

  function renderLoaded(data) {
    var tbody = document.querySelector("#loaded-table tbody");
    tbody.innerHTML = "";
    var rows = data.loaded || [];
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      var tr = document.createElement("tr");
      var cells = [r.model_id, r.pool, r.gb, r.last_used];
      for (var j = 0; j < cells.length; j++) {
        var td = document.createElement("td");
        td.textContent = (cells[j] === null || cells[j] === undefined) ? "-" : cells[j];
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    document.getElementById("counters").textContent =
      "node=" + data.node +
      " in_flight=" + data.in_flight +
      " dropped_events=" + data.dropped_events;
  }

  function drawSeries(svgId, points, valueKey) {
    var svg = document.getElementById(svgId);
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    if (!points || points.length === 0) { return; }

    var w = svg.getAttribute("width") * 1;
    var h = svg.getAttribute("height") * 1;
    var pad = 10;

    var values = [];
    for (var i = 0; i < points.length; i++) {
      var v = points[i][valueKey];
      values.push(typeof v === "number" ? v : 0);
    }
    var minV = Math.min.apply(null, values);
    var maxV = Math.max.apply(null, values);
    if (maxV === minV) { maxV = minV + 1; }

    var n = points.length;
    var stepX = n > 1 ? (w - 2 * pad) / (n - 1) : 0;

    var coords = [];
    for (var k = 0; k < n; k++) {
      var x = pad + k * stepX;
      var norm = (values[k] - minV) / (maxV - minV);
      var y = h - pad - norm * (h - 2 * pad);
      coords.push(x.toFixed(1) + "," + y.toFixed(1));
    }

    var ns = "http://www.w3.org/2000/svg";
    var poly = document.createElementNS(ns, "polyline");
    poly.setAttribute("points", coords.join(" "));
    poly.setAttribute("fill", "none");
    poly.setAttribute("stroke", "#5ad");
    poly.setAttribute("stroke-width", "2");
    svg.appendChild(poly);
  }

  function fetchSummary() {
    fetch("/v1/telemetry/summary", { headers: authHeaders() })
      .then(function (r) {
        if (!r.ok) { throw new Error("summary " + r.status); }
        return r.json();
      })
      .then(function (data) {
        renderLoaded(data);
        setStatus("connected");
      })
      .catch(function (e) {
        setStatus("error: " + e.message);
      });
  }

  function fetchSeries(metric, svgId, valueKey) {
    fetch("/v1/telemetry/series?metric=" + encodeURIComponent(metric) + "&window=3600",
      { headers: authHeaders() })
      .then(function (r) {
        if (!r.ok) { throw new Error("series " + r.status); }
        return r.json();
      })
      .then(function (data) {
        drawSeries(svgId, data.points, valueKey);
      })
      .catch(function () {
        // Series errors are non-fatal to the rest of the dashboard.
      });
  }

  function pollAll() {
    fetchSummary();
    fetchSeries("request_rate", "chart-request-rate", "count");
    fetchSeries("latency", "chart-latency", "avg");
  }

  function startPolling() {
    if (pollTimer) { clearInterval(pollTimer); }
    pollAll();
    pollTimer = setInterval(pollAll, POLL_MS);
  }

  function connectLogs() {
    var modelId = document.getElementById("log-model-id").value || "default";
    if (logSource) { logSource.close(); }
    fetch("/v1/telemetry/logs-ticket", { method: "POST", headers: authHeaders() })
      .then(function (r) {
        if (!r.ok) { throw new Error("logs-ticket " + r.status); }
        return r.json();
      })
      .then(function (data) {
        var url = "/v1/telemetry/logs/" + encodeURIComponent(modelId) +
          "?ticket=" + encodeURIComponent(data.ticket);
        logSource = new EventSource(url);
        var panel = document.getElementById("log-panel");
        logSource.onmessage = function (ev) {
          var line = document.createElement("div");
          line.textContent = ev.data;
          panel.appendChild(line);
          panel.scrollTop = panel.scrollHeight;
        };
        logSource.onerror = function () {
          setStatus("log stream error");
        };
      })
      .catch(function () {
        setStatus("log auth failed");
      });
  }

  document.getElementById("connect-btn").addEventListener("click", function () {
    var t = document.getElementById("token-input").value;
    setToken(t);
    startPolling();
  });

  document.getElementById("log-connect-btn").addEventListener("click", connectLogs);

  var existing = getToken();
  if (existing) {
    document.getElementById("token-input").value = existing;
    startPolling();
  } else {
    setStatus("enter a token to connect");
  }
})();
</script>
"""
