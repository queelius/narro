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
import json
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
        samples = state.telemetry_store.samples(since_ts=0, limit=1)
        latest_sample = samples[-1] if samples else None
        store_status = state.telemetry_store.status()
        return {
            "node": _node_id(state),
            "loaded": loaded,
            "queued": queued,
            "in_flight": in_flight,
            "dropped_events": get_recorder().dropped,
            "resources": latest_sample,
            "event_counts": store_status["counts"],
            "auth_required": bool(config.get("telemetry.require_auth")),
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

    @router.get(
        "/v1/telemetry/report",
        dependencies=[Depends(require_dashboard_auth)],
    )
    def report(window: float = Query(86400)) -> dict:
        if not math.isfinite(window) or window <= 0:
            raise _err(
                400, "invalid_window",
                "Telemetry window must be a positive finite number of seconds",
            )
        rows = state.telemetry_store.request_report(
            since_ts=time.time() - window,
        )
        for row in rows:
            raw = row.get("evicted_models")
            if raw:
                try:
                    row["evicted_models"] = json.loads(raw)
                except (TypeError, ValueError):
                    row["evicted_models"] = [str(raw)]
            else:
                row["evicted_models"] = []
        return {"window": window, "rows": rows}

    @router.get(
        "/v1/telemetry/traces",
        dependencies=[Depends(require_dashboard_auth)],
    )
    def traces(
        window: float = Query(86400),
        limit: int = Query(100, ge=1, le=1000),
        model_id: str | None = Query(default=None),
        modality: str | None = Query(default=None),
    ) -> dict:
        if not math.isfinite(window) or window <= 0:
            raise _err(
                400, "invalid_window",
                "Telemetry window must be a positive finite number of seconds",
            )
        rows = state.telemetry_store.recent_requests(
            since_ts=time.time() - window,
            limit=limit,
            model_id=model_id,
            modality=modality,
        )
        for row in rows:
            raw = row.get("evicted_models")
            if raw:
                try:
                    row["evicted_models"] = json.loads(raw)
                except (TypeError, ValueError):
                    row["evicted_models"] = [str(raw)]
            else:
                row["evicted_models"] = []
            if row.get("cold") is not None:
                row["cold"] = bool(row["cold"])
        return {"window": window, "rows": rows}

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


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Muse telemetry</title>
<style>
:root { --bg:#0b0e13; --panel:#111720; --panel2:#151d28; --line:#263242;
  --text:#e8edf4; --muted:#8998aa; --cyan:#55d9d0; --gold:#f6bd60;
  --pink:#f284a9; --green:#7bd88f; --red:#ff6b6b; }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--text);
  font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
header { position:sticky; top:0; z-index:4; display:flex; align-items:center;
  justify-content:space-between; gap:18px; padding:18px 28px; border-bottom:1px solid var(--line);
  background:rgba(11,14,19,.94); backdrop-filter:blur(12px); }
.brand { display:flex; align-items:center; gap:12px; } .mark { width:13px; height:13px;
  background:var(--cyan); border-radius:50%; box-shadow:0 0 24px var(--cyan); }
h1 { margin:0; font:600 19px/1.1 system-ui,sans-serif; letter-spacing:.02em; }
.sub { color:var(--muted); font-size:11px; margin-top:4px; }
.controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
button,select,input { color:var(--text); background:var(--panel2); border:1px solid var(--line);
  border-radius:7px; padding:8px 10px; font:inherit; } button { cursor:pointer; }
button:hover { border-color:var(--cyan); } main { max-width:1540px; margin:auto; padding:22px 28px 42px; }
.auth { display:none; gap:8px; padding:12px; margin-bottom:16px; border:1px solid #5c4721;
  border-radius:9px; background:#211b12; } .auth.show { display:flex; }
.cards { display:grid; grid-template-columns:repeat(6,minmax(130px,1fr)); gap:10px; margin-bottom:12px; }
.card,.panel { background:linear-gradient(145deg,var(--panel),#0f141c); border:1px solid var(--line);
  border-radius:10px; box-shadow:0 12px 40px rgba(0,0,0,.16); }
.card { padding:14px 16px; min-height:92px; } .label { color:var(--muted); text-transform:uppercase;
  letter-spacing:.1em; font-size:10px; } .value { margin-top:9px; font:600 25px/1 system-ui,sans-serif; }
.unit { color:var(--muted); font-size:11px; margin-left:4px; } .grid { display:grid;
  grid-template-columns:minmax(0,2fr) minmax(310px,1fr); gap:12px; margin-bottom:12px; }
.panel { padding:17px; min-width:0; } .panel-head { display:flex; justify-content:space-between;
  align-items:flex-end; gap:10px; margin-bottom:14px; } h2 { margin:0; font:600 14px system-ui,sans-serif; }
.hint { color:var(--muted); font-size:11px; } svg.chart { width:100%; height:245px; overflow:visible; }
.legend { display:flex; gap:16px; color:var(--muted); font-size:11px; } .dot { display:inline-block;
  width:8px; height:8px; border-radius:50%; margin-right:6px; }
.resident { display:flex; flex-direction:column; gap:8px; max-height:245px; overflow:auto; }
.model { padding:10px 11px; background:var(--panel2); border-radius:7px; border-left:3px solid var(--cyan); }
.model-top { display:flex; justify-content:space-between; gap:10px; } .model-meta { color:var(--muted); margin-top:5px; font-size:11px; }
.wide { margin-bottom:12px; overflow:hidden; } .scroll { overflow:auto; max-height:390px; }
table { width:100%; border-collapse:collapse; font-size:12px; } th { position:sticky; top:0; z-index:1;
  background:var(--panel2); color:var(--muted); text-align:left; text-transform:uppercase;
  letter-spacing:.07em; font-size:9px; } th,td { padding:10px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }
tr:last-child td { border:0; } tbody tr:hover { background:#151c26; } .cold { color:var(--gold); }
.hot { color:var(--green); } .bad { color:var(--red); } .muted { color:var(--muted); }
.two { display:grid; grid-template-columns:1fr 1fr; gap:12px; } #logs { height:220px;
  overflow:auto; background:#080a0e; border:1px solid var(--line); border-radius:7px; padding:10px;
  white-space:pre-wrap; color:#b8c4d1; font-size:11px; }
.empty { color:var(--muted); padding:26px 4px; text-align:center; } .status { color:var(--muted); }
.status.ok { color:var(--green); } .status.error { color:var(--red); }
@media (max-width:1050px) { .cards { grid-template-columns:repeat(3,1fr); } .grid,.two { grid-template-columns:1fr; } }
@media (max-width:620px) { header { padding:14px; align-items:flex-start; } main { padding:14px; }
  .cards { grid-template-columns:repeat(2,1fr); } .controls { justify-content:flex-end; } }
</style></head><body>
<header><div class="brand"><span class="mark"></span><div><h1>Muse telemetry</h1>
<div class="sub" id="node">waiting for node</div></div></div>
<div class="controls"><span id="status" class="status">connecting</span>
<select id="window"><option value="3600">1 hour</option><option value="21600">6 hours</option>
<option value="86400" selected>24 hours</option><option value="604800">7 days</option></select>
<button id="refresh">Refresh</button></div></header>
<main><div id="auth" class="auth"><input id="token" type="password" placeholder="Admin token">
<button id="connect">Connect</button><span class="hint">Stored only in this browser tab.</span></div>
<section class="cards">
<div class="card"><div class="label">GPU working set</div><div class="value" id="gpu">--</div></div>
<div class="card"><div class="label">Free VRAM</div><div class="value" id="free-vram">--</div></div>
<div class="card"><div class="label">Resident models</div><div class="value" id="resident-count">0</div></div>
<div class="card"><div class="label">Requests</div><div class="value" id="requests">0</div></div>
<div class="card"><div class="label">Cold starts</div><div class="value cold" id="cold-starts">0</div></div>
<div class="card"><div class="label">Evictions</div><div class="value" id="evictions">0</div></div>
</section>
<section class="grid"><div class="panel"><div class="panel-head"><div><h2>VRAM over time</h2>
<div class="hint">Device-wide resident working set sampled faster during active requests</div></div>
<div class="legend"><span><i class="dot" style="background:var(--cyan)"></i>peak used</span>
<span><i class="dot" style="background:#526174"></i>free</span></div></div><svg id="vram-chart" class="chart"></svg></div>
<div class="panel"><div class="panel-head"><div><h2>Resident working set</h2><div class="hint">Models currently loaded</div></div></div>
<div id="resident" class="resident"></div></div></section>
<section class="panel wide"><div class="panel-head"><div><h2>Cold vs hot evidence</h2>
<div class="hint">End-to-end request latency, request-linked evictions, and observed peak device VRAM</div></div></div>
<div class="scroll"><table><thead><tr><th>Request</th><th>Model</th><th>Cold latency</th><th>Hot latency</th>
<th>Peak VRAM</th><th>Evicted model</th><th>Samples</th><th>Basis</th></tr></thead><tbody id="report-body"></tbody></table></div></section>
<section class="panel wide"><div class="panel-head"><div><h2>Recent request traces</h2>
<div class="hint">Newest first; streaming latency runs until the response stream closes</div></div></div>
<div class="scroll"><table><thead><tr><th>Time</th><th>Request</th><th>Model</th><th>State</th><th>Total</th>
<th>Load</th><th>Forward</th><th>Queue</th><th>Peak VRAM</th><th>Evicted</th><th>Status</th></tr></thead>
<tbody id="trace-body"></tbody></table></div></section>
<section class="two"><div class="panel"><div class="panel-head"><div><h2>Latency over time</h2>
<div class="hint">Average and maximum end-to-end milliseconds per bucket</div></div></div><svg id="traffic-chart" class="chart"></svg></div>
<div class="panel"><div class="panel-head"><div><h2>Worker logs</h2><div class="hint">Recent buffer followed by live output</div></div>
<div><select id="log-model"><option value="">Select model</option></select> <button id="tail">Tail</button></div></div>
<div id="logs">Choose a model to open its live log stream.</div></div></section></main>
<script>(function(){"use strict";
var KEY="muse_dashboard_token", timer=null, logSource=null, ns="http://www.w3.org/2000/svg";
function el(id){return document.getElementById(id);} function token(){return sessionStorage.getItem(KEY)||"";}
function headers(){var t=token();return t?{"Authorization":"Bearer "+t}:{};}
function windowSeconds(){return Number(el("window").value)||86400;}
function fmtMs(v){return typeof v==="number"?v.toFixed(v<100?1:0)+" ms":"--";}
function fmtGb(v){return typeof v==="number"?v.toFixed(2)+" GB":"--";}
function friendly(v){var m={"audio/speech":"Generate speech","audio/transcriptions":"Transcribe audio",
"audio/transcription":"Transcribe audio","images/segmentations":"Segment image","images/segmentation":"Segment image","images/segment":"Segment image",
"images/generations":"Generate image","chat/completions":"Chat completion","embeddings":"Create embedding"};
return m[v]||String(v||"Unknown").replace(/[_/]/g," ").replace(/\b\w/g,function(c){return c.toUpperCase();});}
function setStatus(msg,kind){el("status").textContent=msg;el("status").className="status "+(kind||"");}
function api(path,opts){opts=opts||{};opts.headers=headers();return fetch(path,opts).then(function(r){
if(r.status===401||r.status===403||r.status===503){el("auth").classList.add("show");}
if(!r.ok){return r.json().catch(function(){return {};}).then(function(b){var e=(b.detail&&b.detail.error)||b.error||{};
throw new Error(e.message||path+" returned "+r.status);});}return r.json();});}
function cell(tr,value,cls){var td=document.createElement("td");td.textContent=value===null||value===undefined?"--":value;
if(cls)td.className=cls;tr.appendChild(td);}
function draw(svgId,series){var svg=el(svgId);while(svg.firstChild)svg.removeChild(svg.firstChild);var all=[];
series.forEach(function(s){s.values.forEach(function(v){if(typeof v==="number")all.push(v);});});
if(!all.length){var e=document.createElementNS(ns,"text");e.setAttribute("x","50%");e.setAttribute("y","50%");
e.setAttribute("text-anchor","middle");e.setAttribute("fill","#8998aa");e.textContent="No samples in this window";svg.appendChild(e);return;}
var w=900,h=245,p=28,max=Math.max.apply(null,all),min=Math.min.apply(null,all);if(max===min){min=0;max=max||1;}
svg.setAttribute("viewBox","0 0 "+w+" "+h);[0,.5,1].forEach(function(q){var y=p+q*(h-2*p),line=document.createElementNS(ns,"line");
line.setAttribute("x1",p);line.setAttribute("x2",w-p);line.setAttribute("y1",y);line.setAttribute("y2",y);line.setAttribute("stroke","#263242");svg.appendChild(line);});
series.forEach(function(s){var pts=[],n=s.values.length;s.values.forEach(function(v,i){if(typeof v!=="number")return;
var x=p+(n<2?0:i/(n-1))*(w-2*p),y=h-p-(v-min)/(max-min)*(h-2*p);pts.push(x.toFixed(1)+","+y.toFixed(1));});
if(!pts.length)return;var poly=document.createElementNS(ns,"polyline");poly.setAttribute("points",pts.join(" "));
poly.setAttribute("fill","none");poly.setAttribute("stroke",s.color);poly.setAttribute("stroke-width",s.width||2);svg.appendChild(poly);});
var top=document.createElementNS(ns,"text");top.setAttribute("x",p);top.setAttribute("y",13);top.setAttribute("fill","#8998aa");
top.setAttribute("font-size","10");top.textContent=max.toFixed(max<10?2:0);svg.appendChild(top);}
function renderSummary(d){el("node").textContent=d.node+(d.auth_required?" / token protected":" / open telemetry");
el("resident-count").textContent=(d.loaded||[]).length;var r=d.resources||{};el("gpu").textContent=fmtGb(r.gpu_used_gb);
el("free-vram").textContent=fmtGb(r.free_vram_gb);el("requests").textContent=(d.event_counts||{}).request||0;
el("evictions").textContent=(d.event_counts||{}).model_evict||0;var box=el("resident");box.innerHTML="";
var models={};(d.loaded||[]).forEach(function(m){models[m.model_id]=true;var row=document.createElement("div");row.className="model";
var top=document.createElement("div");top.className="model-top";var a=document.createElement("span");a.textContent=m.model_id;
var b=document.createElement("span");b.textContent=fmtGb(m.gb);top.appendChild(a);top.appendChild(b);row.appendChild(top);
var meta=document.createElement("div");meta.className="model-meta";meta.textContent=(m.pool||"unknown")+" pool / queue "+(m.queue_depth||0);row.appendChild(meta);box.appendChild(row);});
if(!(d.loaded||[]).length)box.innerHTML='<div class="empty">No models resident</div>';var sel=el("log-model"),chosen=sel.value;
sel.innerHTML='<option value="">Select model</option>';Object.keys(models).sort().forEach(function(m){var o=document.createElement("option");o.value=m;o.textContent=m;sel.appendChild(o);});if(models[chosen])sel.value=chosen;
el("auth").classList.toggle("show",!!d.auth_required&&!token());setStatus("live","ok");}
function renderReport(d){var body=el("report-body");body.innerHTML="";var cold=0,total=0;(d.rows||[]).forEach(function(r){cold+=r.cold_count||0;total+=r.request_count||0;
var tr=document.createElement("tr");cell(tr,friendly(r.modality));cell(tr,r.model_id);cell(tr,fmtMs(r.cold_latency_ms),"cold");
cell(tr,fmtMs(r.hot_latency_ms),"hot");cell(tr,fmtGb(r.peak_vram_gb));cell(tr,(r.evicted_models||[]).join(", ")||"--");
cell(tr,r.request_count);cell(tr,r.basis||"measured",r.basis==="estimated"?"cold":"");body.appendChild(tr);});el("cold-starts").textContent=cold;el("requests").textContent=total;if(!(d.rows||[]).length){var tr=document.createElement("tr");
var td=document.createElement("td");td.colSpan=8;td.className="empty";td.textContent="No request traces in this window";tr.appendChild(td);body.appendChild(tr);}}
function renderTraces(d){var body=el("trace-body");body.innerHTML="";(d.rows||[]).forEach(function(r){var tr=document.createElement("tr");
cell(tr,new Date(r.ts*1000).toLocaleTimeString());cell(tr,friendly(r.modality));cell(tr,r.model_id);cell(tr,r.cold?"cold":"hot",r.cold?"cold":"hot");
cell(tr,fmtMs(r.latency_ms));cell(tr,fmtMs(r.load_ms));cell(tr,fmtMs(r.forward_ms));cell(tr,fmtMs(r.queued_ms));
cell(tr,fmtGb(r.peak_vram_gb));cell(tr,(r.evicted_models||[]).join(", ")||"--");cell(tr,r.status,r.status>=400?"bad":"");body.appendChild(tr);});}
function refresh(){var w=windowSeconds();Promise.all([api("/v1/telemetry/summary"),api("/v1/telemetry/report?window="+w),
api("/v1/telemetry/traces?window="+w+"&limit=100"),api("/v1/telemetry/series?metric=vram&window="+w),
api("/v1/telemetry/series?metric=latency&window="+w),api("/v1/telemetry/series?metric=load_evict&window="+w)])
.then(function(x){renderSummary(x[0]);renderReport(x[1]);renderTraces(x[2]);draw("vram-chart",[
{values:x[3].points.map(function(p){return p.peak;}),color:"#55d9d0",width:3},
{values:x[3].points.map(function(p){return p.free;}),color:"#526174"}]);draw("traffic-chart",[
{values:x[4].points.map(function(p){return p.avg;}),color:"#7bd88f",width:3},
{values:x[4].points.map(function(p){return p.max;}),color:"#f6bd60"}]);
el("evictions").textContent=x[5].points.reduce(function(n,p){return n+(p.evicts||0);},0);})
.catch(function(e){setStatus(e.message,"error");});}
function start(){if(timer)clearInterval(timer);refresh();timer=setInterval(refresh,5000);}
function tail(){var m=el("log-model").value;if(!m)return;if(logSource)logSource.close();el("logs").textContent="opening "+m+"...";
api("/v1/telemetry/logs-ticket",{method:"POST"}).then(function(t){logSource=new EventSource("/v1/telemetry/logs/"+
encodeURIComponent(m)+"?ticket="+encodeURIComponent(t.ticket));el("logs").textContent="";logSource.onmessage=function(e){
var line=document.createElement("div");line.textContent=e.data;el("logs").appendChild(line);el("logs").scrollTop=el("logs").scrollHeight;};
logSource.onerror=function(){setStatus("log stream interrupted","error");};}).catch(function(e){setStatus(e.message,"error");});}
el("connect").onclick=function(){sessionStorage.setItem(KEY,el("token").value);start();};el("refresh").onclick=refresh;
el("window").onchange=refresh;el("tail").onclick=tail;el("token").value=token();start();
})();</script></body></html>"""
