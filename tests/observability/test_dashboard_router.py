"""Tests for muse.observability.dashboard: router + self-contained HTML."""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sse_starlette.sse import EventSourceResponse

from muse.core import config
from muse.observability.dashboard import _stream_model_logs, build_dashboard_router
from muse.observability.logs import LogHub
from muse.observability.store import TelemetryStore
from muse.observability.events import event_to_row


@pytest.fixture(autouse=True)
def _isolate_operator_config(tmp_path, monkeypatch):
    """Dashboard tests must not inherit the developer's real Muse token."""
    monkeypatch.setenv("MUSE_CONFIG", str(tmp_path / "absent-config.yaml"))
    config.reset_config()
    yield
    config.reset_config()


class _FakeDirector:
    def __init__(self):
        self.loaded = {}
        self.in_flight_loads = {}


def _make_state(tmp_path):
    store = TelemetryStore(str(tmp_path / "telemetry.sqlite3"))
    return SimpleNamespace(
        telemetry_store=store,
        log_hub=LogHub(),
        director=_FakeDirector(),
        node_url="test-node",
    )


def _make_app(tmp_path):
    state = _make_state(tmp_path)
    app = FastAPI()
    app.include_router(build_dashboard_router(state))
    return app, state


@pytest.fixture
def client_no_token(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    config.reset_config()
    app, state = _make_app(tmp_path)
    with TestClient(app) as client:
        client.state_ref = state
        yield client
    state.telemetry_store.close()


@pytest.fixture
def client_with_token(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "t")
    config.reset_config()
    app, state = _make_app(tmp_path)
    with TestClient(app) as client:
        client.state_ref = state
        yield client
    state.telemetry_store.close()


def test_summary_requires_token(client_no_token):
    assert client_no_token.get("/v1/telemetry/summary").status_code == 503


def test_summary_ok_with_token(client_with_token):
    r = client_with_token.get(
        "/v1/telemetry/summary", headers={"Authorization": "Bearer t"}
    )
    assert r.status_code == 200
    body = r.json()
    assert "loaded" in body and "node" in body
    assert body["node"] == "test-node"
    assert body["loaded"] == []
    assert body["in_flight"] == 0
    assert body["dropped_events"] == 0


def test_summary_snapshots_director_state_under_its_lock(client_with_token):
    state = client_with_token.state_ref

    class TrackingLock:
        entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *_args):
            self.entered = False

    lock = TrackingLock()

    class LockedMapping(dict):
        def items(self):
            assert lock.entered
            return super().items()

    state.director.lock = lock
    state.director.loaded = LockedMapping()
    state.director.in_flight_loads = LockedMapping()

    r = client_with_token.get(
        "/v1/telemetry/summary", headers={"Authorization": "Bearer t"}
    )
    assert r.status_code == 200


def test_summary_projects_mutable_entry_fields_while_director_is_locked(
    client_with_token,
):
    state = client_with_token.state_ref

    class TrackingLock:
        entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *_args):
            self.entered = False

    lock = TrackingLock()

    class LockCheckedEntry:
        @property
        def pool(self):
            assert lock.entered
            return "cpu"

        @property
        def memory_gb(self):
            assert lock.entered
            return 1.25

        @property
        def last_touched_at(self):
            assert lock.entered
            return 12.0

    state.director.lock = lock
    state.director.loaded = {"m": LockCheckedEntry()}

    response = client_with_token.get(
        "/v1/telemetry/summary", headers={"Authorization": "Bearer t"}
    )

    assert response.status_code == 200
    assert response.json()["loaded"] == [{
        "model_id": "m",
        "pool": "cpu",
        "gb": 1.25,
        "last_used": 12.0,
        "queue_depth": 0,
    }]


def test_summary_includes_queue_depth(client_with_token):
    """loaded[] entries carry the live per-model queue_depth from the
    gateway's ConcurrencyGate, mirroring /v1/admin/memory's discipline:
    present (0 or more) when a gate is bound, so an operator can see
    parked waiters from the dashboard summary too."""
    from muse.cli_impl.queueing import ConcurrencyGate

    state = client_with_token.state_ref
    state.director.loaded["m"] = SimpleNamespace(
        pool=None, memory_gb=1.0, last_touched_at=0.0,
    )
    gate = ConcurrencyGate()
    # cap=1, 3 entered (1 holding + 2 parked) -> depth = 3 - 1 = 2.
    gate._caps["m"] = 1
    gate._sems["m"] = asyncio.Semaphore(1)
    gate._entered["m"] = 3
    assert gate.depth("m") == 2
    state.concurrency_gate = gate

    r = client_with_token.get(
        "/v1/telemetry/summary", headers={"Authorization": "Bearer t"}
    )
    assert r.status_code == 200
    entry = [e for e in r.json()["loaded"] if e["model_id"] == "m"][0]
    assert entry["queue_depth"] == 2


def test_summary_queue_depth_zero_when_no_gate_bound(client_with_token):
    """Bare test state has no concurrency_gate attribute at all (unlike
    production SupervisorState's default field), so summary() must guard
    with getattr and fall back to an empty depths mapping -> queue_depth 0
    for a loaded model, rather than raising AttributeError."""
    state = client_with_token.state_ref
    assert not hasattr(state, "concurrency_gate")
    state.director.loaded["m"] = SimpleNamespace(
        pool=None, memory_gb=1.0, last_touched_at=0.0,
    )

    r = client_with_token.get(
        "/v1/telemetry/summary", headers={"Authorization": "Bearer t"}
    )
    assert r.status_code == 200
    entry = [e for e in r.json()["loaded"] if e["model_id"] == "m"][0]
    assert entry["queue_depth"] == 0


def test_summary_lists_cold_start_queuers(client_with_token):
    """#331: waiters parked on a model during its own cold start (not yet in
    director.loaded) must be visible via a `queued` list -- the loaded[]
    projection alone hides queue pressure exactly when it is deepest."""
    from muse.cli_impl.queueing import ConcurrencyGate

    state = client_with_token.state_ref
    gate = ConcurrencyGate()
    gate._caps["coldy"] = 1
    gate._sems["coldy"] = asyncio.Semaphore(1)
    gate._entered["coldy"] = 3  # 1 holding + 2 parked; NOT in director.loaded
    assert gate.depth("coldy") == 2
    state.concurrency_gate = gate

    r = client_with_token.get(
        "/v1/telemetry/summary", headers={"Authorization": "Bearer t"}
    )
    assert r.status_code == 200
    assert r.json()["queued"] == [{"model_id": "coldy", "queue_depth": 2}]


def test_summary_queued_empty_when_no_gate(client_with_token):
    """Shape stability: `queued` is always present (empty without a gate or
    with no cold-start waiters), so consumers need no key-existence checks."""
    r = client_with_token.get(
        "/v1/telemetry/summary", headers={"Authorization": "Bearer t"}
    )
    assert r.status_code == 200
    assert r.json()["queued"] == []


def test_series_ok(client_with_token):
    r = client_with_token.get(
        "/v1/telemetry/series?metric=request_rate&window=3600",
        headers={"Authorization": "Bearer t"},
    )
    assert r.status_code == 200 and r.json()["metric"] == "request_rate"


def test_series_unknown_metric_returns_400(client_with_token):
    r = client_with_token.get(
        "/v1/telemetry/series?metric=bogus_metric&window=3600",
        headers={"Authorization": "Bearer t"},
    )
    assert r.status_code == 400
    body = r.json()
    # No unwrap handler installed on this bare test app, so FastAPI's
    # default HTTPException handler double-wraps under "detail".
    assert body["detail"]["error"]["code"] == "invalid_metric"


@pytest.mark.parametrize("window", ["0", "-1", "nan", "inf"])
def test_series_rejects_nonpositive_or_nonfinite_window(
    client_with_token, window,
):
    r = client_with_token.get(
        f"/v1/telemetry/series?metric=request_rate&window={window}",
        headers={"Authorization": "Bearer t"},
    )
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "invalid_window"


def test_series_requires_token(client_no_token):
    r = client_no_token.get("/v1/telemetry/series?metric=request_rate&window=3600")
    assert r.status_code == 503


def test_dashboard_html_is_self_contained(client_with_token):
    r = client_with_token.get("/dashboard")
    assert r.status_code == 200
    body = r.text
    # no external resource loads (works internet-exposed / strict CSP)
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', body)
    assert body.isascii()
    assert "VRAM over time" in body
    assert "Cold vs hot evidence" in body
    assert "Recent request traces" in body


def test_report_and_traces_endpoints(client_with_token):
    state = client_with_token.state_ref
    state.telemetry_store.insert_many([
        event_to_row(
            "request", 9999999999.0, request_id="r1", model_id="m",
            modality="audio/speech", cold=True, latency_ms=1000.0,
            load_ms=700.0, peak_vram_gb=3.5,
            evicted_models='["old"]', status=200,
        ),
    ])
    headers = {"Authorization": "Bearer t"}

    report = client_with_token.get(
        "/v1/telemetry/report?window=9999999999", headers=headers,
    )
    traces = client_with_token.get(
        "/v1/telemetry/traces?window=9999999999", headers=headers,
    )

    assert report.status_code == 200
    assert report.json()["rows"][0]["evicted_models"] == ["old"]
    assert traces.status_code == 200
    assert traces.json()["rows"][0]["cold"] is True


def test_dashboard_data_can_be_opened_explicitly(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_TELEMETRY_REQUIRE_AUTH", "false")
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    config.reset_config()
    app, state = _make_app(tmp_path)
    try:
        client = TestClient(app)
        response = client.get("/v1/telemetry/summary")
        assert response.status_code == 200
        assert response.json()["auth_required"] is False
    finally:
        state.telemetry_store.close()


def test_dashboard_is_ungated(client_no_token):
    # The shell must load even with no token configured, so the JS can
    # prompt for one; only the data endpoints are gated.
    r = client_no_token.get("/dashboard")
    assert r.status_code == 200


def test_logs_requires_token(client_no_token):
    r = client_no_token.get("/v1/telemetry/logs/some-model")
    assert r.status_code == 503


def test_logs_ticket_mint_requires_token(client_no_token):
    r = client_no_token.post("/v1/telemetry/logs-ticket")
    assert r.status_code == 503


def test_logs_ticket_mint_wrong_bearer_returns_403(client_with_token):
    r = client_with_token.post(
        "/v1/telemetry/logs-ticket", headers={"Authorization": "Bearer wrong"}
    )
    assert r.status_code == 403


def test_logs_ticket_mint_correct_bearer_returns_ticket(client_with_token):
    r = client_with_token.post(
        "/v1/telemetry/logs-ticket", headers={"Authorization": "Bearer t"}
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["ticket"], str) and len(body["ticket"]) > 0
    assert isinstance(body["expires_in"], int) and body["expires_in"] > 0


def test_logs_stream_rejects_invalid_ticket_with_no_header(client_with_token):
    r = client_with_token.get("/v1/telemetry/logs/some-model?ticket=bogus-ticket")
    assert r.status_code in (401, 403)


def test_logs_stream_rejects_no_ticket_no_header(client_with_token):
    r = client_with_token.get("/v1/telemetry/logs/some-model")
    assert r.status_code == 401


def test_logs_stream_rejects_removed_access_token_query_path(client_with_token):
    # Proves the admin-token-in-URL path is gone: ?access_token=<token>
    # with no header and no valid ticket must NOT open the stream.
    r = client_with_token.get("/v1/telemetry/logs/some-model?access_token=t")
    assert r.status_code != 200
    assert r.status_code == 401


def test_logs_endpoint_opens_stream_with_valid_ticket(tmp_path, monkeypatch):
    # Direct-call pattern (see test_logs_endpoint_returns_event_source_response_with_token
    # below): TestClient/ASGITransport fully drain an unbounded SSE generator
    # before returning, so a real HTTP round-trip through a 200 stream-open
    # would hang. Calling the route function directly avoids the ASGI
    # dispatch while still exercising the real auth branch (ticket path).
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "t")
    config.reset_config()
    state = _make_state(tmp_path)
    router = build_dashboard_router(state)
    route = next(
        r for r in router.routes if getattr(r, "path", None) == "/v1/telemetry/logs/{model_id}"
    )
    mint_route = next(
        r for r in router.routes if getattr(r, "path", None) == "/v1/telemetry/logs-ticket"
    )

    class _UnusedRequest:
        async def is_disconnected(self):
            raise AssertionError("must not be invoked without ASGI dispatch")

    mint_body = mint_route.endpoint()
    ticket = mint_body["ticket"]

    response = asyncio.run(
        route.endpoint(
            model_id="m",
            request=_UnusedRequest(),
            ticket=ticket,
            authorization=None,
        )
    )
    assert isinstance(response, EventSourceResponse)
    assert response.status_code == 200
    state.telemetry_store.close()


def test_logs_endpoint_returns_event_source_response_with_token(tmp_path, monkeypatch):
    # The ASGI test transports available here (Starlette TestClient and
    # httpx.ASGITransport) both fully await the app callable before
    # returning a response, which would hang forever against an
    # intentionally unbounded SSE generator. So this exercises the route
    # function directly (no ASGI invocation, no generator iteration) to
    # confirm an authorized request gets back a 200 EventSourceResponse;
    # the 503-without-token path above already covers real HTTP gating,
    # and the generator's own snapshot/subscribe/disconnect/unsubscribe
    # contract is covered by the _stream_model_logs tests below.
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "t")
    config.reset_config()
    state = _make_state(tmp_path)
    router = build_dashboard_router(state)
    route = next(
        r for r in router.routes if getattr(r, "path", None) == "/v1/telemetry/logs/{model_id}"
    )

    class _UnusedRequest:
        async def is_disconnected(self):
            raise AssertionError("must not be invoked without ASGI dispatch")

    response = asyncio.run(
        route.endpoint(
            model_id="m",
            request=_UnusedRequest(),
            ticket=None,
            authorization="Bearer t",
        )
    )
    assert isinstance(response, EventSourceResponse)
    assert response.status_code == 200
    state.telemetry_store.close()


class _FakeSSERequest:
    """Fake Request exposing is_disconnected(), for direct generator tests."""

    def __init__(self, disconnect_after: int = 0):
        self._calls = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self._calls += 1
        return self._calls > self._disconnect_after


async def test_stream_model_logs_yields_snapshot_then_stops_on_disconnect():
    hub = LogHub()
    hub.append("m", "line1")
    hub.append("m", "line2")
    request = _FakeSSERequest(disconnect_after=0)

    items = [item async for item in _stream_model_logs(hub, "m", request)]

    assert items == [{"data": "line1"}, {"data": "line2"}]
    # unsubscribe in the finally must have run: no leftover subscriber.
    assert not hub._subscribers.get("m")


async def test_stream_model_logs_streams_live_lines_and_unsubscribes_on_close():
    hub = LogHub()
    # Never disconnects on its own; we close the generator explicitly.
    request = _FakeSSERequest(disconnect_after=1000)
    gen = _stream_model_logs(hub, "m", request)

    # No snapshot lines exist yet, so the first value out of the
    # generator must come from the live subscribe-and-poll loop.
    task = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0.05)  # let the generator reach subscribe() + its poll sleep
    hub.append("m", "live-line")
    item = await asyncio.wait_for(task, timeout=2)
    assert item == {"data": "live-line"}

    # Closing early (simulating a cancelled/closed ASGI response) must
    # still run the `finally: hub.unsubscribe(...)` cleanup.
    await gen.aclose()
    assert not hub._subscribers.get("m")
