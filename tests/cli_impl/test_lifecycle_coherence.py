"""Cross-component lifecycle regressions using only in-process fakes."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import Response

from muse.admin.operations import (
    OperationError,
    disable_model,
    remove_model,
    warmup_model,
)
from muse.cli_impl.gateway import (
    ShutdownCancellationMiddleware,
    _forward_with_release,
    _install_response_headers,
    _proxy_headers,
    _route_via_director,
)
from muse.cli_impl.idle_sweeper import IdleSweeper
from muse.cli_impl.load_director import LoadDirector, LoadEntry
from muse.cli_impl.supervisor import SupervisorState


@pytest.fixture(autouse=True)
def _isolated_catalog(tmp_path, monkeypatch):
    from muse.core.catalog import _reset_known_models_cache, _reset_read_catalog_cache

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    yield
    _reset_known_models_cache()
    _reset_read_catalog_cache()


class _Probe:
    def gpu_free_gb(self) -> None:
        return None

    def cpu_free_gb(self) -> float:
        return 100.0


def _manifest(*, memory_gb: float = 1.0) -> dict:
    return {
        "model_id": "model",
        "modality": "audio/speech",
        "capabilities": {"device": "cpu", "memory_gb": memory_gb},
    }


def _entry(
    model_id: str = "model", *, refcount: int = 0, memory_gb: float = 3.0,
) -> LoadEntry:
    now = time.monotonic()
    return LoadEntry(
        model_id=model_id,
        worker_port=9001,
        memory_gb=memory_gb,
        refcount=refcount,
        last_touched_at=now - 60.0,
        loaded_at=now - 60.0,
        pool="cpu",
    )


def _director(*, disable_fn=None, enable_fn=None):
    return LoadDirector(
        enable_fn=enable_fn or MagicMock(return_value=9002),
        disable_fn=disable_fn or MagicMock(),
        memory_probe=_Probe(),
        cpu_budget_gb=4.0,
        cpu_headroom_gb=0.0,
    )


def test_admin_disable_refuses_active_request_without_leaving_model_blocked():
    director = _director()
    resident = _entry(refcount=1)
    director.loaded[resident.model_id] = resident

    with pytest.raises(OperationError) as caught:
        director.begin_model_disable(resident.model_id)

    assert caught.value.code == "model_in_use"
    assert resident.model_id not in director._blocked_models
    assert director.acquire(resident.model_id, manifest=_manifest()) == 9001


def test_eviction_claim_survives_free_memory_probe_failure_and_can_settle():
    class _FailingProbe(_Probe):
        def cpu_free_gb(self) -> float:
            raise RuntimeError("probe unavailable")

    director = LoadDirector(
        enable_fn=MagicMock(return_value=9002),
        disable_fn=MagicMock(),
        memory_probe=_FailingProbe(),
        cpu_headroom_gb=0.0,
    )
    resident = _entry()
    director.loaded[resident.model_id] = resident

    claim = director.begin_model_disable(resident.model_id)

    assert director._eviction_claims[resident.model_id] is claim
    assert resident.model_id in director.evicting
    assert resident.model_id in director._blocked_models
    assert director.finish_eviction(
        claim, success=False, error=RuntimeError("teardown failed"),
    )
    director.allow_model(resident.model_id)
    assert director.loaded[resident.model_id] is resident
    assert resident.model_id not in director.evicting
    assert resident.model_id not in director._blocked_models


def test_interrupted_eviction_claim_publication_rolls_back_every_marker():
    class _InterruptedProbe(_Probe):
        def cpu_free_gb(self) -> float:
            raise KeyboardInterrupt

    director = LoadDirector(
        enable_fn=MagicMock(return_value=9002),
        disable_fn=MagicMock(),
        memory_probe=_InterruptedProbe(),
        cpu_headroom_gb=0.0,
    )
    resident = _entry()
    director.loaded[resident.model_id] = resident

    with pytest.raises(KeyboardInterrupt):
        director.begin_model_disable(resident.model_id)

    assert director.loaded[resident.model_id] is resident
    assert resident.model_id not in director.evicting
    assert resident.model_id not in director._blocked_models
    assert resident.model_id not in director._eviction_claims
    assert resident.model_id not in director._evicting_pools
    assert resident.model_id not in director._evicting_memory_gb


def test_admin_disable_wrapper_settles_director_and_blocks_until_enable():
    enable_fn = MagicMock(return_value=9002)
    director = _director(enable_fn=enable_fn)
    resident = _entry()
    director.loaded[resident.model_id] = resident
    state = SupervisorState()
    state.director = director
    expected = {"model_id": resident.model_id, "loaded": False}

    with patch(
        "muse.admin.operations.known_models", return_value={resident.model_id: object()},
    ), patch(
        "muse.admin.operations._disable_model_worker", return_value=expected,
    ) as worker_disable:
        assert disable_model(resident.model_id, state=state) == expected

    worker_disable.assert_called_once_with(resident.model_id, state=state)
    assert resident.model_id not in director.loaded
    assert resident.model_id not in director.evicting
    assert resident.model_id in director._blocked_models
    with pytest.raises(OperationError) as caught:
        director.acquire(resident.model_id, manifest=_manifest())
    assert caught.value.code == "model_disabled"

    director.allow_model(resident.model_id)
    assert director.acquire(resident.model_id, manifest=_manifest()) == 9002
    enable_fn.assert_called_once_with(resident.model_id)


def test_failed_disable_restores_accounting_and_prior_allow_state():
    director = _director()
    resident = _entry()
    director.loaded[resident.model_id] = resident
    state = SupervisorState()
    state.director = director

    with patch(
        "muse.admin.operations.known_models", return_value={resident.model_id: object()},
    ), patch(
        "muse.admin.operations._disable_model_worker",
        side_effect=RuntimeError("catalog write failed"),
    ), patch("muse.admin.operations.is_enabled", return_value=True):
        with pytest.raises(RuntimeError, match="catalog write failed"):
            disable_model(resident.model_id, state=state)

    assert director.loaded[resident.model_id] is resident
    assert resident.model_id not in director.evicting
    assert resident.model_id not in director._blocked_models


def test_disabled_model_cannot_be_warmed_directly():
    state = SupervisorState()
    state.director = MagicMock()
    with patch("muse.admin.operations.known_models", return_value={"model": object()}), \
         patch("muse.admin.operations.is_pulled", return_value=True), \
         patch("muse.admin.operations.is_enabled", return_value=False):
        with pytest.raises(OperationError) as caught:
            warmup_model("model", state=state)

    assert caught.value.code == "model_disabled"
    state.director.warmup.assert_not_called()


def test_removing_disabled_model_clears_persistent_director_block():
    director = _director()
    director._blocked_models.add("model")
    state = SupervisorState()
    state.director = director
    with patch("muse.admin.operations.known_models", return_value={"model": object()}), \
         patch("muse.admin.operations.is_pulled", return_value=True), \
         patch("muse.admin.operations._read_catalog", return_value={"model": {}}), \
         patch("muse.admin.operations.catalog_remove") as catalog_remove:
        assert remove_model("model", state=state, purge=False) == {
            "model_id": "model", "removed": True, "purged": False,
        }

    catalog_remove.assert_called_once_with("model", purge=False)
    assert "model" not in director._blocked_models


def test_admin_disable_blocks_a_racing_acquire_before_worker_teardown():
    entered_worker_teardown = threading.Event()
    finish_worker_teardown = threading.Event()

    def slow_worker_disable(_model_id: str, *, state: SupervisorState) -> dict:
        entered_worker_teardown.set()
        assert finish_worker_teardown.wait(timeout=2.0)
        return {"model_id": _model_id, "loaded": False}

    enable_fn = MagicMock(return_value=9002)
    director = _director(enable_fn=enable_fn)
    resident = _entry()
    director.loaded[resident.model_id] = resident
    state = SupervisorState()
    state.director = director
    result: list[dict] = []

    with patch(
        "muse.admin.operations.known_models", return_value={resident.model_id: object()},
    ), patch(
        "muse.admin.operations._disable_model_worker", side_effect=slow_worker_disable,
    ):
        thread = threading.Thread(
            target=lambda: result.append(disable_model(resident.model_id, state=state)),
        )
        thread.start()
        assert entered_worker_teardown.wait(timeout=2.0)
        with pytest.raises(OperationError) as caught:
            director.acquire(resident.model_id, manifest=_manifest())
        assert caught.value.code == "model_disabled"
        enable_fn.assert_not_called()
        finish_worker_teardown.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert result == [{"model_id": resident.model_id, "loaded": False}]


def test_idle_eviction_reserves_capacity_and_blocks_racing_reacquire():
    entered_disable = threading.Event()
    finish_disable = threading.Event()

    def slow_disable(_model_id: str) -> None:
        entered_disable.set()
        assert finish_disable.wait(timeout=2.0)

    enable_fn = MagicMock(return_value=9002)
    director = _director(disable_fn=slow_disable, enable_fn=enable_fn)
    resident = _entry()
    director.loaded[resident.model_id] = resident
    notifications: list[str] = []
    director.capacity_listener = lambda: notifications.append("capacity")
    sweeper = IdleSweeper(
        director=director,
        catalog_lookup=lambda _model_id: {
            "capabilities": {"idle_timeout_seconds": 1.0},
        },
    )
    outcome: list[list[str]] = []
    thread = threading.Thread(target=lambda: outcome.append(sweeper.tick()))
    thread.start()
    assert entered_disable.wait(timeout=2.0)

    with director.lock:
        assert resident.model_id not in director.loaded
        assert resident.model_id in director.evicting
        assert director._resident_for_pool("cpu") == pytest.approx(3.0)
        _, available = director._available_for_device("cpu")
        assert available == pytest.approx(1.0)
    with pytest.raises(OperationError) as caught:
        director.acquire(resident.model_id, manifest=_manifest())
    assert caught.value.code == "model_eviction_in_progress"
    assert caught.value.retryable is True
    enable_fn.assert_not_called()

    finish_disable.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert outcome == [[resident.model_id]]
    assert resident.model_id not in director.evicting
    assert resident.model_id not in director._blocked_models
    assert notifications == ["capacity"]
    assert director.recent_decisions[-1].reason == "idle_timeout:1s"


def test_failed_idle_eviction_clears_transient_state_and_restores_capacity():
    director = _director(disable_fn=MagicMock(side_effect=RuntimeError("boom")))
    resident = _entry()
    director.loaded[resident.model_id] = resident
    notifications: list[str] = []
    director.capacity_listener = lambda: notifications.append("capacity")
    sweeper = IdleSweeper(
        director=director,
        catalog_lookup=lambda _model_id: {
            "capabilities": {"idle_timeout_seconds": 1.0},
        },
    )

    assert sweeper.tick() == []

    assert director.loaded[resident.model_id] is resident
    assert resident.model_id not in director.evicting
    assert resident.model_id not in director._blocked_models
    assert director._resident_for_pool("cpu") == pytest.approx(3.0)
    assert notifications == ["capacity"]


async def test_explicit_disabled_model_never_reaches_director():
    state = SimpleNamespace(director=MagicMock())
    with patch(
        "muse.cli_impl.gateway._read_catalog",
        return_value={"model": {"enabled": False}},
    ):
        response = await _route_via_director(
            MagicMock(), "v1/audio/speech", "model", state, 1.0,
        )

    assert response.status_code == 409
    assert json.loads(response.body)["error"]["code"] == "model_disabled"
    state.director.acquire.assert_not_called()


def test_proxy_response_preserves_multiple_set_cookie_header_lines():
    upstream = SimpleNamespace(raw=[
        (b"set-cookie", b"session=one; Path=/"),
        (b"set-cookie", b"csrf=two; Path=/"),
        (b"connection", b"x-worker-hop"),
        (b"x-worker-hop", b"remove-me"),
        (b"x-end-to-end", b"keep-me"),
    ])
    forwarded = _proxy_headers(upstream)
    response = _install_response_headers(Response(b"ok"), forwarded)

    assert [
        value for key, value in response.raw_headers if key == b"set-cookie"
    ] == [b"session=one; Path=/", b"csrf=two; Path=/"]
    assert (b"x-end-to-end", b"keep-me") in response.raw_headers
    assert not any(key == b"x-worker-hop" for key, _value in response.raw_headers)
    assert (b"content-length", b"2") in response.raw_headers


async def test_shutdown_cancellation_before_response_start_returns_clean_503():
    entered = asyncio.Event()

    async def blocked_app(_scope, _receive, _send):
        entered.set()
        await asyncio.Event().wait()

    stop_event = threading.Event()
    middleware = ShutdownCancellationMiddleware(blocked_app, stop_event=stop_event)
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    task = asyncio.create_task(middleware({"type": "http"}, receive, send))
    await entered.wait()
    stop_event.set()
    task.cancel()
    await task

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 503
    body = json.loads(sent[1]["body"])
    assert body["error"]["code"] == "server_shutting_down"


def _request() -> Request:
    delivered = False

    async def receive() -> dict:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        return {"type": "http.disconnect"}

    return Request({
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/speech",
        "raw_path": b"/v1/audio/speech",
        "query_string": b"",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 80),
    }, receive)


class _BlockingWorkerResponse:
    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self, entered: asyncio.Event | None = None) -> None:
        self.entered = entered
        self.iterator_started = False

    async def aiter_raw(self):
        self.iterator_started = True
        if self.entered is not None:
            self.entered.set()
            await asyncio.Event().wait()
        yield b"data: ok\n\n"


class _StreamContext:
    def __init__(self, response: _BlockingWorkerResponse) -> None:
        self.response = response
        self.exit_calls = 0

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *_args):
        self.exit_calls += 1


class _Client:
    def __init__(self, context: _StreamContext) -> None:
        self.context = context
        self.close_calls = 0

    def stream(self, **_kwargs):
        return self.context

    async def aclose(self):
        self.close_calls += 1


def _http_scope() -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/speech",
        "raw_path": b"/v1/audio/speech",
        "query_string": b"",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 80),
    }


async def test_stream_send_failure_before_iteration_releases_all_ownership():
    worker_response = _BlockingWorkerResponse()
    context = _StreamContext(worker_response)
    client = _Client(context)
    director = MagicMock()
    extra_release = MagicMock()
    with patch("muse.cli_impl.gateway.httpx.AsyncClient", return_value=client):
        response = await _forward_with_release(
            _request(), "http://worker.test/v1/audio/speech", 1.0,
            director=director,
            model_id="model",
            extra_release=extra_release,
        )

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    async def fail_start(_message: dict) -> None:
        raise RuntimeError("downstream send failed")

    with pytest.raises(RuntimeError, match="downstream send failed"):
        await response(_http_scope(), receive, fail_start)

    assert worker_response.iterator_started is False
    director.release.assert_called_once_with("model")
    extra_release.assert_called_once_with()
    assert context.exit_calls == 1
    assert client.close_calls == 1


async def test_active_stream_shutdown_cleans_up_without_cancelled_error():
    iterator_entered = asyncio.Event()
    worker_response = _BlockingWorkerResponse(iterator_entered)
    context = _StreamContext(worker_response)
    client = _Client(context)
    director = MagicMock()
    extra_release = MagicMock()
    with patch("muse.cli_impl.gateway.httpx.AsyncClient", return_value=client):
        response = await _forward_with_release(
            _request(), "http://worker.test/v1/audio/speech", 1.0,
            director=director,
            model_id="model",
            extra_release=extra_release,
        )

    stop_event = threading.Event()
    middleware = ShutdownCancellationMiddleware(response, stop_event=stop_event)
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    task = asyncio.create_task(middleware(_http_scope(), receive, send))
    await asyncio.wait_for(iterator_entered.wait(), timeout=1.0)
    stop_event.set()
    task.cancel()
    await task

    assert sent[0]["type"] == "http.response.start"
    director.release.assert_called_once_with("model")
    extra_release.assert_called_once_with()
    assert context.exit_calls == 1
    assert client.close_calls == 1
