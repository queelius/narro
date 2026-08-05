"""Tests for Task 10: gateway request-telemetry recording + dashboard mount.

Two things land here:

  1. Every forwarded request (director-driven path) records one
     `request` telemetry event via `muse.cli_impl.gateway.record`, timed
     around the forward, with the modality derived structurally from
     the request path (no hardcoded per-route lookup table).
  2. `build_gateway` mounts the dashboard router
     (`GET /dashboard` -> 200) whenever `telemetry.enabled` is true AND
     a SupervisorState is passed; the legacy static-routes mode
     (state=None) has no supervisor to serve telemetry from, so it is
     not mounted there.

Reuses the existing director-path test harness from test_gateway_lazy.py
(`_make_state_with_director`, `_patch_get_manifest`, `_wire_async_client_json`)
rather than inventing a new one.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from muse.cli_impl.gateway import build_gateway
from muse.core import config
from tests.cli_impl.test_gateway_lazy import (
    _make_state_with_director,
    _patch_get_manifest,
    _wire_async_client_json,
)


class TestRequestTelemetry:
    def test_forwarded_request_records_one_request_event(self, monkeypatch):
        captured: list[tuple[str, dict]] = []

        def _fake_record(event_type: str, **fields) -> None:
            captured.append((event_type, fields))

        monkeypatch.setattr("muse.cli_impl.gateway.record", _fake_record)

        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls, response_status=200)
            r = client.post(
                "/v1/chat/completions",
                json={"model": "fake-model", "messages": []},
            )

        assert r.status_code == 200
        request_events = [c for c in captured if c[0] == "request"]
        assert len(request_events) == 1
        _, fields = request_events[0]
        assert fields["model_id"] == "fake-model"
        assert isinstance(fields["latency_ms"], (int, float))
        assert fields["latency_ms"] >= 0
        assert fields["status"] == r.status_code
        assert fields["stream"] is False
        assert fields["modality"] == "chat/completions"
        assert isinstance(fields["request_id"], str) and fields["request_id"]
        assert fields["cold"] is False
        assert fields["latency_ms"] >= fields["forward_ms"]
        assert fields["load_ms"] == 0.0
        assert fields["evicted_models"] == "[]"

    def test_record_failure_never_breaks_the_forward(self, monkeypatch):
        """record() raising must not propagate: telemetry is fire-and-forget."""
        def _boom(event_type: str, **fields) -> None:
            raise RuntimeError("telemetry backend exploded")

        monkeypatch.setattr("muse.cli_impl.gateway.record", _boom)

        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls, response_status=200)
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 200

    def test_cold_annotation_crosses_asyncio_to_thread_context(self, monkeypatch):
        from muse.observability.traces import note_model_eviction, note_model_load

        captured = []
        monkeypatch.setattr(
            "muse.cli_impl.gateway.record",
            lambda event_type, **fields: captured.append((event_type, fields)),
        )
        state = _make_state_with_director(acquire_port=9001)

        def acquire(model_id, *, manifest):
            # gateway invokes this through asyncio.to_thread; ContextVar
            # propagation is what links these notes to the outer request.
            note_model_eviction("old-model")
            note_model_load(model_id, 0.5)
            return 9001

        state.director.acquire.side_effect = acquire
        app = build_gateway(state=state)
        client = TestClient(app)
        with _patch_get_manifest(), patch(
            "muse.cli_impl.gateway.httpx.AsyncClient"
        ) as mock_cls:
            _wire_async_client_json(mock_cls, response_status=200)
            response = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert response.status_code == 200
        fields = [fields for event, fields in captured if event == "request"][0]
        assert fields["cold"] is True
        assert fields["load_ms"] == 500.0
        assert fields["evicted_models"] == '["old-model"]'

    def test_stream_trace_finishes_after_the_response_body(self, monkeypatch):
        captured = []
        chunks_seen = []
        finish_at = []
        monkeypatch.setattr(
            "muse.cli_impl.gateway.record",
            lambda event_type, **fields: captured.append((event_type, fields)),
        )
        state = _make_state_with_director(acquire_port=9001)

        class Tracker:
            def begin(self, request_id):
                self.request_id = request_id

            def finish(self, request_id):
                assert request_id == self.request_id
                finish_at.append(len(chunks_seen))
                return 5.5

        state.telemetry_vram_tracker = Tracker()
        app = build_gateway(state=state)
        client = TestClient(app)
        chunks = [b"data: one\n\n", b"event: done\ndata: \n\n"]
        with _patch_get_manifest(), patch(
            "muse.cli_impl.gateway.httpx.AsyncClient"
        ) as mock_cls:
            mock_client = MagicMock()
            mock_client.aclose = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/event-stream"}

            async def aiter_raw():
                for chunk in chunks:
                    chunks_seen.append(chunk)
                    yield chunk

            mock_response.aiter_raw = aiter_raw
            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream.return_value = stream_ctx
            mock_cls.return_value = mock_client

            response = client.post(
                "/v1/chat/completions",
                json={"model": "fake-model", "messages": [], "stream": True},
            )

        assert response.status_code == 200
        assert finish_at == [len(chunks)]
        fields = [fields for event, fields in captured if event == "request"][0]
        assert fields["stream"] is True
        assert fields["peak_vram_gb"] == 5.5

    def test_disabled_telemetry_skips_trace_creation(self, monkeypatch):
        monkeypatch.setenv("MUSE_TELEMETRY_ENABLED", "false")
        config.reset_config()
        monkeypatch.setattr(
            "muse.cli_impl.gateway.begin_request_trace",
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("trace must not be created when telemetry is disabled")
            ),
        )
        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app)
        try:
            with _patch_get_manifest(), patch(
                "muse.cli_impl.gateway.httpx.AsyncClient"
            ) as mock_cls:
                _wire_async_client_json(mock_cls, response_status=200)
                response = client.post(
                    "/v1/chat/completions",
                    json={"model": "fake-model", "messages": []},
                )
            assert response.status_code == 200
        finally:
            config.reset_config()


class TestDashboardMount:
    def test_dashboard_mounted_when_telemetry_enabled_and_state_present(self):
        config.reset_config()
        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_dashboard_absent_when_telemetry_disabled(self, monkeypatch):
        # With no dashboard router mounted, GET /dashboard falls through to
        # the catch-all proxy, which 400s "model_required" (no `model`
        # field) rather than FastAPI's native 404 -- that 400 IS the
        # "not mounted" signal here.
        monkeypatch.setenv("MUSE_TELEMETRY_ENABLED", "false")
        config.reset_config()
        try:
            state = _make_state_with_director(acquire_port=9001)
            app = build_gateway(state=state)
            client = TestClient(app, raise_server_exceptions=False)

            r = client.get("/dashboard")
            assert r.status_code == 400
            assert r.json()["error"]["code"] == "model_required"
        finally:
            config.reset_config()

    def test_dashboard_absent_in_legacy_static_routes_mode(self):
        """state=None (legacy static-routes mode) has no supervisor to
        serve telemetry from, so the dashboard router is not mounted
        even though telemetry.enabled defaults to true. Falls through to
        the catch-all proxy's 400 model_required, same signal as above."""
        config.reset_config()
        app = build_gateway([])
        client = TestClient(app, raise_server_exceptions=False)

        r = client.get("/dashboard")
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "model_required"
