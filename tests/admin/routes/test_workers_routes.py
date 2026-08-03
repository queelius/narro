"""Tests for /v1/admin/workers and /v1/admin/workers/{port}/restart."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from muse.admin.auth import ADMIN_TOKEN_ENV, verify_admin_token
from muse.admin.routes.workers import build_workers_router
from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    clear_supervisor_state,
    set_supervisor_state,
)


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(
        build_workers_router(),
        prefix="/v1/admin",
        dependencies=[Depends(verify_admin_token)],
    )
    return app


@pytest.fixture
def client(app, monkeypatch):
    monkeypatch.setenv(ADMIN_TOKEN_ENV, "test-token")
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def headers():
    return {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def _state_reset():
    clear_supervisor_state()
    yield
    clear_supervisor_state()


class TestListWorkers:
    def test_empty_state_returns_empty_list(self, client, headers):
        set_supervisor_state(SupervisorState(workers=[], device="cpu"))
        r = client.get("/v1/admin/workers", headers=headers)
        assert r.status_code == 200
        assert r.json() == {"workers": []}

    def test_lists_one_worker(self, client, headers):
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/v/bin/python", port=9001,
        )
        spec.status = "running"
        spec.last_spawn_at = 0.0
        spec.process = MagicMock(pid=999)
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.get("/v1/admin/workers", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert len(body["workers"]) == 1
        w = body["workers"][0]
        assert w["port"] == 9001
        assert w["models"] == ["kokoro-82m"]
        assert w["pid"] == 999
        assert w["status"] == "running"


class TestRestartWorker:
    def test_unknown_port_returns_404(self, client, headers):
        set_supervisor_state(SupervisorState(workers=[], device="cpu"))
        r = client.post("/v1/admin/workers/99999/restart", headers=headers)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "worker_not_found"

    def test_no_process_returns_409(self, client, headers):
        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001,
        )
        # process is None by default
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "worker_not_running"

    def test_terminate_called_and_200_returned(self, client, headers):
        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001,
        )
        spec.process = MagicMock()
        spec.process.poll.return_value = None
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 200
        assert r.json()["signal"] == "SIGTERM"
        spec.process.terminate.assert_called_once()

    def test_terminate_failure_returns_500_server_error(self, client, headers):
        """A SIGTERM that raises is a 500 terminate_failed, and its OpenAI
        type is derived from the status (server_error for 5xx), not the
        hardcoded invalid_request_error the old local builder emitted."""
        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001,
        )
        spec.process = MagicMock()
        spec.process.poll.return_value = None
        spec.process.terminate.side_effect = OSError("no such process")
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 500
        body = r.json()
        assert body["error"]["code"] == "terminate_failed"
        assert body["error"]["type"] == "server_error"

    def test_resets_restart_and_failure_counters(self, client, headers):
        """v0.34.0 finding #12: an operator's explicit restart must
        re-arm the restart budget. Without this, a worker 9-deep into
        the 10-restart cap would die permanently on the next failure
        even though the operator just intervened."""
        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001,
        )
        spec.process = MagicMock()
        spec.process.poll.return_value = None
        spec.restart_count = 9
        spec.failure_count = 3
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))

        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["restart_count_reset"] is True
        assert spec.restart_count == 0
        assert spec.failure_count == 0
        spec.process.terminate.assert_called_once()
