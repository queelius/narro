"""End-to-end admin router tests.

Mounts the full /v1/admin/* router into a FastAPI app (the same builder
the gateway uses) and drives it through TestClient. SupervisorState is
stubbed; subprocess calls are mocked at module level so no real workers
or pull operations run.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from muse.admin.auth import ADMIN_TOKEN_ENV
from muse.admin.jobs import reset_default_store
from muse.admin.routes import build_admin_router
from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    clear_supervisor_state,
    set_supervisor_state,
)


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(build_admin_router())
    # Match the gateway wiring: unwrap admin auth's OpenAI-shaped
    # HTTPException details to a bare {"error": {...}} envelope.
    from muse.admin.errors import install_admin_error_handler
    install_admin_error_handler(app)
    return app


@pytest.fixture
def client(app, monkeypatch):
    monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret-test-token")
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def headers():
    return {"Authorization": "Bearer secret-test-token"}


@pytest.fixture(autouse=True)
def _reset_state_and_store():
    clear_supervisor_state()
    reset_default_store()
    yield
    clear_supervisor_state()
    reset_default_store()


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    from muse.core.catalog import _reset_known_models_cache
    _reset_known_models_cache()
    yield tmp_path
    _reset_known_models_cache()


def _seed(data: dict) -> None:
    from muse.core.catalog import _catalog_path, _reset_known_models_cache
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    _reset_known_models_cache()


class TestGatewayAdminErrorEnvelope:
    """Fix #4 (v0.47.4): admin auth errors use the bare OpenAI envelope
    {"error": {...}} on the real gateway, matching the route-level admin
    errors, not the double-wrapped {"detail": {"error": {...}}} that
    FastAPI's default HTTPException handler would produce.
    """

    def test_gateway_admin_auth_error_is_unwrapped(self, monkeypatch):
        from muse.cli_impl.gateway import build_gateway

        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        app = build_gateway(state=SupervisorState(workers=[], device="cpu"))
        c = TestClient(app, raise_server_exceptions=False)
        r = c.get("/v1/admin/workers")
        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "admin_disabled"
        # 503 is a 5xx, so the OpenAI type is server_error (derived from
        # status via core.errors.error_type_for_status), not the hardcoded
        # invalid_request_error the admin auth path used to emit.
        assert body["error"]["type"] == "server_error"
        assert "detail" not in body


class TestAuthEnvelope:
    def test_no_token_configured_returns_503(self, app, monkeypatch):
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        c = TestClient(app, raise_server_exceptions=False)
        # Try every endpoint family; all should 503.
        for path in [
            "/v1/admin/workers",
            "/v1/admin/memory",
            "/v1/admin/jobs",
            "/v1/admin/models/x/status",
        ]:
            r = c.get(path)
            assert r.status_code == 503, path
            assert r.json()["error"]["code"] == "admin_disabled"

    def test_no_header_returns_401(self, client, monkeypatch):
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret-test-token")
        r = client.get("/v1/admin/workers")
        assert r.status_code == 401

    def test_wrong_bearer_returns_403(self, client, headers):
        r = client.get(
            "/v1/admin/workers",
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 403

    def test_token_never_appears_in_error_body(self, client):
        # Force a 403 and assert the secret never bleeds into the body
        r = client.get(
            "/v1/admin/workers",
            headers={"Authorization": "Bearer not-the-token"},
        )
        assert r.status_code == 403
        assert "secret-test-token" not in r.text


class TestEnableDisableLifecycle:
    def test_enable_then_status_then_disable_then_remove(
        self, client, headers, tmp_catalog,
    ):
        # Seed catalog with a bundled-script-known model.
        _seed({
            "kokoro-82m": {
                "pulled_at": "...",
                "hf_repo": "hexgrad/Kokoro-82M",
                "local_dir": "/tmp/k",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": False,
            },
        })
        state = SupervisorState(workers=[], device="cpu")
        set_supervisor_state(state)

        # Stub supervisor calls so enable doesn't fork a real process.
        with patch("muse.admin.operations.spawn_worker") as mock_spawn, \
             patch("muse.admin.operations.wait_for_ready"), \
             patch("muse.admin.operations.find_free_port", return_value=9123):
            r = client.post(
                "/v1/admin/models/kokoro-82m/enable", headers=headers,
            )
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            # Poll the job until done. The thread is daemonic so we can
            # join it via the JobStore for determinism.
            from muse.admin.jobs import get_default_store
            store = get_default_store()
            store.get(job_id).thread.join(timeout=5.0)

            r = client.get(f"/v1/admin/jobs/{job_id}", headers=headers)
            assert r.status_code == 200
            body = r.json()
            assert body["state"] == "done", body
            assert body["result"]["worker_port"] == 9123

        mock_spawn.assert_called_once()

        # Status now reports loaded
        r = client.get("/v1/admin/models/kokoro-82m/status", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["loaded"] is True
        assert body["worker_port"] == 9123

        # Disable -> sync; worker terminated since it was the only model
        with patch("muse.admin.operations._shutdown_workers"):
            r = client.post(
                "/v1/admin/models/kokoro-82m/disable", headers=headers,
            )
        assert r.status_code == 200
        body = r.json()
        assert body["loaded"] is False
        assert body["worker_terminated"] is True

        # Remove -> sync
        r = client.delete(
            "/v1/admin/models/kokoro-82m?purge=false", headers=headers,
        )
        assert r.status_code == 200
        assert r.json()["removed"] is True


class TestRestartWorker:
    def test_restart_calls_terminate_on_process(self, client, headers):
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/v/bin/python", port=9001,
        )
        spec.process = MagicMock()
        spec.process.poll.return_value = None
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        r = client.post("/v1/admin/workers/9001/restart", headers=headers)
        assert r.status_code == 200
        spec.process.terminate.assert_called_once()


class TestJobsList:
    def test_jobs_list_after_enable(self, client, headers, tmp_catalog):
        from muse.admin.jobs import get_default_store
        store = get_default_store()
        store.create("enable", "model-1")
        store.create("pull", "model-2")
        r = client.get("/v1/admin/jobs", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert len(body["jobs"]) == 2
        # Newest first
        assert body["jobs"][0]["op"] == "pull"
        assert body["jobs"][1]["op"] == "enable"
