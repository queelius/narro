"""Tests for /v1/admin/models/{id}/* routes.

The router is mounted under /v1/admin (matching the gateway). Tests use
TestClient with the admin token configured. SupervisorState fixtures
are injected via set_supervisor_state so each test gets a clean state.
"""
from __future__ import annotations

import json

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from muse.admin.auth import ADMIN_TOKEN_ENV, verify_admin_token
from muse.admin.jobs import reset_default_store
from muse.admin.routes.models import build_models_router
from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    clear_supervisor_state,
    set_supervisor_state,
)


@pytest.fixture
def auth_app():
    """A FastAPI app mounting the models router with auth applied."""
    app = FastAPI()
    app.include_router(
        build_models_router(),
        prefix="/v1/admin",
        dependencies=[Depends(verify_admin_token)],
    )
    return app


@pytest.fixture
def client(auth_app, monkeypatch):
    monkeypatch.setenv(ADMIN_TOKEN_ENV, "test-token")
    return TestClient(auth_app, raise_server_exceptions=False)


@pytest.fixture
def headers():
    return {"Authorization": "Bearer test-token"}


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


def _seed_catalog(data: dict) -> None:
    from muse.core.catalog import _catalog_path, _reset_known_models_cache
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    _reset_known_models_cache()


def test_models_route_module_resolves_singletons_like_its_siblings():
    """All /v1/admin route modules must resolve shared singletons
    (SupervisorState, JobStore) the same way.

    jobs.py, workers.py, and memory.py all call
    `get_supervisor_state()` / `get_default_store()` directly. models.py
    used to route through private `_resolve_state()` / `_resolve_store()`
    wrappers whose docstring claimed "so tests can inject a state
    without going through the module-level singleton" -- untrue, since
    no test ever monkeypatched those wrappers; injection happens via
    `set_supervisor_state()` / `reset_default_store()` either way. A
    test that tried to patch the resolution point for models.py would
    have needed a different target than one patching the other three
    modules.
    """
    import muse.admin.routes.models as models_route_module

    assert not hasattr(models_route_module, "_resolve_state")
    assert not hasattr(models_route_module, "_resolve_store")


class TestEnableRoute:
    def test_unknown_model_returns_404(self, client, headers, tmp_catalog):
        _seed_catalog({})
        r = client.post("/v1/admin/models/ghost/enable", headers=headers)
        assert r.status_code == 404
        body = r.json()
        assert body["error"]["code"] == "model_not_found"

    def test_known_model_returns_202_with_job_id(
        self, client, headers, tmp_catalog, monkeypatch,
    ):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": False,
            },
        })
        # Stub launch_async so the route returns immediately without
        # actually spawning a worker thread that calls subprocess.
        from muse.admin.jobs import Job
        captured = {}

        def fake_launch(op, **kwargs):
            job = Job(job_id="abc", op=kwargs["op_name"], model_id=kwargs["model_id"])
            captured["job"] = job
            captured["kwargs"] = kwargs
            return job

        monkeypatch.setattr(
            "muse.admin.routes.models.launch_async", fake_launch,
        )
        state = SupervisorState(workers=[], device="cpu")
        set_supervisor_state(state)

        r = client.post("/v1/admin/models/kokoro-82m/enable", headers=headers)
        assert r.status_code == 202
        body = r.json()
        assert body["job_id"] == "abc"
        assert body["status"] == "pending"
        assert captured["kwargs"]["op_name"] == "enable"
        assert captured["kwargs"]["model_id"] == "kokoro-82m"

    def test_job_capacity_returns_structured_503(
        self, client, headers, tmp_catalog, monkeypatch,
    ):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": False,
            },
        })
        from muse.admin.operations import OperationError

        def reject(*_args, **_kwargs):
            raise OperationError(
                "admin_job_capacity", "admin job capacity reached",
                status=503, retryable=True,
            )

        monkeypatch.setattr("muse.admin.routes.models.launch_async", reject)
        response = client.post(
            "/v1/admin/models/kokoro-82m/enable", headers=headers,
        )

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "admin_job_capacity"


class TestDisableRoute:
    def test_unknown_model_returns_404(self, client, headers, tmp_catalog):
        _seed_catalog({})
        r = client.post("/v1/admin/models/ghost/disable", headers=headers)
        assert r.status_code == 404

    def test_unloaded_returns_200(self, client, headers, tmp_catalog):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
            },
        })
        state = SupervisorState(workers=[], device="cpu")
        set_supervisor_state(state)
        r = client.post("/v1/admin/models/kokoro-82m/disable", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["model_id"] == "kokoro-82m"
        assert body["loaded"] is False


class TestProbeRoute:
    def test_unknown_model_returns_404(self, client, headers, tmp_catalog):
        _seed_catalog({})
        r = client.post(
            "/v1/admin/models/ghost/probe", headers=headers, json={},
        )
        assert r.status_code == 404

    def test_known_returns_202(self, client, headers, tmp_catalog, monkeypatch):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
            },
        })
        from muse.admin.jobs import Job

        def fake_launch(op, **kwargs):
            return Job(job_id="x", op="probe", model_id=kwargs["model_id"])

        monkeypatch.setattr(
            "muse.admin.routes.models.launch_async", fake_launch,
        )
        state = SupervisorState(workers=[], device="cpu")
        set_supervisor_state(state)

        r = client.post(
            "/v1/admin/models/kokoro-82m/probe",
            headers=headers,
            json={"no_inference": True, "device": "cpu"},
        )
        assert r.status_code == 202
        assert r.json()["job_id"] == "x"


class TestPullRoute:
    def test_pull_with_body_identifier(self, client, headers, tmp_catalog, monkeypatch):
        from muse.admin.jobs import Job
        captured = {}

        def fake_launch(op, **kwargs):
            captured["model_id"] = kwargs["model_id"]
            captured["op_args"] = kwargs.get("op_args")
            return Job(job_id="j", op="pull", model_id=kwargs["model_id"])

        monkeypatch.setattr(
            "muse.admin.routes.models.launch_async", fake_launch,
        )
        r = client.post(
            "/v1/admin/models/_/pull",
            headers=headers,
            json={"identifier": "hf://Qwen/Qwen3-8B-GGUF@q4_k_m"},
        )
        assert r.status_code == 202
        assert captured["op_args"] == ("hf://Qwen/Qwen3-8B-GGUF@q4_k_m",)
        assert captured["model_id"] == "hf://Qwen/Qwen3-8B-GGUF@q4_k_m"

    def test_pull_with_path_id_no_body(self, client, headers, tmp_catalog, monkeypatch):
        from muse.admin.jobs import Job
        captured = {}

        def fake_launch(op, **kwargs):
            captured["model_id"] = kwargs["model_id"]
            captured["op_args"] = kwargs.get("op_args")
            return Job(job_id="j", op="pull", model_id=kwargs["model_id"])

        monkeypatch.setattr(
            "muse.admin.routes.models.launch_async", fake_launch,
        )
        r = client.post(
            "/v1/admin/models/qwen3.5-9b-q4/pull",
            headers=headers,
            json={},
        )
        assert r.status_code == 202
        assert captured["op_args"] == ("qwen3.5-9b-q4",)

    def test_pull_with_underscore_path_no_body_returns_400(
        self, client, headers, tmp_catalog,
    ):
        r = client.post(
            "/v1/admin/models/_/pull", headers=headers, json={},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "missing_identifier"


class TestDeleteRoute:
    def test_unknown_returns_404(self, client, headers, tmp_catalog):
        _seed_catalog({})
        r = client.delete("/v1/admin/models/ghost", headers=headers)
        assert r.status_code == 404

    def test_loaded_model_returns_409(self, client, headers, tmp_catalog):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/v/bin/python", port=9001,
        )
        state = SupervisorState(workers=[spec], device="cpu")
        set_supervisor_state(state)
        r = client.delete("/v1/admin/models/kokoro-82m", headers=headers)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "model_loaded"

    def test_unloaded_removes_with_purge(self, client, headers, tmp_catalog):
        venv_path = tmp_catalog / "venvs" / "kokoro-82m"
        weights_path = tmp_catalog / "weights" / "kokoro-82m"
        venv_path.mkdir(parents=True)
        weights_path.mkdir(parents=True)
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k",
                "local_dir": str(weights_path),
                "venv_path": str(venv_path),
                "python_path": str(venv_path / "bin" / "python"),
                "enabled": True,
            },
        })
        state = SupervisorState(workers=[], device="cpu")
        set_supervisor_state(state)
        r = client.delete(
            "/v1/admin/models/kokoro-82m?purge=true", headers=headers,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["removed"] is True
        assert body["purged"] is True
        assert not venv_path.exists()
        assert not weights_path.exists()


class TestStatusRoute:
    def test_unknown_returns_404(self, client, headers, tmp_catalog):
        _seed_catalog({})
        r = client.get("/v1/admin/models/ghost/status", headers=headers)
        assert r.status_code == 404

    def test_known_unloaded_returns_record(self, client, headers, tmp_catalog):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {"cpu": {"weights_bytes": 100}},
            },
        })
        state = SupervisorState(workers=[], device="cpu")
        set_supervisor_state(state)
        r = client.get("/v1/admin/models/kokoro-82m/status", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["model_id"] == "kokoro-82m"
        assert body["enabled"] is True
        assert body["loaded"] is False
        assert body["worker_port"] is None
        assert body["measurements"] == {"cpu": {"weights_bytes": 100}}

    def test_known_loaded_returns_worker_fields(self, client, headers, tmp_catalog):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/v/bin/python", port=9001,
        )
        spec.status = "running"
        spec.last_spawn_at = 0.0  # uptime is None when unset
        # Simulate a process with a pid
        from unittest.mock import MagicMock
        spec.process = MagicMock(pid=4711)
        state = SupervisorState(workers=[spec], device="cpu")
        set_supervisor_state(state)
        r = client.get("/v1/admin/models/kokoro-82m/status", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["loaded"] is True
        assert body["worker_port"] == 9001
        assert body["worker_pid"] == 4711
        assert body["worker_status"] == "running"


class TestAuthRequired:
    def test_no_token_returns_503(self, client, monkeypatch, tmp_path):
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        monkeypatch.setenv("MUSE_CONFIG", str(tmp_path / "absent-config.yaml"))
        from muse.core import config

        config.reset_config()
        try:
            r = client.get("/v1/admin/models/kokoro-82m/status")
            assert r.status_code == 503
        finally:
            config.reset_config()
