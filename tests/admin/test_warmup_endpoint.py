"""Tests for warmup_model operation, /v1/admin/models/{id}/warmup route,
and AdminClient.warmup HTTP wrapper.

Task G of the v0.40.0 lazy-load plan: pre-load a model without serving
a request. Mirrors the enable_model shape but synchronous (no JobStore
wrapping; warmup itself is fast modulo the load duration, but on the
caller's request it returns the load result inline).

The route-level tests use TestClient with the admin token configured
and a mocked SupervisorState whose director.warmup is a MagicMock so
no real worker is spawned. The operation-level tests inject a fake
director so the unit under test is exactly `warmup_model` and nothing
else.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from muse.admin.auth import ADMIN_TOKEN_ENV, verify_admin_token
from muse.admin.client import AdminClient, AdminClientError
from muse.admin.operations import OperationError, warmup_model
from muse.admin.routes.models import build_models_router
from muse.cli_impl.supervisor import (
    SupervisorState,
    clear_supervisor_state,
    set_supervisor_state,
)


# ----------------------------------------------------------------------
# Operation-level: warmup_model directly, no HTTP
# ----------------------------------------------------------------------


class TestWarmupOperation:
    """warmup_model wraps director.warmup with the standard error mapping."""

    def test_returns_model_id_and_worker_port_on_success(self):
        director = MagicMock()
        director.warmup.return_value = 9001
        state = SupervisorState(workers=[], device="cpu")
        state.director = director

        # Need a manifest in the catalog; patch get_manifest at the
        # operations module's import site so the call inside warmup_model
        # resolves cleanly without touching the real catalog.
        with patch(
            "muse.admin.operations.known_models",
            return_value={"fake-model": MagicMock()},
        ), patch(
            "muse.admin.operations.is_pulled",
            return_value=True,
        ), patch(
            "muse.admin.operations.is_enabled",
            return_value=True,
        ), patch(
            "muse.admin.operations.get_manifest",
            return_value={
                "model_id": "fake-model",
                "modality": "audio/speech",
                "capabilities": {"memory_gb": 0.5, "device": "cpu"},
            },
        ):
            result = warmup_model("fake-model", state=state)

        assert result == {"model_id": "fake-model", "worker_port": 9001}
        director.warmup.assert_called_once()

    def test_unknown_model_raises_404_OperationError(self):
        director = MagicMock()
        state = SupervisorState(workers=[], device="cpu")
        state.director = director

        with patch("muse.admin.operations.known_models", return_value={}):
            with pytest.raises(OperationError) as exc_info:
                warmup_model("ghost", state=state)

        err = exc_info.value
        assert err.code == "model_not_found"
        assert err.status == 404
        director.warmup.assert_not_called()

    def test_known_but_unpulled_model_raises_409_before_director(self):
        # warmup_model must mirror enable_model's preflight: validate
        # is_pulled upfront, BEFORE invoking the director. Otherwise the
        # error surfaces only after the director's load phase via its
        # exception cleanup path, which is harder to read in logs.
        director = MagicMock()
        state = SupervisorState(workers=[], device="cpu")
        state.director = director

        with patch(
            "muse.admin.operations.known_models",
            return_value={"fake-model": MagicMock()},
        ), patch(
            "muse.admin.operations.is_pulled",
            return_value=False,
        ):
            with pytest.raises(OperationError) as exc_info:
                warmup_model("fake-model", state=state)

        err = exc_info.value
        assert err.code == "model_not_pulled"
        assert err.status == 409
        # Director must not have been touched; the upfront check fired.
        director.warmup.assert_not_called()

    def test_propagates_director_OperationError(self):
        # When director.warmup raises (e.g. model_too_large_for_device),
        # warmup_model should let the exception propagate so the route
        # handler can map it to the right HTTP status.
        director = MagicMock()
        director.warmup.side_effect = OperationError(
            "model_too_large_for_device",
            "cannot fit",
            status=503,
        )
        state = SupervisorState(workers=[], device="cpu")
        state.director = director

        with patch(
            "muse.admin.operations.known_models",
            return_value={"fake-model": MagicMock()},
        ), patch(
            "muse.admin.operations.is_pulled",
            return_value=True,
        ), patch(
            "muse.admin.operations.is_enabled",
            return_value=True,
        ), patch(
            "muse.admin.operations.get_manifest",
            return_value={
                "model_id": "fake-model",
                "modality": "audio/speech",
                "capabilities": {"memory_gb": 100.0, "device": "cuda"},
            },
        ):
            with pytest.raises(OperationError) as exc_info:
                warmup_model("fake-model", state=state)

        assert exc_info.value.code == "model_too_large_for_device"
        assert exc_info.value.status == 503

    def test_state_without_director_raises_helpful_error(self):
        # Defensive: if the supervisor state has no director (e.g. the
        # supervisor hasn't been booted yet), warmup_model should fail
        # cleanly rather than NPE on `state.director.warmup`.
        state = SupervisorState(workers=[], device="cpu")
        # state.director is None by default

        with patch(
            "muse.admin.operations.known_models",
            return_value={"fake-model": MagicMock()},
        ), patch(
            "muse.admin.operations.is_pulled",
            return_value=True,
        ), patch(
            "muse.admin.operations.is_enabled",
            return_value=True,
        ), patch(
            "muse.admin.operations.get_manifest",
            return_value={
                "model_id": "fake-model",
                "modality": "audio/speech",
                "capabilities": {"memory_gb": 0.5, "device": "cpu"},
            },
        ):
            with pytest.raises(OperationError) as exc_info:
                warmup_model("fake-model", state=state)

        assert exc_info.value.status == 503
        assert "director" in exc_info.value.message.lower()


class TestWarmupBackfillsMemory:
    """Regression: warmup_model must backfill capabilities.memory_gb from
    the catalog sizing ladder (probe measurement, else on-disk weights)
    before invoking director.warmup, exactly like the gateway request
    path does via backfill_manifest_memory. Without this, a never-probed
    model reads memory_gb as fallback 0.0, always "fits", reserves 0
    memory against concurrent loads, and can over-admit -> OOM.
    """

    def test_warmup_sizes_never_probed_model_from_weights_on_disk(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        from muse.core.catalog import _catalog_path, _reset_known_models_cache

        weights_dir = tmp_path / "weights" / "fake-model"
        weights_dir.mkdir(parents=True)
        (weights_dir / "model.bin").write_bytes(b"x" * (50 * 1024 * 1024))

        catalog_path = _catalog_path()
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(json.dumps({
            "fake-model": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "local_dir": str(weights_dir),
                "manifest": {
                    "model_id": "fake-model",
                    "modality": "audio/speech",
                    "hf_repo": "org/fake-model",
                    "backend_path": "muse.fake:FakeModel",
                    "capabilities": {"device": "cpu"},
                },
            },
        }))
        _reset_known_models_cache()

        director = MagicMock()
        director.warmup.return_value = 9001
        state = SupervisorState(workers=[], device="cpu")
        state.director = director

        result = warmup_model("fake-model", state=state)

        assert result == {"model_id": "fake-model", "worker_port": 9001}
        _, kwargs = director.warmup.call_args
        manifest_arg = kwargs.get("manifest")
        assert manifest_arg is not None, "director.warmup must be called with manifest="
        memory_gb = (manifest_arg.get("capabilities") or {}).get("memory_gb")
        assert memory_gb is not None and memory_gb > 0, (
            f"expected memory_gb backfilled > 0 from weights on disk, got {memory_gb!r}"
        )


# ----------------------------------------------------------------------
# Route-level: POST /v1/admin/models/{id}/warmup via TestClient
# ----------------------------------------------------------------------


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
def _state_reset():
    clear_supervisor_state()
    yield
    clear_supervisor_state()


class TestWarmupRoute:
    def test_returns_200_with_operation_result(
        self, client, headers, monkeypatch,
    ):
        # Stub warmup_model itself so we exercise just the route layer.
        captured = {}

        def fake_warmup(model_id: str, *, state):
            captured["model_id"] = model_id
            captured["state"] = state
            return {"model_id": model_id, "worker_port": 9123}

        monkeypatch.setattr(
            "muse.admin.routes.models.warmup_model", fake_warmup,
        )
        state = SupervisorState(workers=[], device="cpu")
        set_supervisor_state(state)

        r = client.post("/v1/admin/models/fake-model/warmup", headers=headers)

        assert r.status_code == 200
        body = r.json()
        assert body == {"model_id": "fake-model", "worker_port": 9123}
        assert captured["model_id"] == "fake-model"
        assert captured["state"] is state

    def test_unknown_model_returns_404(self, client, headers, monkeypatch):
        # warmup_model raises OperationError(404); the route maps it to
        # the OpenAI-shape envelope with status 404.
        def fake_warmup(model_id: str, *, state):
            raise OperationError(
                "model_not_found", f"unknown model {model_id!r}", status=404,
            )

        monkeypatch.setattr(
            "muse.admin.routes.models.warmup_model", fake_warmup,
        )
        state = SupervisorState(workers=[], device="cpu")
        set_supervisor_state(state)

        r = client.post("/v1/admin/models/ghost/warmup", headers=headers)

        assert r.status_code == 404
        body = r.json()
        assert body["error"]["code"] == "model_not_found"

    def test_too_large_returns_503(self, client, headers, monkeypatch):
        def fake_warmup(model_id: str, *, state):
            raise OperationError(
                "model_too_large_for_device",
                "cannot fit",
                status=503,
            )

        monkeypatch.setattr(
            "muse.admin.routes.models.warmup_model", fake_warmup,
        )
        state = SupervisorState(workers=[], device="cpu")
        set_supervisor_state(state)

        r = client.post("/v1/admin/models/oversized/warmup", headers=headers)

        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "model_too_large_for_device"
        # A 5xx OperationError surfaced through the admin route carries the
        # status-derived type (server_error), not a client-error label.
        assert body["error"]["type"] == "server_error"

    def test_requires_bearer_auth(self, client):
        # No headers: 401 (or 503 if env var unset; the fixture sets it).
        r = client.post("/v1/admin/models/fake-model/warmup")
        assert r.status_code in (401, 403)

    def test_wrong_token_returns_403(self, client):
        r = client.post(
            "/v1/admin/models/fake-model/warmup",
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 403


# ----------------------------------------------------------------------
# AdminClient.warmup HTTP behavior
# ----------------------------------------------------------------------


class TestAdminClientWarmup:
    @pytest.fixture
    def admin_client(self, monkeypatch):
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        monkeypatch.delenv("MUSE_SERVER", raising=False)
        return AdminClient(base_url="http://test.example.com", token="tok")

    @pytest.fixture
    def mock_httpx_client(self):
        with patch("muse.admin.client.httpx.Client") as cls:
            ctx = MagicMock()
            cls.return_value = ctx
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=None)
            yield ctx

    def test_warmup_calls_post_with_correct_path(
        self, admin_client, mock_httpx_client,
    ):
        # 200 response on success.
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = {"model_id": "kokoro-82m", "worker_port": 9001}
        mock_httpx_client.request.return_value = r

        out = admin_client.warmup("kokoro-82m")

        assert out == {"model_id": "kokoro-82m", "worker_port": 9001}
        call = mock_httpx_client.request.call_args
        assert call.args[0] == "POST"
        assert call.args[1].endswith("/v1/admin/models/kokoro-82m/warmup")
        # Bearer token forwarded
        assert call.kwargs["headers"]["Authorization"] == "Bearer tok"

    def test_warmup_propagates_503_as_AdminClientError(
        self, admin_client, mock_httpx_client,
    ):
        r = MagicMock()
        r.status_code = 503
        r.json.return_value = {
            "error": {
                "code": "model_too_large_for_device",
                "message": "cannot fit",
                "type": "server_error",
            },
        }
        r.text = "err"
        mock_httpx_client.request.return_value = r

        with pytest.raises(AdminClientError) as exc_info:
            admin_client.warmup("oversized")

        err = exc_info.value
        assert err.status == 503
        assert err.code == "model_too_large_for_device"

    def test_warmup_per_call_timeout_overrides_constructor_default(
        self, monkeypatch,
    ):
        # Constructor default is 30s, which is too short for many cold
        # loads (SDXL ~30s, FLUX ~45s, video ~60s+). The per-call
        # timeout argument lets callers (the CLI verb, programmatic
        # users) raise it without touching the constructor.
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        monkeypatch.delenv("MUSE_SERVER", raising=False)
        client = AdminClient(
            base_url="http://test.example.com",
            token="tok",
            timeout=30.0,
        )
        with patch("muse.admin.client.httpx.Client") as cls:
            ctx = MagicMock()
            cls.return_value = ctx
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=None)
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {"model_id": "sdxl", "worker_port": 9002}
            ctx.request.return_value = r

            client.warmup("sdxl", timeout=300.0)

        # The httpx.Client(...) constructor receives the per-call value,
        # NOT the AdminClient constructor's default.
        cls.assert_called_once_with(timeout=300.0)

    def test_warmup_default_timeout_uses_constructor_value(
        self, monkeypatch,
    ):
        # When the per-call `timeout` is omitted (None), the constructor's
        # value applies. This protects existing call sites that don't
        # know about the new parameter.
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        monkeypatch.delenv("MUSE_SERVER", raising=False)
        client = AdminClient(
            base_url="http://test.example.com",
            token="tok",
            timeout=42.0,
        )
        with patch("muse.admin.client.httpx.Client") as cls:
            ctx = MagicMock()
            cls.return_value = ctx
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=None)
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {"model_id": "sdxl", "worker_port": 9002}
            ctx.request.return_value = r

            client.warmup("sdxl")  # no per-call timeout

        cls.assert_called_once_with(timeout=42.0)
