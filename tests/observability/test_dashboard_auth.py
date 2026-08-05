import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from muse.core import config
from muse.observability.dashboard_auth import check_dashboard_token, require_dashboard_auth


@pytest.fixture(autouse=True)
def _isolate_operator_config(tmp_path, monkeypatch):
    """Auth tests must not inherit the developer's real Muse token."""
    monkeypatch.setenv("MUSE_CONFIG", str(tmp_path / "absent-config.yaml"))
    config.reset_config()
    yield
    config.reset_config()


def test_no_token_configured_returns_503_dashboard_closed(monkeypatch):
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    config.reset_config()
    with pytest.raises(HTTPException) as exc_info:
        check_dashboard_token(None)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"]["code"] == "dashboard_closed"


def test_token_set_no_header_returns_401_missing_token(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
    config.reset_config()
    with pytest.raises(HTTPException) as exc_info:
        check_dashboard_token(None)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "missing_token"


def test_token_set_malformed_header_returns_401_missing_token(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
    config.reset_config()
    with pytest.raises(HTTPException) as exc_info:
        check_dashboard_token("secret123")
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "missing_token"


def test_token_set_wrong_bearer_returns_403_invalid_token(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
    config.reset_config()
    with pytest.raises(HTTPException) as exc_info:
        check_dashboard_token("Bearer wrongtoken")
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"]["code"] == "invalid_token"


def test_token_set_correct_via_bearer_returns_none(monkeypatch):
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
    config.reset_config()
    assert check_dashboard_token("Bearer secret123") is None


def test_auth_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("MUSE_TELEMETRY_REQUIRE_AUTH", "false")
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    config.reset_config()
    assert check_dashboard_token(None) is None


@pytest.fixture
def app():
    """A trivial FastAPI app whose only route requires the dashboard token.

    Unlike the admin auth reference suite (tests/admin/test_auth.py), no
    OpenAI-envelope-unwrapping handler is installed here: dashboard_auth has
    no equivalent to muse.admin.errors.install_admin_error_handler. So the
    real wire shape a bare route sees is FastAPI's default HTTPException
    handler, which wraps our {"error": {...}} detail under a top-level
    "detail" key: {"detail": {"error": {...}}}.
    """
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_dashboard_auth)])
    def protected():
        return {"ok": True}

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestRequireDashboardAuth:
    """Exercises require_dashboard_auth (the FastAPI dependency) through
    real HTTP, so the Header parameter wiring routes will actually mount
    is under test -- not just check_dashboard_token's internal logic.

    Header-only (v1 hardening): there is no ?access_token= path any more.
    """

    def test_no_token_configured_returns_503(self, client, monkeypatch):
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        config.reset_config()
        r = client.get("/protected")
        assert r.status_code == 503
        assert r.json()["detail"]["error"]["code"] == "dashboard_closed"

    def test_token_set_no_credential_returns_401(self, client, monkeypatch):
        monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
        config.reset_config()
        r = client.get("/protected")
        assert r.status_code == 401
        assert r.json()["detail"]["error"]["code"] == "missing_token"

    def test_token_set_correct_via_header_returns_200(self, client, monkeypatch):
        monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
        config.reset_config()
        r = client.get(
            "/protected", headers={"Authorization": "Bearer secret123"}
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_query_param_access_token_is_no_longer_accepted(self, client, monkeypatch):
        """The admin-token-in-URL path is REMOVED: a request that only
        supplies ?access_token= (no header) must be rejected, proving the
        query-param credential path is gone."""
        monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
        config.reset_config()
        r = client.get("/protected?access_token=secret123")
        assert r.status_code == 401
        assert r.json()["detail"]["error"]["code"] == "missing_token"

    def test_token_set_wrong_via_header_returns_403(self, client, monkeypatch):
        monkeypatch.setenv("MUSE_ADMIN_TOKEN", "secret123")
        config.reset_config()
        r = client.get(
            "/protected", headers={"Authorization": "Bearer wrongtoken"}
        )
        assert r.status_code == 403
        assert r.json()["detail"]["error"]["code"] == "invalid_token"
