"""Tests for verify_admin_token: the five auth paths.

The token is read from MUSE_ADMIN_TOKEN; without it, every request is
rejected with 503 (closed-by-default). With it, the bearer must match.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from muse.admin.auth import ADMIN_TOKEN_ENV, verify_admin_token
from muse.admin.errors import install_admin_error_handler
from muse.core import config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Never let auth tests read the operator's real Muse configuration."""
    monkeypatch.setenv("MUSE_CONFIG", str(tmp_path / "absent-config.yaml"))
    config.reset_config()
    try:
        yield
    finally:
        config.reset_config()


@pytest.fixture
def app():
    """A trivial FastAPI app whose only route requires the admin token.

    The OpenAI-envelope-unwrapping handler is installed exactly as the
    gateway installs it, so these tests assert the real wire shape
    (bare {"error": {...}}), not the dependency's raw HTTPException.
    """
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(verify_admin_token)])
    def protected():
        return {"ok": True}

    install_admin_error_handler(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestVerifyAdminToken:
    def test_no_token_configured_returns_503(self, client, monkeypatch):
        """Closed-by-default: missing env var -> 503 admin_disabled."""
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        r = client.get("/protected")
        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "admin_disabled"

    def test_no_token_configured_even_with_bearer_returns_503(self, client, monkeypatch):
        """A request CAN'T succeed if the env var is unset, even with a header."""
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        r = client.get("/protected", headers={"Authorization": "Bearer anything"})
        assert r.status_code == 503

    def test_token_set_no_header_returns_401(self, client, monkeypatch):
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected")
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "missing_token"

    def test_token_set_malformed_header_returns_401(self, client, monkeypatch):
        """An "Authorization: Token X" or other non-Bearer prefix -> 401."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected", headers={"Authorization": "Token secret"})
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "missing_token"

    def test_token_set_wrong_bearer_returns_403(self, client, monkeypatch):
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "invalid_token"

    def test_token_set_correct_bearer_passes(self, client, monkeypatch):
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_token_never_appears_in_403_response(self, client, monkeypatch):
        """The 403 envelope MUST NOT echo the configured secret."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "ultra-secret-12345")
        r = client.get("/protected", headers={"Authorization": "Bearer not-the-token"})
        assert r.status_code == 403
        assert "ultra-secret-12345" not in r.text

    def test_4xx_envelope_has_invalid_request_error_type(self, client, monkeypatch):
        """4xx admin errors (missing/bad token) carry invalid_request_error."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        r = client.get("/protected")
        assert r.status_code == 401
        body = r.json()
        assert body["error"]["type"] == "invalid_request_error"

    def test_503_admin_disabled_has_server_error_type(self, client, monkeypatch):
        """The 503 admin_disabled envelope's type is derived from the status
        (server_error for 5xx), matching core.errors.error_response -- not
        hardcoded invalid_request_error. SDK clients branching on error.type
        must see the same server-side class the core path emits for a 503."""
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        r = client.get("/protected")
        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "admin_disabled"
        assert body["error"]["type"] == "server_error"

    def test_whitespace_only_token_treated_as_disabled(self, client, monkeypatch):
        """A whitespace-only MUSE_ADMIN_TOKEN is an accident, not a secret:
        treat it as unset (503 closed-by-default), even with a Bearer header."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "   ")
        r = client.get("/protected", headers={"Authorization": "Bearer    "})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "admin_disabled"

    def test_token_with_trailing_newline_still_matches(self, client, monkeypatch):
        """`MUSE_ADMIN_TOKEN=$(cat tokenfile)` leaves a trailing newline; the
        operator's `Bearer secret` must still authenticate (the env token is
        stripped before comparison)."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret\n")
        r = client.get("/protected", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_non_ascii_bearer_returns_403_not_500(self, monkeypatch):
        """A non-ASCII bearer must be rejected (403), not crash the server.

        HTTP headers arrive latin-1-decoded, so a raw non-ASCII byte in the
        Authorization header reaches the dependency as a non-ASCII str.
        `secrets.compare_digest` raises TypeError on non-ASCII str args; that
        would surface as a 500. We call the dependency directly because the
        httpx test client refuses to send non-ASCII header bytes (the real
        ASGI layer does not).
        """
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret")
        with pytest.raises(HTTPException) as exc:
            verify_admin_token(authorization="Bearer café")
        assert exc.value.status_code == 403
        assert exc.value.detail["error"]["code"] == "invalid_token"

    def test_non_ascii_env_token_does_not_crash(self, monkeypatch):
        """A non-ASCII configured token must also not crash the compare."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "sécret")
        with pytest.raises(HTTPException) as exc:
            verify_admin_token(authorization="Bearer wrong")
        assert exc.value.status_code == 403

    def test_token_uses_constant_time_compare(self, client, monkeypatch):
        """Token comparison MUST go through secrets.compare_digest, not `!=`.

        A naive `!=` short-circuits on first byte mismatch, leaking the
        prefix one byte at a time via response-time variance. Even though
        the timing window is small for an in-process test, the regression
        guard here is structural: assert the constant-time API is the one
        being called.
        """
        from muse.admin import auth as auth_mod
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "secret-correct")
        calls: list[tuple[str, str]] = []
        original = auth_mod.secrets.compare_digest

        def spy(a, b):
            calls.append((a, b))
            return original(a, b)

        monkeypatch.setattr(auth_mod.secrets, "compare_digest", spy)
        r = client.get("/protected", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 403
        assert calls, "secrets.compare_digest was not called"
        # The two arguments should be presented bearer + expected env value,
        # UTF-8-encoded to bytes so non-ASCII tokens compare without raising.
        assert calls[0] == (b"wrong", b"secret-correct")

    def test_admin_token_from_config_file_unlocks(self, tmp_path, monkeypatch):
        """A token set only via admin.token in config.yaml (no env var) must
        unlock the server-side gate too. Before this fix, verify_admin_token
        read os.environ directly and a config-file-only token silently left
        the server closed while CLI/MCP clients (which go through
        muse.core.config) believed admin was configured."""
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("admin:\n  token: sekret\n")
        monkeypatch.setenv("MUSE_CONFIG", str(config_file))
        config.reset_config()
        try:
            assert verify_admin_token(authorization="Bearer sekret") is None
            with pytest.raises(HTTPException) as exc:
                verify_admin_token(authorization="Bearer wrong")
            assert exc.value.status_code == 403

            # A config file present but with no usable token still 503s:
            # closed-by-default applies to the file source too, not just env.
            blank_file = tmp_path / "blank.yaml"
            blank_file.write_text("admin:\n  token:\n")
            monkeypatch.setenv("MUSE_CONFIG", str(blank_file))
            config.reset_config()
            with pytest.raises(HTTPException) as exc:
                verify_admin_token(authorization="Bearer anything")
            assert exc.value.status_code == 503
            assert exc.value.detail["error"]["code"] == "admin_disabled"
        finally:
            config.reset_config()

    def test_admin_token_env_still_works(self, monkeypatch):
        """The env var path (MUSE_ADMIN_TOKEN) must keep working after the
        gate switched to reading through muse.core.config: config.get()
        checks the live env var before the config file."""
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "envtok")
        config.reset_config()
        try:
            assert verify_admin_token(authorization="Bearer envtok") is None
        finally:
            config.reset_config()

    def test_admin_disabled_when_neither_set(self, tmp_path, monkeypatch):
        """With neither the env var nor a config file token set, admin stays
        closed-by-default (503), regardless of the Authorization header."""
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        monkeypatch.setenv("MUSE_CONFIG", str(tmp_path / "does-not-exist.yaml"))
        config.reset_config()
        try:
            with pytest.raises(HTTPException) as exc:
                verify_admin_token(authorization="Bearer anything")
            assert exc.value.status_code == 503
            assert exc.value.detail["error"]["code"] == "admin_disabled"
        finally:
            config.reset_config()
