"""Tests for the `muse mcp` CLI subcommand.

The CLI is typer-based as of v0.39.0; these tests exercise the
argument-binding behavior by invoking the typer app with a mocked
`run_mcp_server` and asserting the kwargs that arrived. The actual
asyncio.run is mocked out below.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from typer.testing import CliRunner

from muse.cli import app


class TestTyperBindings:
    """Verify typer command parameter binding for `muse mcp`.

    Each test invokes the CLI with argv and asserts the kwargs the
    subcommand received via a patched run_mcp_server stub.
    """

    @staticmethod
    def _invoke(argv):
        captured: dict = {}

        def _stub(**kwargs):
            captured.update(kwargs)
            return 0

        runner = CliRunner()
        with patch("muse.cli_impl.mcp_server.run_mcp_server", side_effect=_stub):
            result = runner.invoke(app, argv)
        return result, captured

    def test_mcp_defaults(self):
        result, kw = self._invoke(["mcp"])
        assert result.exit_code == 0, result.output
        assert kw["http"] is False
        assert kw["port"] == 8088
        assert kw["filter_kind"] == "all"

    def test_mcp_http_flag(self):
        result, kw = self._invoke(["mcp", "--http", "--port", "9999"])
        assert result.exit_code == 0, result.output
        assert kw["http"] is True
        assert kw["port"] == 9999

    def test_mcp_filter_admin(self):
        result, kw = self._invoke(["mcp", "--filter", "admin"])
        assert result.exit_code == 0, result.output
        assert kw["filter_kind"] == "admin"

    def test_mcp_filter_inference(self):
        result, kw = self._invoke(["mcp", "--filter", "inference"])
        assert result.exit_code == 0, result.output
        assert kw["filter_kind"] == "inference"

    def test_mcp_server_and_token_args(self):
        result, kw = self._invoke([
            "mcp",
            "--server", "http://other:8000",
            "--admin-token", "abc",
        ])
        assert result.exit_code == 0, result.output
        assert kw["server_url"] == "http://other:8000"
        assert kw["admin_token"] == "abc"


class TestRunMcpServer:
    def test_runs_stdio_by_default(self, monkeypatch):
        from muse.cli_impl.mcp_server import run_mcp_server

        ran_modes: list[str] = []

        def fake_asyncio_run(coro):
            # The coro is run_stdio(); record and close it.
            ran_modes.append(coro.__qualname__)
            coro.close()
            return None

        monkeypatch.setattr("muse.cli_impl.mcp_server.asyncio.run", fake_asyncio_run)
        # Mock health to short-circuit
        with patch("muse.mcp.client.MuseClient.health", return_value={"status": "ok"}):
            rc = run_mcp_server(
                http=False,
                port=8088,
                server_url="http://test",
                admin_token=None,
                filter_kind="all",
            )
        assert rc == 0
        assert any("run_stdio" in m for m in ran_modes)

    def test_runs_http_when_flag_set(self, monkeypatch):
        from muse.cli_impl.mcp_server import run_mcp_server

        ran_modes: list[str] = []

        def fake_asyncio_run(coro):
            ran_modes.append(coro.__qualname__)
            coro.close()
            return None

        monkeypatch.setattr("muse.cli_impl.mcp_server.asyncio.run", fake_asyncio_run)
        with patch("muse.mcp.client.MuseClient.health", return_value={"status": "ok"}):
            rc = run_mcp_server(
                http=True,
                port=8088,
                server_url="http://test",
                admin_token="t",
                filter_kind="all",
            )
        assert rc == 0
        assert any("run_http" in m for m in ran_modes)

    def test_unreachable_health_warns_but_continues(self, monkeypatch, capsys):
        from muse.cli_impl.mcp_server import run_mcp_server

        def fake_asyncio_run(coro):
            coro.close()
            return None

        monkeypatch.setattr("muse.cli_impl.mcp_server.asyncio.run", fake_asyncio_run)
        # Force health to raise
        with patch(
            "muse.mcp.client.MuseClient.health",
            side_effect=ConnectionError("nope"),
        ):
            rc = run_mcp_server(
                http=False, port=8088,
                server_url="http://test",
                admin_token=None,
                filter_kind="all",
            )
        captured = capsys.readouterr()
        assert "warning: could not reach muse server" in captured.err
        assert rc == 0

    def test_admin_filter_without_token_warns(self, monkeypatch, capsys):
        from muse.cli_impl.mcp_server import run_mcp_server

        def fake_asyncio_run(coro):
            coro.close()
            return None

        monkeypatch.setattr("muse.cli_impl.mcp_server.asyncio.run", fake_asyncio_run)
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        with (
            patch("muse.mcp.client.MuseClient.health", return_value={"status": "ok"}),
            patch("muse.cli_impl.mcp_server.config.get", return_value=None),
        ):
            rc = run_mcp_server(
                http=False, port=8088,
                server_url="http://test",
                admin_token=None,
                filter_kind="admin",
            )
        captured = capsys.readouterr()
        assert "warning: --filter admin without MUSE_ADMIN_TOKEN" in captured.err
        assert rc == 0

    def test_invalid_filter_returns_2(self, monkeypatch, capsys):
        from muse.cli_impl.mcp_server import run_mcp_server

        # Bypass argparse choices=... by calling run_mcp_server directly.
        with patch("muse.mcp.client.MuseClient.health", return_value={"status": "ok"}):
            rc = run_mcp_server(
                http=False, port=8088,
                server_url="http://test",
                admin_token=None,
                filter_kind="bogus",
            )
        captured = capsys.readouterr()
        assert rc == 2
        assert "error" in captured.err

    def test_http_admin_token_arms_auth(self, monkeypatch):
        """CLI-supplied --admin-token must actually arm the /mcp auth gate.

        Regression coverage for a bug where run_mcp_server passed
        admin_token to MuseClient (outbound) but never to run_http, so
        starting `muse mcp --http --admin-token SECRET` without also
        exporting MUSE_ADMIN_TOKEN left the HTTP+SSE endpoint open.

        We intercept the coroutine asyncio.run() would have executed,
        pull the bound `self` (the MCPServer instance) and `admin_token`
        argument out of its not-yet-started frame, then build the real
        Starlette app from them and drive it with TestClient to confirm
        an unauthenticated request is rejected with 401.
        """
        pytest.importorskip(
            "mcp.server.streamable_http_manager",
            reason="mcp SDK without streamable HTTP support",
        )
        from muse.cli_impl.mcp_server import run_mcp_server
        from starlette.testclient import TestClient

        captured: dict = {}

        def fake_asyncio_run(coro):
            frame = coro.cr_frame
            assert frame is not None, "coroutine already started/closed"
            captured["self"] = frame.f_locals.get("self")
            captured["admin_token"] = frame.f_locals.get("admin_token")
            coro.close()
            return None

        monkeypatch.setattr("muse.cli_impl.mcp_server.asyncio.run", fake_asyncio_run)
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        with patch("muse.mcp.client.MuseClient.health", return_value={"status": "ok"}):
            rc = run_mcp_server(
                http=True,
                port=8088,
                server_url="http://test",
                admin_token="t",
                filter_kind="all",
            )
        assert rc == 0
        assert captured["admin_token"] == "t", (
            "run_http did not receive the CLI --admin-token"
        )
        server = captured["self"]
        app = server.build_http_app(admin_token=captured["admin_token"])
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/mcp")
        assert resp.status_code == 401, (
            "unauthenticated /mcp request was not rejected: "
            "CLI --admin-token failed to arm the auth gate"
        )

    def test_keyboard_interrupt_exits_zero(self, monkeypatch):
        from muse.cli_impl.mcp_server import run_mcp_server

        def fake_asyncio_run(coro):
            coro.close()
            raise KeyboardInterrupt

        monkeypatch.setattr("muse.cli_impl.mcp_server.asyncio.run", fake_asyncio_run)
        with patch("muse.mcp.client.MuseClient.health", return_value={"status": "ok"}):
            rc = run_mcp_server(
                http=False, port=8088,
                server_url="http://test",
                admin_token=None,
                filter_kind="all",
            )
        assert rc == 0
