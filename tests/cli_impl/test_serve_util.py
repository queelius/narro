"""Tests for muse.cli_impl.serve_util.

The supervisor gateway (port 8000) and each per-model worker both run a
FastAPI app under uvicorn. uvicorn's default `timeout_graceful_shutdown`
is None, meaning the first Ctrl-C waits FOREVER for any in-flight
connection (an SSE chat stream, a long inference, or an idle keep-alive
socket) to drain before the process exits. Until then the process stays
alive holding the listen port, so `muse serve` cannot be restarted
without manually killing the stuck process.

serve_util centralizes uvicorn construction with a BOUNDED graceful
timeout so the first Ctrl-C always exits within a fixed window.
"""
import os
import signal
import threading
from unittest.mock import MagicMock, patch

import pytest

from muse.cli_impl import serve_util


def test_build_uvicorn_server_sets_bounded_graceful_timeout():
    # The whole point: timeout_graceful_shutdown must be a finite number,
    # never None (the uvicorn default that hangs forever).
    server = serve_util.build_uvicorn_server(
        object(), host="127.0.0.1", port=8000,
    )
    assert server.config.timeout_graceful_shutdown is not None
    assert server.config.timeout_graceful_shutdown == pytest.approx(10.0)
    # host/port must round-trip so the gateway still binds where asked.
    assert server.config.host == "127.0.0.1"
    assert server.config.port == 8000


def test_server_signals_shutdown_event_as_soon_as_signal_arrives(monkeypatch):
    # sse-starlette patches uvicorn.Server.handle_exit and records shutdown
    # in process-global state. Mock the superclass method so this unit test
    # verifies delegation without cancelling later in-process SSE responses.
    delegated: list[tuple[object, int, object]] = []

    def _base_handle_exit(server, sig, frame):
        delegated.append((server, sig, frame))
        server.should_exit = True

    monkeypatch.setattr(serve_util.uvicorn.Server, "handle_exit", _base_handle_exit)
    shutdown_event = threading.Event()
    server = serve_util.build_uvicorn_server(
        object(),
        host="127.0.0.1",
        port=8000,
        shutdown_event=shutdown_event,
    )

    server.handle_exit(signal.SIGTERM, None)

    assert shutdown_event.is_set()
    assert server.should_exit is True
    assert delegated == [(server, signal.SIGTERM, None)]


def test_shutdown_grace_seconds_default():
    with patch.dict(os.environ, {}, clear=True):
        assert serve_util.shutdown_grace_seconds() == pytest.approx(10.0)


def test_shutdown_grace_seconds_env_override():
    with patch.dict(os.environ, {"MUSE_SHUTDOWN_GRACE_SECONDS": "2.5"}):
        assert serve_util.shutdown_grace_seconds() == pytest.approx(2.5)


def test_shutdown_grace_seconds_invalid_falls_back():
    with patch.dict(os.environ, {"MUSE_SHUTDOWN_GRACE_SECONDS": "not-a-number"}):
        assert serve_util.shutdown_grace_seconds() == pytest.approx(10.0)


def test_shutdown_grace_seconds_negative_falls_back():
    # A negative grace is nonsensical; treat it as the default rather than
    # letting it reach uvicorn (which would treat <0 unpredictably).
    with patch.dict(os.environ, {"MUSE_SHUTDOWN_GRACE_SECONDS": "-4"}):
        assert serve_util.shutdown_grace_seconds() == pytest.approx(10.0)


@pytest.mark.parametrize("raw", ["inf", "infinity", "Infinity", "-inf"])
def test_shutdown_grace_seconds_non_finite_falls_back(raw):
    # inf parses via float() and passes `>= 0`, but timeout_graceful_shutdown
    # =inf is exactly the hang-forever behavior this module exists to
    # prevent. Reject anything non-finite so the env var can never
    # re-introduce it.
    with patch.dict(os.environ, {"MUSE_SHUTDOWN_GRACE_SECONDS": raw}):
        assert serve_util.shutdown_grace_seconds() == pytest.approx(10.0)


def test_run_uvicorn_swallows_keyboardinterrupt():
    # uvicorn re-raises the SIGINT it captured after a graceful shutdown;
    # uvicorn.run() swallows that KeyboardInterrupt and returns normally.
    # run_uvicorn claims to be a drop-in replacement, so it must too --
    # otherwise a caller that migrates off uvicorn.run() would suddenly see
    # a KeyboardInterrupt propagate out of a clean Ctrl-C shutdown.
    fake_server = MagicMock()
    fake_server.run.side_effect = KeyboardInterrupt()
    with patch.object(
        serve_util, "build_uvicorn_server", return_value=fake_server,
    ):
        # Must NOT raise.
        serve_util.run_uvicorn(object(), host="127.0.0.1", port=8000)
    fake_server.run.assert_called_once()
