"""Shared uvicorn construction for the supervisor gateway and workers.

Both `muse serve` (the supervisor's gateway on port 8000) and each
per-model worker run a FastAPI app under uvicorn. Historically both
called ``uvicorn.run(app, host=..., port=...)`` directly, which builds a
``uvicorn.Config`` with ``timeout_graceful_shutdown=None``.

A None graceful timeout means the FIRST Ctrl-C (SIGINT) waits FOREVER for
any in-flight connection to drain before the process exits. In muse that
is trivial to trigger: an SSE chat stream, a long-running inference, or
even an idle keep-alive socket held open by an OpenAI-SDK client or a
browser tab. Until the connection closes, the process stays alive holding
its listen port, so `muse serve` cannot be restarted without manually
killing the stuck process.

This module builds the uvicorn server with a BOUNDED graceful-shutdown
timeout so the first Ctrl-C always exits within a fixed window (uvicorn
cancels any still-running request handlers once the window elapses). A
second Ctrl-C remains an immediate force-quit, unchanged.
"""
from __future__ import annotations

import math
import threading

import uvicorn

from muse.core import config

# Default seconds uvicorn waits for in-flight requests to drain on
# shutdown before cancelling them and exiting. Finite (never None), so
# Ctrl-C can never hang forever. Override with MUSE_SHUTDOWN_GRACE_SECONDS
# (e.g. "1" for near-instant dev restarts, or a larger value to let a
# genuine in-flight generation finish during a rolling restart).
_DEFAULT_GRACE_SECONDS = 10.0


class _ShutdownAwareServer(uvicorn.Server):
    """Uvicorn server that exposes signal receipt to supervisor threads."""

    def __init__(
        self,
        config: uvicorn.Config,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        super().__init__(config)
        self._muse_shutdown_event = shutdown_event

    def handle_exit(self, sig: int, frame) -> None:
        if self._muse_shutdown_event is not None:
            self._muse_shutdown_event.set()
        super().handle_exit(sig, frame)


def shutdown_grace_seconds() -> float:
    """Resolve the graceful-shutdown window from the environment.

    Falls back to the default on a missing, non-numeric, or negative
    value, so a fat-fingered env var can never re-introduce the
    hang-forever behavior.
    """
    value = config.get("server.shutdown_grace_seconds")
    if value is None:
        return _DEFAULT_GRACE_SECONDS
    # Reject non-finite (inf/nan) as well as negative: `inf` parses and is
    # `>= 0`, but timeout_graceful_shutdown=inf is precisely the
    # hang-forever behavior this module exists to prevent.
    if not math.isfinite(value) or value < 0:
        return _DEFAULT_GRACE_SECONDS
    return value


def build_uvicorn_server(
    app,
    *,
    host: str,
    port: int,
    shutdown_event: threading.Event | None = None,
) -> uvicorn.Server:
    """Construct a uvicorn.Server with a bounded graceful-shutdown timeout.

    Callers run it with ``.run()`` (blocking, installs SIGINT/SIGTERM
    handlers exactly like ``uvicorn.run``). When supplied, `shutdown_event`
    is set directly from Uvicorn's signal handler before graceful draining.
    """
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=None,
        timeout_graceful_shutdown=shutdown_grace_seconds(),
    )
    return _ShutdownAwareServer(config, shutdown_event=shutdown_event)


def run_uvicorn(
    app,
    *,
    host: str,
    port: int,
    shutdown_event: threading.Event | None = None,
) -> None:
    """Blocking uvicorn run with a bounded graceful-shutdown timeout.

    Drop-in replacement for ``uvicorn.run(app, host=host, port=port,
    log_config=None)`` that guarantees Ctrl-C exits within
    ``shutdown_grace_seconds()`` even with lingering connections. A supplied
    `shutdown_event` is signalled immediately on SIGINT/SIGTERM.
    """
    try:
        build_uvicorn_server(
            app,
            host=host,
            port=port,
            shutdown_event=shutdown_event,
        ).run()
    except KeyboardInterrupt:
        # uvicorn re-raises the SIGINT it captured once the graceful
        # shutdown completes. uvicorn.run() swallows that KeyboardInterrupt
        # and returns normally; mirror it so this stays a true drop-in and
        # a clean Ctrl-C never surfaces as an exception to the caller.
        pass
