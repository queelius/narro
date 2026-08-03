"""Tests for the lazy-load gateway integration (Task F of v0.40.0).

The gateway no longer routes by a static map of (model_id -> worker_url).
On every inference request the gateway:

  1. Extracts `model` from the request (existing logic).
  2. Short-circuits 503 if state.unservable_reasons[model_id] is set,
     before any director call.
  3. Resolves the manifest via muse.core.catalog.get_manifest. KeyError
     -> 404 model_not_found.
  4. Calls state.director.acquire(model_id, manifest=...) to get the
     worker port.
  5. Forwards the request to http://127.0.0.1:<port>/<path>.
  6. Calls state.director.release(model_id) on completion (success
     OR error; for SSE streams, on stream-close not on first chunk).

These tests cover that contract and verify the SSE release-timing.
"""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from muse.cli_impl.gateway import build_gateway
from muse.cli_impl.supervisor import SupervisorState
from muse.core.catalog import CatalogError


def _manifest(memory_gb: float = 0.5, device: str = "cpu") -> dict:
    return {
        "model_id": "fake-model",
        "modality": "audio/speech",
        "capabilities": {"memory_gb": memory_gb, "device": device},
    }


def _make_state_with_director(
    *,
    acquire_port: int = 9001,
    acquire_side_effect=None,
    unservable: dict | None = None,
) -> SupervisorState:
    """Build a SupervisorState with a mocked LoadDirector.

    The director is a MagicMock with .acquire() returning `acquire_port`
    (or raising via `acquire_side_effect`) and .release() recording calls.
    """
    state = SupervisorState(workers=[], device="cpu")
    director = MagicMock()
    if acquire_side_effect is not None:
        director.acquire.side_effect = acquire_side_effect
    else:
        director.acquire.return_value = acquire_port
    state.director = director
    if unservable:
        state.unservable_reasons.update(unservable)
    return state


def _patch_get_manifest(manifest: dict | None = None, raises: bool = False):
    """Return a patch object for muse.cli_impl.gateway.get_manifest.

    The gateway imports get_manifest from muse.core.catalog at the top
    of the module. Tests patch the gateway-module-level alias so the
    request-path lookup is intercepted without touching catalog state.
    """
    if raises:
        return patch(
            "muse.cli_impl.gateway.get_manifest",
            side_effect=KeyError("unknown model"),
        )
    return patch(
        "muse.cli_impl.gateway.get_manifest",
        return_value=manifest if manifest is not None else _manifest(),
    )


def _patch_async_client_for_json(
    *,
    response_body: bytes = b'{"ok": true}',
    response_status: int = 200,
    response_content_type: str = "application/json",
):
    """Patch httpx.AsyncClient inside the gateway to short-circuit a
    non-streaming JSON response.

    Returns the patch context manager. The mock_client is exposed via
    .return_value so the test can assert on .stream / .aclose calls.
    """
    p = patch("muse.cli_impl.gateway.httpx.AsyncClient")
    return p


def _wire_async_client_json(
    mock_client_cls: MagicMock,
    *,
    response_body: bytes = b'{"ok": true}',
    response_status: int = 200,
    response_content_type: str = "application/json",
) -> MagicMock:
    """Configure mock_client_cls to return a non-streaming JSON response."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.aclose = AsyncMock()

    mock_response = MagicMock()
    mock_response.status_code = response_status
    mock_response.headers = {"content-type": response_content_type}
    mock_response.aread = AsyncMock(return_value=response_body)

    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    stream_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(return_value=stream_ctx)

    mock_client_cls.return_value = mock_client
    return mock_client


@pytest.mark.asyncio
async def test_gateway_lifespan_signals_supervisor_shutdown_before_cleanup():
    state = _make_state_with_director()
    app = build_gateway(state=state)

    store = MagicMock()
    with patch("muse.admin.jobs.get_default_store", return_value=store):
        async with app.router.lifespan_context(app):
            assert not state.stop_event.is_set()

    assert state.stop_event.is_set()
    store.shutdown.assert_called_once_with(timeout=5.0)


# =============================================================================
# F1: per-request acquire + release wrap
# =============================================================================


class TestAcquireRelease:
    def test_proxy_calls_director_acquire_with_manifest(self):
        state = _make_state_with_director(acquire_port=9099)
        manifest = _manifest(memory_gb=0.7, device="cpu")
        app = build_gateway(state=state)
        client = TestClient(app)

        with _patch_get_manifest(manifest=manifest), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls)
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 200
        # acquire was called exactly once with the model_id + manifest
        state.director.acquire.assert_called_once_with(
            "fake-model", manifest=manifest,
        )

    def test_proxy_forwards_to_acquired_worker_port(self):
        state = _make_state_with_director(acquire_port=9099)
        app = build_gateway(state=state)
        client = TestClient(app)

        captured_url: dict[str, str] = {}

        def _capture_stream(method, url, **kwargs):
            captured_url["url"] = url
            stream_ctx = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.aread = AsyncMock(return_value=b'{"ok": true}')
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            return stream_ctx

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.aclose = AsyncMock()
            mock_client.stream = MagicMock(side_effect=_capture_stream)
            mock_cls.return_value = mock_client

            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 200
        # Forwarded to the port the director returned, not a static map.
        assert captured_url["url"] == "http://127.0.0.1:9099/v1/audio/speech"

    def test_proxy_calls_release_on_success(self):
        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls)
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 200
        state.director.release.assert_called_once_with("fake-model")

    def test_proxy_calls_release_on_backend_error(self):
        """If the worker connection raises, release MUST still fire."""
        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app, raise_server_exceptions=False)

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.aclose = AsyncMock()

            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(
                side_effect=httpx.ConnectError("worker died"),
            )
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=stream_ctx)
            mock_cls.return_value = mock_client

            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        # The forward raised; FastAPI surfaces 500.
        assert r.status_code == 500
        # Despite the exception, release ran via the finally clause.
        state.director.release.assert_called_once_with("fake-model")
        # Director acquire was called too.
        state.director.acquire.assert_called_once()

    def test_release_runs_even_when_aclose_raises_during_stream_open_failure(self):
        """Cascading-failure regression: stream-open raises AND aclose
        raises during cleanup. release MUST still fire so the refcount
        does not leak.

        Without the fix in `_forward_with_release`, a `aclose()` that
        raises would propagate before `director.release(...)` runs,
        stranding the refcount forever and (eventually) wedging the
        director's refcount-based eviction.
        """
        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app, raise_server_exceptions=False)

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            # aclose() itself raises during cleanup. Without the fix,
            # this exception would skip past director.release(...).
            mock_client.aclose = AsyncMock(
                side_effect=RuntimeError("aclose blew up"),
            )

            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(
                side_effect=httpx.ConnectError("worker died"),
            )
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=stream_ctx)
            mock_cls.return_value = mock_client

            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        # The forward raised; FastAPI surfaces 500.
        assert r.status_code == 500
        # The cascading failure must NOT have stolen the release call.
        state.director.release.assert_called_once_with("fake-model")
        state.director.acquire.assert_called_once()

    def test_release_runs_when_request_body_read_raises(self):
        """L16: director.acquire() bumps the refcount, then
        _forward_with_release reads the request body. A ClientDisconnect
        (client vanished mid-body) during that read happens AFTER acquire
        and BEFORE the stream-open try/except, so without the fix it skips
        director.release and strands the refcount, wedging eviction.

        A body-bearing GET is the reachable path: GET routes extract the
        model from the query string, so the body is unread until
        _forward_with_release -- the read there is the first one and CAN
        disconnect (a POST's body is already read + cached during model
        extraction, so its later read never hits the network). The read
        must release the slot on failure.
        """
        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app, raise_server_exceptions=False)

        async def _boom(self):
            raise ClientDisconnect("client went away mid-body")

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.Request.body", _boom):
            r = client.get(
                "/v1/audio/speech",
                params={"model": "fake-model"},
            )

        assert r.status_code == 500
        state.director.acquire.assert_called_once()
        state.director.release.assert_called_once_with("fake-model")


# =============================================================================
# F2: unservable short-circuit (BEFORE acquire)
# =============================================================================


class TestUnservableShortCircuit:
    # Each test seeds a catalog where `fake-model` stays genuinely unservable
    # (enabled, but no memory estimate and no weights on disk), so the
    # request-path `revalidate_servability` re-derives the same reason and
    # keeps the stamp -- the realistic in-catalog short-circuit. (A stamped
    # model ABSENT from the catalog is now cleared + 404'd; see
    # TestStaleUnservableRevalidation and the supervisor unit tests.)
    _UNSERVABLE_CATALOG = {
        "fake-model": {
            "enabled": True,
            "python_path": "/v/bin/python",
            "local_dir": "/tmp/does-not-exist-fake",  # no weights on disk
            "manifest": {"capabilities": {"device": "cpu"}},  # no estimate
        },
    }

    def test_unservable_returns_503_with_reason(self):
        reason = "no memory estimate; run `muse models probe` to populate"
        state = _make_state_with_director(
            unservable={"fake-model": reason},
        )
        app = build_gateway(state=state)
        client = TestClient(app)

        with patch(
            "muse.cli_impl.supervisor._read_catalog",
            return_value=self._UNSERVABLE_CATALOG,
        ), _patch_get_manifest():
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 503
        body = r.json()
        assert "error" in body
        assert "detail" not in body
        assert reason in body["error"]["message"]
        # The reason text is surfaced to the client.

    def test_unservable_does_not_call_acquire(self):
        state = _make_state_with_director(
            unservable={"fake-model": "no memory estimate"},
        )
        app = build_gateway(state=state)
        client = TestClient(app)

        with patch(
            "muse.cli_impl.supervisor._read_catalog",
            return_value=self._UNSERVABLE_CATALOG,
        ), _patch_get_manifest():
            client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        # Crucial: the director MUST NOT see this request.
        state.director.acquire.assert_not_called()
        state.director.release.assert_not_called()

    def test_unservable_uses_unservable_error_code(self):
        """The 503 envelope's code is `model_unservable`, not 503-default."""
        state = _make_state_with_director(
            unservable={"fake-model": "no memory estimate"},
        )
        app = build_gateway(state=state)
        client = TestClient(app)

        with patch(
            "muse.cli_impl.supervisor._read_catalog",
            return_value=self._UNSERVABLE_CATALOG,
        ), _patch_get_manifest():
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 503
        assert r.json()["error"]["code"] == "model_unservable"


# =============================================================================
# F3: unknown model 404
# =============================================================================


class TestUnknownModel:
    def test_unknown_model_returns_404(self):
        state = _make_state_with_director()
        app = build_gateway(state=state)
        client = TestClient(app)

        with _patch_get_manifest(raises=True):
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "ghost"},
            )

        assert r.status_code == 404
        body = r.json()
        assert body["error"]["code"] == "model_not_found"

    def test_unknown_model_does_not_call_acquire(self):
        state = _make_state_with_director()
        app = build_gateway(state=state)
        client = TestClient(app)

        with _patch_get_manifest(raises=True):
            client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "ghost"},
            )

        state.director.acquire.assert_not_called()
        state.director.release.assert_not_called()


# =============================================================================
# Regression: get_manifest raising CatalogError (corrupt catalog.json, no
# last-known-good cache) must NOT surface as a bare 500 `{"detail": ...}`.
# Pre-fix, a corrupt catalog silently 404'd (get_manifest returned nothing
# useful); the corrupt-guard now raises a distinct CatalogError, and the
# route must translate that into a clean OpenAI-shaped 503, matching the
# model_unservable envelope shape used elsewhere in this function.
# =============================================================================


class TestCatalogErrorOnGetManifest:
    def test_returns_clean_openai_shaped_503(self):
        state = _make_state_with_director()
        app = build_gateway(state=state)
        client = TestClient(app)

        with patch(
            "muse.cli_impl.gateway.get_manifest",
            side_effect=CatalogError("catalog at /tmp/catalog.json is corrupt"),
        ):
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "some-model"},
            )

        assert r.status_code == 503
        body = r.json()
        assert "error" in body
        assert "detail" not in body
        assert body["error"]["code"] == "catalog_unavailable"
        assert body["error"]["type"] == "server_error"

    def test_does_not_call_acquire(self):
        state = _make_state_with_director()
        app = build_gateway(state=state)
        client = TestClient(app)

        with patch(
            "muse.cli_impl.gateway.get_manifest",
            side_effect=CatalogError("catalog is corrupt"),
        ):
            client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "some-model"},
            )

        state.director.acquire.assert_not_called()
        state.director.release.assert_not_called()


# =============================================================================
# F4: acquire-failure -> 503 + no release
# =============================================================================


class TestAcquireRaises:
    def test_director_too_large_raises_503(self):
        from muse.admin.operations import OperationError

        state = _make_state_with_director(
            acquire_side_effect=OperationError(
                "model_too_large_for_device",
                "cannot fit",
                status=503,
            ),
        )
        app = build_gateway(state=state)
        client = TestClient(app, raise_server_exceptions=False)

        with _patch_get_manifest():
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "model_too_large_for_device"
        # acquire was attempted; release was NOT called (no successful acquire).
        state.director.acquire.assert_called_once()
        state.director.release.assert_not_called()

    def test_unexpected_acquire_error_returns_503_not_500(self):
        """A non-OperationError from director.acquire (e.g. a pynvml hiccup
        in _decide) must degrade the SAME way a same-model WAITER sees the
        identical root cause: 503 model_load_failed via _acquire_coalesced's
        mapless-failure branch (see
        TestColdLoadCoalescing::test_mapless_loader_failure_does_not_herd).
        Before the fix, only OperationError was caught here, so the loading
        request itself got a raw 500 while waiters got 503 for the exact
        same failure.
        """
        state = _make_state_with_director(
            acquire_side_effect=RuntimeError("pynvml hiccup"),
        )
        app = build_gateway(state=state)
        client = TestClient(app, raise_server_exceptions=False)

        with _patch_get_manifest():
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "model_load_failed"
        assert "detail" not in body
        state.director.acquire.assert_called_once()
        state.director.release.assert_not_called()


# =============================================================================
# F5: SSE streaming - release on stream-close, not first-chunk
# =============================================================================


class TestStreamingRelease:
    def test_sse_release_fires_at_stream_close(self):
        """The release MUST happen when the stream finishes, not when the
        first chunk is dispatched. We track the order of events: release
        must come after the LAST chunk, not before the first.
        """
        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        # Track the order in which (a) chunks are yielded by the relay,
        # (b) release is called. release_called_at_chunk_index records
        # the index of the chunk that was being processed when release
        # finally fired.
        chunk_count_at_release: list[int] = []
        chunks_yielded: list[bytes] = []

        chunks = [
            b"data: chunk1\n\n",
            b"data: chunk2\n\n",
            b"data: chunk3\n\n",
            b"event: done\ndata: \n\n",
        ]

        def release_recorder(model_id: str) -> None:
            chunk_count_at_release.append(len(chunks_yielded))

        state.director.release = MagicMock(side_effect=release_recorder)

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.aclose = AsyncMock()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/event-stream"}

            async def aiter_raw():
                for c in chunks:
                    chunks_yielded.append(c)
                    yield c

            mock_response.aiter_raw = aiter_raw
            mock_response.aclose = AsyncMock()
            mock_response.aread = AsyncMock(return_value=b"".join(chunks))

            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=stream_ctx)

            mock_cls.return_value = mock_client

            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert r.status_code == 200
        # All chunks reached the client.
        assert b"data: chunk1" in r.content
        assert b"event: done" in r.content
        # Release fired exactly once.
        assert state.director.release.call_count == 1
        # Crucial: release fired AFTER all chunks were yielded, not
        # before the first one. chunk_count_at_release should equal the
        # full chunk count, not 0 or 1.
        assert chunk_count_at_release == [len(chunks)], (
            f"release fired after {chunk_count_at_release} chunks; "
            f"expected after all {len(chunks)}"
        )

    def test_sse_release_fires_even_when_relay_iteration_raises(self):
        """If the upstream stream errors mid-relay (e.g. worker dies), the
        finally on the relay generator must still call release."""
        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app, raise_server_exceptions=False)

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.aclose = AsyncMock()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/event-stream"}

            async def aiter_raw():
                yield b"data: chunk1\n\n"
                raise httpx.ReadError("upstream died")

            mock_response.aiter_raw = aiter_raw
            mock_response.aclose = AsyncMock()
            mock_response.aread = AsyncMock(return_value=b"")

            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=stream_ctx)

            mock_cls.return_value = mock_client

            # Even a mid-stream error must produce a complete request
            # cycle from the gateway's bookkeeping perspective.
            try:
                r = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fake-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                # Some chunks may have made it before the error
                _ = r.content  # drain (may raise too)
            except Exception:
                pass

        # release fired exactly once despite the mid-stream upstream error
        state.director.release.assert_called_once_with("fake-model")


# =============================================================================
# Backward-compat: legacy static-routes path still works (when state is None)
# =============================================================================


# =============================================================================
# v0.47.3 Bug #2: a stale boot unservable stamp is re-checked against the
# live catalog before 503, so `muse models probe` takes effect without a
# supervisor restart.
# =============================================================================


class TestStaleUnservableRevalidation:
    def test_proceeds_to_acquire_when_estimate_appeared(self):
        """Boot stamped 'no memory estimate', but the catalog now carries an
        estimate (probe landed). The request must NOT 503; it proceeds to
        acquire and the stamp is cleared.
        """
        reason = "no memory estimate; run `muse models probe` to populate"
        state = _make_state_with_director(
            acquire_port=9001, unservable={"fake-model": reason},
        )
        app = build_gateway(state=state)
        client = TestClient(app)

        fresh_catalog = {
            "fake-model": {
                "enabled": True,
                "python_path": "/v/bin/python",
                "manifest": {"capabilities": {"memory_gb": 0.5, "device": "cpu"}},
            },
        }
        with patch(
            "muse.cli_impl.supervisor._read_catalog", return_value=fresh_catalog,
        ), _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls)
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 200
        state.director.acquire.assert_called_once()
        assert "fake-model" not in state.unservable_reasons

    def test_still_503_when_estimate_absent(self):
        """If the model still has no estimate, the 503 stands and acquire is
        never called.
        """
        reason = "no memory estimate; run `muse models probe` to populate"
        state = _make_state_with_director(unservable={"fake-model": reason})
        app = build_gateway(state=state)
        client = TestClient(app)

        fresh_catalog = {
            "fake-model": {
                "enabled": True,
                "python_path": "/v/bin/python",
                "manifest": {"capabilities": {"device": "cpu"}},  # no estimate
            },
        }
        with patch(
            "muse.cli_impl.supervisor._read_catalog", return_value=fresh_catalog,
        ), _patch_get_manifest():
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 503
        assert r.json()["error"]["code"] == "model_unservable"
        state.director.acquire.assert_not_called()


class TestRevalidationHonorsDirectorHeadroom:
    """The request-path servability re-check must use the SAME headroom the
    director was built with (state.director.gpu_headroom_gb /
    .cpu_headroom_gb), not the hardcoded 1.0/2.0 defaults baked into
    revalidate_servability's signature. An operator who configures a looser
    headroom (server.cpu_headroom_gb / MUSE_CPU_HEADROOM_GB) must see that
    SAME looser headroom applied here; otherwise the gate can 503 a model
    the director would actually admit.
    """

    class _FakeProbe:
        """Fixed 10 GB free CPU, no live GPU info -- deterministic fit math."""

        def cpu_free_gb(self) -> float:
            return 10.0

        def gpu_free_gb(self, device_id: int = 0):
            return None

    def test_revalidate_uses_director_configured_headroom(self):
        # 8.5 GB model: with the hardcoded 2.0 GB default headroom, only
        # 8.0 GB is available (10.0 - 2.0) -> does NOT fit -> stays
        # unservable. With the director's configured 0.5 GB headroom,
        # 9.5 GB is available -> FITS -> the stamp must clear.
        reason = "exceeds device capacity (stale boot stamp)"
        state = _make_state_with_director(
            acquire_port=9001, unservable={"fake-model": reason},
        )
        state.director.cpu_headroom_gb = 0.5
        state.director.gpu_headroom_gb = 1.0
        app = build_gateway(state=state)
        client = TestClient(app)

        fresh_catalog = {
            "fake-model": {
                "enabled": True,
                "python_path": "/v/bin/python",
                "manifest": {"capabilities": {"memory_gb": 8.5, "device": "cpu"}},
            },
        }
        with patch(
            "muse.cli_impl.supervisor._read_catalog", return_value=fresh_catalog,
        ), patch(
            "muse.cli_impl.supervisor._MemoryProbeAdapter",
            return_value=self._FakeProbe(),
        ), _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls)
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 200
        state.director.acquire.assert_called_once()
        assert "fake-model" not in state.unservable_reasons

    def test_mock_director_without_headroom_attrs_falls_back_to_defaults(self):
        """A director that doesn't expose numeric headroom attrs (e.g. a
        bare MagicMock in older tests) must not crash the servability
        check; it falls back to revalidate_servability's own defaults."""
        reason = "no memory estimate; run `muse models probe` to populate"
        state = _make_state_with_director(unservable={"fake-model": reason})
        app = build_gateway(state=state)
        client = TestClient(app)

        fresh_catalog = {
            "fake-model": {
                "enabled": True,
                "python_path": "/v/bin/python",
                "manifest": {"capabilities": {"device": "cpu"}},  # no estimate
            },
        }
        with patch(
            "muse.cli_impl.supervisor._read_catalog", return_value=fresh_catalog,
        ), _patch_get_manifest():
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 503
        assert r.json()["error"]["code"] == "model_unservable"


# =============================================================================
# v0.47.3 Gap #2b: the manifest handed to the director is sized from the
# catalog measurement / weights when it declares no memory_gb, so the
# director accounts for the load instead of treating it as 0 GB.
# =============================================================================


class TestManifestMemoryBackfill:
    def test_acquire_receives_manifest_sized_from_measurement(self):
        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        bare_manifest = {
            "model_id": "fake-model",
            "modality": "audio/speech",
            "capabilities": {"device": "cpu"},  # no memory_gb
        }
        fresh_catalog = {
            "fake-model": {
                "enabled": True,
                "python_path": "/v/bin/python",
                "manifest": {"capabilities": {"device": "cpu"}},
                "measurements": {
                    "cpu": {"peak_bytes": 800_000_000, "device": "cpu"},
                },
            },
        }
        with patch(
            "muse.cli_impl.supervisor._read_catalog", return_value=fresh_catalog,
        ), _patch_get_manifest(bare_manifest), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls)
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 200
        _, kwargs = state.director.acquire.call_args
        sized = kwargs["manifest"]
        assert sized["capabilities"]["memory_gb"] == pytest.approx(
            800_000_000 / 1024 ** 3, rel=1e-6,
        )


# =============================================================================
# v0.47.3 Bug #1: /v1/models lists enabled-but-unloaded catalog models.
# =============================================================================


class TestV1ModelsListsUnloaded:
    def test_lists_enabled_unloaded_with_loaded_false(self):
        state = _make_state_with_director()  # no workers loaded
        app = build_gateway(state=state)
        client = TestClient(app)

        fresh_catalog = {
            "cold-model": {
                "enabled": True,
                "python_path": "/v/bin/python",
                "manifest": {
                    "model_id": "cold-model",
                    "modality": "embedding/text",
                    "description": "a cold one",
                    "capabilities": {"device": "cpu", "memory_gb": 0.5},
                },
            },
            "disabled-model": {
                "enabled": False,
                "python_path": "/v/bin/python",
                "manifest": {
                    "model_id": "disabled-model",
                    "modality": "embedding/text",
                    "capabilities": {},
                },
            },
        }

        def _gm(mid: str) -> dict:
            return fresh_catalog[mid]["manifest"]

        with patch(
            "muse.cli_impl.gateway._read_catalog", return_value=fresh_catalog,
        ), patch("muse.cli_impl.gateway.get_manifest", side_effect=_gm):
            r = client.get("/v1/models")

        assert r.status_code == 200
        data = r.json()["data"]
        by_id = {e["id"]: e for e in data}
        assert "cold-model" in by_id
        assert by_id["cold-model"]["loaded"] is False
        assert by_id["cold-model"]["last_loaded_at"] is None
        assert by_id["cold-model"]["modality"] == "embedding/text"
        assert by_id["cold-model"]["description"] == "a cold one"
        # disabled models are not advertised
        assert "disabled-model" not in by_id

    def test_unloaded_entry_carries_unservable_reason(self):
        state = _make_state_with_director(
            unservable={"cold-model": "exceeds device capacity"},
        )
        app = build_gateway(state=state)
        client = TestClient(app)

        fresh_catalog = {
            "cold-model": {
                "enabled": True,
                "python_path": "/v/bin/python",
                "manifest": {
                    "model_id": "cold-model",
                    "modality": "embedding/text",
                    "capabilities": {"device": "cpu"},
                },
            },
        }
        with patch(
            "muse.cli_impl.gateway._read_catalog", return_value=fresh_catalog,
        ), patch(
            "muse.cli_impl.gateway.get_manifest",
            side_effect=lambda mid: fresh_catalog[mid]["manifest"],
        ):
            r = client.get("/v1/models")

        by_id = {e["id"]: e for e in r.json()["data"]}
        assert by_id["cold-model"]["unservable_reason"] == "exceeds device capacity"

    def test_empty_catalog_returns_empty_list(self):
        state = _make_state_with_director()
        app = build_gateway(state=state)
        client = TestClient(app)
        with patch(
            "muse.cli_impl.gateway._read_catalog", return_value={},
        ):
            r = client.get("/v1/models")
        assert r.status_code == 200
        assert r.json() == {"object": "list", "data": []}


class TestV1ModelsLoadedAtJoin:
    """Fix #2 (v0.47.4): the gateway fills last_loaded_at on resident
    entries from the director's LoadEntry. Workers self-report
    last_loaded_at=None (they run outside a supervisor and have no
    director); the gateway owns the load timestamps via
    state.director.loaded[id].loaded_at.
    """

    def _running_state_with_loaded(self, model_id: str, port: int):
        import time

        from muse.cli_impl.load_director import LoadEntry
        from muse.cli_impl.supervisor import WorkerSpec

        state = _make_state_with_director()
        state.workers.append(WorkerSpec(
            models=[model_id], python_path="/v/bin/python",
            port=port, status="running",
        ))
        now = time.monotonic()
        state.director.loaded = {
            model_id: LoadEntry(
                model_id=model_id, worker_port=port, memory_gb=0.5,
                refcount=0, last_touched_at=now, loaded_at=now,
            ),
        }
        state.director.lock = threading.RLock()
        return state

    def _patch_worker_models(self, data: list[dict]):
        """Patch httpx.AsyncClient so every worker /v1/models returns `data`."""
        def _factory():
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"object": "list", "data": data}

            async def fake_get(url, **kwargs):
                return resp

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            return mock_client

        p = patch("muse.cli_impl.gateway.httpx.AsyncClient")
        return p, _factory

    def test_resident_entry_gets_last_loaded_at_from_director(self):
        from datetime import datetime

        state = self._running_state_with_loaded("hot-model", 9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        # The worker self-reports last_loaded_at=None (the bug source).
        p, factory = self._patch_worker_models([
            {"id": "hot-model", "modality": "embedding/text",
             "object": "model", "loaded": True, "last_loaded_at": None},
        ])
        with p as mock_cls:
            mock_cls.return_value = factory()
            with patch("muse.cli_impl.gateway._read_catalog", return_value={}):
                r = client.get("/v1/models")

        assert r.status_code == 200
        by_id = {e["id"]: e for e in r.json()["data"]}
        assert by_id["hot-model"]["loaded"] is True
        loaded_at = by_id["hot-model"]["last_loaded_at"]
        assert loaded_at is not None
        assert isinstance(loaded_at, str)
        # Parseable ISO-8601 (raises if not).
        datetime.fromisoformat(loaded_at)

    def test_entry_not_in_director_keeps_self_reported_value(self):
        # hot-model is loaded per the director; ghost-model is reported by
        # a worker but absent from director.loaded -> its null stays null.
        state = self._running_state_with_loaded("hot-model", 9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        p, factory = self._patch_worker_models([
            {"id": "hot-model", "object": "model",
             "loaded": True, "last_loaded_at": None},
            {"id": "ghost-model", "object": "model",
             "loaded": True, "last_loaded_at": None},
        ])
        with p as mock_cls:
            mock_cls.return_value = factory()
            with patch("muse.cli_impl.gateway._read_catalog", return_value={}):
                r = client.get("/v1/models")

        by_id = {e["id"]: e for e in r.json()["data"]}
        assert by_id["hot-model"]["last_loaded_at"] is not None
        assert by_id["ghost-model"]["last_loaded_at"] is None

    def test_existing_non_null_last_loaded_at_not_overwritten(self):
        # If a worker ever reports its own non-null timestamp, the join
        # must not clobber it with the director's value.
        state = self._running_state_with_loaded("hot-model", 9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        sentinel = "2099-01-01T00:00:00+00:00"
        p, factory = self._patch_worker_models([
            {"id": "hot-model", "object": "model",
             "loaded": True, "last_loaded_at": sentinel},
        ])
        with p as mock_cls:
            mock_cls.return_value = factory()
            with patch("muse.cli_impl.gateway._read_catalog", return_value={}):
                r = client.get("/v1/models")

        by_id = {e["id"]: e for e in r.json()["data"]}
        assert by_id["hot-model"]["last_loaded_at"] == sentinel


class TestHealthReflectsCatalog:
    """Fix #6 (v0.47.4): /health lists enabled-but-unloaded catalog models
    and their modalities, not just resident workers, so the serviceable
    surface it reports matches /v1/models.
    """

    def _catalog(self):
        return {
            "hot-model": {  # also resident; must not be double-listed
                "enabled": True, "python_path": "/v/bin/python",
                "manifest": {"model_id": "hot-model",
                             "modality": "audio/speech", "capabilities": {}},
            },
            "cold-model": {  # enabled, not resident -> should appear
                "enabled": True, "python_path": "/v/bin/python",
                "manifest": {"model_id": "cold-model",
                             "modality": "embedding/text", "capabilities": {}},
            },
            "disabled-model": {  # disabled -> must not appear
                "enabled": False, "python_path": "/v/bin/python",
                "manifest": {"model_id": "disabled-model",
                             "modality": "image/generation", "capabilities": {}},
            },
        }

    def _worker_health(self, payload: dict):
        def _factory():
            resp = MagicMock(status_code=200)
            resp.json.return_value = payload

            async def fake_get(url, **kwargs):
                return resp

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            return mock_client

        return patch("muse.cli_impl.gateway.httpx.AsyncClient"), _factory

    def test_health_includes_enabled_unloaded(self):
        from muse.cli_impl.supervisor import WorkerSpec

        state = _make_state_with_director()
        state.workers.append(WorkerSpec(
            models=["hot-model"], python_path="/v/bin/python",
            port=9001, status="running",
        ))
        app = build_gateway(state=state)
        client = TestClient(app)

        catalog = self._catalog()
        p, factory = self._worker_health({
            "status": "ok", "modalities": ["audio/speech"], "models": ["hot-model"],
        })
        with p as mock_cls:
            mock_cls.return_value = factory()
            with patch(
                "muse.cli_impl.gateway._read_catalog", return_value=catalog,
            ), patch(
                "muse.cli_impl.gateway.get_manifest",
                side_effect=lambda mid: catalog[mid]["manifest"],
            ):
                r = client.get("/health")

        body = r.json()
        assert body["status"] == "ok"
        assert "hot-model" in body["models"]
        assert "cold-model" in body["models"]
        assert "disabled-model" not in body["models"]
        assert "audio/speech" in body["modalities"]
        assert "embedding/text" in body["modalities"]
        assert "image/generation" not in body["modalities"]


class TestLegacyStaticRoutesStillWork:
    """The build_gateway(routes=...) signature used by tests must keep
    working for tests that pre-date Task F. The static-routes path is the
    fallback when state is None or state.director is None.
    """

    def test_static_routes_path_does_not_call_get_manifest(self):
        from muse.cli_impl.gateway import WorkerRoute

        routes = [WorkerRoute(model_id="fake-model", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        # If get_manifest were called on the static-routes path, this
        # patch would break the request. The static path predates Task F
        # and must keep working untouched.
        with patch(
            "muse.cli_impl.gateway.get_manifest",
            side_effect=AssertionError("static path should not call get_manifest"),
        ), patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls)
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 200


# =============================================================================
# v0.50.x: acquire runs OFF the event loop (asyncio.to_thread) so a cold load
# no longer blocks the single gateway loop, plus cancellation-safe release.
# =============================================================================


class TestAcquireOffEventLoop:
    def test_acquire_runs_off_the_event_loop(self):
        """director.acquire must run on a worker thread, not the event-loop
        thread. Proven by comparing the thread that runs acquire against the
        thread that runs the forward (which stays on the loop). A synchronous
        acquire would run on the SAME thread as the forward.
        """
        threads: dict[str, int] = {}
        state = _make_state_with_director()

        def record_acquire(model_id, *, manifest):
            threads["acquire"] = threading.get_ident()
            return 9099

        state.director.acquire.side_effect = record_acquire
        app = build_gateway(state=state)
        client = TestClient(app)

        def _capture_stream(method, url, **kwargs):
            threads["forward"] = threading.get_ident()
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"content-type": "application/json"}
            resp.aread = AsyncMock(return_value=b'{"ok": true}')
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=None)
            return ctx

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            mc = MagicMock()
            mc.__aenter__ = AsyncMock(return_value=mc)
            mc.__aexit__ = AsyncMock(return_value=None)
            mc.aclose = AsyncMock()
            mc.stream = MagicMock(side_effect=_capture_stream)
            mock_cls.return_value = mc
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 200
        assert "acquire" in threads and "forward" in threads
        assert threads["acquire"] != threads["forward"]

    async def test_cancelled_request_during_acquire_releases_refcount(self):
        """If the request is cancelled (client disconnect / timeout) while the
        off-loop acquire is still running, and the acquire then SUCCEEDS (it
        bumped refcount), release must fire so the model is not pinned
        non-evictable forever. The forward must never be reached.
        """
        from muse.cli_impl.gateway import _route_via_director

        started = threading.Event()
        gate = threading.Event()
        forward_called: list[bool] = []
        state = _make_state_with_director()

        def slow_acquire(model_id, *, manifest):
            started.set()
            gate.wait(timeout=2)  # stay in-flight until the test releases us
            return 9099

        state.director.acquire.side_effect = slow_acquire

        async def fake_forward(*args, **kwargs):
            forward_called.append(True)
            return MagicMock()

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway._forward_with_release", new=fake_forward):
            task = asyncio.create_task(
                _route_via_director(
                    MagicMock(), "v1/audio/speech", "fake-model", state, 1.0,
                )
            )
            await asyncio.to_thread(started.wait, 2)  # acquire is now running off-loop
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            gate.set()  # let the detached acquire finish -> callback releases
            for _ in range(300):
                if state.director.release.called:
                    break
                await asyncio.sleep(0.01)

        assert not forward_called  # forward never reached (cancelled first)
        state.director.release.assert_called_once_with("fake-model")

    async def test_forward_release_fires_on_cancellation_during_stream_open(self):
        """A CancelledError while opening the worker connection (client
        disconnect) must still release the refcount. CancelledError is a
        BaseException, so an `except Exception` cleanup would leak it and pin
        the model non-evictable.
        """
        from muse.cli_impl.gateway import _forward_with_release

        director = MagicMock()
        req = MagicMock()
        req.body = AsyncMock(return_value=b"")
        req.headers = {}
        req.method = "POST"
        req.query_params = {}

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            mc = MagicMock()
            mc.aclose = AsyncMock()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError())
            mc.stream = MagicMock(return_value=ctx)
            mock_cls.return_value = mc
            with pytest.raises(asyncio.CancelledError):
                await _forward_with_release(
                    req, "http://127.0.0.1:9099/x", 1.0,
                    director=director, model_id="m",
                )
        director.release.assert_called_once_with("m")

    async def test_forward_release_fires_on_cancellation_during_body_read(self):
        """A CancelledError during the request body read must release too."""
        from muse.cli_impl.gateway import _forward_with_release

        director = MagicMock()
        req = MagicMock()
        req.body = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await _forward_with_release(
                req, "http://127.0.0.1:9099/x", 1.0,
                director=director, model_id="m",
            )
        director.release.assert_called_once_with("m")


# =============================================================================
# #319: same-model cold-load coalescing. At most ONE thread parks per model-
# load; concurrent same-model requests await a shared asyncio gate on the loop.
# =============================================================================


class TestColdLoadCoalescing:
    async def test_same_model_burst_elects_one_loader(self):
        """N concurrent requests for one cold model -> only the LOADER
        dispatches director.acquire while the load is in flight; the other
        N-1 await the gate on the loop (no thread each). After the load
        settles, each waiter takes its own refcount via a hot re-acquire."""
        from muse.cli_impl.gateway import _acquire_coalesced

        started = threading.Event()
        release = threading.Event()
        state = _make_state_with_director()

        def blocking_acquire(model_id, *, manifest):
            started.set()
            release.wait(timeout=5)  # the loader's load "in flight"
            return 9001

        state.director.acquire.side_effect = blocking_acquire
        N = 8
        tasks = [asyncio.create_task(_acquire_coalesced(state, "M", _manifest()))
                 for _ in range(N)]
        await asyncio.to_thread(started.wait, 5)   # loader is now loading
        await asyncio.sleep(0.1)                     # let all waiters park on the gate

        assert state.director.acquire.call_count == 1   # ONLY the loader dispatched

        release.set()
        ports = await asyncio.gather(*tasks)
        assert ports == [9001] * N
        assert state.director.acquire.call_count == N   # loader + N-1 hot re-acquires
        assert state.cold_load_gates == {}              # gate cleaned up

    async def test_loader_failure_propagates_to_waiters_without_retry_herd(self):
        """When the loader's load fails, waiters get the SAME 503 and do NOT
        re-acquire (a retry herd would re-park N-1 threads)."""
        from muse.cli_impl.gateway import _acquire_coalesced
        from muse.admin.operations import OperationError

        started = threading.Event()
        release = threading.Event()
        state = _make_state_with_director()

        def failing_acquire(model_id, *, manifest):
            started.set()
            release.wait(timeout=5)
            raise OperationError("model_too_large_for_device", "nope", status=503)

        state.director.acquire.side_effect = failing_acquire
        tasks = [asyncio.create_task(_acquire_coalesced(state, "M", _manifest()))
                 for _ in range(5)]
        await asyncio.to_thread(started.wait, 5)
        await asyncio.sleep(0.1)
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert all(isinstance(r, OperationError) and r.status == 503 for r in results)
        assert state.director.acquire.call_count == 1   # loader only; no retry herd
        assert state.cold_load_gates == {}

    async def test_waiter_receives_loader_retryable_flag(self):
        """v0.57.3 review finding: the loader's retryable capacity 503 must
        reach waiters WITH the retryable flag intact -- otherwise the loader
        parks and retries while every waiter fast-fails, defeating the
        capacity-wait for same-model cold bursts."""
        from muse.cli_impl.gateway import _acquire_coalesced
        from muse.admin.operations import OperationError

        started = threading.Event()
        release = threading.Event()
        state = _make_state_with_director()

        def failing_acquire(model_id, *, manifest):
            started.set()
            release.wait(timeout=5)
            raise OperationError(
                "model_too_large_for_device", "transient", status=503,
                retryable=True,
            )

        state.director.acquire.side_effect = failing_acquire
        tasks = [asyncio.create_task(_acquire_coalesced(state, "M", _manifest()))
                 for _ in range(4)]
        await asyncio.to_thread(started.wait, 5)
        await asyncio.sleep(0.1)
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert all(isinstance(r, OperationError) for r in results)
        assert all(getattr(r, "retryable", False) is True for r in results)

    async def test_cancelling_one_waiter_does_not_poison_the_group(self):
        """Cancelling one waiter (via shield) must not cancel the shared gate;
        the loader and other waiters still complete."""
        from muse.cli_impl.gateway import _acquire_coalesced

        started = threading.Event()
        release = threading.Event()
        state = _make_state_with_director()

        def blocking_acquire(model_id, *, manifest):
            started.set()
            release.wait(timeout=5)
            return 9001

        state.director.acquire.side_effect = blocking_acquire
        loader = asyncio.create_task(_acquire_coalesced(state, "M", _manifest()))
        await asyncio.to_thread(started.wait, 5)
        await asyncio.sleep(0.05)
        w1 = asyncio.create_task(_acquire_coalesced(state, "M", _manifest()))
        w2 = asyncio.create_task(_acquire_coalesced(state, "M", _manifest()))
        await asyncio.sleep(0.05)  # let waiters park on the gate
        w1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await w1
        release.set()

        assert await loader == 9001
        assert await w2 == 9001

    async def test_cancelling_the_loader_still_completes_waiters(self):
        """THE subtle one: the gate is settled from a done-callback, not the
        loader body, so a loader cancelled mid-load (client disconnect) does
        NOT hang its waiters. Waiters still resolve, and the loader's orphaned
        refcount is released exactly once."""
        from muse.cli_impl.gateway import _acquire_coalesced

        started = threading.Event()
        release = threading.Event()
        state = _make_state_with_director()

        def blocking_acquire(model_id, *, manifest):
            started.set()
            release.wait(timeout=5)
            return 9001

        state.director.acquire.side_effect = blocking_acquire
        loader = asyncio.create_task(_acquire_coalesced(state, "M", _manifest()))
        await asyncio.to_thread(started.wait, 5)   # loader mid-load
        await asyncio.sleep(0.05)
        w1 = asyncio.create_task(_acquire_coalesced(state, "M", _manifest()))
        w2 = asyncio.create_task(_acquire_coalesced(state, "M", _manifest()))
        await asyncio.sleep(0.05)                     # waiters park on the gate

        loader.cancel()                               # cancel the LOADER
        with pytest.raises(asyncio.CancelledError):
            await loader
        release.set()                                 # shielded acquire finishes

        assert await w1 == 9001                       # waiters NOT hung
        assert await w2 == 9001
        for _ in range(200):                          # loader's orphan refcount released once
            if state.director.release.called:
                break
            await asyncio.sleep(0.01)
        state.director.release.assert_called_once_with("M")
        assert state.cold_load_gates == {}

    async def test_mapless_loader_failure_does_not_herd(self):
        """A non-OperationError from the loader's acquire must fail waiters
        fast (generic 503) WITHOUT them re-dispatching a cold-acquire herd."""
        from muse.cli_impl.gateway import _acquire_coalesced
        from muse.admin.operations import OperationError

        started = threading.Event()
        release = threading.Event()
        state = _make_state_with_director()

        def raising_acquire(model_id, *, manifest):
            started.set()
            release.wait(timeout=5)
            raise RuntimeError("pynvml hiccup")  # NOT an OperationError

        state.director.acquire.side_effect = raising_acquire
        tasks = [asyncio.create_task(_acquire_coalesced(state, "M", _manifest()))
                 for _ in range(5)]
        await asyncio.to_thread(started.wait, 5)
        await asyncio.sleep(0.1)
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # loader propagates the raw RuntimeError; waiters get a generic 503.
        assert sum(isinstance(r, OperationError) and r.status == 503 for r in results) == 4
        assert any(isinstance(r, RuntimeError) for r in results)
        assert state.director.acquire.call_count == 1   # NO retry herd
