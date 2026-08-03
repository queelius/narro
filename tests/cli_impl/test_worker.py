"""Smoke tests for run_worker (single-worker mode)."""
import asyncio
import os
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muse.cli_impl.worker import run_worker


@pytest.fixture(autouse=True)
def _isolate_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path / "catalog"))


def test_parent_watchdog_terminates_worker_when_supervisor_disappears():
    from muse.cli_impl.worker import _watch_parent

    with patch("muse.cli_impl.worker.os.getppid", return_value=999), \
         patch("muse.cli_impl.worker.os.kill") as mock_kill:
        _watch_parent(expected_parent_pid=123, poll_interval=0)

    mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)


@patch("muse.cli_impl.worker.run_uvicorn")
def test_worker_starts_parent_watchdog_from_supervisor_env(
    mock_run_uvicorn, monkeypatch,
):
    monkeypatch.setenv("MUSE_SUPERVISOR_PID", "12345")

    with patch("muse.cli_impl.worker._start_parent_watchdog") as mock_watchdog:
        run_worker(host="127.0.0.1", port=9999, models=[], device="cpu")

    mock_watchdog.assert_called_once_with(12345)


@patch("muse.cli_impl.worker.run_uvicorn")
@patch("muse.cli_impl.worker.discover_modalities", return_value={})
def test_worker_adds_spawn_nonce_to_response_headers(
    _mock_discover, mock_run_uvicorn, monkeypatch,
):
    from starlette.responses import Response

    monkeypatch.delenv("MUSE_SUPERVISOR_PID", raising=False)
    monkeypatch.setenv("MUSE_WORKER_NONCE", "spawn-nonce")

    run_worker(host="127.0.0.1", port=9999, models=[], device="cpu")

    app = mock_run_uvicorn.call_args.args[0]
    dispatch = next(
        middleware.kwargs["dispatch"]
        for middleware in app.user_middleware
        if "dispatch" in middleware.kwargs
    )
    call_next = AsyncMock(return_value=Response())
    response = asyncio.run(dispatch(MagicMock(), call_next))

    call_next.assert_awaited_once()
    assert response.headers["X-Muse-Worker-Nonce"] == "spawn-nonce"


def _collect_route_paths(app) -> set:
    """Collect all route paths from a FastAPI app.

    Starlette >= 1.3 uses _IncludedRouter wrapper objects (no `.path`
    attribute) instead of flattening include_router() calls to APIRoute
    instances at the app level.  We recurse into original_router.routes
    to reach the actual APIRoute objects.
    """
    paths: set = set()
    for r in app.routes:
        if hasattr(r, "path"):
            paths.add(r.path)
        # _IncludedRouter (Starlette >= 1.3): included routers are wrapped
        # rather than flattened; descend one level to find the real routes.
        elif hasattr(r, "original_router"):
            for sub in r.original_router.routes:
                if hasattr(sub, "path"):
                    paths.add(sub.path)
    return paths


@patch("muse.cli_impl.worker.run_uvicorn")
@patch("muse.cli_impl.worker.load_backend")
@patch("muse.cli_impl.worker.is_pulled", return_value=True)
def test_worker_loads_requested_models_and_runs_uvicorn(mock_pulled, mock_load, mock_run_uvicorn):
    fake_backend = MagicMock(model_id="soprano-80m", sample_rate=32000)
    mock_load.return_value = fake_backend

    run_worker(host="127.0.0.1", port=9999, models=["soprano-80m"], device="cpu")

    # load_backend was called for the model
    mock_load.assert_called_once_with("soprano-80m", device="cpu")
    # uvicorn was told to serve on the requested port
    mock_run_uvicorn.assert_called_once()
    kwargs = mock_run_uvicorn.call_args.kwargs
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9999


@patch("muse.cli_impl.worker.load_backend")
@patch("muse.cli_impl.worker.is_pulled", return_value=True)
def test_worker_holds_resource_lease_through_uvicorn(mock_pulled, mock_load):
    from contextlib import contextmanager

    active: set[str] = set()
    events: list[tuple[str, str]] = []

    @contextmanager
    def fake_lease(model_id, *, wait):
        assert wait is True
        active.add(model_id)
        events.append(("enter", model_id))
        try:
            yield
        finally:
            events.append(("exit", model_id))
            active.remove(model_id)

    def fake_uvicorn(*_args, **_kwargs):
        assert active == {"soprano-80m"}

    with patch("muse.cli_impl.worker._model_resource_lease", new=fake_lease), \
         patch("muse.cli_impl.worker.run_uvicorn", side_effect=fake_uvicorn):
        rc = run_worker(
            host="127.0.0.1",
            port=9999,
            models=["soprano-80m"],
            device="cpu",
        )

    assert rc == 0
    assert events == [("enter", "soprano-80m"), ("exit", "soprano-80m")]
    assert not active


@patch("muse.cli_impl.worker.run_uvicorn")
@patch("muse.cli_impl.worker.is_pulled", return_value=True)
@patch("muse.cli_impl.worker.load_backend")
def test_worker_exits_nonzero_when_assigned_model_fails_to_load(
    mock_load, mock_pulled, mock_run_uvicorn, caplog,
):
    """A worker that cannot load an assigned model must fail loud.

    Previously the worker would log the error and run uvicorn anyway
    with an empty registry, so the supervisor's /health kept reporting
    'ok' while requests 404'd. Fail-fast lets the supervisor's
    restart-then-mark-dead machinery kick in.
    """
    import logging
    caplog.set_level(logging.ERROR)
    mock_load.side_effect = RuntimeError("diffusers not installed")

    rc = run_worker(host="127.0.0.1", port=9999, models=["sd-turbo"], device="cpu")

    assert rc != 0, "worker must exit non-zero when an assigned model fails to load"
    mock_run_uvicorn.assert_not_called()
    assert "sd-turbo" in caplog.text
    assert "diffusers not installed" in caplog.text


@patch("muse.cli_impl.worker.run_uvicorn")
@patch("muse.cli_impl.worker.is_pulled", return_value=False)
def test_worker_exits_nonzero_when_assigned_model_not_pulled(
    mock_pulled, mock_run_uvicorn, caplog,
):
    """An assigned model that isn't pulled (yet) is a configuration bug."""
    import logging
    caplog.set_level(logging.ERROR)

    rc = run_worker(host="127.0.0.1", port=9999, models=["soprano-80m"], device="cpu")

    assert rc != 0
    mock_run_uvicorn.assert_not_called()
    assert "soprano-80m" in caplog.text


@patch("muse.cli_impl.worker.run_uvicorn")
def test_worker_exits_nonzero_when_model_id_unknown(mock_run_uvicorn, caplog):
    """An unknown model-id (not in known_models()) must fail loud.

    v0.11.x bundled scripts removed in v0.12.0 manifest in catalogs
    as ids with stale backend_paths. Silent skip masks the bug;
    the fail-fast path gives the supervisor a chance to surface it.
    """
    import logging
    caplog.set_level(logging.WARNING)

    rc = run_worker(host="127.0.0.1", port=9999, models=["bogus-model-xyz"], device="cpu")

    assert rc != 0
    mock_run_uvicorn.assert_not_called()
    assert "ignoring unknown models" in caplog.text


@patch("muse.cli_impl.worker.run_uvicorn")
@patch("muse.cli_impl.worker.is_pulled", return_value=True)
@patch("muse.cli_impl.worker.load_backend")
def test_worker_exits_nonzero_when_some_models_load_and_others_fail(
    mock_load, mock_pulled, mock_run_uvicorn,
):
    """Partial success is still a failure. If the worker was asked to
    host N models, it must host all N. Failing 1 of 2 exits non-zero."""
    # First call succeeds, second raises
    good = MagicMock(model_id="soprano-80m", sample_rate=32000)
    mock_load.side_effect = [good, RuntimeError("gguf load failed")]

    rc = run_worker(
        host="127.0.0.1", port=9999,
        models=["soprano-80m", "qwen3.5-4b-q4"], device="cpu",
    )

    assert rc != 0
    mock_run_uvicorn.assert_not_called()


@patch("muse.cli_impl.worker.run_uvicorn")
def test_worker_mounts_all_bundled_modality_routers(mock_run_uvicorn):
    """Regression guard: all bundled modality routers mounted regardless of registry content.

    Empty-registry requests must get the OpenAI 404 envelope, not FastAPI's
    default {"detail": "Not Found"}. That requires the router to exist.

    Asserts presence rather than exact count so adding a new bundled
    modality (chat/completion in v0.10.0; future video, transcriptions)
    doesn't break this test.
    """
    run_worker(host="127.0.0.1", port=9999, models=[], device="cpu")

    mock_run_uvicorn.assert_called_once()
    app = mock_run_uvicorn.call_args.args[0]
    route_paths = _collect_route_paths(app)
    assert "/v1/audio/speech" in route_paths
    assert "/v1/images/generations" in route_paths
    assert "/v1/embeddings" in route_paths
    assert "/v1/chat/completions" in route_paths


@patch("muse.cli_impl.worker.run_uvicorn")
@patch("muse.cli_impl.worker.discover_modalities")
def test_worker_mounts_routers_from_discovery(mock_discover, mock_run_uvicorn):
    """Worker delegates router selection to discover_modalities, not hardcoded imports."""
    from fastapi import APIRouter

    sentinel_router = APIRouter()

    @sentinel_router.get("/v1/sentinel/ping")
    def _ping():
        return {"ok": True}

    sentinel_build = MagicMock(return_value=sentinel_router)
    mock_discover.return_value = {"sentinel/type": sentinel_build}

    run_worker(host="127.0.0.1", port=9999, models=[], device="cpu")

    # Discovery was consulted
    mock_discover.assert_called_once()
    # The router from the discovered build function was mounted
    sentinel_build.assert_called_once()
    app = mock_run_uvicorn.call_args.args[0]
    route_paths = _collect_route_paths(app)
    assert "/v1/sentinel/ping" in route_paths


@patch("muse.cli_impl.worker.run_uvicorn")
@patch("muse.cli_impl.worker.discover_modalities")
def test_worker_includes_env_modalities_dir_when_set(
    mock_discover, mock_run_uvicorn, monkeypatch, tmp_path,
):
    """$MUSE_MODALITIES_DIR is appended to the modality scan dirs."""
    mock_discover.return_value = {}
    monkeypatch.setenv("MUSE_MODALITIES_DIR", str(tmp_path))

    run_worker(host="127.0.0.1", port=9999, models=[], device="cpu")

    # The positional arg to discover_modalities is the list of dirs
    dirs_arg = mock_discover.call_args.args[0]
    # Bundled comes first, env second
    assert len(dirs_arg) == 2
    assert dirs_arg[0].name == "modalities"
    assert dirs_arg[1] == tmp_path


@patch("muse.cli_impl.worker.run_uvicorn")
@patch("muse.cli_impl.worker.discover_modalities")
def test_worker_modality_dirs_are_just_bundled_when_env_unset(
    mock_discover, mock_run_uvicorn, monkeypatch,
):
    """Without $MUSE_MODALITIES_DIR, only the bundled dir is scanned."""
    mock_discover.return_value = {}
    monkeypatch.delenv("MUSE_MODALITIES_DIR", raising=False)

    run_worker(host="127.0.0.1", port=9999, models=[], device="cpu")

    dirs_arg = mock_discover.call_args.args[0]
    assert len(dirs_arg) == 1
    assert dirs_arg[0].name == "modalities"
