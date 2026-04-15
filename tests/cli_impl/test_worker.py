"""Smoke tests for run_worker (single-worker mode)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.cli_impl.worker import run_worker


@patch("muse.cli_impl.worker.uvicorn")
@patch("muse.cli_impl.worker.load_backend")
@patch("muse.cli_impl.worker.is_pulled", return_value=True)
def test_worker_loads_requested_models_and_runs_uvicorn(mock_pulled, mock_load, mock_uvicorn):
    fake_backend = MagicMock(model_id="soprano-80m", sample_rate=32000)
    mock_load.return_value = fake_backend

    run_worker(host="127.0.0.1", port=9999, models=["soprano-80m"], device="cpu")

    # load_backend was called for the model
    mock_load.assert_called_once_with("soprano-80m", device="cpu")
    # uvicorn was told to serve on the requested port
    mock_uvicorn.run.assert_called_once()
    kwargs = mock_uvicorn.run.call_args.kwargs
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9999


@patch("muse.cli_impl.worker.uvicorn")
@patch("muse.cli_impl.worker.is_pulled", return_value=True)
@patch("muse.cli_impl.worker.load_backend")
def test_worker_skips_load_failures_without_crashing(mock_load, mock_pulled, mock_uvicorn):
    mock_load.side_effect = RuntimeError("diffusers not installed")
    # Should not raise; worker starts empty
    run_worker(host="127.0.0.1", port=9999, models=["sd-turbo"], device="cpu")
    mock_uvicorn.run.assert_called_once()


@patch("muse.cli_impl.worker.uvicorn")
@patch("muse.cli_impl.worker.is_pulled", return_value=False)
def test_worker_skips_unpulled_models_without_crashing(mock_pulled, mock_uvicorn):
    run_worker(host="127.0.0.1", port=9999, models=["soprano-80m"], device="cpu")
    mock_uvicorn.run.assert_called_once()


@patch("muse.cli_impl.worker.uvicorn")
def test_worker_ignores_unknown_model_ids(mock_uvicorn, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    run_worker(host="127.0.0.1", port=9999, models=["bogus-model-xyz"], device="cpu")
    mock_uvicorn.run.assert_called_once()
    assert "ignoring unknown models" in caplog.text


@patch("muse.cli_impl.worker.uvicorn")
def test_worker_mounts_all_bundled_modality_routers(mock_uvicorn):
    """Regression guard: all bundled modality routers mounted regardless of registry content.

    Empty-registry requests must get the OpenAI 404 envelope, not FastAPI's
    default {"detail": "Not Found"}. That requires the router to exist.

    Asserts presence rather than exact count so adding a new bundled
    modality (chat/completion in v0.10.0; future video, transcriptions)
    doesn't break this test.
    """
    run_worker(host="127.0.0.1", port=9999, models=[], device="cpu")

    mock_uvicorn.run.assert_called_once()
    app = mock_uvicorn.run.call_args.args[0]
    route_paths = {getattr(r, "path", "") for r in app.routes}
    paths_str = "\n".join(route_paths)
    assert "/v1/audio/speech" in paths_str
    assert "/v1/images/generations" in paths_str
    assert "/v1/embeddings" in paths_str
    assert "/v1/chat/completions" in paths_str


@patch("muse.cli_impl.worker.uvicorn")
@patch("muse.cli_impl.worker.discover_modalities")
def test_worker_mounts_routers_from_discovery(mock_discover, mock_uvicorn):
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
    app = mock_uvicorn.run.call_args.args[0]
    route_paths = {getattr(r, "path", "") for r in app.routes}
    assert "/v1/sentinel/ping" in route_paths


@patch("muse.cli_impl.worker.uvicorn")
@patch("muse.cli_impl.worker.discover_modalities")
def test_worker_includes_env_modalities_dir_when_set(
    mock_discover, mock_uvicorn, monkeypatch, tmp_path,
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


@patch("muse.cli_impl.worker.uvicorn")
@patch("muse.cli_impl.worker.discover_modalities")
def test_worker_modality_dirs_are_just_bundled_when_env_unset(
    mock_discover, mock_uvicorn, monkeypatch,
):
    """Without $MUSE_MODALITIES_DIR, only the bundled dir is scanned."""
    mock_discover.return_value = {}
    monkeypatch.delenv("MUSE_MODALITIES_DIR", raising=False)

    run_worker(host="127.0.0.1", port=9999, models=[], device="cpu")

    dirs_arg = mock_discover.call_args.args[0]
    assert len(dirs_arg) == 1
    assert dirs_arg[0].name == "modalities"
