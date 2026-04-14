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
