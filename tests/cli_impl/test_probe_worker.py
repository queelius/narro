"""Tests for `muse _probe_worker` via run_probe_worker().

run_probe_worker is the in-venv side of `muse models probe`. It loads
the model via catalog.load_backend, captures memory before/after, runs
representative inference unless --no-inference, and prints a JSON record
on stdout's last line.

These tests stub catalog.load_backend with fake backends so we don't
need real ML deps, and patch the modality-PROBE_DEFAULTS lookup to
inject our own callable.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from muse.cli_impl.probe_worker import run_probe_worker
from muse.core.catalog import CatalogEntry, _reset_known_models_cache


@pytest.fixture(autouse=True)
def _isolate_catalog_cache():
    _reset_known_models_cache()
    yield
    _reset_known_models_cache()


def _fake_entry(model_id="kokoro-82m", modality="audio/speech", device_cap=None):
    extra = {}
    if device_cap is not None:
        extra["device"] = device_cap
    return CatalogEntry(
        model_id=model_id,
        modality=modality,
        backend_path="muse.models.kokoro_82m:Model",
        hf_repo="hexgrad/Kokoro-82M",
        description="",
        pip_extras=(),
        system_packages=(),
        extra=extra,
    )


def test_probe_worker_load_only_path(capsys):
    """run_inference=False: weights captured, ran_inference is False."""
    fake_backend = MagicMock()
    fake_entry = _fake_entry(device_cap="cpu")
    from muse.core import catalog as catalog_mod
    with patch.object(catalog_mod, "known_models", return_value={"kokoro-82m": fake_entry}), \
         patch.object(catalog_mod, "load_backend", return_value=fake_backend):
        rc = run_probe_worker(
            model_id="kokoro-82m",
            device="cpu",
            run_inference=False,
        )
    assert rc == 0
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    record = json.loads(lines[-1])
    assert record["model_id"] == "kokoro-82m"
    assert record["device"] == "cpu"
    assert record["ran_inference"] is False
    assert record["weights_bytes"] >= 0
    assert "shape" not in record


def test_probe_worker_with_inference_records_shape(capsys):
    """run_inference=True: PROBE_DEFAULTS.call invoked; shape recorded."""
    fake_backend = MagicMock()
    fake_entry = _fake_entry(modality="embedding/text", device_cap="cpu")

    # Sentinel callable so we can verify it ran.
    invoked = {}
    def _probe_call(m):
        invoked["called"] = True
        invoked["arg"] = m

    fake_defaults = {"shape": "1 short string", "call": _probe_call}

    from muse.core import catalog as catalog_mod
    with patch.object(catalog_mod, "known_models", return_value={"my-emb": fake_entry}), \
         patch.object(catalog_mod, "load_backend", return_value=fake_backend), \
         patch("muse.cli_impl.probe_worker._read_probe_defaults", return_value=fake_defaults):
        rc = run_probe_worker(
            model_id="my-emb",
            device="cpu",
            run_inference=True,
        )
    assert rc == 0
    assert invoked.get("called") is True
    assert invoked["arg"] is fake_backend

    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    record = json.loads(lines[-1])
    assert record["ran_inference"] is True
    assert record["shape"] == "1 short string"
    # peak_bytes should be at least weights_bytes (the worker enforces max)
    assert record["peak_bytes"] >= record["weights_bytes"]


def test_probe_worker_load_failure_emits_error_record(capsys):
    """Load failure: JSON record carries `error` key, returncode 2."""
    fake_entry = _fake_entry(device_cap="cpu")

    from muse.core import catalog as catalog_mod
    with patch.object(catalog_mod, "known_models", return_value={"x": fake_entry}), \
         patch.object(catalog_mod, "load_backend", side_effect=RuntimeError("boom")):
        rc = run_probe_worker(
            model_id="x",
            device="cpu",
            run_inference=False,
        )
    assert rc == 2
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    record = json.loads(lines[-1])
    assert "error" in record
    assert "boom" in record["error"]
    assert record["ran_inference"] is False


def test_probe_worker_load_failure_never_emits_blank_exception(capsys):
    fake_entry = _fake_entry(device_cap="cpu")

    from muse.core import catalog as catalog_mod
    with patch.object(catalog_mod, "known_models", return_value={"x": fake_entry}), \
         patch.object(catalog_mod, "load_backend", side_effect=AssertionError()):
        rc = run_probe_worker(
            model_id="x",
            device="cpu",
            run_inference=False,
        )

    assert rc == 2
    captured = capsys.readouterr()
    assert "load failed: AssertionError" in captured.err
    record = json.loads(captured.out.splitlines()[-1])
    assert record["error"] == "load failed: AssertionError"


def test_probe_worker_inference_failure_records_error_but_keeps_weights(capsys):
    """Inference exception: load result preserved, inference_error captured."""
    fake_backend = MagicMock()
    fake_entry = _fake_entry(modality="embedding/text", device_cap="cpu")

    def _failing_call(m):
        raise RuntimeError("inference exploded")

    fake_defaults = {"shape": "1 short string", "call": _failing_call}

    from muse.core import catalog as catalog_mod
    with patch.object(catalog_mod, "known_models", return_value={"x": fake_entry}), \
         patch.object(catalog_mod, "load_backend", return_value=fake_backend), \
         patch("muse.cli_impl.probe_worker._read_probe_defaults", return_value=fake_defaults):
        rc = run_probe_worker(
            model_id="x",
            device="cpu",
            run_inference=True,
        )
    assert rc == 0
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    record = json.loads(lines[-1])
    assert record["ran_inference"] is False
    assert "inference_error" in record
    assert "inference exploded" in record["inference_error"]


def test_probe_worker_resolves_capability_device_over_request():
    """Capability device pin overrides the requested device."""
    from muse.cli_impl.probe_worker import _resolve_device
    entry = _fake_entry(device_cap="cpu")
    # Even when caller asks for cuda, capability cpu wins.
    assert _resolve_device("cuda", entry) == "cpu"


def test_probe_worker_resolves_auto_when_no_cap():
    """No capability device: requested wins."""
    from muse.cli_impl.probe_worker import _resolve_device
    entry = _fake_entry(device_cap=None)
    assert _resolve_device("mps", entry) == "mps"


def test_probe_worker_device_override_beats_capability_pin():
    """M9: an operator `set-device` pin (catalog device_override) is tier 1
    in load_backend's precedence, above the manifest capability pin. The
    probe must resolve the same device it will actually load on, else it
    measures the wrong memory pool and records a bogus peak."""
    from muse.cli_impl.probe_worker import _resolve_device
    entry = _fake_entry(device_cap="cpu")
    # Model is cpu-pinned in its manifest, but the operator forced cuda.
    assert _resolve_device("auto", entry, override="cuda") == "cuda"


def test_probe_worker_device_override_beats_request():
    """The override also beats the --device flag (tier 3)."""
    from muse.cli_impl.probe_worker import _resolve_device
    entry = _fake_entry(device_cap=None)
    assert _resolve_device("cuda", entry, override="cpu") == "cpu"


def test_probe_worker_hardcoded_defaults_per_modality():
    """Fallback defaults exist for every bundled modality."""
    from muse.cli_impl.probe_worker import _hardcoded_defaults_for
    for modality in (
        "audio/speech", "audio/transcription", "embedding/text",
        "image/generation", "image/animation", "text/classification",
        "chat/completion",
    ):
        d = _hardcoded_defaults_for(modality)
        assert "shape" in d
        assert callable(d["call"])


def test_probe_worker_hardcoded_defaults_unknown_modality_raises():
    from muse.cli_impl.probe_worker import _hardcoded_defaults_for
    with pytest.raises(NotImplementedError):
        _hardcoded_defaults_for("video/synthesis")


def test_probe_worker_reads_modality_probe_defaults_when_present():
    """_read_probe_defaults reads PROBE_DEFAULTS off the modality module."""
    from muse.cli_impl.probe_worker import _read_probe_defaults
    # audio_speech has PROBE_DEFAULTS declared
    d = _read_probe_defaults("audio/speech")
    assert "shape" in d
    assert callable(d["call"])
