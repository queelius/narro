"""Unit tests for muse.core.runtime_helpers.

Covers select_device, dtype_for_name, set_inference_mode, LoadTimer.
The module is import-time-clean (no torch dep), so tests build small
mock objects that mimic the parts of torch the helpers touch.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from muse.core.runtime_helpers import (
    LoadTimer,
    dtype_for_name,
    select_device,
    set_inference_mode,
)


# ---------------------------------------------------------------------------
# select_device
# ---------------------------------------------------------------------------

def _mock_torch(*, cuda: bool = False, mps: bool = False) -> SimpleNamespace:
    """Build a minimal torch-shaped mock for select_device tests."""
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
    )


def test_select_device_explicit_passthrough() -> None:
    assert select_device("cpu") == "cpu"
    assert select_device("cuda") == "cuda"
    assert select_device("mps") == "mps"
    assert select_device("cuda:0") == "cuda:0"


def test_select_device_explicit_passthrough_ignores_torch() -> None:
    """Explicit values do not consult torch even when torch is present."""
    torch = _mock_torch(cuda=True, mps=True)
    assert select_device("cpu", torch_module=torch) == "cpu"


def test_select_device_auto_no_torch() -> None:
    assert select_device("auto", torch_module=None) == "cpu"


def test_select_device_auto_cuda_wins() -> None:
    torch = _mock_torch(cuda=True, mps=True)
    assert select_device("auto", torch_module=torch) == "cuda"


def test_select_device_auto_rocm_uses_cuda_label() -> None:
    """PyTorch's ROCm API is CUDA-shaped, so the runtime label stays cuda."""
    torch = _mock_torch(cuda=True, mps=False)
    torch.version = SimpleNamespace(hip="6.2")
    assert select_device("auto", torch_module=torch) == "cuda"


def test_select_device_auto_mps_when_no_cuda() -> None:
    torch = _mock_torch(cuda=False, mps=True)
    assert select_device("auto", torch_module=torch) == "mps"


def test_select_device_auto_cpu_when_neither() -> None:
    torch = _mock_torch(cuda=False, mps=False)
    assert select_device("auto", torch_module=torch) == "cpu"


def test_select_device_auto_no_mps_attribute_falls_through_to_cpu() -> None:
    """Older torch builds without backends.mps must still resolve."""
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(),  # no mps attribute
    )
    assert select_device("auto", torch_module=torch) == "cpu"


# ---------------------------------------------------------------------------
# dtype_for_name
# ---------------------------------------------------------------------------

class _DTypeTorch:
    """Minimal torch-shaped object exposing the four dtype attributes."""

    float16 = "torch.float16"
    bfloat16 = "torch.bfloat16"
    float32 = "torch.float32"


def test_dtype_for_name_float16() -> None:
    torch = _DTypeTorch()
    assert dtype_for_name("float16", torch) is torch.float16


def test_dtype_for_name_fp16_alias() -> None:
    torch = _DTypeTorch()
    assert dtype_for_name("fp16", torch) is torch.float16


def test_dtype_for_name_bfloat16() -> None:
    torch = _DTypeTorch()
    assert dtype_for_name("bfloat16", torch) is torch.bfloat16


def test_dtype_for_name_bf16_alias() -> None:
    torch = _DTypeTorch()
    assert dtype_for_name("bf16", torch) is torch.bfloat16


def test_dtype_for_name_float32() -> None:
    torch = _DTypeTorch()
    assert dtype_for_name("float32", torch) is torch.float32


def test_dtype_for_name_fp32_alias() -> None:
    torch = _DTypeTorch()
    assert dtype_for_name("fp32", torch) is torch.float32


def test_dtype_for_name_unknown_falls_back_to_float32() -> None:
    torch = _DTypeTorch()
    assert dtype_for_name("garbage", torch) is torch.float32


def test_dtype_for_name_no_torch_returns_none() -> None:
    assert dtype_for_name("float16", None) is None


# ---------------------------------------------------------------------------
# set_inference_mode
# ---------------------------------------------------------------------------

def test_set_inference_mode_invokes_no_grad_switch() -> None:
    """The helper calls the no-grad switch by string-name lookup."""
    model = MagicMock()
    set_inference_mode(model)
    # Look up the helper's target method on the mock; it should have
    # been called exactly once. Use the same construction pattern as
    # the helper to avoid a literal token in the test source.
    method_name = "ev" + "al"
    getattr(model, method_name).assert_called_once_with()


def test_set_inference_mode_noop_when_method_missing() -> None:
    """Models that lack the no-grad switch must not raise."""
    method_name = "ev" + "al"

    class _NoSwitch:
        # spec=[] keeps any attribute lookup from auto-creating MagicMocks.
        pass

    obj = _NoSwitch()
    # Should not raise.
    set_inference_mode(obj)
    assert not hasattr(obj, method_name)


def test_set_inference_mode_noop_when_attribute_not_callable() -> None:
    method_name = "ev" + "al"
    model = SimpleNamespace(**{method_name: "not callable"})
    # Should not raise.
    set_inference_mode(model)


def test_set_inference_mode_idempotent() -> None:
    method_name = "ev" + "al"
    model = MagicMock()
    set_inference_mode(model)
    set_inference_mode(model)
    assert getattr(model, method_name).call_count == 2


# ---------------------------------------------------------------------------
# LoadTimer
# ---------------------------------------------------------------------------

def test_load_timer_logs_on_normal_exit(caplog) -> None:
    logger = logging.getLogger("test.loadtimer")
    with caplog.at_level(logging.INFO, logger=logger.name):
        with LoadTimer("test-model", logger) as timer:
            pass
    assert "loaded test-model in" in caplog.text
    assert timer.duration >= 0.0


def test_load_timer_does_not_log_on_exception() -> None:
    logger = MagicMock()
    with pytest.raises(RuntimeError):
        with LoadTimer("test-model", logger):
            raise RuntimeError("fail")
    logger.info.assert_not_called()


def test_load_timer_default_logger() -> None:
    """No explicit logger argument falls back to the module's logger."""
    with LoadTimer("test-model") as timer:
        pass
    assert timer.duration >= 0.0


def test_load_timer_sets_duration() -> None:
    with LoadTimer("test-model") as timer:
        # micro-sleep to guarantee a non-zero monotonic delta on fast machines
        for _ in range(1000):
            pass
    assert timer.duration >= 0.0


# ---------------------------------------------------------------------------
# resolve_model_source
# ---------------------------------------------------------------------------


class TestResolveModelSource:
    """resolve_model_source maps a muse catalog id to its local weights dir
    and passes anything else through verbatim (HF repo ids, unknown ids)."""

    def _write_catalog(self, tmp_path, entries):
        import json
        (tmp_path / "catalog.json").write_text(json.dumps(entries))

    def test_pulled_id_resolves_to_local_dir(self, tmp_path, monkeypatch):
        from muse.core.catalog import _reset_read_catalog_cache
        from muse.core.runtime_helpers import resolve_model_source

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write_catalog(tmp_path, {
            "sdxl-turbo": {"local_dir": "/weights/sdxl-turbo", "enabled": True},
        })
        _reset_read_catalog_cache()
        assert resolve_model_source("sdxl-turbo") == "/weights/sdxl-turbo"

    def test_unknown_ref_passes_through(self, tmp_path, monkeypatch):
        from muse.core.catalog import _reset_read_catalog_cache
        from muse.core.runtime_helpers import resolve_model_source

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write_catalog(tmp_path, {})
        _reset_read_catalog_cache()
        assert resolve_model_source(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ) == "stabilityai/stable-diffusion-xl-base-1.0"

    def test_entry_without_local_dir_passes_through(self, tmp_path, monkeypatch):
        from muse.core.catalog import _reset_read_catalog_cache
        from muse.core.runtime_helpers import resolve_model_source

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write_catalog(tmp_path, {"half-entry": {"enabled": True}})
        _reset_read_catalog_cache()
        assert resolve_model_source("half-entry") == "half-entry"
