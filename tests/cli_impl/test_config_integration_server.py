"""Task 5: server/supervisor/paths/fetch/admin config-registry integration.

Covers the behavior-wiring half of the migration (not just "does
config.get replace os.environ.get", which is exercised implicitly by
every existing env-var test at each migrated call site):

  - LoadDirector budgets/headroom actually flow from config (the
    doc-drift fix: MUSE_GPU_BUDGET_GB / MUSE_CPU_BUDGET_GB /
    MUSE_GPU_HEADROOM_GB / MUSE_CPU_HEADROOM_GB were documented but
    never read).
  - The idle-timeout default flip (None -> 600.0).
  - The new server.device knob and its precedence (--device flag >
    MUSE_DEVICE > "auto" default).
"""
from __future__ import annotations

import pytest

from muse.core import config as cfg


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.delenv("MUSE_CONFIG", raising=False)
    cfg.reset_config()
    yield
    cfg.reset_config()


class TestLoadDirectorBudgetsAndHeadroom:
    """The v0.5x doc-drift fix: supervisor.py's sole LoadDirector
    construction site now reads the four budget/headroom knobs from
    config instead of leaving them at the LoadDirector's hardcoded
    (None, None, 1.0, 2.0) defaults."""

    def test_director_gets_headroom_from_config(self, monkeypatch):
        monkeypatch.setenv("MUSE_GPU_HEADROOM_GB", "3.0")
        cfg.reset_config()
        from muse.cli_impl import supervisor

        director = supervisor.build_load_director(
            enable_fn=lambda model_id: 9001,
            disable_fn=lambda model_id: None,
            memory_probe=object(),
        )
        assert director.gpu_headroom_gb == 3.0

    def test_director_cpu_headroom_from_config(self, monkeypatch):
        monkeypatch.setenv("MUSE_CPU_HEADROOM_GB", "5.5")
        cfg.reset_config()
        from muse.cli_impl import supervisor

        director = supervisor.build_load_director(
            enable_fn=lambda model_id: 9001,
            disable_fn=lambda model_id: None,
            memory_probe=object(),
        )
        assert director.cpu_headroom_gb == 5.5

    def test_director_budgets_and_default_headroom_when_unset(self, monkeypatch):
        monkeypatch.delenv("MUSE_GPU_BUDGET_GB", raising=False)
        monkeypatch.delenv("MUSE_CPU_BUDGET_GB", raising=False)
        monkeypatch.delenv("MUSE_GPU_HEADROOM_GB", raising=False)
        monkeypatch.delenv("MUSE_CPU_HEADROOM_GB", raising=False)
        cfg.reset_config()
        from muse.cli_impl import supervisor

        director = supervisor.build_load_director(
            enable_fn=lambda model_id: 9001,
            disable_fn=lambda model_id: None,
            memory_probe=object(),
        )
        # Defaults match today's hardcoded LoadDirector.__init__ values,
        # so a deployment that sets nothing sees identical behavior.
        assert director.gpu_budget_gb is None
        assert director.cpu_budget_gb is None
        assert director.gpu_headroom_gb == 1.0
        assert director.cpu_headroom_gb == 2.0

    def test_director_gpu_budget_from_config(self, monkeypatch):
        monkeypatch.setenv("MUSE_GPU_BUDGET_GB", "8.0")
        cfg.reset_config()
        from muse.cli_impl import supervisor

        director = supervisor.build_load_director(
            enable_fn=lambda model_id: 9001,
            disable_fn=lambda model_id: None,
            memory_probe=object(),
        )
        assert director.gpu_budget_gb == 8.0


class TestIdleTimeoutDefault:
    """server.idle_timeout_seconds now defaults to 600.0 (was None /
    "off"). An operator who wants the old "never idle-evict" behavior
    sets MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS=0 explicitly."""

    def test_default_idle_timeout_is_600(self, monkeypatch):
        monkeypatch.delenv("MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS", raising=False)
        cfg.reset_config()
        assert cfg.get("server.idle_timeout_seconds") == 600.0

    def test_idle_timeout_explicit_zero_is_preserved(self, monkeypatch):
        monkeypatch.setenv("MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS", "0")
        cfg.reset_config()
        assert cfg.get("server.idle_timeout_seconds") == 0.0

    def test_idle_timeout_env_override(self, monkeypatch):
        monkeypatch.setenv("MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS", "120")
        cfg.reset_config()
        assert cfg.get("server.idle_timeout_seconds") == 120.0


class TestServerDeviceResolution:
    """server.device (MUSE_DEVICE) is a new knob. `muse serve --device`
    flips from a hardcoded "auto" default to None ("unset -> consult
    config"); an explicit flag still wins over the env/config value."""

    def test_device_env_used_when_flag_unset(self, monkeypatch):
        monkeypatch.setenv("MUSE_DEVICE", "cuda")
        cfg.reset_config()
        from muse.cli import _resolve_serve_device

        assert _resolve_serve_device(None) == "cuda"

    def test_explicit_flag_beats_env(self, monkeypatch):
        monkeypatch.setenv("MUSE_DEVICE", "cuda")
        cfg.reset_config()
        from muse.cli import Device, _resolve_serve_device

        assert _resolve_serve_device(Device.cpu) == "cpu"

    def test_default_is_auto_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("MUSE_DEVICE", raising=False)
        cfg.reset_config()
        from muse.cli import _resolve_serve_device

        assert _resolve_serve_device(None) == "auto"
