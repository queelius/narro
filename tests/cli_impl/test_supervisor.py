"""Tests for the supervisor: catalog -> worker specs."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from muse.cli_impl.supervisor import (
    WorkerSpec,
    plan_workers,
)


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    return tmp_path


def _seed_catalog(data):
    """Write catalog.json directly."""
    from muse.core.catalog import _catalog_path
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


class TestPlanWorkers:
    def test_empty_catalog_yields_no_workers(self, tmp_catalog):
        _seed_catalog({})
        specs = plan_workers()
        assert specs == []

    def test_one_model_yields_one_worker(self, tmp_catalog):
        _seed_catalog({
            "soprano-80m": {
                "pulled_at": "2026-04-13T00:00:00Z",
                "hf_repo": "ekwek/Soprano-1.1-80M",
                "local_dir": "/fake/local",
                "venv_path": "/home/user/.muse/venvs/soprano-80m",
                "python_path": "/home/user/.muse/venvs/soprano-80m/bin/python",
            },
        })
        specs = plan_workers()
        assert len(specs) == 1
        spec = specs[0]
        assert spec.models == ["soprano-80m"]
        assert spec.python_path == "/home/user/.muse/venvs/soprano-80m/bin/python"
        assert isinstance(spec.port, int)
        assert 9001 <= spec.port <= 9999

    def test_models_in_same_venv_share_a_worker(self, tmp_catalog):
        """If two models share venv_path (rare in M1 but supported), one worker."""
        shared_venv = "/home/user/.muse/venvs/shared"
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": shared_venv,
                "python_path": f"{shared_venv}/bin/python",
            },
            "model-b": {
                "pulled_at": "...", "hf_repo": "b", "local_dir": "/b",
                "venv_path": shared_venv,
                "python_path": f"{shared_venv}/bin/python",
            },
        })
        specs = plan_workers()
        assert len(specs) == 1
        assert set(specs[0].models) == {"model-a", "model-b"}

    def test_different_venvs_yield_different_workers(self, tmp_catalog):
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
            },
            "model-b": {
                "pulled_at": "...", "hf_repo": "b", "local_dir": "/b",
                "venv_path": "/venvs/b",
                "python_path": "/venvs/b/bin/python",
            },
        })
        specs = plan_workers()
        assert len(specs) == 2
        assert specs[0].port != specs[1].port

    def test_skips_pre_worker_entries_without_python_path(self, tmp_catalog, caplog):
        """Old catalog entries (no python_path field) are skipped with warning."""
        _seed_catalog({
            "legacy-model": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                # No venv_path / python_path - pre-worker entry
            },
            "new-model": {
                "pulled_at": "...", "hf_repo": "y", "local_dir": "/y",
                "venv_path": "/venvs/y",
                "python_path": "/venvs/y/bin/python",
            },
        })
        import logging
        caplog.set_level(logging.WARNING)
        specs = plan_workers()
        all_models = {m for s in specs for m in s.models}
        assert "legacy-model" not in all_models
        assert "new-model" in all_models
        # Warning should mention the legacy model id or re-pulling
        assert "legacy-model" in caplog.text or "re-pull" in caplog.text.lower() or "re-run" in caplog.text.lower()
