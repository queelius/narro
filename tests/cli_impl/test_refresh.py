"""Tests for `muse models refresh` (#140)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.cli_impl.refresh import (
    MODALITY_EXTRAS,
    RefreshResult,
    _infer_extras,
    _muse_repo_root,
    _pip_target,
    _pip_target_args,
    _select_targets,
    refresh_one,
    run_refresh,
)


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    return tmp_path


def _seed_catalog(data):
    from muse.core.catalog import _catalog_path
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    p.chmod(0o600)


def _make_python_path(tmp_path: Path, name: str) -> str:
    """Create a fake python_path file so Path(p).exists() is True."""
    from muse.core.venv import _venv_site_packages

    venv_path = tmp_path / "venvs" / name
    venv_dir = venv_path / "bin"
    venv_dir.mkdir(parents=True, exist_ok=True)
    _venv_site_packages(venv_path).mkdir(parents=True, exist_ok=True)
    p = venv_dir / "python"
    p.write_text("#!/bin/sh\necho fake\n")
    p.chmod(0o755)
    return str(p)


class TestInferExtras:
    def test_known_modality_returns_mapped_extras(self):
        assert _infer_extras("audio/speech") == ["audio"]
        assert _infer_extras("image/generation") == ["images"]
        assert _infer_extras("embedding/text") == ["embeddings"]

    def test_unknown_modality_returns_empty(self):
        assert _infer_extras("totally/unknown") == []

    def test_modality_with_no_extras_returns_empty(self):
        assert _infer_extras("text/rerank") == []
        assert "audio/alignment" in MODALITY_EXTRAS
        assert _infer_extras("audio/alignment") == []


class TestPipTarget:
    def test_includes_server_unconditionally(self):
        target = _pip_target([])
        assert "[server]" in target

    def test_appends_modality_extras(self):
        target = _pip_target(["audio"])
        assert "[server,audio]" in target

    def test_multiple_extras_comma_separated(self):
        target = _pip_target(["images", "embeddings"])
        assert "[server,images,embeddings]" in target

    def test_path_points_at_repo_root(self):
        target = _pip_target([])
        # The path part precedes the bracket
        path = target.split("[", 1)[0]
        assert (Path(path) / "pyproject.toml").exists()


class TestPipTargetArgs:
    def test_editable_flag_when_source_tree(self):
        with patch(
            "muse.cli_impl.refresh._muse_repo_root",
            return_value=Path("/src/muse"),
        ):
            args = _pip_target_args(["audio"])
        assert args == ["-e", "/src/muse[server,audio]"]

    def test_pypi_dist_when_no_source_tree(self):
        """M2: on a PyPI install of museq there is no pyproject.toml in any
        parent of __file__, so `_muse_repo_root()` returns None and refresh
        must install the published `museq` distribution (no -e, no bogus
        cwd path that would editable-install whatever project sits in the
        current directory)."""
        with patch(
            "muse.cli_impl.refresh._muse_repo_root", return_value=None,
        ), patch(
            "muse.core.catalog.importlib_metadata.version", return_value="1.2.3",
        ):
            args = _pip_target_args(["images", "embeddings"])
        assert args == ["museq[server,images,embeddings]==1.2.3"]
        assert "-e" not in args

    def test_refresh_one_installs_museq_from_pypi_when_not_source_tree(
        self, tmp_catalog, tmp_path,
    ):
        """End-to-end: a wheel/PyPI install refresh must shell out to
        `pip install --upgrade museq[server,audio]` with no -e flag."""
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": py, "enabled": True,
            },
        })
        manifest = {"modality": "audio/speech", "pip_extras": ()}
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh._muse_repo_root", return_value=None), \
             patch(
                 "muse.core.catalog.importlib_metadata.version",
                 return_value="1.2.3",
             ), \
             patch("muse.cli_impl.refresh.run_owned_command") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = refresh_one("x")
        assert result.state == "ok"
        cmd = mock_run.call_args_list[0].args[0]
        assert cmd[:5] == [py, "-m", "pip", "install", "--upgrade"]
        assert cmd[5:] == ["museq[server,audio]==1.2.3"]
        assert "-e" not in cmd


class TestRefreshOne:
    def test_invokes_pip_install_with_muse_extras(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "kokoro-82m")
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "venv_path": str(tmp_path / "venvs" / "kokoro-82m"),
                "python_path": py,
                "enabled": True,
            },
        })
        manifest = {
            "model_id": "kokoro-82m",
            "modality": "audio/speech",
            "pip_extras": ("kokoro", "soundfile"),
        }
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.run_owned_command") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = refresh_one("kokoro-82m")
        assert result.state == "ok"
        # Two calls: museq[server,audio] then pip_extras
        assert mock_run.call_count == 2
        first_cmd = mock_run.call_args_list[0].args[0]
        assert first_cmd[0] == py
        assert first_cmd[1:5] == ["-m", "pip", "install", "--upgrade"]
        assert first_cmd[5] == "-e"
        target = first_cmd[6]
        assert "[server,audio]" in target

    def test_appends_pip_extras_in_second_call(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "venv_path": str(tmp_path / "venvs" / "x"),
                "python_path": py,
                "enabled": True,
            },
        })
        manifest = {
            "model_id": "x",
            "modality": "audio/speech",
            "pip_extras": ("kokoro", "misaki[en]"),
        }
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.run_owned_command") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            refresh_one("x")
        second_cmd = mock_run.call_args_list[1].args[0]
        assert second_cmd[0] == py
        assert second_cmd[1:5] == ["-m", "pip", "install", "--upgrade"]
        assert "kokoro" in second_cmd
        assert "misaki[en]" in second_cmd

    def test_no_extras_flag_skips_extras_step(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "venv_path": str(tmp_path / "venvs" / "x"),
                "python_path": py,
                "enabled": True,
            },
        })
        manifest = {
            "modality": "audio/speech",
            "pip_extras": ("kokoro",),
            "python_sources": ({"type": "git"},),
        }
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.run_owned_command") as mock_run, \
             patch("muse.cli_impl.refresh.install_python_sources") as mock_sources:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = refresh_one("x", no_extras=True)
        assert result.state == "ok"
        # Only ONE call (the museq[server,audio] one); extras call skipped
        assert mock_run.call_count == 1
        mock_sources.assert_not_called()

    def test_refreshes_reviewed_python_sources_in_catalog_venv(
        self, tmp_catalog, tmp_path,
    ):
        py = _make_python_path(tmp_path, "x")
        venv_path = tmp_path / "venvs" / "x"
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "venv_path": str(venv_path),
                "python_path": py,
                "enabled": True,
            },
        })
        sources = ({"type": "git", "name": "reviewed"},)
        manifest = {
            "modality": "3d/generation",
            "pip_extras": (),
            "python_sources": sources,
        }
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.run_owned_command") as mock_run, \
             patch("muse.cli_impl.refresh.install_python_sources") as mock_sources:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = refresh_one("x")

        assert result.state == "ok"
        mock_sources.assert_called_once_with(venv_path, list(sources))

    def test_python_source_failure_is_reported(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": py,
                "enabled": True,
            },
        })
        manifest = {
            "modality": "3d/generation",
            "pip_extras": (),
            "python_sources": ({"type": "git"},),
        }
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.run_owned_command") as mock_run, \
             patch(
                 "muse.cli_impl.refresh.install_python_sources",
                 side_effect=RuntimeError("checkout identity changed"),
             ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = refresh_one("x")

        assert result.state == "failed"
        assert "python_sources install failed" in result.message
        assert "checkout identity changed" in result.message

    def test_skips_missing_catalog_entry(self, tmp_catalog):
        _seed_catalog({})
        result = refresh_one("does-not-exist")
        assert result.state == "skipped"
        assert "not in catalog" in result.message

    def test_skips_missing_python_path(self, tmp_catalog):
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": "/nonexistent/python",
                "enabled": True,
            },
        })
        result = refresh_one("x")
        assert result.state == "failed"
        assert "python_path" in result.message

    def test_skips_missing_canonical_python_path(self, tmp_catalog):
        expected = tmp_catalog / "venvs" / "x" / "bin" / "python"
        _seed_catalog({
            "x": {
                "python_path": str(expected),
                "venv_path": str(tmp_catalog / "venvs" / "x"),
                "enabled": True,
            },
        })

        result = refresh_one("x")

        assert result.state == "skipped"
        assert "not found" in result.message

    def test_rejects_existing_python_outside_owned_model_venv(
        self, tmp_catalog, tmp_path,
    ):
        external = tmp_path / "unrelated" / "bin" / "python"
        external.parent.mkdir(parents=True)
        external.write_text("#!/bin/sh\n", encoding="utf-8")
        external.chmod(0o700)
        _seed_catalog({
            "x": {
                "python_path": str(external),
                "venv_path": str(external.parent.parent),
                "enabled": True,
            },
        })

        with patch("muse.cli_impl.refresh.run_owned_command") as runner:
            result = refresh_one("x")

        assert result.state == "failed"
        assert "unsafe catalog venv" in result.message
        runner.assert_not_called()

    def test_rejects_symlinked_owned_venvs_root(self, tmp_catalog, tmp_path):
        outside = tmp_path / "outside-venvs"
        outside.mkdir()
        (tmp_catalog / "venvs").symlink_to(outside, target_is_directory=True)
        python = outside / "x" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        _seed_catalog({
            "x": {
                "python_path": str(tmp_catalog / "venvs" / "x" / "bin" / "python"),
                "venv_path": str(tmp_catalog / "venvs" / "x"),
                "enabled": True,
            },
        })

        with patch("muse.cli_impl.refresh.run_owned_command") as runner:
            result = refresh_one("x")

        assert result.state == "failed"
        assert "unsafe catalog venv" in result.message
        runner.assert_not_called()

    def test_failed_pip_returns_failed_with_output(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": py,
                "enabled": True,
            },
        })
        with patch("muse.cli_impl.refresh.get_manifest", return_value={"modality": ""}), \
             patch("muse.cli_impl.refresh.run_owned_command") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="ERROR: Could not find muse",
            )
            result = refresh_one("x")
        assert result.state == "failed"
        assert "museq[server] install failed" in result.message
        assert "Could not find muse" in result.pip_output

    def test_failed_extras_install_returns_failed(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": py,
                "enabled": True,
            },
        })
        manifest = {
            "modality": "audio/speech",
            "pip_extras": ("kokoro",),
        }
        # First call ok, second fails
        results = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=1, stdout="", stderr="kokoro install failed"),
        ]
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.run_owned_command", side_effect=results):
            result = refresh_one("x")
        assert result.state == "failed"
        assert "pip_extras install failed" in result.message
        assert "kokoro install failed" in result.pip_output

    def test_pip_install_timeout_returns_failed(self, tmp_catalog, tmp_path):
        """A hung PyPI mirror must surface as a failed RefreshResult,
        not block the parent process indefinitely."""
        import subprocess
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": py,
                "enabled": True,
            },
        })
        with patch("muse.cli_impl.refresh.get_manifest",
                   return_value={"modality": "", "pip_extras": ()}), \
             patch(
                 "muse.cli_impl.refresh.run_owned_command",
                 side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=1800),
             ):
            result = refresh_one("x")
        assert result.state == "failed"
        assert "timed out" in result.message
        assert "1800" in result.message

    def test_pip_subprocess_called_with_timeout(self, tmp_catalog, tmp_path):
        """Regression guard: the owned runner MUST receive a timeout.

        The pre-fix code omitted timeout, so a hung mirror would never
        surface."""
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": py,
                "enabled": True,
            },
        })
        with patch("muse.cli_impl.refresh.get_manifest",
                   return_value={"modality": "", "pip_extras": ()}), \
             patch("muse.cli_impl.refresh.run_owned_command") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            refresh_one("x")
        assert mock_run.call_count >= 1
        kwargs = mock_run.call_args_list[0].kwargs
        assert "timeout" in kwargs, f"refresh owned runner lacks timeout kwarg: {kwargs}"
        assert kwargs["timeout"] >= 60

    def test_no_pip_extras_in_manifest_skips_second_step(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "python_path": py,
                "enabled": True,
            },
        })
        manifest = {"modality": "audio/speech", "pip_extras": ()}
        with patch("muse.cli_impl.refresh.get_manifest", return_value=manifest), \
             patch("muse.cli_impl.refresh.run_owned_command") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            refresh_one("x")
        # Only the museq[server,audio] call; no extras pass.
        assert mock_run.call_count == 1


class TestRefreshLocking:
    def test_live_worker_lease_returns_failed_without_running_pip(
        self, tmp_catalog, tmp_path,
    ):
        from muse.core.catalog import ModelInUseError

        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "python_path": py,
                "venv_path": str(tmp_path / "venvs" / "x"),
                "enabled": True,
            },
        })
        with patch(
            "muse.cli_impl.refresh._model_resource_lease",
            side_effect=ModelInUseError("model 'x' is in use"),
        ), patch("muse.cli_impl.refresh.run_owned_command") as runner:
            result = refresh_one("x")

        assert result.state == "failed"
        assert "is in use" in result.message
        runner.assert_not_called()

    def test_refresh_waits_behind_same_model_pull_lock(
        self, tmp_catalog, tmp_path,
    ):
        from contextlib import contextmanager
        import threading

        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "python_path": py,
                "venv_path": str(tmp_path / "venvs" / "x"),
                "enabled": True,
            },
        })
        shared = threading.Lock()
        refresh_started = threading.Event()
        installer_entered = threading.Event()
        refresh_done = threading.Event()
        results = []

        @contextmanager
        def same_model_lock(identity):
            assert identity == "x"
            with shared:
                yield

        def fake_run(*_args, **_kwargs):
            installer_entered.set()
            return MagicMock(returncode=0, stdout="", stderr="")

        def run_refresh_thread():
            refresh_started.set()
            results.append(refresh_one("x"))
            refresh_done.set()

        with patch("muse.cli_impl.refresh._model_pull_lock", new=same_model_lock), \
             patch(
                 "muse.cli_impl.refresh.get_manifest",
                 return_value={"modality": "", "pip_extras": ()},
             ), \
             patch("muse.cli_impl.refresh.run_owned_command", side_effect=fake_run):
            # Simulate pull() already holding the exact shared model identity.
            with same_model_lock("x"):
                thread = threading.Thread(target=run_refresh_thread)
                thread.start()
                assert refresh_started.wait(1)
                assert not installer_entered.wait(0.05)
                assert not refresh_done.is_set()
            assert installer_entered.wait(1)
            thread.join(1)

        assert not thread.is_alive()
        assert refresh_done.is_set()
        assert [result.state for result in results] == ["ok"]

    def test_two_same_model_refreshes_do_not_enter_installer_together(
        self, tmp_catalog, tmp_path,
    ):
        from contextlib import contextmanager
        import threading

        py = _make_python_path(tmp_path, "x")
        _seed_catalog({
            "x": {
                "python_path": py,
                "venv_path": str(tmp_path / "venvs" / "x"),
                "enabled": True,
            },
        })
        shared = threading.Lock()
        calls_guard = threading.Lock()
        first_entered = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        second_entered = threading.Event()
        call_count = 0
        results = []

        @contextmanager
        def same_model_lock(identity):
            assert identity == "x"
            with shared:
                yield

        def fake_run(*_args, **_kwargs):
            nonlocal call_count
            with calls_guard:
                call_count += 1
                current = call_count
            if current == 1:
                first_entered.set()
                assert release_first.wait(1)
            else:
                second_entered.set()
            return MagicMock(returncode=0, stdout="", stderr="")

        def run_one(started=None):
            if started is not None:
                started.set()
            results.append(refresh_one("x"))

        with patch("muse.cli_impl.refresh._model_pull_lock", new=same_model_lock), \
             patch(
                 "muse.cli_impl.refresh.get_manifest",
                 return_value={"modality": "", "pip_extras": ()},
             ), \
             patch("muse.cli_impl.refresh.run_owned_command", side_effect=fake_run):
            first = threading.Thread(target=run_one)
            second = threading.Thread(target=run_one, args=(second_started,))
            first.start()
            assert first_entered.wait(1)
            second.start()
            assert second_started.wait(1)
            try:
                assert not second_entered.wait(0.05)
            finally:
                release_first.set()
            first.join(1)
            second.join(1)

        assert not first.is_alive()
        assert not second.is_alive()
        assert second_entered.is_set()
        assert [result.state for result in results] == ["ok", "ok"]


class TestSelectTargets:
    def test_all_returns_alphabetical(self, tmp_catalog):
        _seed_catalog({
            "zebra": {"python_path": "/x", "enabled": True},
            "alpha": {"python_path": "/y", "enabled": True},
            "mango": {"python_path": "/z", "enabled": False},
        })
        targets = _select_targets(model_id=None, all_=True, enabled_only=False)
        assert targets == ["alpha", "mango", "zebra"]

    def test_enabled_only_filters_disabled(self, tmp_catalog):
        _seed_catalog({
            "yes1": {"python_path": "/x", "enabled": True},
            "yes2": {"python_path": "/y", "enabled": True},
            "no1": {"python_path": "/z", "enabled": False},
        })
        targets = _select_targets(model_id=None, all_=False, enabled_only=True)
        assert targets == ["yes1", "yes2"]

    def test_single_id_returns_singleton(self, tmp_catalog):
        _seed_catalog({"x": {}})
        targets = _select_targets(model_id="x", all_=False, enabled_only=False)
        assert targets == ["x"]

    def test_no_flags_returns_none(self, tmp_catalog):
        _seed_catalog({})
        targets = _select_targets(model_id=None, all_=False, enabled_only=False)
        assert targets is None


class TestRunRefresh:
    def test_no_targets_prints_usage_returns_2(self, tmp_catalog, capsys):
        _seed_catalog({})
        rc = run_refresh()
        assert rc == 2
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    def test_empty_catalog_with_all_returns_0(self, tmp_catalog, capsys):
        _seed_catalog({})
        rc = run_refresh(all_=True)
        assert rc == 0
        captured = capsys.readouterr()
        assert "no targets" in captured.out.lower()

    def test_all_iterates_alphabetically(self, tmp_catalog, tmp_path):
        py_a = _make_python_path(tmp_path, "alpha")
        py_z = _make_python_path(tmp_path, "zebra")
        _seed_catalog({
            "zebra": {"python_path": py_z, "enabled": True},
            "alpha": {"python_path": py_a, "enabled": True},
        })
        manifest = {"modality": "", "pip_extras": ()}
        called: list[str] = []

        def fake_refresh_one(mid, **kw):
            called.append(mid)
            return RefreshResult(mid, "ok")

        with patch("muse.cli_impl.refresh.refresh_one", side_effect=fake_refresh_one):
            rc = run_refresh(all_=True)
        assert rc == 0
        assert called == ["alpha", "zebra"]

    def test_enabled_only_filters(self, tmp_catalog, tmp_path):
        py = _make_python_path(tmp_path, "yes")
        py2 = _make_python_path(tmp_path, "no")
        _seed_catalog({
            "yes-id": {"python_path": py, "enabled": True},
            "no-id": {"python_path": py2, "enabled": False},
        })
        called: list[str] = []
        with patch(
            "muse.cli_impl.refresh.refresh_one",
            side_effect=lambda mid, **kw: (called.append(mid), RefreshResult(mid, "ok"))[1],
        ):
            rc = run_refresh(enabled_only=True)
        assert rc == 0
        assert called == ["yes-id"]

    def test_continues_past_failures(self, tmp_catalog, tmp_path):
        py_a = _make_python_path(tmp_path, "a")
        py_b = _make_python_path(tmp_path, "b")
        py_c = _make_python_path(tmp_path, "c")
        _seed_catalog({
            "a": {"python_path": py_a, "enabled": True},
            "b": {"python_path": py_b, "enabled": True},
            "c": {"python_path": py_c, "enabled": True},
        })

        def fake(mid, **kw):
            if mid == "b":
                return RefreshResult(mid, "failed", "boom", "boom output")
            return RefreshResult(mid, "ok")

        called: list[str] = []
        with patch("muse.cli_impl.refresh.refresh_one",
                   side_effect=lambda mid, **kw: (called.append(mid), fake(mid))[1]):
            rc = run_refresh(all_=True)
        # All three were attempted
        assert called == ["a", "b", "c"]
        # Exit code is 1 because one failed
        assert rc == 1

    def test_json_output_is_parseable(self, tmp_catalog, tmp_path, capsys):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({"x": {"python_path": py, "enabled": True}})
        with patch(
            "muse.cli_impl.refresh.refresh_one",
            return_value=RefreshResult("x", "ok", extras=["audio"]),
        ):
            run_refresh(model_id="x", as_json=True)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["model_id"] == "x"
        assert parsed[0]["state"] == "ok"
        assert parsed[0]["extras"] == ["audio"]

    def test_human_output_includes_per_target_and_summary(self, tmp_catalog, tmp_path, capsys):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({"x": {"python_path": py, "enabled": True}})
        with patch(
            "muse.cli_impl.refresh.refresh_one",
            return_value=RefreshResult("x", "ok"),
        ):
            run_refresh(model_id="x")
        captured = capsys.readouterr()
        assert "x" in captured.out
        assert "ok" in captured.out
        assert "1 ok" in captured.out

    def test_failed_result_includes_pip_tail_in_human_output(self, tmp_catalog, tmp_path, capsys):
        py = _make_python_path(tmp_path, "x")
        _seed_catalog({"x": {"python_path": py, "enabled": True}})
        bad = RefreshResult("x", "failed", "boom", "line1\nline2\nline3\nline4\nline5\nline6")
        with patch("muse.cli_impl.refresh.refresh_one", return_value=bad):
            run_refresh(model_id="x")
        captured = capsys.readouterr()
        assert "failed" in captured.out
        assert "line6" in captured.out
        # First line was clipped (only last 5 shown)
        assert "line1" not in captured.out


class TestModalityExtrasMap:
    def test_all_modality_keys_have_list_values(self):
        for k, v in MODALITY_EXTRAS.items():
            assert isinstance(v, list), f"{k} maps to non-list {type(v)}"

    def test_known_muse_modalities_are_mapped(self):
        """Sanity: every modality the muse server can serve has a row."""
        # Spot-check the public set; not a hard guard against new modalities.
        for mod in (
            "audio/speech",
            "audio/transcription",
            "audio/quality",
            "image/generation",
            "embedding/text",
            "chat/completion",
            "video/generation",
        ):
            assert mod in MODALITY_EXTRAS


def test_muse_repo_root_finds_pyproject():
    root = _muse_repo_root()
    assert (root / "pyproject.toml").exists()


class TestTransactionalRefresh:
    """Refresh promotes a clone only after every update step succeeds."""

    @staticmethod
    def _seed_environment(tmp_path: Path) -> tuple[Path, str, Path]:
        python = _make_python_path(tmp_path, "transactional")
        venv_path = Path(python).parent.parent
        (venv_path / "marker").write_text("prior")
        pth = venv_path / "lib" / "python" / "site-packages" / "reviewed.pth"
        pth.parent.mkdir(parents=True)
        pth.write_text("prior-checkout\n")
        _seed_catalog({
            "transactional": {
                "enabled": True,
                "venv_path": str(venv_path),
                "python_path": python,
            },
        })
        return venv_path, python, pth

    def test_pip_failure_restores_prior_marker_and_pth(
        self, tmp_catalog, tmp_path,
    ):
        venv_path, python, pth = self._seed_environment(tmp_path)

        def fail_pip(cmd, **_kwargs):
            assert cmd[0] == python
            (venv_path / "marker").write_text("pip-mutated")
            pth.write_text("pip-mutated-checkout\n")
            return MagicMock(returncode=1, stdout="", stderr="dependency failed")

        with patch(
            "muse.cli_impl.refresh.get_manifest",
            return_value={"modality": "", "pip_extras": ()},
        ), patch(
            "muse.cli_impl.refresh.run_owned_command", side_effect=fail_pip,
        ):
            result = refresh_one("transactional")

        assert result.state == "failed"
        assert (venv_path / "marker").read_text() == "prior"
        assert pth.read_text() == "prior-checkout\n"

    def test_staging_failure_is_reported_without_running_installer(
        self, tmp_catalog, tmp_path,
    ):
        self._seed_environment(tmp_path)

        with patch(
            "muse.cli_impl.refresh.get_manifest",
            return_value={"modality": "", "pip_extras": ()},
        ), patch(
            "muse.cli_impl.refresh.venv_transaction",
            side_effect=RuntimeError("insufficient disk space"),
        ), patch("muse.cli_impl.refresh.run_owned_command") as runner:
            result = refresh_one("transactional")

        assert result.state == "failed"
        assert "transactional venv refresh failed" in result.message
        assert "insufficient disk space" in result.message
        runner.assert_not_called()

    def test_source_failure_restores_prior_marker_and_pth(
        self, tmp_catalog, tmp_path,
    ):
        venv_path, python, pth = self._seed_environment(tmp_path)
        source = {"type": "git", "name": "reviewed"}

        def successful_pip(cmd, **_kwargs):
            assert cmd[0] == python
            (venv_path / "marker").write_text("pip-updated")
            return MagicMock(returncode=0, stdout="", stderr="")

        def fail_source(path, sources):
            assert path == venv_path
            assert sources == [source]
            pth.write_text("source-mutated-checkout\n")
            (venv_path / "marker").write_text("source-mutated")
            raise RuntimeError("source activation failed")

        with patch(
            "muse.cli_impl.refresh.get_manifest",
            return_value={
                "modality": "3d/generation",
                "pip_extras": (),
                "python_sources": (source,),
            },
        ), patch(
            "muse.cli_impl.refresh.run_owned_command", side_effect=successful_pip,
        ), patch(
            "muse.cli_impl.refresh.install_python_sources", side_effect=fail_source,
        ):
            result = refresh_one("transactional")

        assert result.state == "failed"
        assert (venv_path / "marker").read_text() == "prior"
        assert pth.read_text() == "prior-checkout\n"

    def test_success_commits_updated_marker_and_pth(
        self, tmp_catalog, tmp_path,
    ):
        venv_path, python, pth = self._seed_environment(tmp_path)
        source = {"type": "git", "name": "reviewed"}

        def successful_pip(cmd, **_kwargs):
            assert cmd[0] == python
            (venv_path / "marker").write_text("pip-updated")
            return MagicMock(returncode=0, stdout="", stderr="")

        def successful_source(path, sources):
            assert path == venv_path
            assert sources == [source]
            pth.write_text("committed-checkout\n")
            (venv_path / "marker").write_text("committed")

        with patch(
            "muse.cli_impl.refresh.get_manifest",
            return_value={
                "modality": "3d/generation",
                "pip_extras": (),
                "python_sources": (source,),
            },
        ), patch(
            "muse.cli_impl.refresh.run_owned_command", side_effect=successful_pip,
        ), patch(
            "muse.cli_impl.refresh.install_python_sources",
            side_effect=successful_source,
        ):
            result = refresh_one("transactional")

        assert result.state == "ok"
        assert (venv_path / "marker").read_text() == "committed"
        assert pth.read_text() == "committed-checkout\n"
