"""Tests for the supervisor: catalog -> worker specs."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call

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

    def test_skips_disabled_models(self, tmp_catalog):
        """Disabled models are filtered out of plan_workers results."""
        _seed_catalog({
            "enabled-model": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
            "disabled-model": {
                "pulled_at": "...", "hf_repo": "b", "local_dir": "/b",
                "venv_path": "/venvs/b",
                "python_path": "/venvs/b/bin/python",
                "enabled": False,
            },
        })
        specs = plan_workers()
        all_models = {m for s in specs for m in s.models}
        assert "enabled-model" in all_models
        assert "disabled-model" not in all_models

    def test_legacy_entries_without_enabled_field_treated_as_enabled(self, tmp_catalog):
        """Pre-flag entries (no `enabled` key) must still be planned."""
        _seed_catalog({
            "legacy-model": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                # no 'enabled' key
            },
        })
        specs = plan_workers()
        all_models = {m for s in specs for m in s.models}
        assert "legacy-model" in all_models


class TestSpawnWorker:
    @patch("muse.cli_impl.supervisor.subprocess.Popen")
    def test_spawn_worker_invokes_venv_python_with_worker_subcommand(self, mock_popen):
        mock_popen.return_value = MagicMock(pid=12345)
        spec = WorkerSpec(
            models=["soprano-80m"],
            python_path="/venvs/soprano-80m/bin/python",
            port=9001,
        )
        from muse.cli_impl.supervisor import spawn_worker
        spawn_worker(spec, device="cpu")
        mock_popen.assert_called_once()
        args = mock_popen.call_args.args[0]
        assert args[0] == "/venvs/soprano-80m/bin/python"
        assert args[1:4] == ["-m", "muse.cli", "_worker"]
        assert "--port" in args and "9001" in args
        assert "--model" in args and "soprano-80m" in args
        assert "--device" in args and "cpu" in args
        assert spec.process is mock_popen.return_value

    @patch("muse.cli_impl.supervisor.subprocess.Popen")
    def test_spawn_worker_passes_all_models_in_group(self, mock_popen):
        spec = WorkerSpec(
            models=["model-a", "model-b"],
            python_path="/venvs/shared/bin/python",
            port=9001,
        )
        from muse.cli_impl.supervisor import spawn_worker
        spawn_worker(spec, device="cuda")
        args = mock_popen.call_args.args[0]
        # Each model passed via separate --model
        model_values = [args[i+1] for i, v in enumerate(args) if v == "--model"]
        assert set(model_values) == {"model-a", "model-b"}


class TestWaitForReady:
    def test_returns_when_health_responds_200(self):
        from muse.cli_impl.supervisor import wait_for_ready

        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            # Should return cleanly
            wait_for_ready(port=9001, timeout=5.0, poll_interval=0.01)

    def test_raises_timeouterror_when_worker_never_responds(self):
        from muse.cli_impl.supervisor import wait_for_ready
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            import httpx
            mock_get.side_effect = httpx.ConnectError("nope", request=None)
            with pytest.raises(TimeoutError, match="did not become ready"):
                wait_for_ready(port=9001, timeout=0.1, poll_interval=0.01)

    def test_polls_multiple_times_before_success(self):
        from muse.cli_impl.supervisor import wait_for_ready
        import httpx
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            # First two calls fail, third succeeds
            mock_get.side_effect = [
                httpx.ConnectError("not yet", request=None),
                httpx.ConnectError("not yet", request=None),
                MagicMock(status_code=200),
            ]
            wait_for_ready(port=9001, timeout=5.0, poll_interval=0.001)
            assert mock_get.call_count == 3


class TestRunSupervisor:
    def test_supervisor_spawns_all_workers_and_waits_for_all_ready(self, tmp_catalog):
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
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.wait_for_ready") as mock_wait, \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers") as mock_shutdown:
            # Simulate graceful shutdown by raising KeyboardInterrupt from uvicorn.run
            mock_uvicorn.run.side_effect = KeyboardInterrupt()

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # Two workers planned, so spawn + wait called twice
            assert mock_spawn.call_count == 2
            assert mock_wait.call_count == 2
            mock_uvicorn.run.assert_called_once()
            mock_shutdown.assert_called_once()

    def test_supervisor_tears_down_workers_if_gateway_fails(self, tmp_catalog):
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers") as mock_shutdown:
            mock_uvicorn.run.side_effect = RuntimeError("uvicorn died")

            with pytest.raises(RuntimeError, match="uvicorn died"):
                run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            mock_shutdown.assert_called_once()


class TestWorkerSpecExtensions:
    def test_worker_spec_has_device_field_with_default(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.device == "auto"

    def test_worker_spec_accepts_explicit_device(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cuda")
        assert spec.device == "cuda"

    def test_worker_spec_default_status_is_pending(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.status == "pending"

    def test_worker_spec_default_restart_and_failure_counts(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.restart_count == 0
        assert spec.failure_count == 0

    def test_worker_spec_has_last_spawn_at_default(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.last_spawn_at == 0.0


class TestAttemptRestart:
    def test_respawns_after_process_death(self):
        """If process exited, terminate (no-op if dead) + respawn."""
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001, device="cpu",
        )
        spec.process = MagicMock(poll=MagicMock(return_value=1))  # already exited
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.wait_for_ready") as mock_wait:
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        mock_spawn.assert_called_once_with(spec, device="cpu")
        mock_wait.assert_called_once()
        assert spec.restart_count == 1
        assert spec.failure_count == 0
        assert spec.status == "running"

    def test_terminates_still_running_process_before_respawn(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        old_process = MagicMock(poll=MagicMock(return_value=None))  # still alive
        spec.process = old_process
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready"):
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        old_process.terminate.assert_called_once()

    def test_marks_dead_after_max_restarts(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001, device="cpu",
            restart_count=10,  # already at budget
        )
        spec.process = MagicMock(poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn:
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        mock_spawn.assert_not_called()
        assert spec.status == "dead"

    def test_spawn_failure_keeps_status_unhealthy(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready") as mock_wait:
            mock_wait.side_effect = TimeoutError("never ready")
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        assert spec.restart_count == 1  # counter still increments
        assert spec.status != "running"  # spawn tried, but didn't become ready

    def test_respects_stop_event_during_backoff(self):
        """If stop_event is set during backoff wait, skip the restart."""
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=1))
        stop_event = threading.Event()
        stop_event.set()  # shutdown already requested

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn:
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=1)

        # With stop_event set, we don't respawn
        mock_spawn.assert_not_called()


class TestMonitorLoop:
    def test_monitor_calls_restart_after_threshold_failures(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=None))  # alive
        stop_event = threading.Event()

        # First 3 checks fail, then we stop
        health_calls = {"count": 0}
        def health_side_effect(**kwargs):
            health_calls["count"] += 1
            if health_calls["count"] >= 4:
                stop_event.set()
            return False

        with patch("muse.cli_impl.supervisor.check_worker_health", side_effect=health_side_effect), \
             patch("muse.cli_impl.supervisor._attempt_restart") as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        # After 3 consecutive unhealthy polls, restart should be invoked at least once
        assert mock_restart.called

    def test_monitor_resets_failure_count_on_success(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=None))
        spec.failure_count = 2  # close to threshold
        stop_event = threading.Event()

        call_count = {"n": 0}
        def health_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                stop_event.set()
            return True  # healthy

        with patch("muse.cli_impl.supervisor.check_worker_health", side_effect=health_side_effect), \
             patch("muse.cli_impl.supervisor._attempt_restart") as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        assert spec.failure_count == 0
        assert spec.status == "running"
        mock_restart.assert_not_called()

    def test_monitor_stops_when_event_set(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=None))
        stop_event = threading.Event()
        stop_event.set()  # already stopped

        with patch("muse.cli_impl.supervisor.check_worker_health", return_value=True) as mock_health:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        # Loop should exit immediately without any health checks
        mock_health.assert_not_called()

    def test_monitor_skips_dead_workers(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        alive_spec = WorkerSpec(models=["a"], python_path="/p", port=9001, device="cpu")
        alive_spec.process = MagicMock(poll=MagicMock(return_value=None))
        dead_spec = WorkerSpec(models=["b"], python_path="/p", port=9002, device="cpu")
        dead_spec.status = "dead"

        stop_event = threading.Event()
        checked_ports = []

        def health_side_effect(**kwargs):
            checked_ports.append(kwargs["port"])
            if len(checked_ports) >= 1:
                stop_event.set()
            return True

        with patch("muse.cli_impl.supervisor.check_worker_health", side_effect=health_side_effect):
            _monitor_workers(
                [alive_spec, dead_spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        # Only the alive worker should have been polled
        assert 9001 in checked_ports
        assert 9002 not in checked_ports


class TestCheckWorkerHealth:
    def test_returns_true_on_200(self):
        from muse.cli_impl.supervisor import check_worker_health
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert check_worker_health(port=9001) is True

    def test_returns_false_on_non_200(self):
        from muse.cli_impl.supervisor import check_worker_health
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=500)
            assert check_worker_health(port=9001) is False

    def test_returns_false_on_connection_error(self):
        from muse.cli_impl.supervisor import check_worker_health
        import httpx
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("down", request=None)
            assert check_worker_health(port=9001) is False

    def test_returns_false_on_timeout(self):
        from muse.cli_impl.supervisor import check_worker_health
        import httpx
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("slow", request=None)
            assert check_worker_health(port=9001) is False


class TestRunSupervisorMonitor:
    def test_run_supervisor_starts_monitor_thread(self, tmp_catalog):
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"), \
             patch("muse.cli_impl.supervisor.threading.Thread") as mock_thread_cls:
            mock_uvicorn.run.side_effect = KeyboardInterrupt()
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # A daemon thread was created and started
            mock_thread_cls.assert_called_once()
            kwargs = mock_thread_cls.call_args.kwargs
            assert kwargs.get("daemon") is True
            assert kwargs.get("target") is not None
            mock_thread.start.assert_called_once()

    def test_run_supervisor_sets_stop_event_on_exit(self, tmp_catalog):
        """On shutdown path, the monitor must be told to stop."""
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor
        import threading

        captured_events = []
        real_event_cls = threading.Event
        def capture_event(*a, **kw):
            e = real_event_cls(*a, **kw)
            captured_events.append(e)
            return e

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"), \
             patch("muse.cli_impl.supervisor.threading.Event", side_effect=capture_event), \
             patch("muse.cli_impl.supervisor.threading.Thread"):
            mock_uvicorn.run.side_effect = KeyboardInterrupt()
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # The shutdown Event was set
        assert captured_events, "no threading.Event was created"
        assert any(e.is_set() for e in captured_events)
