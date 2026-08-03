"""Tests for the supervisor: catalog -> worker specs."""
import json
import os
import signal
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    clear_supervisor_state,
    get_supervisor_state,
    set_supervisor_state,
)


@pytest.fixture(autouse=True)
def _reset_supervisor_state():
    clear_supervisor_state()
    yield
    clear_supervisor_state()


@pytest.fixture(autouse=True)
def _mock_resource_registry(monkeypatch):
    """Keep supervisor unit tests off the host process registry."""
    monkeypatch.setattr(
        "muse.cli_impl.supervisor.register_process",
        lambda **kwargs: f"test-resource-{kwargs['pid']}",
    )
    monkeypatch.setattr(
        "muse.cli_impl.supervisor.unregister_process",
        lambda resource_id: bool(resource_id),
    )


@pytest.fixture(autouse=True)
def _isolate_pynvml_sentinels():
    """Prevent supervisor tests from polluting memory_probe module-level
    pynvml sentinels via run_supervisor's lazy-load validate_catalog_at_boot
    path. On exit, reset to fresh-untried state so subsequent test files
    (notably tests/core/test_memory_probe.py) see a clean module.
    """
    import muse.core.memory_probe as mod
    try:
        yield
    finally:
        mod.pynvml, mod._init_attempted, mod._init_ok = None, False, False


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
        kwargs = mock_popen.call_args.kwargs
        assert kwargs["env"]["MUSE_SUPERVISOR_PID"] == str(os.getpid())
        assert kwargs["start_new_session"] is (os.name == "posix")

    @patch("muse.cli_impl.supervisor.subprocess.Popen")
    def test_spawn_worker_assigns_a_fresh_readiness_nonce(
        self, mock_popen, monkeypatch,
    ):
        from muse.cli_impl.supervisor import spawn_worker

        monkeypatch.delenv("MUSE_WORKER_NONCE", raising=False)
        processes = [MagicMock(pid=12345), MagicMock(pid=12346)]
        mock_popen.side_effect = processes
        specs = [
            WorkerSpec(models=["model-a"], python_path="/a/python", port=9001),
            WorkerSpec(models=["model-b"], python_path="/b/python", port=9002),
        ]

        with patch(
            "muse.cli_impl.supervisor.secrets.token_urlsafe",
            side_effect=["nonce-a", "nonce-b"],
        ) as token_urlsafe, patch(
            "muse.cli_impl.supervisor._validated_worker_process_group",
            return_value=None,
        ):
            for spec in specs:
                spawn_worker(spec, device="cpu")

        assert token_urlsafe.call_args_list == [call(32), call(32)]
        child_envs = [popen_call.kwargs["env"] for popen_call in mock_popen.call_args_list]
        assert [env["MUSE_WORKER_NONCE"] for env in child_envs] == [
            "nonce-a", "nonce-b",
        ]
        assert [spec.worker_nonce for spec in specs] == ["nonce-a", "nonce-b"]
        assert "MUSE_WORKER_NONCE" not in os.environ

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

    def test_accepts_only_the_matching_nonce_for_the_exact_generation(self):
        from muse.cli_impl.supervisor import wait_for_ready

        process = MagicMock(pid=12345, returncode=None)
        process.poll.return_value = None
        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001,
            process=process, worker_nonce="current-nonce",
        )
        response = MagicMock(
            status_code=200,
            headers={"X-Muse-Worker-Nonce": "current-nonce"},
        )

        with patch("muse.cli_impl.supervisor.httpx.get", return_value=response):
            wait_for_ready(
                port=9001,
                timeout=5.0,
                poll_interval=0,
                expected_nonce=spec.worker_nonce,
                worker=spec,
            )

    def test_rejects_a_healthy_response_with_a_mismatched_nonce(self):
        from muse.cli_impl.supervisor import WorkerIdentityError, wait_for_ready

        process = MagicMock(pid=12345, returncode=None)
        process.poll.return_value = None
        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001,
            process=process, worker_nonce="current-nonce",
        )
        response = MagicMock(
            status_code=200,
            headers={"X-Muse-Worker-Nonce": "stale-nonce"},
        )

        with patch("muse.cli_impl.supervisor.httpx.get", return_value=response):
            with pytest.raises(WorkerIdentityError, match="nonce mismatch"):
                wait_for_ready(
                    port=9001,
                    timeout=5.0,
                    poll_interval=0,
                    expected_nonce=spec.worker_nonce,
                    worker=spec,
                )

    def test_rejects_a_generation_swap_during_the_health_request(self):
        from muse.cli_impl.supervisor import WorkerIdentityError, wait_for_ready

        original = MagicMock(pid=12345, returncode=None)
        original.poll.return_value = None
        replacement = MagicMock(pid=12346, returncode=None)
        replacement.poll.return_value = None
        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001,
            process=original, worker_nonce="current-nonce",
        )
        response = MagicMock(
            status_code=200,
            headers={"X-Muse-Worker-Nonce": "current-nonce"},
        )

        def swap_generation(*_args, **_kwargs):
            spec.process = replacement
            return response

        with patch(
            "muse.cli_impl.supervisor.httpx.get",
            side_effect=swap_generation,
        ):
            with pytest.raises(WorkerIdentityError, match="changed process generation"):
                wait_for_ready(
                    port=9001,
                    timeout=5.0,
                    poll_interval=0,
                    expected_nonce=spec.worker_nonce,
                    worker=spec,
                )

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

    def test_aborts_when_supervisor_shutdown_is_requested(self):
        from muse.cli_impl.supervisor import wait_for_ready
        import threading

        stop_event = threading.Event()
        stop_event.set()
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            with pytest.raises(RuntimeError, match="shutdown requested"):
                wait_for_ready(
                    port=9001,
                    timeout=120.0,
                    poll_interval=0.01,
                    stop_event=stop_event,
                )
        mock_get.assert_not_called()


class TestShutdownWorkers:
    @pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
    def test_signals_worker_process_group_on_posix(self):
        from muse.cli_impl.supervisor import _shutdown_workers

        proc = MagicMock(pid=12345)
        proc.poll.return_value = None
        proc.wait.return_value = 0
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        spec.process = proc
        spec.process_group_id = 12345

        with patch("muse.cli_impl.supervisor.os.getpgid", return_value=12345), \
             patch("muse.cli_impl.supervisor.os.getpgrp", return_value=54321), \
             patch(
                 "muse.cli_impl.supervisor._supports_pinned_worker_leader",
                 return_value=True,
             ), \
             patch(
                 "muse.cli_impl.supervisor._worker_process_state_locked",
                 side_effect=(True, False, False, False, False),
             ), \
             patch("muse.cli_impl.supervisor.os.killpg") as mock_killpg:
            _shutdown_workers([spec], grace=0.0)

        assert [item.args for item in mock_killpg.call_args_list] == [
            (12345, signal.SIGTERM),
            (12345, signal.SIGKILL),
        ]
        proc.wait.assert_called_once_with(timeout=0.0)
        proc.terminate.assert_not_called()

    def test_mock_pid_cannot_broadcast_to_process_group_one(self):
        from muse.cli_impl.supervisor import _signal_worker_process

        proc = MagicMock()
        proc.poll.return_value = None
        # A non-concrete mock PID cannot reach killpg.  Because the exact
        # generation is proven alive, the Popen-scoped fallback is safe.
        with patch("muse.cli_impl.supervisor.os.killpg") as mock_killpg:
            assert _signal_worker_process(proc, signal.SIGTERM) is True

        mock_killpg.assert_not_called()
        proc.terminate.assert_called_once_with()

    def test_ambiguous_process_state_is_never_signalled(self):
        from muse.cli_impl.supervisor import _signal_worker_process

        proc = MagicMock(pid=12345)
        proc.poll.side_effect = OSError("cannot inspect child")

        with patch("muse.cli_impl.supervisor.os.killpg") as mock_killpg:
            assert _signal_worker_process(proc, signal.SIGTERM) is False

        mock_killpg.assert_not_called()
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()

    def test_concurrent_reap_finishes_before_shutdown_without_stale_signal(self):
        """The shared identity lock closes the poll/reap -> PGID reuse race."""
        from muse.cli_impl.supervisor import _shutdown_workers
        import threading

        proc = MagicMock(pid=12345)
        proc.poll.return_value = 0
        proc.wait.return_value = 0
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        spec.process = proc
        poll_locked = threading.Event()
        release_poll = threading.Event()

        def reap_first():
            with spec.process_lock:
                poll_locked.set()
                assert release_poll.wait(timeout=1.0)
                assert proc.poll() == 0

        reaper = threading.Thread(target=reap_first)
        reaper.start()
        assert poll_locked.wait(timeout=1.0)

        shutdown = threading.Thread(target=_shutdown_workers, args=([spec],))
        with patch("muse.cli_impl.supervisor.os.killpg") as mock_killpg:
            shutdown.start()
            release_poll.set()
            reaper.join(timeout=1.0)
            shutdown.join(timeout=1.0)

        assert not reaper.is_alive()
        assert not shutdown.is_alive()
        mock_killpg.assert_not_called()
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()
        assert spec.process is None

    def test_kills_and_reaps_worker_that_ignores_sigterm(self):
        from muse.cli_impl.supervisor import _shutdown_workers

        proc = MagicMock(pid=12345)
        proc.wait.return_value = 0
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        spec.process = proc

        with patch(
            "muse.cli_impl.supervisor._worker_process_state_locked",
            side_effect=(True, True, True, False, False),
        ):
            _shutdown_workers([spec], grace=0.0)

        proc.terminate.assert_called_once_with()
        proc.kill.assert_called_once_with()
        proc.wait.assert_called_once_with(timeout=0.0)
        assert spec.process is None

    def test_bulk_shutdown_uses_shared_term_and_kill_deadlines(self):
        from muse.cli_impl.supervisor import _shutdown_workers

        clock = {"now": 0.0}

        processes = [MagicMock(pid=1001), MagicMock(pid=1002)]
        specs = [
            WorkerSpec(models=["a"], python_path="/a", port=9001),
            WorkerSpec(models=["b"], python_path="/b", port=9002),
        ]
        for spec, proc in zip(specs, processes):
            proc.poll.return_value = None
            spec.process = proc

        with patch(
            "muse.cli_impl.supervisor.time.monotonic",
            side_effect=lambda: clock["now"],
        ), patch(
            "muse.cli_impl.supervisor.time.sleep",
            side_effect=lambda seconds: clock.__setitem__(
                "now", clock["now"] + seconds,
            ),
        ):
            result = _shutdown_workers(specs, grace=0.05)

        # TERM (0.05s) and final-KILL (minimum 1.0s) are each one shared
        # deadline for the batch, not multiplied by two targets.
        assert 1.04 <= clock["now"] <= 1.06
        for proc in processes:
            proc.terminate.assert_called_once_with()
            proc.kill.assert_called_once_with()
        assert result.retained == tuple(specs)

    def test_bulk_shutdown_shares_one_log_reader_join_deadline(self):
        from muse.cli_impl.supervisor import _shutdown_workers

        clock = {"now": 0.0}
        join_timeouts: list[list[float]] = [[], []]
        processes = [MagicMock(pid=1001), MagicMock(pid=1002)]
        specs = [
            WorkerSpec(models=["a"], python_path="/a", port=9001),
            WorkerSpec(models=["b"], python_path="/b", port=9002),
        ]
        for index, (spec, proc) in enumerate(zip(specs, processes)):
            proc.wait.return_value = 0
            thread = MagicMock()
            thread.is_alive.return_value = True

            def join(*, timeout, worker=index):
                join_timeouts[worker].append(timeout)
                clock["now"] += timeout

            thread.join.side_effect = join
            spec.process = proc
            spec.log_thread = thread

        with patch(
            "muse.cli_impl.supervisor.time.monotonic",
            side_effect=lambda: clock["now"],
        ), patch(
            "muse.cli_impl.supervisor._signal_worker_process",
            return_value=True,
        ), patch(
            "muse.cli_impl.supervisor._BACKGROUND_JOIN_TIMEOUT_SECONDS",
            3.0,
        ):
            result = _shutdown_workers(specs)

        assert join_timeouts == [[3.0], [0.0]]
        assert result.retained == tuple(specs)
        for proc in processes:
            proc.stdout.close.assert_not_called()


class TestRunSupervisor:
    def test_supervisor_does_not_spawn_workers_at_boot(self, tmp_catalog):
        """v0.40.0 lazy-load: no eager worker spawn. Models load on
        first request via the LoadDirector.
        """
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
             patch("muse.cli_impl.supervisor.threading.Thread"), \
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers") as mock_shutdown:
            mock_run_uvicorn.side_effect = KeyboardInterrupt()

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # No eager spawn under lazy load.
            mock_spawn.assert_not_called()
            # Gateway still starts; shutdown still runs.
            mock_run_uvicorn.assert_called_once()
            mock_shutdown.assert_called_once()

    def test_supervisor_tears_down_workers_if_gateway_fails(self, tmp_catalog):
        """Crash path: shutdown_workers runs even when uvicorn raises.

        Under lazy load, state.workers is empty at this point unless the
        director already loaded something. Either way, the teardown call
        must happen so the exit path is consistent.
        """
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers") as mock_shutdown:
            mock_run_uvicorn.side_effect = RuntimeError("uvicorn died")

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
        spec.process = MagicMock(
            pid=12345, poll=MagicMock(return_value=1),
        )  # already exited
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.wait_for_ready") as mock_wait:
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        mock_spawn.assert_called_once_with(spec, device="cpu", log_hub=None)
        mock_wait.assert_called_once()
        # restart_count counts UNSUCCESSFUL restart attempts (see
        # test_many_successful_restarts_do_not_exhaust_budget below); a
        # restart that succeeds must not bump it.
        assert spec.restart_count == 0
        assert spec.failure_count == 0
        assert spec.status == "running"

    def test_successful_recovery_resets_prior_restart_failures(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001, device="cpu",
            restart_count=4,
        )
        spec.process = MagicMock(pid=12345, poll=MagicMock(return_value=1))

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready"):
            _attempt_restart(
                spec,
                stop_event=threading.Event(),
                max_restarts=10,
                backoff_base=0,
            )

        assert spec.restart_count == 0
        assert spec.status == "running"

    def test_many_successful_restarts_do_not_exhaust_budget(self):
        """restart_count must only count UNSUCCESSFUL restart attempts, per
        the documented '10 unsuccessful restart attempts' cap. A worker
        that cleanly recovers well beyond max_restarts times over its
        lifetime (zero failures) must never be marked dead."""
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready"):
            for _ in range(15):  # more than _MAX_RESTARTS=10, all successful
                spec.process = MagicMock(
                    pid=12345, poll=MagicMock(return_value=1),
                )
                _attempt_restart(
                    spec, stop_event=stop_event, max_restarts=10, backoff_base=0,
                )

        assert spec.status == "running"
        assert spec.status != "dead"
        assert spec.restart_count == 0

    def test_terminates_still_running_process_before_respawn(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        old_process = MagicMock(
            pid=12345,
            poll=MagicMock(side_effect=(None, 0)),
        )  # exits after TERM
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
        spec.process = MagicMock(pid=12345, poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn:
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        mock_spawn.assert_not_called()
        assert spec.status == "dead"

    def test_spawn_failure_keeps_status_unhealthy(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(pid=12345, poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready") as mock_wait:
            mock_wait.side_effect = TimeoutError("never ready")
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        assert spec.restart_count == 1  # counter still increments
        assert spec.status != "running"  # spawn tried, but didn't become ready

    def test_readiness_failure_terminates_partial_respawn(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        old_process = MagicMock(poll=MagicMock(return_value=1))
        new_process = MagicMock(poll=MagicMock(return_value=None))
        spec.process = old_process
        stop_event = threading.Event()

        stopped_processes = []

        def shutdown_exact(_specs, **_kwargs):
            stopped_processes.append(spec.process)
            spec.process = None

        def spawn_replacement(_spec, **_kwargs):
            assert spec.process is None
            spec.process = new_process
            spec.worker_nonce = "replacement-nonce"

        with patch(
            "muse.cli_impl.supervisor.spawn_worker",
            side_effect=spawn_replacement,
        ), \
             patch(
                 "muse.cli_impl.supervisor.wait_for_ready",
                 side_effect=TimeoutError("never ready"),
             ) as mock_wait, \
             patch(
                 "muse.cli_impl.supervisor._shutdown_workers",
                 side_effect=shutdown_exact,
             ):
            _attempt_restart(
                spec, stop_event=stop_event, max_restarts=10, backoff_base=0,
            )

        mock_wait.assert_called_once_with(
            port=9001, timeout=60.0, stop_event=stop_event,
            expected_nonce="replacement-nonce", worker=spec,
        )
        assert stopped_processes == [old_process, new_process]
        assert spec.process is None

    def test_oserror_from_spawn_does_not_propagate(self):
        """M10: a broken/missing venv python makes subprocess.Popen raise
        FileNotFoundError (an OSError, NOT a SubprocessError). It must be
        caught and marked unhealthy, not propagate and kill the monitor
        thread (which would disable supervision for ALL workers)."""
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/gone/python", port=9001,
                          device="cpu")
        spec.process = MagicMock(pid=12345, poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn:
            mock_spawn.side_effect = FileNotFoundError(
                2, "No such file or directory", "/gone/python",
            )
            # Must not raise.
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10,
                             backoff_base=0)

        assert spec.restart_count == 1
        assert spec.status == "unhealthy"

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
        spec.process = MagicMock(
            pid=12345, poll=MagicMock(return_value=None),
        )  # alive
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
        spec.process = MagicMock(pid=12345, poll=MagicMock(return_value=None))
        spec.failure_count = 2  # close to threshold
        spec.restart_count = 4
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
        assert spec.restart_count == 0
        assert spec.status == "running"
        mock_restart.assert_not_called()

    def test_monitor_stops_when_event_set(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(pid=12345, poll=MagicMock(return_value=None))
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
        alive_spec.process = MagicMock(
            pid=12345, poll=MagicMock(return_value=None),
        )
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


class TestResolveIdleSweepInterval:
    """A 0/negative/non-finite MUSE_IDLE_SWEEP_INTERVAL_SECONDS would make
    `_stop_event.wait(interval)` return immediately, busy-looping
    IdleSweeper.tick() against the director lock. The adjacent
    default_idle_timeout resolution already guards <= 0; the sweep
    interval must too, falling back to the documented 30.0s default.
    """
    def test_default_is_thirty(self):
        from muse.cli_impl.supervisor import _resolve_idle_sweep_interval
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MUSE_IDLE_SWEEP_INTERVAL_SECONDS", None)
            assert _resolve_idle_sweep_interval() == pytest.approx(30.0)

    def test_positive_value_passes_through(self):
        from muse.cli_impl.supervisor import _resolve_idle_sweep_interval
        with patch.dict(os.environ, {"MUSE_IDLE_SWEEP_INTERVAL_SECONDS": "5"}):
            assert _resolve_idle_sweep_interval() == pytest.approx(5.0)

    def test_zero_falls_back_to_default(self):
        from muse.cli_impl.supervisor import _resolve_idle_sweep_interval
        with patch.dict(os.environ, {"MUSE_IDLE_SWEEP_INTERVAL_SECONDS": "0"}):
            assert _resolve_idle_sweep_interval() == pytest.approx(30.0)

    def test_negative_falls_back_to_default(self):
        from muse.cli_impl.supervisor import _resolve_idle_sweep_interval
        with patch.dict(os.environ, {"MUSE_IDLE_SWEEP_INTERVAL_SECONDS": "-5"}):
            assert _resolve_idle_sweep_interval() == pytest.approx(30.0)

    def test_nan_falls_back_to_default(self):
        from muse.cli_impl.supervisor import _resolve_idle_sweep_interval
        with patch.dict(os.environ, {"MUSE_IDLE_SWEEP_INTERVAL_SECONDS": "nan"}):
            assert _resolve_idle_sweep_interval() == pytest.approx(30.0)

    def test_infinite_falls_back_to_default(self):
        from muse.cli_impl.supervisor import _resolve_idle_sweep_interval
        with patch.dict(os.environ, {"MUSE_IDLE_SWEEP_INTERVAL_SECONDS": "inf"}):
            assert _resolve_idle_sweep_interval() == pytest.approx(30.0)


class TestRunSupervisorMonitor:
    def test_run_supervisor_starts_monitor_thread(self, tmp_catalog):
        """Even at empty-catalog boot, the monitor thread is started so
        director-spawned workers are watched as soon as they appear.
        """
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
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"), \
             patch("muse.cli_impl.supervisor.threading.Thread") as mock_thread_cls:
            mock_run_uvicorn.side_effect = KeyboardInterrupt()
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # The monitor daemon thread is unconditionally started at
            # boot under lazy load (it watches state.workers, which can
            # grow at any time).
            assert mock_thread_cls.call_count >= 1
            for call in mock_thread_cls.call_args_list:
                assert call.kwargs.get("daemon") is True
                assert call.kwargs.get("target") is not None
            mock_thread.start.assert_called()

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
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"), \
             patch("muse.cli_impl.supervisor.threading.Event", side_effect=capture_event), \
             patch("muse.cli_impl.supervisor.threading.Thread"):
            mock_run_uvicorn.side_effect = KeyboardInterrupt()
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # The shutdown Event was set
        assert captured_events, "no threading.Event was created"
        assert any(e.is_set() for e in captured_events)


class TestSupervisorState:
    def test_default_state_has_empty_workers(self):
        s = SupervisorState()
        assert s.workers == []
        assert s.device == "auto"
        assert s.started_at >= 0.0

    def test_default_state_has_rlock(self):
        import threading
        s = SupervisorState()
        # An RLock can be acquired twice from the same thread without deadlock
        with s.lock:
            with s.lock:
                pass

    def test_get_supervisor_state_returns_sentinel_when_unset(self):
        clear_supervisor_state()
        s = get_supervisor_state()
        assert isinstance(s, SupervisorState)
        assert s.workers == []

    def test_set_and_get_singleton_round_trip(self):
        s = SupervisorState(device="cuda")
        set_supervisor_state(s)
        assert get_supervisor_state() is s

    def test_clear_supervisor_state_drops_singleton(self):
        s = SupervisorState(device="mps")
        set_supervisor_state(s)
        clear_supervisor_state()
        # Subsequent get yields a fresh sentinel, not the cleared one
        out = get_supervisor_state()
        assert out is not s


class TestRunSupervisorRegistersState:
    def test_run_supervisor_registers_state_during_run(self, tmp_catalog):
        """run_supervisor should set the singleton before uvicorn.run.

        Under lazy load, workers list is empty at boot; the director is
        non-None on the registered state.
        """
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        seen_state = {"value": None}

        def capture_state(*args, **kwargs):
            seen_state["value"] = get_supervisor_state()
            raise KeyboardInterrupt()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = capture_state
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # State was non-None during the run
        assert seen_state["value"] is not None
        assert seen_state["value"].device == "cpu"
        # Lazy load: zero workers at boot.
        assert seen_state["value"].workers == []
        # The director is wired up.
        assert seen_state["value"].director is not None
        # And it's been cleared on exit
        cleared = get_supervisor_state()
        assert cleared.workers == []
        assert cleared.director is None

    def test_run_supervisor_clears_state_on_exception(self, tmp_catalog):
        """Crash path: state must still get cleared in the finally block."""
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
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = RuntimeError("uvicorn boom")
            with pytest.raises(RuntimeError):
                run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        cleared = get_supervisor_state()
        assert cleared.workers == []


class TestRunSupervisorLazyBootOrdering:
    """Tests for the lazy-boot ordering. Replaced the v0.39.x
    eager-boot ordering tests when v0.40.0 lazy load made first-ready
    waits irrelevant.
    """

    def test_gateway_starts_immediately_without_worker_wait(self, tmp_catalog):
        """The gateway must boot immediately on lazy load: no worker spawn
        and no readiness wait before uvicorn.run. With one or N enabled
        models in the catalog, the supervisor should still proceed to
        uvicorn.run.
        """
        _seed_catalog({
            "fast": {
                "pulled_at": "...", "hf_repo": "f", "local_dir": "/f",
                "venv_path": "/venvs/fast",
                "python_path": "/venvs/fast/bin/python",
                "enabled": True,
            },
            "slow": {
                "pulled_at": "...", "hf_repo": "s", "local_dir": "/s",
                "venv_path": "/venvs/slow",
                "python_path": "/venvs/slow/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        events: list[str] = []

        def gateway_side(*a, **kw):
            events.append("gateway_started")
            raise KeyboardInterrupt()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor._monitor_workers"), \
             patch("muse.cli_impl.supervisor.threading.Thread"), \
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = gateway_side

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # No eager spawn under lazy load.
        mock_spawn.assert_not_called()
        assert "gateway_started" in events

    def test_lazy_boot_does_not_invoke_eager_helpers(self, tmp_catalog):
        """spawn_worker is silent at boot under lazy load."""
        _seed_catalog({
            "x": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                "venv_path": "/venvs/x",
                "python_path": "/venvs/x/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor._monitor_workers"), \
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = KeyboardInterrupt()
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        mock_spawn.assert_not_called()

    def test_monitor_thread_started_at_boot(self, tmp_catalog):
        """The monitor thread is unconditionally started at boot since it
        watches state.workers, which can grow at any time after startup.
        """
        _seed_catalog({
            "fast": {
                "pulled_at": "...", "hf_repo": "f", "local_dir": "/f",
                "venv_path": "/venvs/fast",
                "python_path": "/venvs/fast/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        seen_thread_targets: list = []

        def fake_thread_init(*args, **kw):
            target = kw.get("target")
            if target is not None:
                seen_thread_targets.append(target.__name__)
            t = MagicMock()
            t.start = MagicMock()
            return t

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.threading.Thread",
                   side_effect=fake_thread_init), \
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = KeyboardInterrupt()

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # _monitor_workers is wired in regardless of catalog size.
        assert "_monitor_workers" in seen_thread_targets


class TestGatewayStateRoutes:
    def test_routes_only_running_workers(self):
        """Gateway derives routes from state.workers, filters out non-running."""
        from muse.cli_impl.gateway import build_gateway

        running = WorkerSpec(models=["a"], python_path="/p", port=9001,
                             status="running")
        pending = WorkerSpec(models=["b"], python_path="/p", port=9002,
                             status="pending")
        state = SupervisorState(workers=[running, pending])

        app = build_gateway(state=state)
        routes = app.state.routes_now()
        assert "a" in routes
        assert "b" not in routes

    def test_routes_update_when_pending_promotes(self):
        """A pending spec that becomes running shows up in routes_now."""
        from muse.cli_impl.gateway import build_gateway

        slow = WorkerSpec(models=["slow"], python_path="/p", port=9001,
                          status="pending")
        state = SupervisorState(workers=[slow])

        app = build_gateway(state=state)
        assert "slow" not in app.state.routes_now()

        with state.lock:
            slow.status = "running"
        assert "slow" in app.state.routes_now()

    def test_static_routes_still_supported(self):
        """When state= isn't passed, build_gateway uses the legacy routes list."""
        from muse.cli_impl.gateway import build_gateway, WorkerRoute

        routes = [WorkerRoute(model_id="m", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes=routes)
        cur = app.state.routes_now()
        assert "m" in cur
        assert cur["m"].worker_url == "http://127.0.0.1:9001"
