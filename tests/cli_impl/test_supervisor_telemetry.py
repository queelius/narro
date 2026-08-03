"""Tests for Task 11: supervisor telemetry lifecycle + per-worker log piping.

Covers the testable units factored out of run_supervisor/spawn_worker:
  - `_pump_worker_logs`: the daemon reader loop that pipes a worker's
    stdout lines into a LogHub (and re-emits them to the aggregate log).
  - `_init_telemetry`: the boot-time wiring that creates a TelemetryStore,
    initializes the recorder, builds a LogHub, and starts a Sampler plus
    a retention-prune daemon -- all attached to SupervisorState.
  - `_attempt_restart` / `_monitor_workers`: the auto-restart respawn path
    must keep forwarding a live `state.log_hub` to `spawn_worker`, same as
    every other spawn site (muse.admin.operations), so a respawned
    worker's stdout keeps flowing into the dashboard log tail instead of
    going silent after a crash-restart.

None of these tests drive uvicorn or `run_supervisor` itself; all call the
factored-out functions directly, per the brief.
"""
from __future__ import annotations

import io
import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    _attempt_restart,
    _init_telemetry,
    _monitor_workers,
    _pump_worker_logs,
    _shutdown_telemetry,
    _shutdown_workers,
)
from muse.core import config
from muse.observability.logs import LogHub
from muse.observability.recorder import get_recorder, reset_recorder
from muse.observability.store import TelemetryStore


class TestPumpWorkerLogs:
    def test_lines_land_in_hub(self):
        stream = io.BytesIO(b"hello\nworld\n")
        proc = types.SimpleNamespace(stdout=stream)
        hub = LogHub()

        _pump_worker_logs(proc, "m", hub)

        assert hub.snapshot("m") == ["hello\n", "world\n"]
        assert stream.closed

    def test_reader_exits_on_eof_without_raising(self):
        stream = io.BytesIO()
        proc = types.SimpleNamespace(stdout=stream)
        hub = LogHub()

        _pump_worker_logs(proc, "m", hub)  # must not raise

        assert hub.snapshot("m") == []
        assert stream.closed

    def test_exception_from_stdout_iteration_is_swallowed(self):
        class _BoomStream:
            closed = False

            def read(self, _size):
                raise RuntimeError("boom")

            def close(self):
                self.closed = True

        stream = _BoomStream()
        proc = types.SimpleNamespace(stdout=stream)
        hub = LogHub()

        _pump_worker_logs(proc, "m", hub)  # must not raise

        assert stream.closed

    def test_huge_no_newline_record_is_bounded_and_closed_by_reader(self):
        import muse.cli_impl.supervisor as supervisor

        reader_thread = threading.get_ident()

        class _HugeStream:
            def __init__(self):
                self.remaining = supervisor._WORKER_LOG_LINE_BYTES * 100
                self.read_sizes: list[int] = []
                self.close_thread: int | None = None

            def read(self, size):
                self.read_sizes.append(size)
                take = min(size, self.remaining)
                self.remaining -= take
                return b"x" * take

            def close(self):
                self.close_thread = threading.get_ident()

        stream = _HugeStream()
        proc = types.SimpleNamespace(stdout=stream)

        with patch.object(supervisor, "_publish_worker_log") as publish:
            _pump_worker_logs(proc, "m", MagicMock())

        publish.assert_called_once()
        raw = publish.call_args.args[2]
        assert len(raw) == supervisor._WORKER_LOG_LINE_BYTES
        assert raw.endswith(supervisor._WORKER_LOG_TRUNCATION)
        assert set(stream.read_sizes) == {supervisor._WORKER_LOG_READ_BYTES}
        assert stream.close_thread == reader_thread


class TestInitTelemetry:
    @pytest.fixture(autouse=True)
    def _cleanup_recorder(self):
        yield
        reset_recorder()

    def test_wires_store_hub_sampler_and_recorder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        config.reset_config()

        state = SupervisorState()
        state.director = types.SimpleNamespace(loaded={}, in_flight_loads={})

        try:
            _init_telemetry(state)

            assert isinstance(state.telemetry_store, TelemetryStore)
            assert isinstance(state.log_hub, LogHub)
            assert (tmp_path / "telemetry.db").exists()

            # The recorder is now the real (non-noop) recorder: recording
            # and flushing an event must not raise, and dropped stays 0
            # for a single event on a fresh queue.
            recorder = get_recorder()
            recorder.record("request", model_id="m", latency_ms=1.0)
            recorder.flush()
            assert recorder.dropped == 0

            # The sampler shares state.stop_event (mirrors IdleSweeper): once
            # the shared event is set and the sampler is stopped, its thread
            # must actually exit -- this is the shutdown path run_supervisor's
            # finally block now performs (sampler.stop() + store.close()).
            assert state.telemetry_sampler is not None
            assert state.telemetry_sampler._stop is state.stop_event
        finally:
            sampler = state.telemetry_sampler
            sampler_thread = sampler._thread if sampler is not None else None
            prune_thread = state.telemetry_prune_thread
            assert _shutdown_telemetry(state) is True
            assert sampler_thread is not None and not sampler_thread.is_alive()
            assert prune_thread is not None and not prune_thread.is_alive()
            config.reset_config()

    def test_prune_loop_uses_state_stop_event(self, tmp_path, monkeypatch):
        # Regression guard: _init_telemetry must not crash when
        # state.stop_event is already a real threading.Event (as run_supervisor
        # sets it before calling _init_telemetry).
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        config.reset_config()

        state = SupervisorState()
        state.director = types.SimpleNamespace(loaded={}, in_flight_loads={})

        try:
            _init_telemetry(state)
            assert not state.stop_event.is_set()
        finally:
            assert _shutdown_telemetry(state) is True
            config.reset_config()

    def test_tracks_and_joins_every_owned_telemetry_thread(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        config.reset_config()
        state = SupervisorState()
        state.director = types.SimpleNamespace(loaded={}, in_flight_loads={})

        _init_telemetry(state)
        sampler_thread = state.telemetry_sampler._thread
        prune_thread = state.telemetry_prune_thread
        store = state.telemetry_store

        assert sampler_thread is not None and sampler_thread.is_alive()
        assert prune_thread is not None and prune_thread.is_alive()
        assert _shutdown_telemetry(state) is True
        assert not sampler_thread.is_alive()
        assert not prune_thread.is_alive()
        assert state.telemetry_sampler is None
        assert state.telemetry_prune_thread is None
        assert state.telemetry_recorder is None
        assert state.telemetry_store is None
        assert store._closed is True
        config.reset_config()

    def test_partial_start_failure_rolls_back_recorder_threads_and_store(
        self, tmp_path, monkeypatch,
    ):
        import muse.cli_impl.supervisor as supervisor

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        config.reset_config()
        state = SupervisorState()
        state.director = types.SimpleNamespace(loaded={}, in_flight_loads={})
        created_stores = []
        real_store = supervisor.TelemetryStore
        real_start = threading.Thread.start

        def _capture_store(path):
            store = real_store(path)
            created_stores.append(store)
            return store

        def _fail_only_prune_start(thread):
            if thread.name == "muse-telemetry-prune":
                raise RuntimeError("thread start failed")
            return real_start(thread)

        monkeypatch.setattr(supervisor, "TelemetryStore", _capture_store)
        monkeypatch.setattr(threading.Thread, "start", _fail_only_prune_start)

        with pytest.raises(RuntimeError, match="thread start failed"):
            _init_telemetry(state)

        assert state.stop_event.is_set()
        assert state.telemetry_sampler is None
        assert state.telemetry_prune_thread is None
        assert state.telemetry_recorder is None
        assert state.telemetry_store is None
        assert created_stores and created_stores[0]._closed is True
        assert get_recorder().dropped == 0
        config.reset_config()

    def test_shutdown_retains_store_while_a_producer_is_still_live(self):
        state = SupervisorState()
        prune_thread = MagicMock()
        prune_thread.is_alive.return_value = True
        store = MagicMock()
        state.telemetry_prune_thread = prune_thread
        state.telemetry_store = store

        assert _shutdown_telemetry(state) is False

        prune_thread.join.assert_called_once_with(timeout=5.0)
        assert state.telemetry_prune_thread is prune_thread
        assert state.telemetry_store is store
        store.close.assert_not_called()

    def test_init_refuses_to_clear_shutdown_requested_before_start(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        config.reset_config()
        state = SupervisorState()
        state.director = types.SimpleNamespace(loaded={}, in_flight_loads={})
        state.stop_event.set()

        with pytest.raises(RuntimeError, match="shutdown requested"):
            _init_telemetry(state)

        assert state.stop_event.is_set()
        assert state.telemetry_store is None
        config.reset_config()

    def test_shutdown_racing_sampler_start_rolls_back_all_owners(
        self, tmp_path, monkeypatch,
    ):
        import muse.cli_impl.supervisor as supervisor

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        config.reset_config()
        state = SupervisorState()
        state.director = types.SimpleNamespace(loaded={}, in_flight_loads={})
        sampler = MagicMock()

        def _start_and_signal():
            state.stop_event.set()
            return True

        sampler.start.side_effect = _start_and_signal
        sampler.stop.return_value = True
        monkeypatch.setattr(supervisor, "Sampler", MagicMock(return_value=sampler))

        with pytest.raises(RuntimeError, match="shutdown requested"):
            _init_telemetry(state)

        sampler.stop.assert_called_once_with()
        assert state.stop_event.is_set()
        assert state.telemetry_sampler is None
        assert state.telemetry_prune_thread is None
        assert state.telemetry_recorder is None
        assert state.telemetry_store is None
        config.reset_config()

    def test_supervisor_unwinds_monitor_and_sweeper_when_telemetry_boot_fails(
        self, tmp_path, monkeypatch,
    ):
        import muse.cli_impl.supervisor as supervisor

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        monkeypatch.setenv("MUSE_CONFIG", str(tmp_path / "absent-config.yaml"))
        monkeypatch.setenv("MUSE_TELEMETRY_ENABLED", "true")
        config.reset_config()
        director = types.SimpleNamespace(
            gpu_budget_gb=None,
            cpu_budget_gb=8.0,
            gpu_headroom_gb=0.0,
            cpu_headroom_gb=0.0,
            capacity_listener=None,
        )
        monitor_thread = MagicMock()
        monitor_thread.is_alive.return_value = False
        sweeper_thread = MagicMock()
        sweeper_thread.is_alive.return_value = False
        sweeper = MagicMock()
        sweeper.start.return_value = sweeper_thread

        with patch.object(supervisor, "_build_load_director", return_value=director), \
             patch.object(supervisor, "validate_catalog_at_boot"), \
             patch.object(supervisor.threading, "Thread", return_value=monitor_thread) \
                 as thread_cls, \
             patch.object(supervisor, "IdleSweeper", return_value=sweeper), \
             patch.object(
                 supervisor,
                 "_init_telemetry",
                 side_effect=RuntimeError("telemetry failed"),
             ), \
             patch.object(supervisor, "run_uvicorn") as run_uvicorn, \
             patch.object(
                 supervisor,
                 "clear_supervisor_state",
                 wraps=supervisor.clear_supervisor_state,
             ) as clear_state:
            with pytest.raises(RuntimeError, match="telemetry failed"):
                supervisor.run_supervisor(
                    host="127.0.0.1", port=8000, device="cpu",
                )

        stop_event = thread_cls.call_args.kwargs["args"][1]
        assert stop_event.is_set()
        monitor_thread.join.assert_called_once_with(timeout=5.0)
        sweeper_thread.join.assert_called_once_with(timeout=5.0)
        clear_state.assert_called_once_with()
        run_uvicorn.assert_not_called()
        config.reset_config()

    @pytest.mark.parametrize("failure_point", ["validation", "monitor", "sweeper"])
    def test_every_startup_failure_uses_the_same_transactional_cleanup(
        self, tmp_path, monkeypatch, failure_point,
    ):
        import muse.cli_impl.supervisor as supervisor

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        monkeypatch.setenv("MUSE_CONFIG", str(tmp_path / "absent-config.yaml"))
        monkeypatch.setenv("MUSE_TELEMETRY_ENABLED", "false")
        config.reset_config()

        director = types.SimpleNamespace(
            gpu_budget_gb=None,
            cpu_budget_gb=8.0,
            gpu_headroom_gb=0.0,
            cpu_headroom_gb=0.0,
            capacity_listener=None,
        )
        job_store = MagicMock()
        monitor_thread = MagicMock(name="monitor_thread")
        monitor_thread.is_alive.return_value = False
        if failure_point == "monitor":
            monitor_thread.start.side_effect = RuntimeError("monitor failed")
        sweeper_thread = MagicMock(name="sweeper_thread")
        sweeper_thread.is_alive.return_value = False
        sweeper = MagicMock(name="sweeper")
        if failure_point == "sweeper":
            sweeper.start.side_effect = RuntimeError("sweeper failed")
        else:
            sweeper.start.return_value = sweeper_thread

        validation_effect = (
            RuntimeError("validation failed")
            if failure_point == "validation"
            else None
        )

        with patch("muse.admin.jobs.reset_default_store"), \
             patch("muse.admin.jobs.get_default_store", return_value=job_store), \
             patch.object(supervisor, "_build_load_director", return_value=director), \
             patch.object(
                 supervisor,
                 "validate_catalog_at_boot",
                 side_effect=validation_effect,
             ), \
             patch.object(
                 supervisor.threading,
                 "Thread",
                 return_value=monitor_thread,
             ) as thread_cls, \
             patch.object(supervisor, "IdleSweeper", return_value=sweeper) \
                 as sweeper_cls, \
             patch.object(supervisor, "register_process") as register, \
             patch.object(supervisor, "run_uvicorn") as run_uvicorn, \
             patch.object(supervisor, "_shutdown_workers") as shutdown_workers, \
             patch.object(
                 supervisor,
                 "clear_supervisor_state",
                 wraps=supervisor.clear_supervisor_state,
             ) as clear_state:
            with pytest.raises(RuntimeError, match=f"{failure_point} failed"):
                supervisor.run_supervisor(
                    host="127.0.0.1", port=8000, device="cpu",
                )

        job_store.shutdown.assert_called_once_with()
        shutdown_workers.assert_called_once_with([])
        clear_state.assert_called_once_with()
        assert supervisor.get_supervisor_state().director is None
        register.assert_not_called()
        run_uvicorn.assert_not_called()

        if failure_point == "validation":
            thread_cls.assert_not_called()
            sweeper_cls.assert_not_called()
        elif failure_point == "monitor":
            assert monitor_thread.join.called
            sweeper_cls.assert_not_called()
        else:
            assert monitor_thread.join.called
            sweeper.start.assert_called_once_with()
            # start() raised before it could return an owned thread handle.
            sweeper_thread.join.assert_not_called()

        config.reset_config()


class TestLogPumpLifecycle:
    def test_worker_shutdown_joins_the_exact_log_pump_after_mocked_exit(self):
        proc = MagicMock(pid=12345)
        proc.poll.return_value = 0
        proc.wait.return_value = 0
        log_thread = MagicMock()
        log_thread.is_alive.return_value = False
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        spec.process = proc
        spec.log_thread = log_thread

        with patch("muse.cli_impl.supervisor.unregister_process"):
            _shutdown_workers([spec])

        log_thread.join.assert_called_once()
        timeout = log_thread.join.call_args.kwargs["timeout"]
        assert 0.0 < timeout <= 5.0
        assert spec.log_thread is None

    def test_log_thread_start_failure_cleans_up_mocked_child(self):
        import muse.cli_impl.supervisor as supervisor

        proc = MagicMock(pid=12345)
        proc.poll.return_value = None
        proc.wait.return_value = 0
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        thread = MagicMock()
        thread.start.side_effect = RuntimeError("cannot start reader")
        thread.is_alive.return_value = False

        with patch.object(supervisor.subprocess, "Popen", return_value=proc), \
             patch.object(supervisor.threading, "Thread", return_value=thread), \
             patch.object(supervisor.os, "getpgid", return_value=12345), \
             patch.object(supervisor.os, "getpgrp", return_value=54321), \
             patch.object(
                 supervisor,
                 "_supports_pinned_worker_leader",
                 return_value=True,
             ), \
             patch.object(
                 supervisor,
                 "_worker_process_state_locked",
                 side_effect=(True, False, False, False, False),
             ), \
             patch.object(supervisor.os, "killpg") as killpg:
            with pytest.raises(RuntimeError, match="cannot start reader"):
                supervisor.spawn_worker(spec, device="cpu", log_hub=LogHub())

        assert [item.args for item in killpg.call_args_list] == [
            (12345, supervisor.signal.SIGTERM),
            (12345, supervisor.signal.SIGKILL),
        ]
        proc.wait.assert_called_once_with(timeout=0.0)
        assert spec.process is None
        assert spec.log_thread is None


class TestRestartForwardsLogHub:
    """Finding 1 (final whole-branch review, observability dashboard).

    Every OTHER spawn_worker call site forwards
    `log_hub=getattr(state, "log_hub", None)` (see muse.admin.operations'
    enable_model / _restart_worker_inplace). Before this fix, the
    auto-restart monitor's respawn path (`_attempt_restart` ->
    `spawn_worker(spec, device=spec.device)`) passed no `log_hub` at all,
    so a crashed-and-respawned worker's stdout stopped flowing into the
    LogHub and its dashboard log tail went silent permanently.

    `_monitor_workers` is started (from `run_supervisor`) before
    `_init_telemetry` populates `state.log_hub`, so it must accept the
    live `state` object (not a value snapshotted at thread-start) and
    read `state.log_hub` at the moment it decides to restart -- these
    tests assert that live threading end-to-end.
    """

    def test_attempt_restart_forwards_log_hub_to_spawn_worker(self):
        hub = LogHub()
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(
            pid=12345, poll=MagicMock(return_value=1),
        )  # already exited
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.wait_for_ready"):
            _attempt_restart(
                spec, stop_event=stop_event, max_restarts=10, backoff_base=0,
                log_hub=hub,
            )

        mock_spawn.assert_called_once_with(spec, device="cpu", log_hub=hub)

    def test_attempt_restart_defaults_log_hub_to_none(self):
        """Regression guard: callers that don't pass log_hub (e.g. direct
        unit tests elsewhere in the suite) keep today's telemetry-disabled
        behavior -- spawn_worker still gets an explicit log_hub=None."""
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(pid=12345, poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.wait_for_ready"):
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        mock_spawn.assert_called_once_with(spec, device="cpu", log_hub=None)

    def test_monitor_workers_forwards_state_log_hub_to_attempt_restart(self):
        """_monitor_workers, given a `state` carrying a LogHub, must thread
        `log_hub=state.log_hub` into `_attempt_restart` at restart time."""
        hub = LogHub()
        state = SupervisorState()
        state.log_hub = hub

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(
            pid=12345, poll=MagicMock(return_value=1),
        )  # already exited
        stop_event = threading.Event()

        def _restart_side_effect(*args, **kwargs):
            stop_event.set()  # stop the loop after the first restart attempt

        with patch(
            "muse.cli_impl.supervisor._attempt_restart",
            side_effect=_restart_side_effect,
        ) as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=1, max_restarts=10,
                state=state,
            )

        mock_restart.assert_called_once()
        _, kwargs = mock_restart.call_args
        assert kwargs["log_hub"] is hub

    def test_monitor_workers_without_state_forwards_none(self):
        """Callers that don't pass `state` (the many existing
        _monitor_workers tests elsewhere in the suite that call with just
        (specs, stop_event)) keep today's behavior: no log_hub forwarded."""
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(pid=12345, poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        def _restart_side_effect(*args, **kwargs):
            stop_event.set()

        with patch(
            "muse.cli_impl.supervisor._attempt_restart",
            side_effect=_restart_side_effect,
        ) as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=1, max_restarts=10,
            )

        mock_restart.assert_called_once()
        _, kwargs = mock_restart.call_args
        assert kwargs["log_hub"] is None

    def test_monitor_workers_reads_log_hub_live_not_at_thread_start(self):
        """The real bug: run_supervisor starts the monitor thread BEFORE
        `_init_telemetry` populates `state.log_hub`. If `_monitor_workers`
        captured `state.log_hub` once (e.g. at thread-creation time via
        `kwargs={"log_hub": state.log_hub}`), a later-populated hub would
        never reach a subsequent restart. Simulate that ordering: log_hub
        starts None, gets populated by a side effect mid-tick (standing in
        for _init_telemetry finishing shortly after the monitor thread
        starts), and the eventual restart must still see the populated hub
        because `state` (not a snapshotted value) is what's threaded
        through.
        """
        hub = LogHub()
        state = SupervisorState()
        state.log_hub = None  # not yet populated, like at monitor-thread-start

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(
            pid=12345, poll=MagicMock(return_value=None),
        )  # still alive
        stop_event = threading.Event()

        call_count = {"n": 0}

        def _health_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Simulate _init_telemetry populating state.log_hub between
                # the first (non-restarting) tick and the second.
                state.log_hub = hub
            return False  # always unhealthy

        def _restart_side_effect(*args, **kwargs):
            stop_event.set()

        with patch(
            "muse.cli_impl.supervisor.check_worker_health",
            side_effect=_health_side_effect,
        ), patch(
            "muse.cli_impl.supervisor._attempt_restart",
            side_effect=_restart_side_effect,
        ) as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=2, max_restarts=10,
                state=state,
            )

        mock_restart.assert_called_once()
        _, kwargs = mock_restart.call_args
        assert kwargs["log_hub"] is hub
