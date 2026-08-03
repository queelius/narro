"""Tests for admin operations: enable / disable / remove / probe / pull.

All tests mock subprocess.Popen + spawn_worker + wait_for_ready so no
actual workers are spawned. SupervisorState instances are local to each
test (never the singleton) so tests don't bleed into each other.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.admin.jobs import JobStore
from muse.admin.operations import (
    OperationError,
    disable_model,
    enable_model,
    find_worker_for_model,
    launch_async,
    probe_model,
    pull_model,
    remove_model,
)
from muse.cli_impl.supervisor import SupervisorState, WorkerSpec


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    return tmp_path


def _seed_catalog(data: dict) -> None:
    from muse.core.catalog import _catalog_path, _reset_known_models_cache
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    _reset_known_models_cache()


@pytest.fixture
def state():
    return SupervisorState(workers=[], device="cpu")


@pytest.fixture
def store():
    return JobStore()


class TestFindWorkerForModel:
    def test_finds_model_in_worker(self, state):
        spec = WorkerSpec(
            models=["soprano-80m"], python_path="/p", port=9001,
        )
        state.workers.append(spec)
        assert find_worker_for_model(state, "soprano-80m") is spec

    def test_returns_none_when_unhosted(self, state):
        assert find_worker_for_model(state, "unknown") is None


class TestEnableModel:
    def test_unknown_model_marks_failed(self, tmp_catalog, state, store):
        _seed_catalog({})
        job = store.create("enable", "ghost")
        enable_model("ghost", state=state, store=store, job=job)
        assert job.state == "failed"
        assert "unknown model" in job.error

    def test_already_loaded_marks_done_no_spawn(self, tmp_catalog, state, store, monkeypatch):
        # Seed with a bundled-style model so known_models() picks it up.
        from muse.core import catalog as catalog_mod
        # Inject a fake known_models entry.
        monkeypatch.setattr(catalog_mod, "_known_models_cache", None)
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...",
                "hf_repo": "hexgrad/Kokoro-82M",
                "local_dir": "/tmp/kokoro",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        # Pre-load it into a worker.
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        state.workers.append(spec)
        job = store.create("enable", "kokoro-82m")
        with patch("muse.admin.operations.spawn_worker") as mock_spawn:
            enable_model("kokoro-82m", state=state, store=store, job=job)
        assert job.state == "done"
        assert job.result["spawned_new"] is False
        assert job.result["worker_port"] == 9001
        mock_spawn.assert_not_called()

    def test_spawns_new_worker_when_no_venv_match(self, tmp_catalog, state, store):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...",
                "hf_repo": "hexgrad/Kokoro-82M",
                "local_dir": "/tmp/kokoro",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": False,
            },
        })
        job = store.create("enable", "kokoro-82m")
        with patch("muse.admin.operations.spawn_worker") as mock_spawn, \
             patch("muse.admin.operations.wait_for_ready") as mock_wait, \
             patch("muse.admin.operations.find_free_port", return_value=9123):
            enable_model("kokoro-82m", state=state, store=store, job=job)
        assert job.state == "done"
        assert job.result["spawned_new"] is True
        assert job.result["worker_port"] == 9123
        mock_spawn.assert_called_once()
        mock_wait.assert_called_once_with(
            port=9123, timeout=120.0, stop_event=state.stop_event,
        )
        assert len(state.workers) == 1
        assert state.workers[0].port == 9123

    def test_joins_existing_venv_group(self, tmp_catalog, state, store):
        # Two bundled models share a venv path; one is already running,
        # the second enable should restart-in-place.
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": True,
            },
            "soprano-80m": {
                "pulled_at": "...", "hf_repo": "s", "local_dir": "/s",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": False,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"],
            python_path="/venv/shared/bin/python",
            port=9001,
        )
        state.workers.append(spec)
        job = store.create("enable", "soprano-80m")
        with patch("muse.admin.operations._restart_worker_inplace") as mock_restart:
            enable_model("soprano-80m", state=state, store=store, job=job)
        assert job.state == "done", f"expected done, got {job.state} (error={job.error})"
        assert job.result["spawned_new"] is False
        assert "soprano-80m" in spec.models
        assert "kokoro-82m" in spec.models
        mock_restart.assert_called_once_with(
            spec,
            device="cpu",
            log_hub=None,
            stop_event=state.stop_event,
        )

    def test_unpulled_model_marks_failed(self, tmp_catalog, state, store):
        # Bundled known but not yet in catalog.json.
        _seed_catalog({})
        job = store.create("enable", "kokoro-82m")
        # known_models scans bundled scripts; kokoro-82m is bundled.
        enable_model("kokoro-82m", state=state, store=store, job=job)
        assert job.state == "failed"
        assert "not pulled" in job.error

    def test_enable_respawns_dead_worker(self, tmp_catalog, state, store):
        """H1: a worker that exhausted its restart budget lingers in
        state.workers with status='dead' and job_id=None. enable_model must
        respawn it, not report it as already-loaded."""
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        dead = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        dead.status = "dead"
        dead.job_id = None
        state.workers.append(dead)
        job = store.create("enable", "kokoro-82m")
        with patch("muse.admin.operations.spawn_worker") as mock_spawn, \
             patch("muse.admin.operations.wait_for_ready"), \
             patch("muse.admin.operations.find_free_port", return_value=9200):
            enable_model("kokoro-82m", state=state, store=store, job=job)
        assert job.state == "done", f"got {job.state} err={job.error}"
        # Must actually respawn, not silently claim the dead spec.
        mock_spawn.assert_called_once()
        assert job.result["spawned_new"] is True
        assert job.result["worker_port"] == 9200

    def test_enable_terminates_dropped_unhealthy_worker_process(
        self, tmp_catalog, state, store,
    ):
        """H1 follow-up: an 'unhealthy' spec (spawn ok, wait_for_ready timed
        out) still owns a LIVE subprocess holding VRAM. Dropping it must
        terminate that process, else it orphans (untracked by monitor +
        shutdown) and leaks memory."""
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        stale = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        stale.status = "unhealthy"
        stale.job_id = None
        stale.process = MagicMock()  # a still-live subprocess
        state.workers.append(stale)
        job = store.create("enable", "kokoro-82m")
        with patch("muse.admin.operations.spawn_worker"), \
             patch("muse.admin.operations.wait_for_ready"), \
             patch("muse.admin.operations.find_free_port", return_value=9200), \
             patch("muse.admin.operations._shutdown_workers") as mock_shutdown:
            enable_model("kokoro-82m", state=state, store=store, job=job)
        mock_shutdown.assert_called_once()
        assert stale in mock_shutdown.call_args.args[0]


class TestLoadModelIntoWorkerDeadSpec:
    def test_refuses_cold_load_after_shutdown_starts(self, state):
        from muse.admin.operations import load_model_into_worker

        state.stop_event.set()
        with patch("muse.admin.operations.spawn_worker") as mock_spawn:
            with pytest.raises(OperationError) as exc:
                load_model_into_worker("kokoro-82m", state=state)

        assert exc.value.code == "server_shutting_down"
        assert exc.value.status == 503
        mock_spawn.assert_not_called()

    def test_readiness_failure_terminates_partial_worker(
        self, tmp_catalog, state,
    ):
        from muse.admin.operations import load_model_into_worker

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        child = MagicMock()

        def _spawn(spec, **_kwargs):
            spec.process = child

        with patch("muse.admin.operations.spawn_worker", side_effect=_spawn), \
             patch(
                 "muse.admin.operations.wait_for_ready",
                 side_effect=TimeoutError("never ready"),
             ) as mock_wait, \
             patch("muse.admin.operations.find_free_port", return_value=9200), \
             patch("muse.admin.operations._shutdown_workers") as mock_shutdown:
            with pytest.raises(TimeoutError, match="never ready"):
                load_model_into_worker("kokoro-82m", state=state)

        spec = state.workers[0]
        mock_wait.assert_called_once_with(
            port=9200, timeout=120.0, stop_event=state.stop_event,
        )
        mock_shutdown.assert_called_once_with([spec])
        assert spec.status == "dead"
        assert spec.job_id is None

    def test_shutdown_during_readiness_returns_service_unavailable(
        self, tmp_catalog, state,
    ):
        from muse.admin.operations import load_model_into_worker

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })

        def _cancel_wait(**_kwargs):
            state.stop_event.set()
            raise RuntimeError("worker startup cancelled")

        with patch("muse.admin.operations.spawn_worker"), \
             patch("muse.admin.operations.wait_for_ready", side_effect=_cancel_wait), \
             patch("muse.admin.operations.find_free_port", return_value=9200), \
             patch("muse.admin.operations._shutdown_workers") as mock_shutdown:
            with pytest.raises(OperationError) as exc:
                load_model_into_worker("kokoro-82m", state=state)

        assert exc.value.code == "server_shutting_down"
        assert exc.value.status == 503
        mock_shutdown.assert_called_once_with([state.workers[0]])

    def test_respawns_dead_worker_instead_of_returning_dead_port(
        self, tmp_catalog, state,
    ):
        """H1: load_model_into_worker (the director's cold-load path) must
        not commit a hot LoadEntry pointing at a dead worker's port. A dead
        spec with job_id=None must trigger a fresh spawn."""
        from muse.admin.operations import load_model_into_worker

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        dead = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        dead.status = "dead"
        dead.job_id = None
        state.workers.append(dead)
        with patch("muse.admin.operations.spawn_worker") as mock_spawn, \
             patch("muse.admin.operations.wait_for_ready"), \
             patch("muse.admin.operations.find_free_port", return_value=9200):
            port = load_model_into_worker("kokoro-82m", state=state)
        mock_spawn.assert_called_once()
        assert port == 9200

    def test_cold_load_excludes_ports_held_by_pending_specs(
        self, tmp_catalog, state,
    ):
        """M1: two concurrent cold loads of DIFFERENT models must not both
        pick the same not-yet-bound port. A pending spec already holds 9001;
        a new load must skip it even though find_free_port (which only probes
        the OS) reports 9001 as free, because the pending worker has not
        bound yet. Otherwise the loser fails to bind and wait_for_ready times
        out despite ~999 free ports."""
        from muse.admin.operations import load_model_into_worker

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        # A pending spec for a DIFFERENT venv already reserved 9001.
        pending = WorkerSpec(
            models=["other-model"], python_path="/venv/other/bin/python",
            port=9001,
        )
        pending.status = "pending"
        pending.job_id = "job-a"
        state.workers.append(pending)

        with patch("muse.admin.operations.spawn_worker") as mock_spawn, \
             patch("muse.admin.operations.wait_for_ready"), \
             patch(
                 "muse.admin.operations.find_free_port",
                 side_effect=[9001, 9002],
             ):
            port = load_model_into_worker("kokoro-82m", state=state)

        mock_spawn.assert_called_once()
        # 9001 is held by the pending spec; the new load must skip it.
        assert port == 9002
        new_spec = next(s for s in state.workers if "kokoro-82m" in s.models)
        assert new_spec.port == 9002

    def test_concurrent_enable_coalesces_to_one_spawn(
        self, tmp_catalog, state, store, monkeypatch,
    ):
        """Two concurrent enables for the same model MUST NOT spawn two
        workers. The second caller observes the first caller's pending
        spec and coalesces onto its job_id (γ-flavor idempotency).

        Closes findings #7 + #8 from the v0.32.0 review.
        """
        import threading
        import time

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": False,
            },
        })

        spawn_count = {"n": 0}

        def _slow_spawn(spec, device, **_kwargs):
            spawn_count["n"] += 1
            # Simulate a slow GGUF load. The fix releases state.lock
            # before this point, so the second concurrent enable can
            # observe spec.status == "pending" + job_id and coalesce.
            time.sleep(0.3)

        monkeypatch.setattr("muse.admin.operations.spawn_worker", _slow_spawn)
        monkeypatch.setattr(
            "muse.admin.operations.wait_for_ready", lambda *a, **k: None,
        )
        monkeypatch.setattr(
            "muse.admin.operations.find_free_port", lambda *a, **k: 9123,
        )

        job1 = store.create("enable", "kokoro-82m")
        job2 = store.create("enable", "kokoro-82m")

        def _call(j):
            enable_model("kokoro-82m", state=state, store=store, job=j)

        t1 = threading.Thread(target=_call, args=(job1,))
        t1.start()
        # Brief delay so t1 grabs state.lock first and appends the
        # pending spec before t2 enters the planning block.
        time.sleep(0.05)
        t2 = threading.Thread(target=_call, args=(job2,))
        t2.start()
        t1.join()
        t2.join()

        assert spawn_count["n"] == 1, (
            f"expected 1 spawn (coalesce), got {spawn_count['n']}"
        )
        assert len(state.workers) == 1
        # Both jobs are done. The first did the spawn; the second
        # coalesced and surfaces the first's job_id.
        assert job1.state == "done"
        assert job2.state == "done"
        assert job2.result.get("coalesced_job_id") == job1.job_id
        assert job2.result["spawned_new"] is False

    def test_state_lock_released_during_spawn(
        self, tmp_catalog, state, store, monkeypatch,
    ):
        """Other admin ops (e.g. /v1/admin/workers, /v1/admin/memory)
        must not block while enable_model's spawn is in flight. Hold
        time on the lock during the slow spawn window must be near-zero.

        Closes finding #7 from the v0.32.0 review.
        """
        import threading
        import time

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": False,
            },
        })

        def _slow_spawn(spec, device, **_kwargs):
            time.sleep(0.4)

        monkeypatch.setattr("muse.admin.operations.spawn_worker", _slow_spawn)
        monkeypatch.setattr(
            "muse.admin.operations.wait_for_ready", lambda *a, **k: None,
        )
        monkeypatch.setattr(
            "muse.admin.operations.find_free_port", lambda *a, **k: 9234,
        )

        job = store.create("enable", "kokoro-82m")
        enable_done = threading.Event()

        def _enable():
            enable_model("kokoro-82m", state=state, store=store, job=job)
            enable_done.set()

        threading.Thread(target=_enable).start()
        # Give the enable thread a head start so it's mid-spawn.
        time.sleep(0.1)

        # Grab the lock and time it. With the bug, this would block
        # for the full slow-spawn duration (0.3s+).
        t0 = time.perf_counter()
        with state.lock:
            snapshot = list(state.workers)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.05, (
            f"state.lock was held during spawn for {elapsed:.3f}s; "
            "should release between append-pending and spawn"
        )
        # The pending spec is visible to readers during the spawn
        # window (auto-restart monitor and admin reads filter by
        # status; pending workers are harmless).
        assert len(snapshot) == 1
        assert snapshot[0].status == "pending"

        enable_done.wait(timeout=2.0)
        assert job.state == "done"
        assert state.workers[0].status == "running"
        assert state.workers[0].job_id is None


class TestRestartWorkerCleanup:
    def test_readiness_failure_terminates_replacement(self, state):
        from muse.admin.operations import _restart_worker_inplace

        old_process = MagicMock(name="old_process")
        new_process = MagicMock(name="new_process")
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        spec.process = old_process
        stopped_processes = []

        def _shutdown(specs):
            stopped_processes.append(specs[0].process)

        def _spawn(worker_spec, **_kwargs):
            worker_spec.process = new_process

        with patch(
            "muse.admin.operations._shutdown_workers",
            side_effect=_shutdown,
        ), patch(
            "muse.admin.operations.spawn_worker", side_effect=_spawn,
        ), patch(
            "muse.admin.operations.wait_for_ready",
            side_effect=TimeoutError("never ready"),
        ) as mock_wait:
            with pytest.raises(TimeoutError, match="never ready"):
                _restart_worker_inplace(
                    spec,
                    device="cpu",
                    stop_event=state.stop_event,
                )

        assert stopped_processes == [old_process, new_process]
        mock_wait.assert_called_once_with(
            port=9001, timeout=120.0, stop_event=state.stop_event,
        )


class TestDisableModel:
    def test_unknown_raises_operation_error(self, tmp_catalog, state):
        _seed_catalog({})
        with pytest.raises(OperationError) as exc:
            disable_model("ghost", state=state)
        assert exc.value.status == 404

    def test_unloaded_returns_unloaded_record(self, tmp_catalog, state):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        out = disable_model("kokoro-82m", state=state)
        assert out["model_id"] == "kokoro-82m"
        assert out["loaded"] is False
        assert out["worker_terminated"] is False

    def test_terminates_worker_when_only_model(self, tmp_catalog, state):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        state.workers.append(spec)
        with patch("muse.admin.operations._shutdown_workers") as mock_sd:
            out = disable_model("kokoro-82m", state=state)
        assert out["worker_terminated"] is True
        assert out["worker_port"] == 9001
        assert state.workers == []
        mock_sd.assert_called_once_with([spec])

    def test_restarts_worker_when_other_models_remain(self, tmp_catalog, state):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": True,
            },
            "soprano-80m": {
                "pulled_at": "...", "hf_repo": "s", "local_dir": "/s",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m", "soprano-80m"],
            python_path="/venv/shared/bin/python", port=9001,
        )
        state.workers.append(spec)
        with patch("muse.admin.operations._restart_worker_inplace") as mock_restart:
            out = disable_model("soprano-80m", state=state)
        assert out["worker_terminated"] is False
        assert "kokoro-82m" in out["remaining_models_in_worker"]
        assert "soprano-80m" not in spec.models
        mock_restart.assert_called_once_with(
            spec,
            device="cpu",
            log_hub=None,
            stop_event=state.stop_event,
        )


class TestOrphanRespawnGuard:
    """Sole-tenant worker removal (unload_model_from_worker / disable_model)
    must stamp spec.job_id under state.lock BEFORE the outside-lock
    shutdown runs. Without it, an auto-restart monitor tick that snapshotted
    the spec earlier (see _monitor_workers' `list(specs)` snapshot) still
    holds a reference to it after the removal; once the outside-lock
    shutdown SIGTERMs the process, that stale-snapshot tick sees
    process.poll() != None, ratchets failure_count to threshold, and
    _attempt_restart spawns a brand-new subprocess on the freed port that
    is never tracked in state.workers again (orphan, leaked VRAM). Setting
    job_id makes the monitor's `if spec.job_id is not None: continue` guard
    skip the spec, matching the restart-in-place paths' existing contract.
    """

    def test_unload_sets_job_id_before_outside_lock_shutdown(
        self, tmp_catalog, state,
    ):
        from muse.admin.operations import unload_model_from_worker

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        seen_job_id = {}

        def _capture_shutdown(specs):
            seen_job_id["value"] = specs[0].job_id

        with patch(
            "muse.admin.operations._shutdown_workers",
            side_effect=_capture_shutdown,
        ) as mock_sd:
            unload_model_from_worker("kokoro-82m", state=state)

        mock_sd.assert_called_once()
        assert seen_job_id["value"] is not None, (
            "spec.job_id must be set before the outside-lock shutdown so "
            "a monitor tick that snapshotted the spec earlier skips it"
        )

    def test_disable_sets_job_id_before_outside_lock_shutdown(
        self, tmp_catalog, state,
    ):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        seen_job_id = {}

        def _capture_shutdown(specs):
            seen_job_id["value"] = specs[0].job_id

        with patch(
            "muse.admin.operations._shutdown_workers",
            side_effect=_capture_shutdown,
        ) as mock_sd:
            disable_model("kokoro-82m", state=state)

        mock_sd.assert_called_once()
        assert seen_job_id["value"] is not None, (
            "spec.job_id must be set before the outside-lock shutdown so "
            "a monitor tick that snapshotted the spec earlier skips it"
        )

    def test_monitor_tick_over_presnapshotted_removed_spec_does_not_respawn(
        self, tmp_catalog, state,
    ):
        """Direct simulation of the race: a monitor tick snapshots
        state.workers BEFORE unload_model_from_worker removes the
        sole-tenant spec. The operation then removes it (stamping job_id
        under the fix). Running _monitor_workers over the STALE
        pre-removal snapshot -- which still references the removed spec,
        mirroring _monitor_workers' own per-tick `list(specs)` snapshot --
        must NOT trigger _attempt_restart for it, even though its process
        looks exited.
        """
        import threading
        import time

        from muse.admin.operations import unload_model_from_worker
        from muse.cli_impl.supervisor import _monitor_workers

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        spec.status = "running"
        spec.process = MagicMock(poll=MagicMock(return_value=1))  # exited
        state.workers.append(spec)

        # Simulate the monitor's own per-tick snapshot, taken BEFORE the
        # removal below.
        pre_removal_snapshot = list(state.workers)

        with patch("muse.admin.operations._shutdown_workers"):
            unload_model_from_worker("kokoro-82m", state=state)

        stop_event = threading.Event()
        with patch("muse.cli_impl.supervisor._attempt_restart") as mock_restart:
            t = threading.Thread(
                target=_monitor_workers,
                args=(pre_removal_snapshot, stop_event),
                kwargs={"interval": 0.01, "failure_threshold": 1, "max_restarts": 10},
            )
            t.start()
            time.sleep(0.1)
            stop_event.set()
            t.join(timeout=2.0)

        mock_restart.assert_not_called()


class TestRemoveModel:
    def test_unknown_model_raises_404(self, tmp_catalog, state):
        _seed_catalog({})
        with pytest.raises(OperationError) as exc:
            remove_model("ghost", state=state, purge=False)
        assert exc.value.status == 404

    def test_loaded_model_raises_409(self, tmp_catalog, state):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        state.workers.append(spec)
        with pytest.raises(OperationError) as exc:
            remove_model("kokoro-82m", state=state, purge=False)
        assert exc.value.status == 409

    def test_unloaded_model_is_removed(self, tmp_catalog, state):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        out = remove_model("kokoro-82m", state=state, purge=False)
        assert out == {"model_id": "kokoro-82m", "removed": True, "purged": False}

    def test_dead_worker_spec_does_not_block_removal(self, tmp_catalog, state):
        # A dead worker (exhausted its restart budget) lingers in
        # state.workers with the model still in spec.models, but its process
        # is gone: it holds no FDs against the venv. remove_model must NOT
        # 409 telling the operator to "disable it first".
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        dead = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        dead.status = "dead"
        state.workers.append(dead)
        out = remove_model("kokoro-82m", state=state, purge=False)
        assert out == {"model_id": "kokoro-82m", "removed": True, "purged": False}

    def test_unhealthy_worker_spec_still_blocks_removal(self, tmp_catalog, state):
        # An unhealthy worker MAY still own a live subprocess holding FDs
        # against the venv, so removal must still 409 until the operator
        # disables it (which reaps the process).
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        unhealthy = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python", port=9001,
        )
        unhealthy.status = "unhealthy"
        state.workers.append(unhealthy)
        with pytest.raises(OperationError) as exc:
            remove_model("kokoro-82m", state=state, purge=False)
        assert exc.value.status == 409


class TestProbeAndPull:
    def test_probe_runs_subprocess(self, tmp_catalog, store):
        job = store.create("probe", "kokoro-82m")
        with patch("muse.admin.operations.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="ok", stderr="",
            )
            probe_model(
                "kokoro-82m", no_inference=True, device=None,
                store=store, job=job,
            )
        assert job.state == "done"
        assert job.result["op"] == "probe"
        assert "--no-inference" in mock_run.call_args.args[0]

    def test_probe_failure_marks_failed(self, tmp_catalog, store):
        job = store.create("probe", "kokoro-82m")
        with patch("muse.admin.operations.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="boom",
            )
            probe_model(
                "kokoro-82m", no_inference=False, device="cpu",
                store=store, job=job,
            )
        assert job.state == "failed"
        assert "boom" in job.error

    def test_pull_runs_subprocess(self, tmp_catalog, store):
        job = store.create("pull", "qwen3-9b-q4")
        with patch("muse.admin.operations.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="pulled", stderr="",
            )
            pull_model("qwen3-9b-q4", store=store, job=job)
        assert job.state == "done"
        cmd = mock_run.call_args.args[0]
        assert "pull" in cmd
        assert "qwen3-9b-q4" in cmd

    def test_pull_timeout_marks_failed(self, tmp_catalog, store):
        import subprocess as sp
        job = store.create("pull", "qwen3-9b-q4")
        with patch("muse.admin.operations.subprocess.run") as mock_run:
            mock_run.side_effect = sp.TimeoutExpired(cmd="x", timeout=1)
            pull_model("qwen3-9b-q4", store=store, job=job)
        assert job.state == "failed"
        assert "timed out" in job.error

    def test_pull_cmd_has_dashdash_terminator_before_identifier(
        self, tmp_catalog, store,
    ):
        """A caller-influenced identifier beginning with '-' must not be
        parseable as a click option: '--' must appear immediately before
        the positional identifier."""
        job = store.create("pull", "--evil-id")
        with patch("muse.admin.operations.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="pulled", stderr="",
            )
            pull_model("--evil-id", store=store, job=job)
        cmd = mock_run.call_args.args[0]
        assert cmd[-2:] == ["--", "--evil-id"], cmd

    def test_probe_cmd_has_dashdash_terminator_before_identifier(
        self, tmp_catalog, store,
    ):
        job = store.create("probe", "--evil-id")
        with patch("muse.admin.operations.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="ok", stderr="",
            )
            probe_model(
                "--evil-id", no_inference=True, device="cpu",
                store=store, job=job,
            )
        cmd = mock_run.call_args.args[0]
        assert cmd[-2:] == ["--", "--evil-id"], cmd
        # options must still precede the terminator so they parse as options
        assert "--no-inference" in cmd[:cmd.index("--")]
        assert "--device" in cmd[:cmd.index("--")]


class TestLaunchAsync:
    def test_creates_job_and_thread(self, store):
        ran = {}

        def op(model_id, *, job, store, **_kwargs):  # noqa: ARG001
            ran["job_id"] = job.job_id
            ran["model_id"] = model_id

        job = launch_async(
            op, op_name="enable", model_id="m", store=store,
        )
        job.thread.join(timeout=2.0)
        assert ran["job_id"] == job.job_id
        assert ran["model_id"] == "m"
        assert job.thread is not None

    def test_thread_is_daemon(self, store):
        def op(model_id, *, job, store, **_kwargs):  # noqa: ARG001
            pass

        job = launch_async(op, op_name="enable", model_id="m", store=store)
        job.thread.join(timeout=2.0)
        assert job.thread.daemon is True

    def test_op_args_override_default_positional(self, store):
        captured = {}

        def op(a, b, *, job, store):  # noqa: ARG001
            captured["a"] = a
            captured["b"] = b

        job = launch_async(
            op, op_name="enable", model_id="ignored",
            store=store, op_args=("x", "y"),
        )
        job.thread.join(timeout=2.0)
        assert captured == {"a": "x", "b": "y"}
