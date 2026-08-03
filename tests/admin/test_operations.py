"""Tests for admin operations: enable / disable / remove / probe / pull.

All tests mock subprocess.Popen + spawn_worker + wait_for_ready so no
actual workers are spawned. SupervisorState instances are local to each
test (never the singleton) so tests don't bleed into each other.
"""
from __future__ import annotations

import io
import json
import signal
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.admin.jobs import JobStore
from muse.admin.operations import (
    OperationError,
    _ProcessOutputCapture,
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


def _fake_admin_process(*, returncode=0, stdout="", stderr=""):
    process = MagicMock()
    process.pid = 4242
    process.returncode = returncode
    process.stdout = io.BytesIO(stdout.encode("utf-8"))
    process.stderr = io.BytesIO(stderr.encode("utf-8"))
    process.wait.return_value = returncode
    process.poll.return_value = returncode
    return process


@contextmanager
def _mock_admin_popen(process):
    """Mock child creation and all identity reads; never touch host PIDs."""
    with patch("muse.admin.jobs.subprocess.Popen", return_value=process) as popen, \
         patch("muse.admin.jobs.os.getpgid", return_value=process.pid), \
         patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
         patch("muse.admin.jobs.register_process", return_value="resource-1"), \
         patch("muse.admin.jobs.os.killpg", side_effect=ProcessLookupError), \
         patch("muse.admin.jobs.unregister_process"):
        yield popen


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

    def test_catalog_write_runs_outside_central_routing_lock(
        self, tmp_catalog, state, store,
    ):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...",
                "hf_repo": "k",
                "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": False,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"],
            python_path="/venv/k/bin/python",
            port=9001,
            status="running",
        )
        state.workers.append(spec)
        job = store.create("enable", "kokoro-82m")
        lock_states = []

        def set_enabled_without_routing_lock(_model_id, _enabled):
            lock_states.append(state.lock._is_owned())

        with patch(
            "muse.admin.operations.set_enabled",
            side_effect=set_enabled_without_routing_lock,
        ):
            enable_model(
                "kokoro-82m", state=state, store=store, job=job,
            )

        assert job.state == "done"
        assert lock_states == [False]

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
            status="running",
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
        def _spawn(spec, **_kwargs):
            spec.worker_nonce = "enable-nonce"

        with patch(
            "muse.admin.operations.spawn_worker", side_effect=_spawn,
        ) as mock_spawn, \
             patch("muse.admin.operations.wait_for_ready") as mock_wait, \
             patch("muse.admin.operations.find_free_port", return_value=9123):
            enable_model("kokoro-82m", state=state, store=store, job=job)
        assert job.state == "done"
        assert job.result["spawned_new"] is True
        assert job.result["worker_port"] == 9123
        mock_spawn.assert_called_once()
        spec = state.workers[0]
        mock_wait.assert_called_once_with(
            port=9123, timeout=120.0, stop_event=state.stop_event,
            expected_nonce="enable-nonce", worker=spec,
        )
        assert len(state.workers) == 1
        assert spec.port == 9123

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
            status="running",
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
            models=("kokoro-82m", "soprano-80m"),
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
            spec.worker_nonce = "load-nonce"

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
            expected_nonce="load-nonce", worker=spec,
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

    def test_retained_dead_worker_blocks_replacement_and_stays_tracked(
        self, tmp_catalog, state,
    ):
        from muse.admin.operations import load_model_into_worker
        from muse.cli_impl.supervisor import WorkerShutdownResult

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        stale = WorkerSpec(
            models=["kokoro-82m"],
            python_path="/venv/k/bin/python",
            port=9001,
            status="dead",
        )
        stale.process = MagicMock(name="retained_process")
        state.workers.append(stale)
        retained = WorkerShutdownResult(released=(), retained=(stale,))

        with patch(
            "muse.admin.operations._shutdown_workers",
            return_value=retained,
        ), patch("muse.admin.operations.spawn_worker") as spawn:
            with pytest.raises(OperationError) as exc:
                load_model_into_worker("kokoro-82m", state=state)

        assert exc.value.code == "worker_shutdown_incomplete"
        spawn.assert_not_called()
        assert state.workers == [stale]
        assert stale.status == "dead"
        assert stale.job_id is None
        assert state.worker_operations == {}

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
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k", "python_path": "/venv/k/bin/python",
                "enabled": False,
            },
        })

        spawn_count = {"n": 0}
        spawn_started = threading.Event()
        release_spawn = threading.Event()

        def _slow_spawn(spec, device, **_kwargs):
            spawn_count["n"] += 1
            spawn_started.set()
            assert release_spawn.wait(timeout=2.0)

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
            try:
                enable_model("kokoro-82m", state=state, store=store, job=j)
            finally:
                done[j.job_id].set()

        done = {
            job1.job_id: threading.Event(),
            job2.job_id: threading.Event(),
        }
        t1 = threading.Thread(target=_call, args=(job1,), daemon=True)
        t1.start()
        assert spawn_started.wait(timeout=1.0)
        t2 = threading.Thread(target=_call, args=(job2,), daemon=True)
        t2.start()
        try:
            assert not done[job2.job_id].wait(timeout=0.05), (
                "second enable returned before the owning generation settled"
            )
        finally:
            release_spawn.set()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
        assert not t1.is_alive()
        assert not t2.is_alive()

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


class TestWorkerOperationInterleavings:
    def test_generations_are_monotonic_and_stale_finish_is_identity_safe(
        self, state,
    ):
        from muse.cli_impl.supervisor import (
            claim_worker_operation,
            finish_worker_operation,
        )

        first, first_claimed = claim_worker_operation(
            state, python_path="/venv/shared/bin/python", owner="job-a",
        )
        assert first_claimed is True
        assert first.generation == 1
        finish_worker_operation(state, first)

        second, second_claimed = claim_worker_operation(
            state, python_path="/venv/shared/bin/python", owner="job-a",
        )
        assert second_claimed is True
        assert second.generation == 2
        assert second.token != first.token

        # A delayed cleanup from generation 1 must not release generation 2.
        finish_worker_operation(state, first)
        current, current_claimed = claim_worker_operation(
            state, python_path="/venv/shared/bin/python", owner="job-b",
        )
        assert current_claimed is False
        assert current is second

        finish_worker_operation(state, second)
        assert state.worker_operations == {}

    def test_admin_enable_and_director_load_share_one_spawn(
        self, tmp_catalog, state, store,
    ):
        """Cross-owner same-model loads wait and revalidate one generation."""
        from muse.admin.operations import load_model_into_worker

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": False,
            },
        })
        spawn_started = threading.Event()
        release_spawn = threading.Event()
        director_entered = threading.Event()
        director_done = threading.Event()
        spawn_calls: list[WorkerSpec] = []
        director_result: list[int] = []
        director_errors: list[BaseException] = []

        def _spawn(spec, **_kwargs):
            spawn_calls.append(spec)
            spawn_started.set()
            assert release_spawn.wait(timeout=2.0)

        job = store.create("enable", "kokoro-82m")
        admin_thread = threading.Thread(
            target=enable_model,
            kwargs={
                "model_id": "kokoro-82m",
                "state": state,
                "store": store,
                "job": job,
            },
            daemon=True,
        )

        def _director_load():
            director_entered.set()
            try:
                director_result.append(
                    load_model_into_worker("kokoro-82m", state=state),
                )
            except BaseException as exc:  # test thread must surface failures
                director_errors.append(exc)
            finally:
                director_done.set()

        with patch(
            "muse.admin.operations.spawn_worker", side_effect=_spawn,
        ), patch(
            "muse.admin.operations.wait_for_ready",
        ), patch(
            "muse.admin.operations.find_free_port", return_value=9123,
        ):
            admin_thread.start()
            assert spawn_started.wait(timeout=1.0)
            director_thread = threading.Thread(
                target=_director_load, daemon=True,
            )
            director_thread.start()
            assert director_entered.wait(timeout=1.0)
            try:
                assert not director_done.wait(timeout=0.05)
                assert len(spawn_calls) == 1
            finally:
                release_spawn.set()
            admin_thread.join(timeout=2.0)
            director_thread.join(timeout=2.0)

        assert not admin_thread.is_alive()
        assert not director_thread.is_alive()
        assert director_errors == []
        assert director_result == [9123]
        assert job.state == "done"
        assert len(spawn_calls) == 1
        assert len(state.workers) == 1
        assert state.workers[0].models == ["kokoro-82m"]
        assert state.worker_operations == {}

    @pytest.mark.parametrize("remove_kind", ["disable", "unload"])
    def test_shared_venv_removal_waits_for_immutable_restart_plan(
        self, tmp_catalog, state, store, remove_kind,
    ):
        """Disable/unload cannot edit the command an enable already owns."""
        from muse.admin.operations import unload_model_from_worker

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
            status="running",
        )
        state.workers.append(spec)
        first_restart_started = threading.Event()
        release_first_restart = threading.Event()
        removal_entered = threading.Event()
        removal_done = threading.Event()
        restart_models: list[tuple[str, ...]] = []
        removal_errors: list[BaseException] = []

        def _restart(worker_spec, *, models, **_kwargs):
            restart_models.append(tuple(models))
            if len(restart_models) == 1:
                first_restart_started.set()
                assert release_first_restart.wait(timeout=2.0)
            worker_spec.status = "running"

        enable_job = store.create("enable", "soprano-80m")
        enable_thread = threading.Thread(
            target=enable_model,
            kwargs={
                "model_id": "soprano-80m",
                "state": state,
                "store": store,
                "job": enable_job,
            },
            daemon=True,
        )

        def _remove_first_model():
            removal_entered.set()
            try:
                if remove_kind == "disable":
                    disable_model("kokoro-82m", state=state)
                else:
                    unload_model_from_worker("kokoro-82m", state=state)
            except BaseException as exc:
                removal_errors.append(exc)
            finally:
                removal_done.set()

        with patch(
            "muse.admin.operations._restart_worker_inplace",
            side_effect=_restart,
        ):
            enable_thread.start()
            assert first_restart_started.wait(timeout=1.0)
            removal_thread = threading.Thread(
                target=_remove_first_model, daemon=True,
            )
            removal_thread.start()
            assert removal_entered.wait(timeout=1.0)
            try:
                assert not removal_done.wait(timeout=0.05)
                assert spec.models == ["kokoro-82m", "soprano-80m"]
                assert restart_models == [
                    ("kokoro-82m", "soprano-80m"),
                ]
            finally:
                release_first_restart.set()
            enable_thread.join(timeout=2.0)
            removal_thread.join(timeout=2.0)

        assert not enable_thread.is_alive()
        assert not removal_thread.is_alive()
        assert removal_errors == []
        assert enable_job.state == "done"
        assert restart_models == [
            ("kokoro-82m", "soprano-80m"),
            ("soprano-80m",),
        ]
        assert spec.models == ["soprano-80m"]
        assert spec.status == "running"
        assert state.worker_operations == {}

    def test_load_owner_blocks_remove_until_live_recheck(
        self, tmp_catalog, state,
    ):
        """A load that owns the venv makes concurrent deletion return 409."""
        from muse.admin.operations import load_model_into_worker

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        spawn_started = threading.Event()
        release_spawn = threading.Event()
        remove_entered = threading.Event()
        remove_done = threading.Event()
        load_errors: list[BaseException] = []
        remove_errors: list[BaseException] = []

        def _spawn(_spec, **_kwargs):
            spawn_started.set()
            assert release_spawn.wait(timeout=2.0)

        def _load():
            try:
                load_model_into_worker("kokoro-82m", state=state)
            except BaseException as exc:
                load_errors.append(exc)

        def _remove():
            remove_entered.set()
            try:
                remove_model("kokoro-82m", state=state, purge=True)
            except BaseException as exc:
                remove_errors.append(exc)
            finally:
                remove_done.set()

        with patch(
            "muse.admin.operations.spawn_worker", side_effect=_spawn,
        ), patch(
            "muse.admin.operations.wait_for_ready",
        ), patch(
            "muse.admin.operations.find_free_port", return_value=9123,
        ), patch(
            "muse.admin.operations.catalog_remove",
        ) as catalog_remove_mock:
            load_thread = threading.Thread(target=_load, daemon=True)
            load_thread.start()
            assert spawn_started.wait(timeout=1.0)
            remove_thread = threading.Thread(target=_remove, daemon=True)
            remove_thread.start()
            assert remove_entered.wait(timeout=1.0)
            try:
                assert not remove_done.wait(timeout=0.05)
                catalog_remove_mock.assert_not_called()
            finally:
                release_spawn.set()
            load_thread.join(timeout=2.0)
            remove_thread.join(timeout=2.0)

        assert load_errors == []
        assert len(remove_errors) == 1
        assert isinstance(remove_errors[0], OperationError)
        assert remove_errors[0].code == "model_loaded"
        catalog_remove_mock.assert_not_called()
        assert state.worker_operations == {}

    def test_remove_owner_prevents_stale_catalog_load(
        self, tmp_catalog, state,
    ):
        """A later load waits for deletion and never spawns from stale paths."""
        from muse.admin.operations import load_model_into_worker

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        remove_at_delete = threading.Event()
        release_delete = threading.Event()
        load_entered = threading.Event()
        load_done = threading.Event()
        remove_errors: list[BaseException] = []
        load_errors: list[BaseException] = []

        def _catalog_remove(_model_id, *, purge):
            assert purge is True
            remove_at_delete.set()
            assert release_delete.wait(timeout=2.0)
            _seed_catalog({})

        def _remove():
            try:
                remove_model("kokoro-82m", state=state, purge=True)
            except BaseException as exc:
                remove_errors.append(exc)

        def _load():
            load_entered.set()
            try:
                load_model_into_worker("kokoro-82m", state=state)
            except BaseException as exc:
                load_errors.append(exc)
            finally:
                load_done.set()

        with patch(
            "muse.admin.operations.catalog_remove",
            side_effect=_catalog_remove,
        ), patch(
            "muse.admin.operations.spawn_worker",
        ) as spawn_mock, patch(
            "muse.admin.operations.wait_for_ready",
        ), patch(
            "muse.admin.operations.find_free_port", return_value=9123,
        ):
            remove_thread = threading.Thread(target=_remove, daemon=True)
            remove_thread.start()
            assert remove_at_delete.wait(timeout=1.0)
            load_thread = threading.Thread(target=_load, daemon=True)
            load_thread.start()
            assert load_entered.wait(timeout=1.0)
            try:
                assert not load_done.wait(timeout=0.05)
                spawn_mock.assert_not_called()
            finally:
                release_delete.set()
            remove_thread.join(timeout=2.0)
            load_thread.join(timeout=2.0)

        assert remove_errors == []
        assert len(load_errors) == 1
        assert isinstance(load_errors[0], OperationError)
        assert load_errors[0].code == "model_not_pulled"
        spawn_mock.assert_not_called()
        assert state.worker_operations == {}

    def test_remove_rekeys_ownership_after_catalog_path_change(
        self, tmp_catalog, state,
    ):
        from muse.cli_impl.supervisor import claim_worker_operation

        def _entry(python_path):
            return {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": str(Path(python_path).parent.parent),
                "python_path": python_path,
                "enabled": True,
            }

        old_path = "/venv/old/bin/python"
        new_path = "/venv/new/bin/python"
        _seed_catalog({"kokoro-82m": _entry(old_path)})
        claim_paths: list[str] = []
        delete_owned_keys: list[set[str]] = []
        replaced = {"value": False}

        def _claim(current_state, *, python_path, owner):
            operation, claimed = claim_worker_operation(
                current_state, python_path=python_path, owner=owner,
            )
            claim_paths.append(python_path)
            if claimed and python_path == old_path and not replaced["value"]:
                replaced["value"] = True
                _seed_catalog({"kokoro-82m": _entry(new_path)})
            return operation, claimed

        def _delete(_model_id, *, purge):
            assert purge is True
            delete_owned_keys.append(set(state.worker_operations))

        with patch(
            "muse.admin.operations.claim_worker_operation",
            side_effect=_claim,
        ), patch(
            "muse.admin.operations.catalog_remove", side_effect=_delete,
        ):
            result = remove_model("kokoro-82m", state=state, purge=True)

        assert result["removed"] is True
        assert claim_paths == [old_path, new_path]
        assert delete_owned_keys == [{new_path}]
        assert state.worker_operations == {}

    def test_monitor_restart_owns_venv_before_admin_mutation(
        self, tmp_catalog, state, store,
    ):
        """Monitor and admin transitions use the same generation registry."""
        from muse.cli_impl.supervisor import _monitor_workers

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
            status="running",
        )
        spec.process = MagicMock()
        spec.process.poll.return_value = 1
        spec.process.returncode = 1
        state.workers.append(spec)

        monitor_stop = threading.Event()
        monitor_restart_started = threading.Event()
        release_monitor_restart = threading.Event()
        enable_entered = threading.Event()
        enable_done = threading.Event()
        admin_restart_models: list[tuple[str, ...]] = []

        def _monitor_restart(worker_spec, **_kwargs):
            monitor_restart_started.set()
            assert release_monitor_restart.wait(timeout=2.0)
            worker_spec.status = "running"
            monitor_stop.set()

        def _admin_restart(worker_spec, *, models, **_kwargs):
            admin_restart_models.append(tuple(models))
            worker_spec.status = "running"

        job = store.create("enable", "soprano-80m")

        def _enable():
            enable_entered.set()
            try:
                enable_model(
                    "soprano-80m", state=state, store=store, job=job,
                )
            finally:
                enable_done.set()

        with patch(
            "muse.cli_impl.supervisor._attempt_restart",
            side_effect=_monitor_restart,
        ), patch(
            "muse.admin.operations._restart_worker_inplace",
            side_effect=_admin_restart,
        ):
            monitor_thread = threading.Thread(
                target=_monitor_workers,
                args=(state.workers, monitor_stop),
                kwargs={
                    "interval": 0.001,
                    "failure_threshold": 1,
                    "max_restarts": 10,
                    "state": state,
                },
                daemon=True,
            )
            monitor_thread.start()
            assert monitor_restart_started.wait(timeout=1.0)
            enable_thread = threading.Thread(target=_enable, daemon=True)
            enable_thread.start()
            assert enable_entered.wait(timeout=1.0)
            try:
                assert not enable_done.wait(timeout=0.05)
                assert spec.models == ["kokoro-82m"]
                assert admin_restart_models == []
            finally:
                release_monitor_restart.set()
            monitor_thread.join(timeout=2.0)
            enable_thread.join(timeout=2.0)

        assert not monitor_thread.is_alive()
        assert not enable_thread.is_alive()
        assert job.state == "done"
        assert admin_restart_models == [
            ("kokoro-82m", "soprano-80m"),
        ]
        assert spec.models == ["kokoro-82m", "soprano-80m"]
        assert spec.job_id is None
        assert state.worker_operations == {}


class TestRestartWorkerCleanup:
    def test_successful_inplace_restart_resets_auto_restart_failures(self, state):
        from muse.admin.operations import _restart_worker_inplace

        old_process = MagicMock(name="old_process")
        new_process = MagicMock(name="new_process")
        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001, restart_count=6,
        )
        spec.process = old_process

        def _shutdown(specs):
            assert specs == [spec]
            spec.process = None

        def _spawn(worker_spec, **_kwargs):
            worker_spec.process = new_process

        with patch(
            "muse.admin.operations._shutdown_workers",
            side_effect=_shutdown,
        ), patch(
            "muse.admin.operations.spawn_worker", side_effect=_spawn,
        ), patch("muse.admin.operations.wait_for_ready"):
            _restart_worker_inplace(
                spec,
                device="cpu",
                stop_event=state.stop_event,
            )

        assert spec.process is new_process
        assert spec.restart_count == 0
        assert spec.status == "running"

    def test_readiness_failure_terminates_replacement(self, state):
        from muse.admin.operations import _restart_worker_inplace

        old_process = MagicMock(name="old_process")
        new_process = MagicMock(name="new_process")
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        spec.process = old_process
        stopped_processes = []

        def _shutdown(specs):
            stopped_processes.append(specs[0].process)
            specs[0].process = None

        def _spawn(worker_spec, **_kwargs):
            worker_spec.process = new_process
            worker_spec.worker_nonce = "restart-nonce"

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
            expected_nonce="restart-nonce", worker=spec,
        )


class TestDisableModel:
    def test_catalog_write_runs_outside_central_routing_lock(
        self, tmp_catalog, state,
    ):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...",
                "hf_repo": "k",
                "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"],
            python_path="/venv/k/bin/python",
            port=9001,
            status="running",
        )
        state.workers.append(spec)
        lock_states = []

        def set_enabled_without_routing_lock(_model_id, _enabled):
            lock_states.append(state.lock._is_owned())

        with patch(
            "muse.admin.operations.set_enabled",
            side_effect=set_enabled_without_routing_lock,
        ), patch("muse.admin.operations._shutdown_workers"):
            result = disable_model("kokoro-82m", state=state)

        assert result["loaded"] is False
        assert lock_states == [False]

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
            status="running",
        )
        state.workers.append(spec)
        with patch("muse.admin.operations._restart_worker_inplace") as mock_restart:
            out = disable_model("soprano-80m", state=state)
        assert out["worker_terminated"] is False
        assert "kokoro-82m" in out["remaining_models_in_worker"]
        assert "soprano-80m" not in spec.models
        mock_restart.assert_called_once_with(
            spec,
            models=("kokoro-82m",),
            device="cpu",
            log_hub=None,
            stop_event=state.stop_event,
        )


class TestIncompleteSoleWorkerTeardown:
    def _seed_loaded_worker(self, state):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"],
            python_path="/venv/k/bin/python",
            port=9001,
            status="running",
        )
        spec.process = MagicMock(name="retained_process")
        state.workers.append(spec)
        return spec

    def test_unload_reinserts_retained_worker_and_reports_503(
        self, tmp_catalog, state,
    ):
        from muse.admin.operations import unload_model_from_worker
        from muse.cli_impl.supervisor import WorkerShutdownResult

        spec = self._seed_loaded_worker(state)
        retained = WorkerShutdownResult(released=(), retained=(spec,))

        with patch(
            "muse.admin.operations._shutdown_workers",
            return_value=retained,
        ):
            with pytest.raises(OperationError) as exc:
                unload_model_from_worker("kokoro-82m", state=state)

        assert exc.value.code == "worker_shutdown_incomplete"
        assert exc.value.status == 503
        assert state.workers == [spec]
        assert spec.models == ["kokoro-82m"]
        assert spec.process is not None
        assert spec.status == "dead"
        assert spec.job_id is None
        assert state.worker_operations == {}

    def test_disable_restart_failure_reinserts_retained_multi_model_worker(
        self, tmp_catalog, state,
    ):
        from muse.cli_impl.supervisor import WorkerShutdownResult

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/shared",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": True,
            },
            "soprano-80m": {
                "pulled_at": "...", "hf_repo": "s", "local_dir": "/shared",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": True,
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m", "soprano-80m"],
            python_path="/venv/shared/bin/python",
            port=9001,
            status="running",
        )
        spec.process = MagicMock(name="retained_replacement")
        state.workers.append(spec)
        retained = WorkerShutdownResult(released=(), retained=(spec,))

        with patch(
            "muse.admin.operations._restart_worker_inplace",
            side_effect=RuntimeError("replacement failed"),
        ), patch(
            "muse.admin.operations._shutdown_workers",
            return_value=retained,
        ):
            with pytest.raises(RuntimeError, match="replacement failed"):
                disable_model("soprano-80m", state=state)

        assert state.workers == [spec]
        assert spec.status == "dead"
        assert spec.job_id is None
        assert spec.process is not None
        assert state.worker_operations == {}

    def test_disable_reinserts_retained_worker_and_does_not_claim_termination(
        self, tmp_catalog, state,
    ):
        from muse.cli_impl.supervisor import WorkerShutdownResult

        spec = self._seed_loaded_worker(state)
        retained = WorkerShutdownResult(released=(), retained=(spec,))

        with patch(
            "muse.admin.operations._shutdown_workers",
            return_value=retained,
        ):
            with pytest.raises(OperationError) as exc:
                disable_model("kokoro-82m", state=state)

        assert exc.value.code == "worker_shutdown_incomplete"
        assert exc.value.status == 503
        assert state.workers == [spec]
        assert spec.models == ["kokoro-82m"]
        assert spec.process is not None
        assert spec.status == "dead"
        assert spec.job_id is None
        assert state.worker_operations == {}


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

    def test_dead_status_with_live_process_still_blocks_removal(
        self, tmp_catalog, state,
    ):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })
        dead_but_alive = WorkerSpec(
            models=["kokoro-82m"],
            python_path="/venv/k/bin/python",
            port=9001,
            status="dead",
        )
        dead_but_alive.process = MagicMock()
        dead_but_alive.process.poll.return_value = None
        state.workers.append(dead_but_alive)

        with patch("muse.admin.operations.catalog_remove") as catalog_remove_mock:
            with pytest.raises(OperationError) as exc:
                remove_model("kokoro-82m", state=state, purge=True)

        assert exc.value.code == "model_loaded"
        catalog_remove_mock.assert_not_called()

    def test_purge_blocks_while_shared_venv_sibling_is_running(
        self, tmp_catalog, state,
    ):
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": False,
            },
            "soprano-80m": {
                "pulled_at": "...", "hf_repo": "s", "local_dir": "/s",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": True,
            },
        })
        sibling = WorkerSpec(
            models=["soprano-80m"],
            python_path="/venv/shared/bin/python",
            port=9001,
            status="running",
        )
        state.workers.append(sibling)

        with patch("muse.admin.operations.catalog_remove") as catalog_remove_mock:
            with pytest.raises(OperationError) as exc:
                remove_model("kokoro-82m", state=state, purge=True)

        assert exc.value.code == "model_loaded"
        assert "shared environment" in exc.value.message
        catalog_remove_mock.assert_not_called()

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
        process = _fake_admin_process(stdout="ok")
        with _mock_admin_popen(process) as mock_popen:
            probe_model(
                "kokoro-82m", no_inference=True, device=None,
                store=store, job=job,
            )
        assert job.state == "done"
        assert job.result["op"] == "probe"
        assert "--no-inference" in mock_popen.call_args.args[0]
        assert mock_popen.call_args.kwargs["start_new_session"] is True

    def test_probe_failure_marks_failed(self, tmp_catalog, store):
        job = store.create("probe", "kokoro-82m")
        process = _fake_admin_process(returncode=1, stderr="boom")
        with _mock_admin_popen(process):
            probe_model(
                "kokoro-82m", no_inference=False, device="cpu",
                store=store, job=job,
            )
        assert job.state == "failed"
        assert "boom" in job.error

    def test_pull_runs_subprocess(self, tmp_catalog, store):
        job = store.create("pull", "qwen3-9b-q4")
        process = _fake_admin_process(stdout="pulled")
        with _mock_admin_popen(process) as mock_popen:
            pull_model("qwen3-9b-q4", store=store, job=job)
        assert job.state == "done"
        cmd = mock_popen.call_args.args[0]
        assert "pull" in cmd
        assert "qwen3-9b-q4" in cmd
        assert job.process is None

    def test_pull_timeout_marks_failed(self, tmp_catalog, store):
        job = store.create("pull", "qwen3-9b-q4")
        process = _fake_admin_process(stdout="partial", stderr="waiting")
        with _mock_admin_popen(process), \
             patch.object(
                 store,
                 "wait_process",
                 side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1),
             ), \
             patch.object(store, "terminate_process") as terminate:
            pull_model("qwen3-9b-q4", store=store, job=job)
        assert job.state == "failed"
        assert "timed out" in job.error
        assert job.log_lines == ["partial", "waiting"]
        terminate.assert_called_once_with(job.job_id, process, timeout=5.0)
        assert job.process is None

    def test_large_dual_stream_output_is_bounded_and_preserves_tails(
        self, tmp_catalog, store,
    ):
        job = store.create("pull", "qwen3-9b-q4")
        process = _fake_admin_process(
            stdout="stdout-head\n" + ("o" * 400_000) + "\nstdout-tail",
            stderr="stderr-head\n" + ("e" * 400_000) + "\nstderr-tail",
        )

        with _mock_admin_popen(process):
            pull_model("qwen3-9b-q4", store=store, job=job)

        assert job.state == "done"
        assert "stdout truncated" in job.result["stdout"]
        assert "stdout-head" not in job.result["stdout"]
        assert job.result["stdout"].endswith("stdout-tail")
        assert any("stderr truncated" in line for line in job.log_lines)
        assert job.log_lines[-1] == "stderr-tail"
        assert len(job.log_lines) <= 2000
        process.communicate.assert_not_called()
        process.wait.assert_not_called()

    def test_stdout_and_stderr_are_drained_concurrently(
        self, tmp_catalog, store,
    ):
        class PeerGatedStream(io.BytesIO):
            def __init__(self, payload, own_started, peer_started):
                super().__init__(payload)
                self._own_started = own_started
                self._peer_started = peer_started
                self._first_read = True

            def read(self, size=-1):
                if self._first_read:
                    self._first_read = False
                    self._own_started.set()
                    assert self._peer_started.wait(timeout=1.0)
                return super().read(size)

        stdout_started = threading.Event()
        stderr_started = threading.Event()
        process = _fake_admin_process()
        process.stdout = PeerGatedStream(
            b"stdout-tail", stdout_started, stderr_started,
        )
        process.stderr = PeerGatedStream(
            b"stderr-tail", stderr_started, stdout_started,
        )
        job = store.create("pull", "qwen3-9b-q4")

        with _mock_admin_popen(process):
            pull_model("qwen3-9b-q4", store=store, job=job)

        assert job.state == "done"
        assert job.result["stdout"] == "stdout-tail"
        assert job.log_lines == ["stdout-tail", "stderr-tail"]

    def test_pathological_line_count_is_bounded_with_marker(
        self, tmp_catalog, store,
    ):
        job = store.create("probe", "kokoro-82m")
        process = _fake_admin_process(
            stdout="".join(f"stdout-{i}\n" for i in range(2500)),
            stderr="".join(f"stderr-{i}\n" for i in range(2500)),
        )

        with _mock_admin_popen(process):
            probe_model(
                "kokoro-82m", no_inference=True, device=None,
                store=store, job=job,
            )

        assert job.state == "done"
        assert len(job.log_lines) == 2000
        assert "earlier combined log line(s) omitted" in job.log_lines[0]
        assert job.log_lines[-1] == "stderr-2499"

    def test_reader_start_failure_is_terminal_without_joining_unstarted_thread(
        self, tmp_catalog, store,
    ):
        job = store.create("pull", "qwen3-9b-q4")
        process = _fake_admin_process(stdout="partial", stderr="not-read")
        real_start = threading.Thread.start

        def start_reader(thread):
            if thread.name == "muse-admin-stderr-drain":
                raise RuntimeError("cannot start stderr reader")
            return real_start(thread)

        with _mock_admin_popen(process), patch(
            "muse.admin.operations.threading.Thread.start",
            new=start_reader,
        ), patch.object(store, "terminate_process") as terminate:
            pull_model("qwen3-9b-q4", store=store, job=job)

        assert job.state == "failed"
        assert job.error == "cannot start stderr reader"
        terminate.assert_called_once_with(job.job_id, process, timeout=5.0)
        assert job.process is None

    def test_capture_closes_blocked_owned_stream_after_leader_exit(self):
        class CloseReleasedStream:
            def __init__(self):
                self.read_started = threading.Event()
                self.closed = threading.Event()

            def read(self, _size=-1):
                self.read_started.set()
                assert self.closed.wait(timeout=1.0)
                return b""

            def close(self):
                self.closed.set()

        stdout = CloseReleasedStream()
        capture = _ProcessOutputCapture(stdout, io.BytesIO(b""))
        capture.start()
        assert stdout.read_started.wait(timeout=1.0)

        assert capture.finish(timeout=0.2) is False
        assert stdout.closed.is_set()
        # Production performs this bounded final-drain pass after a forced
        # close. Mirror that contract before asserting scheduler settlement.
        capture.finish(timeout=1.0)
        assert all(not thread.is_alive() for thread in capture._threads.values())

    def test_inherited_output_fd_never_resignals_reaped_leader(
        self, tmp_catalog, store,
    ):
        class CloseReleasedStream:
            def __init__(self):
                self.closed = threading.Event()

            def read(self, _size=-1):
                assert self.closed.wait(timeout=1.0)
                return b""

            def close(self):
                self.closed.set()

        job = store.create("pull", "qwen3-9b-q4")
        process = _fake_admin_process(stdout="")
        process.stdout = CloseReleasedStream()

        with _mock_admin_popen(process), patch(
            "muse.admin.operations._SUBPROCESS_OUTPUT_DRAIN_SECONDS", 0.05,
        ), patch(
            "muse.admin.operations._SUBPROCESS_FINAL_DRAIN_SECONDS", 0.05,
        ), patch.object(store, "terminate_process") as terminate:
            pull_model("qwen3-9b-q4", store=store, job=job)

        assert job.state == "failed"
        assert job.error == "subprocess output streams did not close"
        assert process.stdout.closed.is_set()
        terminate.assert_not_called()

    def test_pull_cmd_has_dashdash_terminator_before_identifier(
        self, tmp_catalog, store,
    ):
        """A caller-influenced identifier beginning with '-' must not be
        parseable as a click option: '--' must appear immediately before
        the positional identifier."""
        job = store.create("pull", "--evil-id")
        process = _fake_admin_process(stdout="pulled")
        with _mock_admin_popen(process) as mock_popen:
            pull_model("--evil-id", store=store, job=job)
        cmd = mock_popen.call_args.args[0]
        assert cmd[-2:] == ["--", "--evil-id"], cmd

    def test_probe_cmd_has_dashdash_terminator_before_identifier(
        self, tmp_catalog, store,
    ):
        job = store.create("probe", "--evil-id")
        process = _fake_admin_process(stdout="ok")
        with _mock_admin_popen(process) as mock_popen:
            probe_model(
                "--evil-id", no_inference=True, device="cpu",
                store=store, job=job,
            )
        cmd = mock_popen.call_args.args[0]
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

    def test_full_store_becomes_retryable_operation_error(self):
        bounded_store = JobStore(max_jobs=1)
        bounded_store.create("pull", "already-running")

        with pytest.raises(OperationError) as exc:
            launch_async(
                lambda *_args, **_kwargs: None,
                op_name="probe",
                model_id="new",
                store=bounded_store,
            )

        assert exc.value.code == "admin_job_capacity"
        assert exc.value.status == 503
        assert exc.value.retryable is True

    def test_closed_store_does_not_start_thread(self):
        closed_store = JobStore()
        closed_store.shutdown(timeout=0.0)
        ran = MagicMock()

        with pytest.raises(OperationError) as exc:
            launch_async(
                ran, op_name="pull", model_id="m", store=closed_store,
            )

        assert exc.value.code == "server_shutting_down"
        ran.assert_not_called()

    def test_thread_start_failure_is_terminal_and_retryable(self):
        bounded_store = JobStore(max_jobs=1)

        with patch(
            "muse.admin.operations.threading.Thread.start",
            side_effect=RuntimeError("can't start new thread"),
        ), pytest.raises(OperationError) as exc:
            launch_async(
                lambda *_args, **_kwargs: None,
                op_name="pull",
                model_id="m",
                store=bounded_store,
            )

        assert exc.value.code == "admin_job_start_failed"
        assert exc.value.status == 503
        assert exc.value.retryable is True
        failed = bounded_store.list_recent()[0]
        assert failed.state == "failed"
        assert failed.thread is None
        replacement = bounded_store.create("probe", "replacement")
        assert bounded_store.get(replacement.job_id) is replacement

    def test_shutdown_cancels_owned_pull_and_releases_registry_record(self):
        owned_store = JobStore()
        process = MagicMock()
        process.pid = 4242
        process.returncode = None
        poll_started = threading.Event()
        alive = {"value": True}
        signals = []

        def poll():
            poll_started.set()
            return None if alive["value"] else -signal.SIGTERM

        def killpg(pgid, sig):
            signals.append((pgid, sig))
            if sig == signal.SIGTERM:
                alive["value"] = False

        process.stdout = io.BytesIO(b"")
        process.stderr = io.BytesIO(b"terminated")
        process.poll.side_effect = poll

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=4242), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch(
                 "muse.admin.jobs.register_process", return_value="resource-1",
             ), \
             patch("muse.admin.jobs.unregister_process") as unregister, \
             patch("muse.admin.jobs.os.killpg", side_effect=killpg):
            job = launch_async(
                pull_model,
                op_name="pull",
                model_id="m",
                store=owned_store,
            )
            assert poll_started.wait(timeout=1.0)
            owned_store.shutdown(timeout=0.5)

        assert (4242, signal.SIGTERM) in signals
        assert (4242, signal.SIGKILL) not in signals
        assert job.thread.is_alive() is False
        assert job.state == "failed"
        assert job.error == "subprocess cancelled during server shutdown"
        assert job.process is None
        assert job.resource_id is None
        process.wait.assert_not_called()
        unregister.assert_called_once_with("resource-1")

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
