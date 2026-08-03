"""Tests for the in-memory JobStore."""
from __future__ import annotations

import os
import signal
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from muse.admin.jobs import (
    Job,
    JobStore,
    JobStoreFullError,
    JobStoreShuttingDownError,
    get_default_store,
    reset_default_store,
)


class TestJobStore:
    def test_create_assigns_uuid_job_id_and_pending_state(self):
        store = JobStore()
        job = store.create(op="enable", model_id="soprano-80m")
        assert isinstance(job.job_id, str)
        assert len(job.job_id) == 32  # uuid4 hex
        assert job.op == "enable"
        assert job.model_id == "soprano-80m"
        assert job.state == "pending"
        assert job.started_at  # iso timestamp present
        assert job.finished_at is None

    def test_get_returns_job_by_id(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        fetched = store.get(job.job_id)
        assert fetched is job

    def test_get_returns_none_for_unknown_id(self):
        store = JobStore()
        assert store.get("nonexistent") is None

    def test_update_running_then_done_sets_finished_at(self):
        store = JobStore()
        job = store.create(op="enable", model_id="m")
        store.update(job.job_id, state="running")
        assert job.state == "running"
        assert job.finished_at is None
        store.update(job.job_id, state="done", result={"worker_port": 9001})
        assert job.state == "done"
        assert job.finished_at is not None
        assert job.result == {"worker_port": 9001}

    def test_update_failed_state_sets_finished_at_and_error(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        store.update(job.job_id, state="failed", error="subprocess crashed")
        assert job.state == "failed"
        assert job.error == "subprocess crashed"
        assert job.finished_at is not None

    def test_update_unknown_id_returns_none(self):
        store = JobStore()
        assert store.update("nope", state="done") is None

    def test_list_recent_returns_newest_first(self):
        store = JobStore()
        a = store.create(op="enable", model_id="a")
        b = store.create(op="enable", model_id="b")
        c = store.create(op="enable", model_id="c")
        listing = [j.job_id for j in store.list_recent()]
        assert listing == [c.job_id, b.job_id, a.job_id]

    def test_list_recent_caps_at_max_jobs(self):
        store = JobStore(max_jobs=3)
        created = []
        for i in range(5):
            job = store.create(op="enable", model_id=f"m{i}")
            store.update(job.job_id, state="done")
            created.append(job)
        listing = store.list_recent()
        # Terminal history is evicted from both the deque and dict.
        assert len(listing) == 3
        assert [job.model_id for job in listing] == ["m4", "m3", "m2"]
        assert store.get(created[0].job_id) is None

    def test_active_jobs_enforce_hard_bound_without_eviction(self):
        store = JobStore(max_jobs=2)
        first = store.create(op="pull", model_id="a")
        second = store.create(op="probe", model_id="b")

        with pytest.raises(JobStoreFullError, match="2 active jobs"):
            store.create(op="enable", model_id="c")

        assert store.get(first.job_id) is first
        assert store.get(second.job_id) is second
        assert len(store.list_recent()) == 2

    def test_terminal_job_with_live_thread_is_not_evicted(self):
        store = JobStore(retention_seconds=0.01, max_jobs=1)
        release = threading.Event()
        job = store.create(op="pull", model_id="a")
        thread = threading.Thread(target=release.wait, daemon=True)
        store.start_thread(job.job_id, thread)
        store.update(job.job_id, state="failed", error="finishing")
        time.sleep(0.02)

        with pytest.raises(JobStoreFullError):
            store.create(op="probe", model_id="b")
        assert store.get(job.job_id) is job

        release.set()
        thread.join(timeout=1.0)
        replacement = store.create(op="probe", model_id="b")
        assert store.get(job.job_id) is None
        assert store.get(replacement.job_id) is replacement

    def test_jobs_expire_after_retention(self):
        """Lazy reap on get/list_recent. Use a near-zero retention to test."""
        store = JobStore(retention_seconds=0.01)
        job = store.create(op="enable", model_id="m")
        store.update(job.job_id, state="done", result={"ok": True})
        time.sleep(0.05)
        # After retention, the job is reaped on next list call.
        assert store.list_recent() == []
        assert store.get(job.job_id) is None

    def test_pending_jobs_never_expire(self):
        """A pending job has finished_at_monotonic = None; reap skips it."""
        store = JobStore(retention_seconds=0.01)
        job = store.create(op="enable", model_id="m")
        time.sleep(0.05)
        assert store.get(job.job_id) is job

    def test_to_dict_excludes_thread_and_monotonic(self):
        store = JobStore()
        job = store.create(op="enable", model_id="m")
        job.thread = threading.Thread(target=lambda: None)
        store.update(job.job_id, state="done", result={"ok": True})
        d = job.to_dict()
        assert "thread" not in d
        assert "finished_at_monotonic" not in d
        assert "process" not in d
        assert "process_pid" not in d
        assert "process_group_id" not in d
        assert "resource_id" not in d
        assert "process_cancel_requested" not in d
        assert "process_lock" not in d
        assert d["state"] == "done"
        assert d["result"] == {"ok": True}
        assert d["job_id"] == job.job_id

    def test_shutdown_joins_threads(self):
        store = JobStore()
        ran = {"n": 0}
        def work():
            time.sleep(0.01)
            ran["n"] += 1
        job = store.create(op="enable", model_id="m")
        t = threading.Thread(target=work, daemon=True)
        job.thread = t
        t.start()
        store.shutdown(timeout=1.0)
        assert ran["n"] == 1

    def test_shutdown_uses_shared_deadline_not_per_thread(self):
        # Three threads that block past the budget. A per-thread timeout would
        # make shutdown take ~3*budget; the shared deadline caps it near
        # 1*budget so a few hung jobs can't stall gateway exit.
        store = JobStore()
        release = threading.Event()
        threads = []
        for i in range(3):
            job = store.create(op="enable", model_id=f"m{i}")
            t = threading.Thread(target=release.wait, daemon=True)
            job.thread = t
            t.start()
            threads.append(t)
        budget = 0.3
        start = time.monotonic()
        store.shutdown(timeout=budget)
        elapsed = time.monotonic() - start
        release.set()  # let the daemon threads exit cleanly
        for t in threads:
            t.join(timeout=1.0)
        # Shared deadline => ~1*budget, not 3*budget. Generous upper bound
        # (budget*2) tolerates scheduler jitter on a loaded CI box.
        assert elapsed < budget * 2, (
            f"shutdown took {elapsed:.2f}s for a {budget}s budget; "
            f"per-thread timeout regression?"
        )

    def test_spawn_process_registers_validated_isolated_group(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock()
        process.pid = 4242
        process.poll.return_value = 0

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process) as popen, \
             patch("muse.admin.jobs.os.getpgid", return_value=4242), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch(
                 "muse.admin.jobs.register_process", return_value="resource-1",
             ) as register, \
             patch("muse.admin.jobs.os.killpg") as killpg, \
             patch("muse.admin.jobs.unregister_process") as unregister:
            returned = store.spawn_process(
                job.job_id,
                ["muse", "pull", "m"],
                env={"MUSE_MANAGED_JOB_PROCESS_GROUP": "0", "CUSTOM": "yes"},
            )
            assert returned is process
            assert job.process is process
            assert job.process_pid == 4242
            assert job.process_group_id == 4242
            assert job.resource_id == "resource-1"
            assert popen.call_args.kwargs["start_new_session"] is True
            assert (
                popen.call_args.kwargs["env"]["MUSE_MANAGED_JOB_PROCESS_GROUP"]
                == "1"
            )
            assert popen.call_args.kwargs["env"]["CUSTOM"] == "yes"
            register.assert_called_once_with(
                kind="admin_job",
                pid=4242,
                owner_pid=os.getpid(),
                models=["m"],
            )

            store.release_process(job.job_id, process)
            unregister.assert_called_once_with("resource-1")
            killpg.assert_not_called()
        assert job.process is None
        assert job.process_pid is None
        assert job.process_group_id is None
        assert job.resource_id is None

    def test_release_retains_live_child_and_registry_identity(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock()
        process.pid = 4242
        process.poll.return_value = None

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=4242), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch(
                 "muse.admin.jobs.register_process", return_value="resource-1",
             ), \
             patch("muse.admin.jobs.os.killpg") as killpg, \
             patch("muse.admin.jobs.unregister_process") as unregister:
            store.spawn_process(job.job_id, ["muse", "pull", "m"])
            store.release_process(job.job_id, process)

        assert job.process is process
        assert job.resource_id == "resource-1"
        killpg.assert_not_called()
        unregister.assert_not_called()

    def test_exited_leader_never_probes_or_signals_stored_group(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock()
        process.pid = 4242
        process.poll.return_value = 0

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=4242), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch("muse.admin.jobs.register_process", return_value="resource-1"), \
             patch("muse.admin.jobs.os.killpg") as killpg:
            store.spawn_process(job.job_id, ["muse", "pull", "m"])
            store.terminate_process(job.job_id, process, timeout=0.0)

        killpg.assert_not_called()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_ambiguous_leader_state_retains_registry_without_signaling(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock()
        process.pid = 4242
        process.poll.side_effect = OSError("poll unavailable")

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=4242), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch("muse.admin.jobs.register_process", return_value="resource-1"), \
             patch("muse.admin.jobs.unregister_process") as unregister, \
             patch("muse.admin.jobs.os.killpg") as killpg:
            store.spawn_process(job.job_id, ["muse", "pull", "m"])
            store.terminate_process(job.job_id, process, timeout=0.0)
            store.release_process(job.job_id, process)

        assert job.process is process
        assert job.resource_id == "resource-1"
        unregister.assert_not_called()
        killpg.assert_not_called()

    def test_signal_revalidates_current_group_for_live_leader(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock()
        process.pid = 4242
        process.poll.return_value = None
        group_reads = iter((4242, 7777, 7777))

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", side_effect=group_reads), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch("muse.admin.jobs.register_process", return_value="resource-1"), \
             patch("muse.admin.jobs.os.killpg") as killpg:
            store.spawn_process(job.job_id, ["muse", "pull", "m"])
            store.terminate_process(job.job_id, process, timeout=0.0)

        killpg.assert_not_called()

    def test_wait_and_signal_checks_share_per_job_identity_lock(self):
        class GuardLock:
            def __init__(self):
                self.held = False

            def __enter__(self):
                assert self.held is False
                self.held = True
                return self

            def __exit__(self, *_exc):
                self.held = False

        store = JobStore()
        job = store.create(op="pull", model_id="m")
        guard = GuardLock()
        job.process_lock = guard
        process = MagicMock()
        process.pid = 4242

        observations = iter((None, 0))
        def poll():
            assert guard.held is True
            return next(observations, 0)

        process.returncode = None
        process.poll.side_effect = poll

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=4242), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch("muse.admin.jobs.register_process", return_value="resource-1"), \
             patch("muse.admin.jobs.os.killpg") as killpg:
            store.spawn_process(job.job_id, ["muse", "pull", "m"])
            assert store.wait_process(job.job_id, process, timeout=1.0) == 0
            store.terminate_process(job.job_id, process, timeout=0.0)

        killpg.assert_not_called()
        process.wait.assert_not_called()

    def test_shutdown_terminates_only_registered_process_group(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock()
        process.pid = 4242
        alive = {"value": True}
        signals = []
        process.poll.side_effect = lambda: None if alive["value"] else -signal.SIGTERM

        def fake_killpg(pgid, sig):
            signals.append((pgid, sig))
            if sig == signal.SIGTERM:
                alive["value"] = False

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=4242), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch(
                 "muse.admin.jobs.register_process", return_value="resource-1",
             ), \
             patch("muse.admin.jobs.os.killpg", side_effect=fake_killpg):
            store.spawn_process(job.job_id, ["muse", "pull", "m"])
            store.shutdown(timeout=0.1)

        assert (4242, signal.SIGTERM) in signals
        assert (4242, signal.SIGKILL) not in signals
        assert job.process_cancel_requested is True
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_shutdown_escalates_owned_group_that_ignores_term(self):
        store = JobStore()
        job = store.create(op="probe", model_id="m")
        process = MagicMock()
        process.pid = 5252
        alive = {"value": True}
        signals = []
        process.poll.side_effect = lambda: None if alive["value"] else -signal.SIGKILL

        def fake_killpg(pgid, sig):
            signals.append((pgid, sig))
            if sig == signal.SIGKILL:
                alive["value"] = False

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=5252), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch(
                 "muse.admin.jobs.register_process", return_value="resource-2",
             ), \
             patch("muse.admin.jobs.os.killpg", side_effect=fake_killpg):
            store.spawn_process(job.job_id, ["muse", "models", "probe", "m"])
            store.shutdown(timeout=0.05)

        assert (5252, signal.SIGTERM) in signals
        assert (5252, signal.SIGKILL) in signals

    @pytest.mark.skipif(
        not (
            callable(getattr(os, "waitid", None))
            and all(
                getattr(os, name, None) is not None
                for name in ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
            )
        ),
        reason="requires waitid with WNOWAIT",
    )
    def test_pinned_group_gets_final_kill_before_leader_reap(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock()
        process.pid = 4242
        process.returncode = None
        process.wait.return_value = 0
        leader_exited = {"value": False}
        signals: list[tuple[int, signal.Signals]] = []

        def fake_waitid(*_args):
            if leader_exited["value"]:
                return SimpleNamespace(si_pid=4242)
            return None

        def fake_killpg(pgid, sig):
            signals.append((pgid, sig))
            if sig == signal.SIGTERM:
                leader_exited["value"] = True

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=4242), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch("muse.admin.jobs.register_process", return_value="resource-1"), \
             patch("muse.admin.jobs.unregister_process"), \
             patch("muse.admin.jobs._supports_pinned_target", return_value=True), \
             patch("muse.admin.jobs.os.waitid", side_effect=fake_waitid), \
             patch("muse.admin.jobs.os.killpg", side_effect=fake_killpg):
            store.spawn_process(job.job_id, ["muse", "pull", "m"])
            assert store.shutdown(timeout=0.1) is True

        assert signals == [
            (4242, signal.SIGTERM),
            (4242, signal.SIGKILL),
        ]
        process.wait.assert_called_once_with(timeout=0.0)
        assert job.process is None

    def test_shutdown_retains_exact_handles_when_registry_release_fails(self):
        from muse.core.resource_registry import ResourceRegistryError

        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock()
        process.pid = 4242
        process.returncode = 0

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=4242), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch("muse.admin.jobs.register_process", return_value="resource-1"), \
             patch(
                 "muse.admin.jobs.unregister_process",
                 side_effect=ResourceRegistryError("registry busy"),
             ), \
             patch("muse.admin.jobs.os.killpg") as killpg:
            store.spawn_process(job.job_id, ["muse", "pull", "m"])
            job.process_group_released = True
            assert store.shutdown(timeout=0.0) is False

        assert job.process is process
        assert job.process_pid == 4242
        assert job.process_group_id == 4242
        assert job.resource_id == "resource-1"
        killpg.assert_not_called()

    def test_shutdown_refuses_non_concrete_process_identity(self):
        from muse.core.resource_registry import ResourceRegistryError

        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock()
        process.pid = MagicMock(name="unsafe-pid")
        process.poll.return_value = None

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs._terminate_targets") as terminate, \
             patch("muse.admin.jobs.register_process") as register, \
             patch("muse.admin.jobs.os.killpg") as killpg:
            with pytest.raises(ResourceRegistryError, match="unsafe process identity"):
                store.spawn_process(job.job_id, ["muse", "pull", "m"])
            store.shutdown(timeout=0.01)

        terminate.assert_called()
        register.assert_not_called()
        assert job.process is process
        killpg.assert_not_called()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_registration_failure_rolls_back_exited_child_and_raises(self):
        from muse.core.resource_registry import ResourceRegistryError

        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock(pid=4242)
        process.poll.return_value = 0

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=4242), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch(
                 "muse.admin.jobs.register_process",
                 side_effect=ResourceRegistryError("disk unavailable"),
             ):
            with pytest.raises(ResourceRegistryError, match="child was rolled back"):
                store.spawn_process(job.job_id, ["muse", "pull", "m"])

        assert job.process is None
        assert job.process_pid is None
        assert job.process_group_id is None
        assert job.resource_id is None

    def test_registration_failure_retains_ambiguous_child_for_shutdown(self):
        from muse.core.resource_registry import ResourceRegistryError

        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock(pid=4242)
        process.poll.side_effect = OSError("poll unavailable")

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=4242), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch(
                 "muse.admin.jobs.register_process",
                 side_effect=ResourceRegistryError("disk unavailable"),
             ), \
             patch("muse.admin.jobs._terminate_targets") as terminate:
            with pytest.raises(ResourceRegistryError, match="retained for shutdown"):
                store.spawn_process(job.job_id, ["muse", "pull", "m"])

        terminate.assert_called_once()
        assert job.process is process
        assert job.process_pid == 4242
        assert job.process_group_id == 4242
        assert job.resource_id is None

    def test_shutdown_uses_owned_popen_when_group_cannot_be_verified(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        process = MagicMock()
        process.pid = 6262
        alive = {"value": True}
        process.poll.side_effect = lambda: None if alive["value"] else 0
        process.terminate.side_effect = lambda: alive.update(value=False)

        with patch("muse.admin.jobs.subprocess.Popen", return_value=process), \
             patch("muse.admin.jobs.os.getpgid", return_value=31337), \
             patch("muse.admin.jobs.os.getpgrp", return_value=31337), \
             patch(
                 "muse.admin.jobs.register_process", return_value="resource-3",
             ), \
             patch("muse.admin.jobs.os.killpg") as killpg:
            store.spawn_process(job.job_id, ["muse", "pull", "m"])
            store.shutdown(timeout=0.1)

        assert job.process_group_id is None
        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        killpg.assert_not_called()

    def test_shutdown_closes_thread_and_process_registration_races(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        thread = threading.Thread(target=lambda: None, daemon=True)
        store.shutdown(timeout=0.0)

        with pytest.raises(JobStoreShuttingDownError):
            store.start_thread(job.job_id, thread)
        with patch("muse.admin.jobs.subprocess.Popen") as popen:
            with pytest.raises(JobStoreShuttingDownError):
                store.spawn_process(job.job_id, ["muse", "pull", "m"])

        assert not thread.is_alive()
        popen.assert_not_called()


class TestDefaultStore:
    def test_get_default_store_returns_singleton(self):
        reset_default_store()
        s1 = get_default_store()
        s2 = get_default_store()
        assert s1 is s2

    def test_reset_default_store_creates_new_instance(self):
        reset_default_store()
        s1 = get_default_store()
        reset_default_store()
        s2 = get_default_store()
        assert s1 is not s2


@pytest.fixture(autouse=True)
def _reset_default_store():
    reset_default_store()
    yield
    reset_default_store()
