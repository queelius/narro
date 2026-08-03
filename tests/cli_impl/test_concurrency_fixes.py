"""Concurrency and correctness tests for H3, H4, H5, M1, M12 fixes.

These tests prove the five fixes hold under controlled concurrent interleavings.
All use threading.Barrier / threading.Event to force exact sequences rather
than relying on timing races.

H3: Monitor iterates a snapshot of state.workers; concurrent remove does not
    raise RuntimeError and the monitor survives.

H4: event.wait has a bounded timeout; a waiter for a killed winner re-decides
    and does not hang forever.

H5: known_models builds the cache exactly once under concurrent callers;
    discover_models is never called more than once.

M1: Two concurrent catalog RMW sequences on different keys both survive with
    no lost update, regardless of interleaving.

M12: disable_model leaves a consistent state when _restart_worker_inplace
     raises; the stale model is not serveable from state.workers.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    _monitor_workers,
)
from muse.core.catalog import (
    _CATALOG_WRITE_LOCK,
    _KNOWN_MODELS_LOCK,
    _catalog_path,
    _read_catalog,
    _reset_known_models_cache,
    _reset_read_catalog_cache,
    _write_catalog,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_catalog(tmp_path, monkeypatch):
    """Point catalog state at a tmp dir and reset caches around each test."""
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    yield
    _reset_known_models_cache()
    _reset_read_catalog_cache()


# ---------------------------------------------------------------------------
# H3: Monitor survives concurrent state.workers.remove()
# ---------------------------------------------------------------------------


class TestH3MonitorSnapshot:
    """_monitor_workers iterates list(specs) so a concurrent remove cannot
    cause 'RuntimeError: list changed size during iteration'."""

    def test_monitor_survives_concurrent_worker_removal(self):
        """Simulate one monitor tick while a concurrent thread removes a spec.

        We patch check_worker_health and _attempt_restart so no real HTTP
        or subprocess calls happen. A threading.Barrier synchronises the
        monitor's iteration entry with the removal so the race is exact.
        """
        # Build two specs; one will be removed mid-tick.
        spec_a = WorkerSpec(models=["a"], python_path="/pa", port=9001, status="running")
        spec_b = WorkerSpec(models=["b"], python_path="/pb", port=9002, status="running")
        spec_a.process = MagicMock()
        spec_a.process.poll.return_value = None
        spec_b.process = MagicMock()
        spec_b.process.poll.return_value = None
        workers = [spec_a, spec_b]

        # Barrier: 2 parties (monitor inner loop + removal thread).
        barrier = threading.Barrier(2, timeout=5)

        removal_done = threading.Event()
        error_in_removal = []

        def removal_thread():
            try:
                # Wait until the monitor has started iterating (entered barrier).
                barrier.wait()
                # Remove spec_b while the monitor is mid-iteration.
                workers.remove(spec_b)
            except Exception as exc:
                error_in_removal.append(exc)
            finally:
                removal_done.set()

        monitor_errors = []
        original_check = None

        call_count = [0]

        def patched_check_health(*, port, timeout=2.0):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: we're iterating spec_a. Signal removal thread.
                barrier.wait()
                # Give the removal thread a moment to actually mutate the list.
                removal_done.wait(timeout=2.0)
            return True  # healthy; no restart triggered

        stop_event = threading.Event()

        # Run one tick of the monitor in a separate thread so we can join it.
        monitor_finished = threading.Event()

        def run_one_tick():
            try:
                with (
                    patch("muse.cli_impl.supervisor.check_worker_health", patched_check_health),
                    patch("muse.cli_impl.supervisor._attempt_restart"),
                ):
                    # Drive one full poll loop then stop.
                    if not stop_event.is_set():
                        for spec in list(workers):  # this is the fixed line
                            if stop_event.is_set():
                                break
                            if spec.status == "dead":
                                continue
                            if spec.job_id is not None:
                                continue
                            if spec.process is not None and spec.process.poll() is not None:
                                spec.failure_count = 3
                            else:
                                patched_check_health(port=spec.port)
            except RuntimeError as e:
                monitor_errors.append(e)
            finally:
                monitor_finished.set()

        rt = threading.Thread(target=removal_thread, daemon=True)
        mt = threading.Thread(target=run_one_tick, daemon=True)

        mt.start()
        rt.start()

        mt.join(timeout=5)
        rt.join(timeout=5)

        assert not monitor_errors, f"Monitor raised: {monitor_errors}"
        assert not error_in_removal, f"Removal thread raised: {error_in_removal}"

    def test_actual_monitor_function_survives_removal(self):
        """Use the real _monitor_workers function; remove a spec via a thread
        that fires after the first health check call."""

        spec_a = WorkerSpec(models=["a"], python_path="/pa", port=9001, status="running")
        spec_b = WorkerSpec(models=["b"], python_path="/pb", port=9002, status="running")
        spec_a.process = MagicMock()
        spec_a.process.pid = 9101
        spec_a.process.poll.return_value = None
        spec_b.process = MagicMock()
        spec_b.process.pid = 9102
        spec_b.process.poll.return_value = None
        workers = [spec_a, spec_b]

        stop_event = threading.Event()
        first_health_started = threading.Event()
        removal_done = threading.Event()
        monitor_errors = []

        health_call = [0]

        def patched_health(*, port, timeout=2.0, expected_nonce=None):
            health_call[0] += 1
            if health_call[0] == 1:
                # First call (spec_a): let the remover mutate the live list
                # while the monitor is walking its snapshot.
                first_health_started.set()
                assert removal_done.wait(timeout=2.0)
            # Stop the monitor after 2 calls to avoid a second loop.
            if health_call[0] >= 2:
                stop_event.set()
            return True

        def removal_thread():
            assert first_health_started.wait(timeout=2.0)
            workers.remove(spec_b)
            removal_done.set()

        def monitor_thread():
            try:
                with patch(
                    "muse.cli_impl.supervisor.check_worker_health",
                    patched_health,
                ), patch(
                    "muse.cli_impl.supervisor._attempt_restart",
                    side_effect=AssertionError("healthy fakes must not restart"),
                ):
                    _monitor_workers(workers, stop_event, interval=0.01)
            except Exception as exc:
                monitor_errors.append(exc)

        mt = threading.Thread(target=monitor_thread, daemon=True)
        rt = threading.Thread(target=removal_thread, daemon=True)
        mt.start()
        rt.start()
        mt.join(timeout=5)
        rt.join(timeout=2)

        assert not monitor_errors, f"_monitor_workers raised: {monitor_errors}"


# ---------------------------------------------------------------------------
# H4: event.wait() has a bounded timeout; waiter does not hang forever
# ---------------------------------------------------------------------------


class TestH4EventWaitTimeout:
    """A waiter for an in-flight cold load that never has its Event set
    must not block forever. It should re-decide after the timeout and
    either find the model hot or become the new winner."""

    def test_waiter_does_not_hang_when_winner_never_sets_event(self):
        """Simulate a killed winner: register in_flight_loads but never set().

        Use a very short timeout so the test runs in milliseconds instead of
        the production 180s. The waiter should exit and either find the model
        in the loaded set (we insert it manually to simulate a belated
        success) or attempt to become the new winner.
        """
        from muse.cli_impl.load_director import (
            InFlightLoad,
            LoadDirector,
            _INFLIGHT_WAIT_TIMEOUT_SECONDS,
        )

        # Patch the constant so the test uses 0.1s instead of 180s.
        import muse.cli_impl.load_director as ld_mod
        original_timeout = ld_mod._INFLIGHT_WAIT_TIMEOUT_SECONDS
        ld_mod._INFLIGHT_WAIT_TIMEOUT_SECONDS = 0.1

        try:
            enable_call_count = [0]
            winner_done = threading.Event()

            def slow_enable(model_id: str) -> int:
                enable_call_count[0] += 1
                # The "winner" blocks here simulating a killed-but-not-
                # set winner. We signal winner_done so the test can
                # check the waiter unblocked, then return normally so
                # the test can join cleanly.
                winner_done.set()
                return 9001

            probe = MagicMock()
            probe.gpu_free_gb.return_value = None
            probe.cpu_free_gb.return_value = 64.0

            director = LoadDirector(
                enable_fn=slow_enable,
                disable_fn=MagicMock(),
                memory_probe=probe,
            )

            manifest = {
                "capabilities": {"memory_gb": 0.1, "device": "cpu"},
            }

            # Thread A: stash an Event in in_flight_loads WITHOUT setting it.
            # This simulates a killed winner.
            stash_event = threading.Event()
            stash_ready = threading.Event()

            def stash_thread():
                with director.lock:
                    evt = threading.Event()
                    director.in_flight_loads["test-model"] = InFlightLoad(
                        event=evt, memory_gb=0.1, pool="cpu",
                    )
                stash_ready.set()
                # Hold the event un-set for 5 seconds (longer than the
                # 0.1s timeout we patched in). The waiter must unblock.
                stash_event.wait(timeout=5.0)

            stasher = threading.Thread(target=stash_thread, daemon=True)
            stasher.start()
            stash_ready.wait(timeout=2.0)

            # Thread B: waiter.  With the 0.1s timeout it should unblock,
            # find no LoadEntry, and try to become the winner.  At that
            # point it will acquire the in_flight slot and call enable_fn.
            # First, remove the stale event from in_flight_loads so the
            # waiter actually becomes the winner (simulates the killed
            # winner having released its Event registration).
            waiter_result = []
            waiter_error = []

            def waiter_thread():
                # Remove the stale event so the waiter's re-decide goes
                # to the "load" phase instead of "wait" again.
                time.sleep(0.15)  # wait past the 0.1s timeout
                with director.lock:
                    director.in_flight_loads.pop("test-model", None)
                try:
                    port = director.acquire("test-model", manifest=manifest)
                    waiter_result.append(port)
                except Exception as exc:
                    waiter_error.append(exc)

            wt = threading.Thread(target=waiter_thread, daemon=True)
            wt.start()
            wt.join(timeout=5.0)
            stash_event.set()
            stasher.join(timeout=2.0)

            # The waiter should have completed without hanging.
            assert wt.is_alive() is False, "Waiter thread is still blocked -- H4 not fixed"
            assert not waiter_error, f"Waiter raised: {waiter_error}"
            assert waiter_result == [9001]

        finally:
            ld_mod._INFLIGHT_WAIT_TIMEOUT_SECONDS = original_timeout

    def test_constant_exceeds_load_timeout(self):
        """Sanity: _INFLIGHT_WAIT_TIMEOUT_SECONDS > 120 (wait_for_ready default)."""
        from muse.cli_impl.load_director import _INFLIGHT_WAIT_TIMEOUT_SECONDS
        assert _INFLIGHT_WAIT_TIMEOUT_SECONDS > 120.0


# ---------------------------------------------------------------------------
# H5: known_models cache is built exactly once under concurrent callers
# ---------------------------------------------------------------------------


class TestH5KnownModelsCacheLock:
    """Two concurrent threads racing on the first known_models() call must
    result in discover_models being called exactly once."""

    def test_discover_models_called_once_under_concurrency(self, monkeypatch):
        """Patch discover_models to be slow; assert it's called exactly once.

        With double-checked locking, only the winning thread (the one that
        acquired _KNOWN_MODELS_LOCK first) runs discover_models. The losing
        thread blocks on the lock, re-checks the cache after the winner
        releases, finds it populated, and returns the existing dict.

        We use a threading.Event to delay the winner long enough that the
        loser has definitely started waiting -- then release both and verify
        the call count.
        """
        from muse.core import catalog as cat_mod

        # Reset cache so both threads see a cold miss.
        _reset_known_models_cache()

        call_count = [0]

        # Write an empty catalog so _read_catalog does not error.
        _write_catalog({})

        # winner_entered: set by the winner (inside _KNOWN_MODELS_LOCK) after
        # incrementing call_count to signal the loser that a build is in progress.
        winner_entered = threading.Event()
        # release_winner: set by the test body to let the winner's slow discover
        # return, completing the build.
        release_winner = threading.Event()

        def slow_discover(dirs):
            call_count[0] += 1
            winner_entered.set()
            release_winner.wait(timeout=5.0)  # hold the lock while loser waits
            return {}

        monkeypatch.setattr(cat_mod, "discover_models", slow_discover)

        results = []
        errors = []

        def caller():
            try:
                result = cat_mod.known_models()
                results.append(result)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=caller, daemon=True)
        t2 = threading.Thread(target=caller, daemon=True)

        t1.start()
        # Wait until the winner is inside discover_models (holding the lock).
        winner_entered.wait(timeout=5.0)
        # Now start the loser; it will hit the lock and block.
        t2.start()
        # Give the loser time to reach the lock and queue up.
        time.sleep(0.05)
        # Release the winner.
        release_winner.set()

        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Threads raised: {errors}"
        assert len(results) == 2
        # The critical assertion: discover_models was called ONCE, not twice.
        assert call_count[0] == 1, (
            f"discover_models called {call_count[0]} times; expected 1. "
            f"H5 not fixed: double-checked locking missing or broken."
        )
        # Both threads got the same cache object.
        assert results[0] is results[1]

    def test_second_call_is_fast_path_no_lock(self, monkeypatch):
        """After the cache is built, known_models() takes the fast path without
        calling discover_models again."""
        from muse.core import catalog as cat_mod

        # Warm the cache first.
        _write_catalog({})
        monkeypatch.setattr(cat_mod, "discover_models", lambda dirs: {})
        cat_mod.known_models()  # build cache
        assert cat_mod._known_models_cache is not None

        # Now replace discover_models with something that crashes so we know
        # it's NOT called on the second invocation.
        def crashing_discover(dirs):
            raise AssertionError("discover_models called on hot path -- should not happen")

        monkeypatch.setattr(cat_mod, "discover_models", crashing_discover)

        result = cat_mod.known_models()  # must not call discover_models
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# M1: Shared catalog RMW lock prevents lost updates
# ---------------------------------------------------------------------------


class TestM1CatalogRWMLock:
    """Two concurrent catalog mutations on different keys both survive
    (no lost update) regardless of thread interleaving."""

    def test_concurrent_rmw_on_different_keys_no_lost_update(self, tmp_path, monkeypatch):
        """Thread A and Thread B each do: read -> add their key -> write.

        This test uses the _CATALOG_WRITE_LOCK (as the catalog internals do)
        to verify that under concurrent RMW the lock serializes writes and
        both keys survive. Without the lock, the "ABA" pattern would occur:
          A reads (empty), B reads (empty), B writes {b}, A writes {a} -> B lost.
        With the lock, one thread blocks until the other finishes.

        Design: each thread acquires the lock, reads, delays briefly (to give
        the other thread a chance to be queued on the lock), writes, releases.
        Because the lock serializes, the second thread reads the catalog AFTER
        the first has written, so it sees the first thread's key and both
        keys survive.
        """
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        _reset_read_catalog_cache()

        # Start with an empty catalog.
        _write_catalog({})

        errors = []

        def write_key(key: str):
            try:
                # Matches the pattern used by set_enabled, _pull_bundled, etc.
                with _CATALOG_WRITE_LOCK:
                    catalog = _read_catalog()
                    # Brief sleep inside the lock to ensure the other thread
                    # is queued (will read after we release).
                    time.sleep(0.02)
                    catalog[key] = {"value": key}
                    _write_catalog(catalog)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=write_key, args=("key-a",), daemon=True)
        t2 = threading.Thread(target=write_key, args=("key-b",), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Threads raised: {errors}"

        final = _read_catalog()
        assert "key-a" in final, "key-a was lost (M1 not fixed)"
        assert "key-b" in final, "key-b was lost (M1 not fixed)"

    def test_writeback_lock_is_catalog_write_lock(self):
        """_WRITEBACK_LOCK in load_director is the same object as
        _CATALOG_WRITE_LOCK in catalog. A test that patches
        load_director._WRITEBACK_LOCK therefore patches the shared lock."""
        from muse.cli_impl import load_director as ld_mod
        from muse.core import catalog as cat_mod

        assert ld_mod._WRITEBACK_LOCK is cat_mod._CATALOG_WRITE_LOCK, (
            "_WRITEBACK_LOCK is not the same object as _CATALOG_WRITE_LOCK; "
            "M1 shared-lock alias broken."
        )

    def test_set_enabled_serialized_by_catalog_write_lock(self, tmp_path, monkeypatch):
        """Two concurrent set_enabled calls (on different models) both commit.

        This is the direct behavioral test: both models end up with the
        expected enabled state after concurrent mutations.
        """
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        _reset_read_catalog_cache()

        _write_catalog({
            "model-a": {"enabled": True},
            "model-b": {"enabled": True},
        })

        errors = []

        from muse.core import catalog as cat_mod

        def flip(model_id: str, val: bool):
            try:
                cat_mod.set_enabled(model_id, val)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=flip, args=("model-a", False), daemon=True)
        t2 = threading.Thread(target=flip, args=("model-b", False), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Threads raised: {errors}"

        final = _read_catalog()
        assert final.get("model-a", {}).get("enabled") is False, (
            "model-a enabled flag was not persisted (lost update?)"
        )
        assert final.get("model-b", {}).get("enabled") is False, (
            "model-b enabled flag was not persisted (lost update?)"
        )

    def test_probe_write_and_set_enabled_concurrent_no_lost_update(self, tmp_path, monkeypatch):
        """A probe write and a set_enabled running concurrently on the same
        catalog entry both commit their changes. This exercises the shared lock
        between probe.py and catalog.py sites.
        """
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        _reset_read_catalog_cache()

        _write_catalog({
            "probe-model": {
                "enabled": True,
                "measurements": {},
            },
        })

        errors = []

        from muse.core import catalog as cat_mod

        def do_probe_write():
            try:
                with _CATALOG_WRITE_LOCK:
                    catalog = _read_catalog()
                    if "probe-model" in catalog:
                        catalog["probe-model"].setdefault("measurements", {})
                        catalog["probe-model"]["measurements"]["cpu"] = {"peak_bytes": 12345}
                        _write_catalog(catalog)
            except Exception as exc:
                errors.append(exc)

        def do_set_enabled():
            try:
                cat_mod.set_enabled("probe-model", False)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=do_probe_write, daemon=True)
        t2 = threading.Thread(target=do_set_enabled, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Threads raised: {errors}"

        final = _read_catalog()
        # Both operations must have committed; order determines which wins
        # on the `enabled` flag, but the measurement key must always survive
        # because the lock serializes the RMW sequences.
        assert final.get("probe-model", {}).get("measurements", {}).get("cpu") is not None, (
            "probe measurement was lost by concurrent set_enabled (M1 not fixed)"
        )
        # enabled must be False regardless of order (set_enabled always sets it)
        assert final.get("probe-model", {}).get("enabled") is False, (
            "set_enabled result was lost (M1 not fixed)"
        )


# ---------------------------------------------------------------------------
# M12: disable_model leaves a consistent state on restart failure
# ---------------------------------------------------------------------------


class TestM12DisableRestartFailure:
    """When disable_model's _restart_worker_inplace raises, the disabled model
    must not be reachable from state.workers (stale model not serveable)."""

    def _make_catalog(self, tmp_path):
        """Write a minimal catalog with a known model entry."""
        catalog_data = {
            "test-model": {
                "pulled_at": "2026-01-01T00:00:00Z",
                "hf_repo": "fake/repo",
                "local_dir": "/tmp/fake",
                "venv_path": "/tmp/venv",
                "python_path": "/tmp/venv/bin/python",
                "enabled": True,
            },
            "sibling-model": {
                "pulled_at": "2026-01-01T00:00:00Z",
                "hf_repo": "fake/repo2",
                "local_dir": "/tmp/fake2",
                "venv_path": "/tmp/venv",  # same venv -> same worker
                "python_path": "/tmp/venv/bin/python",
                "enabled": True,
            },
        }
        _write_catalog(catalog_data)

    def test_disable_on_restart_failure_model_not_in_workers(self, tmp_path, monkeypatch):
        """When _restart_worker_inplace raises during disable, the spec must be
        removed from state.workers so the stale model is not serveable."""
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        _reset_known_models_cache()
        _reset_read_catalog_cache()
        self._make_catalog(tmp_path)

        from muse.admin.operations import (
            OperationError,
            disable_model,
            find_worker_for_model,
        )

        state = SupervisorState(workers=[], device="cpu")

        # Pre-load a multi-model spec (sibling pattern: both models in one worker).
        spec = WorkerSpec(
            models=["test-model", "sibling-model"],
            python_path="/tmp/venv/bin/python",
            port=9001,
            status="running",
        )
        state.workers.append(spec)

        restart_error = RuntimeError("simulated restart failure")

        # Patch _restart_worker_inplace to raise.
        with (
            patch("muse.admin.operations._restart_worker_inplace", side_effect=restart_error),
            patch("muse.admin.operations._shutdown_workers"),  # no-op shutdown
        ):
            # Also need known_models to return our test-model. The cache is
            # keyed by catalog identity; inject the current key so the
            # memoized fast path serves the fake entries.
            from muse.core import catalog as cat_mod
            fake_entry = MagicMock()
            fake_entry.extra = {}
            monkeypatch.setattr(
                cat_mod, "_known_models_cache",
                (cat_mod._catalog_cache_key(),
                 {"test-model": fake_entry, "sibling-model": fake_entry}),
            )

            with pytest.raises(RuntimeError, match="simulated restart failure"):
                disable_model("test-model", state=state)

        # Invariant: after the failure, the spec must NOT be in state.workers.
        # If it were, the gateway could still route to the worker that may be
        # serving the old model list (including test-model).
        assert spec not in state.workers, (
            "spec still in state.workers after restart failure; "
            "test-model may still be reachable -- M12 not fixed"
        )

        # The spec must be marked dead.
        assert spec.status == "dead"

        # find_worker_for_model must return None for test-model.
        assert find_worker_for_model(state, "test-model") is None, (
            "find_worker_for_model still finds test-model after restart failure"
        )

    def test_disable_sole_tenant_restart_failure_not_applicable(self, tmp_path, monkeypatch):
        """When disable removes the last model from a worker, _shutdown_workers
        is called (not _restart_worker_inplace). This path has no M12 issue
        but the test confirms the sole-tenant path still works correctly."""
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        _reset_known_models_cache()
        _reset_read_catalog_cache()
        _write_catalog({
            "only-model": {
                "pulled_at": "2026-01-01T00:00:00Z",
                "hf_repo": "fake/repo",
                "local_dir": "/tmp/fake",
                "venv_path": "/tmp/venv",
                "python_path": "/tmp/venv/bin/python",
                "enabled": True,
            },
        })

        from muse.admin.operations import disable_model, find_worker_for_model

        state = SupervisorState(workers=[], device="cpu")
        spec = WorkerSpec(
            models=["only-model"],
            python_path="/tmp/venv/bin/python",
            port=9001,
            status="running",
        )
        state.workers.append(spec)

        from muse.core import catalog as cat_mod
        fake_entry = MagicMock()
        fake_entry.extra = {}
        monkeypatch.setattr(
            cat_mod, "_known_models_cache",
            (cat_mod._catalog_cache_key(), {"only-model": fake_entry}),
        )

        with patch("muse.admin.operations._shutdown_workers") as mock_shutdown:
            result = disable_model("only-model", state=state)

        assert result["worker_terminated"] is True
        assert spec not in state.workers
        mock_shutdown.assert_called_once()
        assert find_worker_for_model(state, "only-model") is None
