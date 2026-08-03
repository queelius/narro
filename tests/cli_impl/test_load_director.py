"""Tests for LoadDirector: hot/cold acquire, refcount, singleton-load coordination,
on-demand LRU eviction, and observed-peak writeback.

Task B and Task C of the v0.40.0 lazy-load plan. Task B covers the three-phase
acquire (decide / load / commit). Task C adds the on-demand LRU eviction loop:
when a cold acquire does not fit, the director picks the least-recently-used
loaded model with refcount == 0, calls the injected disable_fn, polls the
memory probe for release, and retries the fit check. Task D adds observed-peak
writeback: every cold load measures `free_before - free_after` and writes the
delta back to `measurements.<device>.peak_bytes` if it exceeds the recorded
value; estimates self-heal upward toward reality.

Concurrency tests use threading.Barrier to make two sync threads enter the
decision phase as close to simultaneously as the GIL allows. Both threads
must see the same worker_port and the injected enable_fn must be called
exactly once.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from muse.admin.operations import OperationError
from muse.cli_impl.load_director import (
    DecisionLogEntry,
    InFlightLoad,
    LoadDirector,
    LoadEntry,
)
from muse.core.catalog import _read_catalog, _write_catalog


def _make_probe(gpu_free: float = 32.0, cpu_free: float = 64.0) -> MagicMock:
    """Default probe: lots of room. The 'fits' branch is exercised."""
    probe = MagicMock()
    probe.gpu_free_gb.return_value = gpu_free
    probe.cpu_free_gb.return_value = cpu_free
    return probe


def _manifest(memory_gb: float = 0.5, device: str = "cpu") -> dict:
    return {
        "model_id": "fake-model",
        "modality": "audio/speech",
        "capabilities": {
            "memory_gb": memory_gb,
            "device": device,
        },
    }


# ----------------------------------------------------------------------
# B3: hot path
# ----------------------------------------------------------------------

class TestHotAcquire:
    def test_returns_existing_port_increments_refcount(self):
        enable_fn = MagicMock(return_value=9001)
        disable_fn = MagicMock()
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=_make_probe(),
        )

        # Cold acquire (loads the model)
        port1 = director.acquire("fake-model", manifest=_manifest())
        assert port1 == 9001
        assert enable_fn.call_count == 1

        # Hot acquire returns the same port without calling enable_fn again
        port2 = director.acquire("fake-model", manifest=_manifest())
        assert port2 == 9001
        assert enable_fn.call_count == 1

        # Refcount incremented twice: once per acquire
        snapshot = director.status()
        assert snapshot["fake-model"]["refcount"] == 2

    def test_hot_acquire_updates_last_touched(self):
        enable_fn = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        first_touched = director.status()["fake-model"]["last_touched_at"]

        # Sleep enough that monotonic advances past time-resolution noise
        time.sleep(0.01)

        director.acquire("fake-model", manifest=_manifest())
        second_touched = director.status()["fake-model"]["last_touched_at"]

        assert second_touched > first_touched

    def test_hot_acquire_does_not_invoke_enable_fn(self):
        enable_fn = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        enable_fn.reset_mock()

        for _ in range(5):
            director.acquire("fake-model", manifest=_manifest())

        enable_fn.assert_not_called()


# ----------------------------------------------------------------------
# B4: cold path that fits
# ----------------------------------------------------------------------

class TestColdAcquireFits:
    def test_calls_enable_fn_once_returns_port(self):
        enable_fn = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        port = director.acquire("fake-model", manifest=_manifest())

        assert port == 9001
        enable_fn.assert_called_once_with("fake-model")

    def test_populates_load_entry_with_manifest_memory(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest(memory_gb=0.5))

        snapshot = director.status()
        assert "fake-model" in snapshot
        assert snapshot["fake-model"]["worker_port"] == 9001
        assert snapshot["fake-model"]["loaded"] is True
        assert snapshot["fake-model"]["refcount"] == 1

    def test_load_entry_has_loaded_at_and_last_touched_at(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        before = time.monotonic()
        director.acquire("fake-model", manifest=_manifest())
        after = time.monotonic()

        snapshot = director.status()
        loaded_at = snapshot["fake-model"]["last_touched_at"]
        # last_touched_at is monotonic seconds, recorded inside acquire
        assert before <= loaded_at <= after

    def test_default_budgets_and_headrooms_used_when_unset(self):
        # Sanity check: probe returning generous numbers + manifest
        # demanding 0.5 GB triggers the fits branch with all defaults.
        enable_fn = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )
        port = director.acquire("fake-model", manifest=_manifest())
        assert port == 9001


class TestCapacityInputValidation:
    @pytest.mark.parametrize(
        "memory_gb", [-1.0, float("nan"), float("inf"), "bad", True],
    )
    def test_invalid_manifest_memory_never_reaches_worker_or_eviction(
        self, memory_gb,
    ):
        enable = MagicMock(return_value=9001)
        disable = MagicMock()
        director = LoadDirector(
            enable_fn=enable, disable_fn=disable, memory_probe=_make_probe(),
        )

        with pytest.raises(OperationError) as exc_info:
            director.acquire(
                "invalid", manifest=_manifest(memory_gb=memory_gb),
            )

        assert exc_info.value.code == "invalid_model_manifest"
        assert exc_info.value.retryable is False
        enable.assert_not_called()
        disable.assert_not_called()
        assert director.in_flight_loads == {}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gpu_budget_gb": -1.0},
            {"cpu_budget_gb": float("nan")},
            {"gpu_headroom_gb": float("inf")},
            {"cpu_headroom_gb": True},
        ],
    )
    def test_invalid_capacity_knobs_fail_at_construction(self, kwargs):
        with pytest.raises(ValueError, match="finite non-negative"):
            LoadDirector(
                enable_fn=MagicMock(), disable_fn=MagicMock(),
                memory_probe=_make_probe(), **kwargs,
            )


# ----------------------------------------------------------------------
# C: on-demand LRU eviction
# ----------------------------------------------------------------------

class TestColdAcquireEvictsLRU:
    """Cold acquire on a tight budget evicts an LRU candidate via disable_fn,
    polls memory_probe for release, then loads the new model in the freed slot.
    """

    def test_evicts_one_lru_candidate_to_make_room(self):
        # Set up: probe whose CPU-free reading mutates as loads + evicts
        # happen, simulating live memory pressure. Initially the host has
        # 10 GB free; the victim load drops it to 4 GB; the eviction
        # disable_fn restores it to 9 GB.
        cpu_free_state = {"value": 10.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = 32.0
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        # enable_fn returns distinct ports per model and consumes memory.
        def enable_side_effect(model_id: str) -> int:
            if model_id == "victim":
                cpu_free_state["value"] = 4.0
                return 9001
            if model_id == "newcomer":
                return 9002
            raise AssertionError(f"unexpected enable: {model_id}")

        enable_fn = MagicMock(side_effect=enable_side_effect)

        # disable_fn flips the probe so the next poll shows freed memory.
        def disable_side_effect(model_id: str) -> None:
            cpu_free_state["value"] = 9.0

        disable_fn = MagicMock(side_effect=disable_side_effect)

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=2.0,
        )

        # Pre-load "victim" with a manifest claiming 5 GB. cpu_free=10,
        # headroom=2, available=8, fits. enable_fn drops cpu_free to 4.
        # After release, refcount goes to 0 (evictable).
        director.acquire("victim", manifest=_manifest(memory_gb=5.0, device="cpu"))
        director.release("victim")
        assert director.status()["victim"]["refcount"] == 0

        # Now cold-acquire "newcomer" at 6 GB. cpu_free=4, headroom=2,
        # available=2, shortfall=4. Director must evict "victim".
        # disable_fn restores cpu_free to 9; available=7; fits.
        port = director.acquire("newcomer", manifest=_manifest(memory_gb=6.0, device="cpu"))

        assert port == 9002
        disable_fn.assert_called_once_with("victim")
        snapshot = director.status()
        assert "newcomer" in snapshot
        assert "victim" not in snapshot

    def test_evicts_multiple_in_lru_order_until_fits(self):
        # Two evictable victims with different last_touched_at. Shortfall
        # larger than either alone forces both out, oldest first.
        cpu_free_state = {"value": 12.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = 32.0
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        evict_calls: list[str] = []

        # Each load consumes 4 GB; each evict releases 4 GB.
        def enable_side_effect(model_id: str) -> int:
            if model_id == "oldest":
                cpu_free_state["value"] -= 4.0
                return 9001
            if model_id == "middle":
                cpu_free_state["value"] -= 4.0
                return 9002
            if model_id == "newcomer":
                return 9003
            raise AssertionError(f"unexpected enable: {model_id}")

        enable_fn = MagicMock(side_effect=enable_side_effect)

        def disable_side_effect(model_id: str) -> None:
            evict_calls.append(model_id)
            cpu_free_state["value"] += 4.0

        disable_fn = MagicMock(side_effect=disable_side_effect)

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=1.0,
        )

        # Pre-load "oldest" first, then "middle". cpu_free goes 12 -> 8
        # -> 4 across the two loads. Touch them so monotonic timestamps
        # differ.
        director.acquire("oldest", manifest=_manifest(memory_gb=4.0, device="cpu"))
        director.release("oldest")
        time.sleep(0.01)
        director.acquire("middle", manifest=_manifest(memory_gb=4.0, device="cpu"))
        director.release("middle")

        # Now cpu_free=4, headroom=1, available=3. Cold-acquire newcomer
        # at 10 GB. shortfall=7. Evicting one frees 4 (cumulative 4 < 7);
        # second eviction brings cumulative to 8 >= 7; fits.
        port = director.acquire("newcomer", manifest=_manifest(memory_gb=10.0, device="cpu"))

        assert port == 9003
        # Both victims evicted, oldest first.
        assert evict_calls == ["oldest", "middle"]
        snapshot = director.status()
        assert "newcomer" in snapshot
        assert "oldest" not in snapshot
        assert "middle" not in snapshot

    def test_raises_503_when_no_evictable_candidates(self):
        # All loaded models have refcount > 0. The new acquire can't evict
        # anyone and must raise OperationError(model_too_large_for_device).
        probe = _make_probe(gpu_free=32.0, cpu_free=4.0)
        disable_fn = MagicMock()

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=2.0,
        )

        # Pre-load a model and DO NOT release it. refcount stays at 1.
        director.acquire("busy", manifest=_manifest(memory_gb=1.0, device="cpu"))
        assert director.status()["busy"]["refcount"] == 1

        with pytest.raises(OperationError) as exc_info:
            director.acquire("oversized", manifest=_manifest(memory_gb=100.0, device="cpu"))

        err = exc_info.value
        assert err.code == "model_too_large_for_device"
        assert err.status == 503
        # No eviction happened: refcount > 0 protected the only candidate.
        disable_fn.assert_not_called()
        # Busy model still loaded.
        assert "busy" in director.status()

    def test_skips_refcount_positive_candidates(self):
        # Mix of evictable + non-evictable candidates. Only the evictable
        # one is used for eviction; refcount > 0 model stays put.
        cpu_free_state = {"value": 8.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = 32.0
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        evict_calls: list[str] = []

        def enable_side_effect(model_id: str) -> int:
            if model_id == "busy":
                cpu_free_state["value"] -= 2.0
                return 9001
            if model_id == "evictable":
                cpu_free_state["value"] -= 2.0
                return 9002
            if model_id == "newcomer":
                return 9003
            raise AssertionError(f"unexpected enable: {model_id}")

        enable_fn = MagicMock(side_effect=enable_side_effect)

        def disable_side_effect(model_id: str) -> None:
            evict_calls.append(model_id)
            cpu_free_state["value"] += 2.0

        disable_fn = MagicMock(side_effect=disable_side_effect)

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=1.0,
        )

        # Load "busy" first (older), keep refcount at 1. cpu_free 8 -> 6.
        director.acquire("busy", manifest=_manifest(memory_gb=2.0, device="cpu"))
        time.sleep(0.01)
        # Load "evictable" newer, then release: refcount = 0. cpu_free 6 -> 4.
        director.acquire("evictable", manifest=_manifest(memory_gb=2.0, device="cpu"))
        director.release("evictable")

        # cpu_free=4, headroom=1, available=3. Cold-acquire newcomer at
        # 4 GB. shortfall=1. Evicting "evictable" frees 2; fits. "busy"
        # stays put even though it is older, because refcount > 0.
        port = director.acquire("newcomer", manifest=_manifest(memory_gb=4.0, device="cpu"))

        assert port == 9003
        assert evict_calls == ["evictable"]
        snapshot = director.status()
        assert "busy" in snapshot
        assert snapshot["busy"]["refcount"] == 1
        assert "evictable" not in snapshot

    def test_memory_release_polling_converges(self):
        # gpu_free_gb returns rising values across calls, simulating
        # asynchronous SIGTERM -> VRAM-release on the GPU driver.
        # The eviction loop's _wait_for_memory_release should converge
        # once enough memory frees up.
        #
        # Sequence: pre-load reads 10 (fits), 9 (post-load capture), then
        # decide for newcomer reads 2 (oversize), then the eviction
        # snapshot reads 2 (free_before of eviction), then poll reads
        # rising values (2, 2, 5, 5...). Use an iterator with a long tail
        # of 5.0 so subsequent reads keep showing freed memory.
        gpu_free_seq = iter([10.0, 9.0, 2.0, 2.0, 2.0, 2.0, 5.0] + [5.0] * 50)
        probe = MagicMock()
        probe.gpu_free_gb.side_effect = lambda: next(gpu_free_seq)
        probe.cpu_free_gb.return_value = 64.0

        disable_fn = MagicMock()
        enable_results = {"victim": 9001, "newcomer": 9002}
        enable_fn = MagicMock(side_effect=lambda mid: enable_results[mid])

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            gpu_headroom_gb=0.5,
        )

        # Pre-load victim. gpu_free=10, headroom=0.5, available=9.5, fits.
        director.acquire("victim", manifest=_manifest(memory_gb=1.0, device="cuda"))
        director.release("victim")

        # Cold-acquire newcomer at 4 GB on cuda.
        # Decision-phase reads gpu_free_gb (2.0), available=1.5, shortfall=2.5.
        # Evict victim -> disable_fn called -> poll loop consumes more
        # iterations until rises to 5.0; freed = 5.0 - 2.0 = 3.0 >= 2.5;
        # converges; retry decide; available=4.5; fits.
        port = director.acquire("newcomer", manifest=_manifest(memory_gb=4.0, device="cuda"))
        assert port == 9002
        disable_fn.assert_called_once_with("victim")

    def test_eviction_records_decision_log_entries(self):
        # Each eviction produces a DecisionLogEntry with action="evict".
        cpu_free_state = {"value": 10.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = 32.0
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        def enable_side_effect(model_id: str) -> int:
            if model_id == "victim":
                cpu_free_state["value"] = 4.0
                return 9001
            if model_id == "newcomer":
                return 9002
            raise AssertionError(f"unexpected enable: {model_id}")

        enable_fn = MagicMock(side_effect=enable_side_effect)

        def disable_side_effect(model_id: str) -> None:
            cpu_free_state["value"] = 9.0

        disable_fn = MagicMock(side_effect=disable_side_effect)

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=2.0,
        )

        director.acquire("victim", manifest=_manifest(memory_gb=5.0, device="cpu"))
        director.release("victim")

        director.acquire("newcomer", manifest=_manifest(memory_gb=6.0, device="cpu"))

        # The recent_decisions deque must have an "evict" entry for victim
        # plus "load" entries for both models.
        decisions = list(director.recent_decisions)
        evict_entries = [d for d in decisions if d.action == "evict"]
        assert len(evict_entries) == 1

        e = evict_entries[0]
        assert e.model_id == "victim"
        assert e.evicted == ["victim"]
        assert e.reason == "evicted_for_newcomer"
        # memory_gb on the evict entry is the victim's memory, not the
        # incoming model's.
        assert e.memory_gb == 5.0
        # free_after_gb populated once the post-eviction poll completed.
        assert e.free_after_gb is not None
        assert e.free_before_gb <= e.free_after_gb

    def test_no_in_flight_event_after_503_oversize(self):
        # Even when eviction fails to find candidates, the in-flight event
        # for the requested model must not be left behind.
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(gpu_free=4.0, cpu_free=8.0),
        )

        with pytest.raises(OperationError):
            director.acquire("huge", manifest=_manifest(memory_gb=100.0, device="cpu"))

        # No stranded Event keeps a future acquire from progressing.
        assert director.in_flight_loads == {}

    def test_disable_fn_failure_reinserts_victim_in_loaded_set(self):
        # If disable_fn raises while evicting an LRU victim, the director
        # must NOT count that victim's memory_gb toward cumulative_freed_gb
        # (the worker is still alive holding memory). The victim's
        # LoadEntry must be re-inserted into self.loaded so the directory's
        # accounting stays consistent with reality. The eviction loop
        # then proceeds to the next candidate.
        cpu_free_state = {"value": 12.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = 32.0
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        # Two evictable victims. First disable_fn call raises (worker
        # stuck); second succeeds. The director must skip the failed
        # victim and evict the next one.
        def enable_side_effect(model_id: str) -> int:
            if model_id == "stuck":
                cpu_free_state["value"] -= 4.0
                return 9001
            if model_id == "good":
                cpu_free_state["value"] -= 4.0
                return 9002
            if model_id == "newcomer":
                return 9003
            raise AssertionError(f"unexpected enable: {model_id}")

        enable_fn = MagicMock(side_effect=enable_side_effect)

        # disable_fn fails on the first call ("stuck"), succeeds on
        # the second ("good") and frees 4 GB.
        def disable_side_effect(model_id: str) -> None:
            if model_id == "stuck":
                raise RuntimeError("worker stuck")
            cpu_free_state["value"] += 4.0

        disable_fn = MagicMock(side_effect=disable_side_effect)

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=1.0,
        )

        # Pre-load "stuck" first (older), release. cpu_free 12 -> 8.
        director.acquire("stuck", manifest=_manifest(memory_gb=4.0, device="cpu"))
        director.release("stuck")
        assert director.status()["stuck"]["refcount"] == 0
        time.sleep(0.01)
        # Pre-load "good" newer, release. cpu_free 8 -> 4.
        director.acquire("good", manifest=_manifest(memory_gb=4.0, device="cpu"))
        director.release("good")
        assert director.status()["good"]["refcount"] == 0

        # Now cpu_free=4, headroom=1, available=3. Cold-acquire newcomer
        # at 6 GB. shortfall=3. LRU is "stuck"; disable_fn raises;
        # director must re-insert "stuck" and proceed to "good".
        # disable_fn("good") frees 4 GB; cumulative_freed=4 >= 3; fits.
        port = director.acquire("newcomer", manifest=_manifest(memory_gb=6.0, device="cpu"))

        assert port == 9003
        # Both disable_fn invocations happened, in LRU order.
        assert disable_fn.call_args_list[0][0] == ("stuck",)
        assert disable_fn.call_args_list[1][0] == ("good",)
        assert disable_fn.call_count == 2

        snapshot = director.status()
        # Newcomer loaded.
        assert "newcomer" in snapshot
        # The first victim ("stuck") was re-inserted because disable_fn
        # raised. Its refcount was preserved at 0.
        assert "stuck" in snapshot
        assert snapshot["stuck"]["refcount"] == 0
        # The second victim ("good") was successfully evicted.
        assert "good" not in snapshot

        # The decision log records the failure with action="evict" and
        # a "disable_fn_raised" reason.
        decisions = list(director.recent_decisions)
        raised_entries = [
            d for d in decisions
            if d.action == "evict" and d.reason.startswith("disable_fn_raised")
        ]
        assert len(raised_entries) == 1
        e = raised_entries[0]
        assert e.model_id == "stuck"
        assert e.evicted == []
        assert e.memory_gb == 4.0
        # free_after_gb is None because the poll never happened.
        assert e.free_after_gb is None

    def test_eviction_does_not_block_concurrent_hot_acquire(self):
        # Central correctness claim of the eviction design: while a
        # cold-acquire thread is inside _evict_lru_until_fits running
        # the slow disable_fn + poll, OTHER threads can still hot-acquire
        # already-loaded models without serializing on the eviction.
        #
        # This test pre-loads X (LRU) and Y. Thread A cold-acquires Z;
        # its disable_fn(X) blocks on an event so we can synchronize.
        # Thread B hot-acquires Y. B must complete quickly (well under
        # the 1-second disable_fn delay), proving the lock is released
        # during eviction.

        # Probe state: starts with 12 GB, drops 4 per load, restores
        # 4 per disable_fn.
        cpu_free_state = {"value": 12.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = 32.0
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        def enable_side_effect(model_id: str) -> int:
            if model_id == "X":
                cpu_free_state["value"] -= 4.0
                return 9001
            if model_id == "Y":
                cpu_free_state["value"] -= 4.0
                return 9002
            if model_id == "Z":
                return 9003
            raise AssertionError(f"unexpected enable: {model_id}")

        enable_fn = MagicMock(side_effect=enable_side_effect)

        # disable_fn waits on eviction_block (up to 1 second). During
        # that wait, another thread (B) must be able to hot-acquire Y.
        eviction_in_progress = threading.Event()
        eviction_block = threading.Event()

        def disable_side_effect(model_id: str) -> None:
            assert model_id == "X"
            eviction_in_progress.set()
            eviction_block.wait(timeout=1.0)
            cpu_free_state["value"] += 4.0

        disable_fn = MagicMock(side_effect=disable_side_effect)

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=1.0,
        )

        # Pre-load X first (older), then Y. Both refcount=0 after release
        # so both are evictable; X is LRU.
        director.acquire("X", manifest=_manifest(memory_gb=4.0, device="cpu"))
        director.release("X")
        time.sleep(0.01)
        # Y will stay loaded after thread B re-acquires it; we release
        # it for now so the LRU pick during A's eviction picks X.
        director.acquire("Y", manifest=_manifest(memory_gb=4.0, device="cpu"))
        director.release("Y")
        # cpu_free=4, headroom=1, available=3.

        # Synchronize thread A's "I'm in eviction now" via eviction_in_progress.

        a_results: list[int] = []
        a_errors: list[BaseException] = []
        b_results: list[int] = []
        b_errors: list[BaseException] = []
        b_elapsed: list[float] = []

        def thread_a():
            try:
                # Cold-acquire Z at 6 GB. cpu_free=4, available=3,
                # shortfall=3. Director enters _evict_lru_until_fits;
                # disable_fn(X) blocks until eviction_block is set.
                p = director.acquire(
                    "Z", manifest=_manifest(memory_gb=6.0, device="cpu"),
                )
                a_results.append(p)
            except BaseException as exc:  # noqa: BLE001
                a_errors.append(exc)

        def thread_b():
            try:
                # Wait for A to be inside disable_fn before starting.
                assert eviction_in_progress.wait(timeout=2.0)
                start = time.monotonic()
                # Hot-acquire Y. Y is already loaded; this should NOT
                # serialize on A's eviction lock.
                p = director.acquire(
                    "Y", manifest=_manifest(memory_gb=4.0, device="cpu"),
                )
                elapsed = time.monotonic() - start
                b_results.append(p)
                b_elapsed.append(elapsed)
            except BaseException as exc:  # noqa: BLE001
                b_errors.append(exc)

        a = threading.Thread(target=thread_a)
        b = threading.Thread(target=thread_b)
        a.start()
        b.start()

        # B is waiting on eviction_in_progress. Once it sees that, it
        # times its hot-acquire and we expect the call to return well
        # under 200ms despite A's disable_fn delay.
        b.join(timeout=3.0)

        # Now release A's eviction_block so A can finish.
        eviction_block.set()
        a.join(timeout=5.0)

        assert not a_errors, a_errors
        assert not b_errors, b_errors
        # B got Y's worker_port without waiting for A.
        assert b_results == [9002]
        # B's elapsed time well under A's 1s disable_fn delay; we use
        # 200ms as a generous bound that still proves the point.
        assert b_elapsed[0] < 0.2, (
            f"hot-acquire serialized on eviction; took {b_elapsed[0]:.3f}s"
        )
        # A eventually completes too.
        assert a_results == [9003]

    def test_evict_rechecks_live_fit_before_503_when_no_candidates(self):
        """M5: two concurrent evict-needing acquires for the same model both
        enter _evict_lru_until_fits (neither claims an in_flight slot). The
        loser finds the single LRU victim already evicted by the winner and
        hits 'no candidates', but the winner's eviction already freed enough
        memory. It must re-check live availability and return (letting
        acquire re-decide -> load), not 503 spuriously.

        Modeled directly: no loaded candidates, but the live pool now fits
        the model (cpu_free=10, headroom=1 -> available 9 >= required 4)."""
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(cpu_free=10.0),
            cpu_headroom_gb=1.0,
        )
        # Must NOT raise: the model fits against live memory now.
        director._evict_lru_until_fits(
            model_id="newcomer",
            shortfall_gb=4.0,
            device="cpu",
            required_gb=4.0,
        )
        assert director.in_flight_loads == {}

    def test_evict_still_503s_when_no_candidates_and_still_too_large(self):
        """The re-check must NOT mask a genuine oversize: no candidates AND
        the model still does not fit live memory -> 503 (else acquire would
        spin re-deciding forever)."""
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(cpu_free=2.0),
            cpu_headroom_gb=1.0,
        )
        with pytest.raises(OperationError):
            director._evict_lru_until_fits(
                model_id="huge",
                shortfall_gb=98.0,
                device="cpu",
                required_gb=100.0,
            )
        assert director.in_flight_loads == {}


# ----------------------------------------------------------------------
# release: refcount decrement, no auto-evict
# ----------------------------------------------------------------------

class TestRelease:
    def test_decrements_refcount(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        director.acquire("fake-model", manifest=_manifest())
        assert director.status()["fake-model"]["refcount"] == 2

        director.release("fake-model")
        assert director.status()["fake-model"]["refcount"] == 1

        director.release("fake-model")
        assert director.status()["fake-model"]["refcount"] == 0

    def test_does_not_call_disable_fn_on_release(self):
        # v0.40.0 Task B: release does not auto-evict. Eviction is on-demand
        # only (Task C, triggered from a *cold* acquire that needs room).
        disable_fn = MagicMock()
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=disable_fn,
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        director.release("fake-model")

        disable_fn.assert_not_called()

    def test_release_updates_last_touched(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        first = director.status()["fake-model"]["last_touched_at"]

        time.sleep(0.01)
        director.release("fake-model")
        second = director.status()["fake-model"]["last_touched_at"]
        assert second > first

    def test_release_unknown_model_is_noop(self):
        # Defensive: a release for a model that's no longer in the loaded
        # set (e.g. evicted between acquire and release on another thread)
        # should not crash.
        director = LoadDirector(
            enable_fn=MagicMock(),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )
        # Should not raise.
        director.release("never-loaded")


# ----------------------------------------------------------------------
# B6: singleton load coordination across threads
# ----------------------------------------------------------------------

class TestConcurrentAcquireSameModel:
    def test_two_threads_share_one_load(self):
        # Two threads call acquire(X) where X is unloaded. They MUST share
        # one enable_fn invocation; the loser awaits the winner's Event
        # and reads the freshly populated loaded entry.
        barrier = threading.Barrier(2)
        slow_load_started = threading.Event()
        slow_load_release = threading.Event()

        def slow_enable(model_id: str) -> int:
            slow_load_started.set()
            # Hold inside the load phase (lock NOT held) long enough for
            # the second thread to enter the decision phase, see the
            # in-flight event, and start awaiting.
            slow_load_release.wait(timeout=5.0)
            return 9001

        director = LoadDirector(
            enable_fn=slow_enable,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        results: list[int] = []
        errors: list[BaseException] = []

        def worker():
            try:
                barrier.wait(timeout=5.0)
                p = director.acquire("fake-model", manifest=_manifest())
                results.append(p)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()

        # Wait for the first thread to actually begin enable_fn so we
        # know it has surrendered the lock and the second thread is
        # blocked on the Event (not still racing in the decision phase).
        assert slow_load_started.wait(timeout=5.0)
        # Tiny pause so the second thread has time to reach the wait.
        time.sleep(0.05)
        slow_load_release.set()

        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert not errors, errors
        assert results == [9001, 9001], f"both threads must see the same port; got {results}"

    def test_enable_fn_called_exactly_once_under_contention(self):
        barrier = threading.Barrier(2)
        slow_release = threading.Event()
        call_count = {"n": 0}
        count_lock = threading.Lock()

        def slow_enable(model_id: str) -> int:
            with count_lock:
                call_count["n"] += 1
            slow_release.wait(timeout=5.0)
            return 9001

        director = LoadDirector(
            enable_fn=slow_enable,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        def worker():
            barrier.wait(timeout=5.0)
            director.acquire("fake-model", manifest=_manifest())

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()

        # Give both threads a chance to reach the decision phase. The
        # winner reaches the load phase and increments call_count; the
        # loser blocks on the Event. Releasing now lets the winner
        # complete; the loser then re-enters the decision phase and
        # sees a hot entry, so it does NOT call enable_fn.
        time.sleep(0.1)
        slow_release.set()

        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert call_count["n"] == 1, f"enable_fn was called {call_count['n']} times; expected 1"

        # Refcount must be 2: each acquire bumps it.
        assert director.status()["fake-model"]["refcount"] == 2


# ----------------------------------------------------------------------
# DecisionLogEntry recorded on every load
# ----------------------------------------------------------------------

class TestDecisionLog:
    def test_decision_recorded_on_load(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest(memory_gb=0.5))

        # recent_decisions exposes the same deque the admin endpoint reads
        decisions = list(director.recent_decisions)
        assert len(decisions) == 1

        d = decisions[0]
        assert isinstance(d, DecisionLogEntry)
        assert d.model_id == "fake-model"
        assert d.action == "load"
        assert d.memory_gb == 0.5
        assert d.evicted == []
        assert d.reason  # free-form, but must be non-empty

    def test_decision_has_free_before_and_after(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(gpu_free=32.0, cpu_free=64.0),
        )

        director.acquire("fake-model", manifest=_manifest(memory_gb=0.5, device="cpu"))

        d = list(director.recent_decisions)[0]
        # CPU model: free_before should reflect cpu_free_gb
        assert d.free_before_gb == 64.0
        # free_after_gb is captured after the load returns; with the
        # MagicMock probe it stays at 64.0
        assert d.free_after_gb == 64.0

    def test_recent_decisions_capped_at_20(self):
        enable_fn = MagicMock(side_effect=lambda mid: 9000 + hash(mid) % 1000)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        for i in range(25):
            director.acquire(f"model-{i}", manifest=_manifest())

        # deque(maxlen=20) caps the recent log at 20 entries
        assert len(director.recent_decisions) == 20

    def test_no_decision_logged_on_hot_acquire(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())  # cold; logs
        director.acquire("fake-model", manifest=_manifest())  # hot; no log

        assert len(director.recent_decisions) == 1


# ----------------------------------------------------------------------
# Exception during enable_fn: cleanup + waiter wake
# ----------------------------------------------------------------------

class TestEnableFnException:
    def test_preload_probe_failure_aborts_claim_before_enable(self):
        failure = RuntimeError("memory probe temporarily unavailable")
        probe = _make_probe()
        reads = iter((64.0, failure, 64.0, 64.0, 64.0))

        def cpu_free():
            value = next(reads)
            if isinstance(value, BaseException):
                raise value
            return value

        probe.cpu_free_gb.side_effect = cpu_free
        enable = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable,
            disable_fn=MagicMock(),
            memory_probe=probe,
        )

        with pytest.raises(OperationError, match="memory probe") as caught:
            director.acquire("fake-model", manifest=_manifest())

        assert caught.value.__cause__ is failure
        assert director.in_flight_loads == {}
        enable.assert_not_called()
        assert director.acquire("fake-model", manifest=_manifest()) == 9001

    def test_exception_propagates_in_flight_cleared(self):
        boom = RuntimeError("worker spawn died")

        def failing_enable(model_id: str) -> int:
            raise boom

        director = LoadDirector(
            enable_fn=failing_enable,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        # v0.50.1: unexpected load failures surface as OperationError(503)
        # (envelope-able) with the original exception chained as __cause__.
        with pytest.raises(OperationError, match="worker spawn died") as ei:
            director.acquire("fake-model", manifest=_manifest())
        assert ei.value.status == 503
        assert ei.value.__cause__ is boom

        # in_flight_loads must be empty so the next acquire isn't
        # blocked indefinitely waiting on a dead Event.
        assert director.in_flight_loads == {}
        # And no LoadEntry was inserted on failure.
        assert "fake-model" not in director.status()

    def test_failed_rollback_stays_blocked_and_accounted_until_disable(self):
        cleanup_failure = RuntimeError("worker teardown uncertain")
        director = LoadDirector(
            enable_fn=MagicMock(return_value=0),
            disable_fn=MagicMock(side_effect=cleanup_failure),
            memory_probe=_make_probe(),
        )

        with pytest.raises(OperationError, match="invalid worker port"):
            director.acquire("fake-model", manifest=_manifest(memory_gb=2.0))

        assert director.in_flight_loads == {}
        stranded = director.loaded["fake-model"]
        assert stranded.worker_port == 0
        assert stranded.refcount == 0
        assert stranded.memory_gb == 2.0
        assert "fake-model" in director._blocked_models
        assert "fake-model" in director._failed_rollbacks

        with pytest.raises(OperationError) as caught:
            director.acquire("fake-model", manifest=_manifest(memory_gb=2.0))
        assert caught.value.code == "model_disabled"
        with pytest.raises(OperationError) as caught:
            director.allow_model("fake-model")
        assert caught.value.code == "model_cleanup_required"

        claim = director.begin_model_disable("fake-model")
        assert claim.entry is stranded
        assert director.finish_eviction(claim, success=True)
        assert "fake-model" not in director.loaded
        assert "fake-model" not in director._failed_rollbacks
        director.allow_model("fake-model")
        assert "fake-model" not in director._blocked_models

    def test_concurrent_waiter_wakes_and_reattempts(self):
        # Two threads racing on a cold acquire: the winner's enable_fn
        # raises. The loser was awaiting the winner's Event; on wake
        # it must re-enter the decision phase. With enable_fn now
        # patched to succeed, the loser should win the second round.
        attempt_count = {"n": 0}
        attempt_lock = threading.Lock()
        first_started = threading.Event()
        first_release = threading.Event()

        def enable_fn(model_id: str) -> int:
            with attempt_lock:
                attempt_count["n"] += 1
                this_attempt = attempt_count["n"]
            if this_attempt == 1:
                # The first attempt waits for our signal, then raises.
                # This holds the lock-free "load phase" open long enough
                # for the second thread to await on the in-flight Event.
                first_started.set()
                first_release.wait(timeout=5.0)
                raise RuntimeError("first attempt fails")
            # Subsequent attempts succeed.
            return 9001

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        results: list[object] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def worker():
            try:
                barrier.wait(timeout=5.0)
                results.append(director.acquire("fake-model", manifest=_manifest()))
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()

        assert first_started.wait(timeout=5.0)
        # Brief pause so the second thread reaches the Event-await.
        time.sleep(0.05)
        first_release.set()

        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        # One thread (the winner) raised. The other thread saw an empty
        # entry on wakeup, re-decided, and successfully loaded. v0.50.1:
        # the failure arrives wrapped as OperationError(503) with the
        # RuntimeError chained as its cause.
        assert len(errors) == 1
        assert isinstance(errors[0], OperationError)
        assert isinstance(errors[0].__cause__, RuntimeError)
        assert results == [9001]
        # enable_fn was called twice: once for the failed attempt, once
        # for the retry by the woken waiter.
        assert attempt_count["n"] == 2


class TestInFlightLeaseRecovery:
    def test_abandoned_claim_is_reclaimed_after_bounded_wait(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )
        abandoned = InFlightLoad(
            event=threading.Event(),
            memory_gb=0.5,
            pool="cpu",
            claimed_at=time.monotonic() - 10.0,
            owner_thread=None,
        )
        director.in_flight_loads["fake-model"] = abandoned

        with patch(
            "muse.cli_impl.load_director._INFLIGHT_WAIT_TIMEOUT_SECONDS",
            0.0,
        ):
            assert director.acquire("fake-model", manifest=_manifest()) == 9001

        assert abandoned.event.is_set()
        assert director.in_flight_loads == {}
        director.enable_fn.assert_called_once_with("fake-model")

    def test_live_owner_timeout_is_retryable_and_never_duplicates_load(self):
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )
        active = InFlightLoad(
            event=threading.Event(),
            memory_gb=0.5,
            pool="cpu",
            claimed_at=time.monotonic() - 10.0,
            owner_thread=threading.current_thread(),
        )
        director.in_flight_loads["fake-model"] = active

        with patch(
            "muse.cli_impl.load_director._INFLIGHT_WAIT_TIMEOUT_SECONDS",
            0.0,
        ), pytest.raises(OperationError) as exc:
            director.acquire("fake-model", manifest=_manifest())

        assert exc.value.code == "model_load_timeout"
        assert exc.value.retryable is True
        assert director.in_flight_loads["fake-model"] is active
        director.enable_fn.assert_not_called()


class TestPostEnableSettlement:
    def test_observation_failure_keeps_successful_load_usable(self):
        probe = _make_probe()
        probe.cpu_free_gb.side_effect = [64.0, 64.0, RuntimeError("probe lost")]
        disable = MagicMock()
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=disable,
            memory_probe=probe,
        )

        assert director.acquire("fake-model", manifest=_manifest()) == 9001

        assert director.loaded["fake-model"].worker_port == 9001
        assert director.in_flight_loads == {}
        assert director.recent_decisions[-1].free_after_gb is None
        disable.assert_not_called()

    def test_commit_failure_rolls_back_enabled_worker_and_claim(self):
        class ExplodingDecisions:
            def append(self, _entry):
                raise RuntimeError("decision sink failed")

        disable = MagicMock()
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=disable,
            memory_probe=_make_probe(),
        )
        director.recent_decisions = ExplodingDecisions()

        with pytest.raises(OperationError, match="decision sink failed"):
            director.acquire("fake-model", manifest=_manifest())

        disable.assert_called_once_with("fake-model")
        assert director.in_flight_loads == {}
        assert "fake-model" not in director.loaded

    @pytest.mark.parametrize("bad_port", [None, True, 0, 65536, "9001"])
    def test_invalid_enable_port_rolls_back(self, bad_port):
        disable = MagicMock()
        director = LoadDirector(
            enable_fn=MagicMock(return_value=bad_port),
            disable_fn=disable,
            memory_probe=_make_probe(),
        )

        with pytest.raises(OperationError, match="invalid worker port"):
            director.acquire("fake-model", manifest=_manifest())

        disable.assert_called_once_with("fake-model")
        assert director.in_flight_loads == {}
        assert director.loaded == {}

    def test_writeback_thread_start_failure_does_not_fail_loaded_request(self):
        probe = _make_probe()
        probe.cpu_free_gb.side_effect = [64.0, 64.0, 63.0]
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
        )
        fake_thread = MagicMock()
        fake_thread.start.side_effect = RuntimeError("no thread capacity")

        with patch(
            "muse.cli_impl.load_director.threading.Thread",
            return_value=fake_thread,
        ):
            assert director.acquire("fake-model", manifest=_manifest()) == 9001

        assert director.loaded["fake-model"].worker_port == 9001
        assert director.in_flight_loads == {}


# ----------------------------------------------------------------------
# LoadEntry shape sanity (the dataclass is part of the public API)
# ----------------------------------------------------------------------

class TestLoadEntryShape:
    def test_fields_present(self):
        e = LoadEntry(
            model_id="x",
            worker_port=9001,
            memory_gb=0.5,
            refcount=0,
            last_touched_at=time.monotonic(),
            loaded_at=time.monotonic(),
        )
        assert e.model_id == "x"
        assert e.worker_port == 9001
        assert e.memory_gb == 0.5
        assert e.refcount == 0
        assert e.last_touched_at > 0
        assert e.loaded_at > 0


class TestDecisionLogEntryShape:
    def test_fields_present(self):
        d = DecisionLogEntry(
            timestamp=time.time(),
            model_id="x",
            action="load",
            memory_gb=0.5,
            free_before_gb=10.0,
            free_after_gb=9.5,
            reason="fit",
            evicted=[],
        )
        assert d.model_id == "x"
        assert d.action == "load"
        assert d.evicted == []


# ----------------------------------------------------------------------
# Task D: observed-peak writeback
# ----------------------------------------------------------------------


def test_observed_peak_coalesces_while_single_drain_thread_is_busy():
    director = LoadDirector(
        enable_fn=MagicMock(return_value=9001),
        disable_fn=MagicMock(),
        memory_probe=_make_probe(),
    )
    entered = threading.Event()
    release = threading.Event()
    calls: list[tuple[str, int, str]] = []

    def blocked_writeback(model_id, observed_peak_bytes, device_key):
        calls.append((model_id, observed_peak_bytes, device_key))
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=5.0)

    director._observed_peak_writeback = blocked_writeback
    thread = director.observed_peak(
        "fake-model",
        observed_peak_bytes=1,
        device="cpu",
    )
    assert thread is not None
    assert entered.wait(timeout=5.0)

    returned = [
        director.observed_peak(
            "fake-model",
            observed_peak_bytes=value,
            device="cpu",
        )
        for value in range(2, 102)
    ]
    assert all(candidate is thread for candidate in returned)
    assert len(director._pending_writebacks) == 1

    release.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert calls == [
        ("fake-model", 1, "cpu"),
        ("fake-model", 101, "cpu"),
    ]
    assert director._pending_writebacks == {}
    assert director._writeback_thread is None

@pytest.fixture
def catalog_dir(tmp_path, monkeypatch):
    """Scope the muse catalog file to a temp dir for the test.

    `_read_catalog` / `_write_catalog` consult the `MUSE_CATALOG_DIR`
    env var (falling back to `~/.muse`); we redirect to tmp_path so each
    test gets a clean catalog.json without touching the user's real one.
    """
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    return tmp_path


def _seed_catalog(model_id: str, *, device: str, peak_bytes: int | None) -> None:
    """Write a minimal catalog.json with the given measurement seed.

    `peak_bytes=None` writes the catalog entry but no `measurements.<device>`
    record at all (simulates first cold load before any probe ran).
    """
    catalog = _read_catalog()
    entry = catalog.setdefault(model_id, {})
    entry["enabled"] = True
    if peak_bytes is not None:
        entry.setdefault("measurements", {})
        entry["measurements"][device] = {
            "peak_bytes": peak_bytes,
            "weights_bytes": peak_bytes,
            "device": device,
        }
    _write_catalog(catalog)


class TestObservedPeakWriteback:
    """observed_peak fires a fire-and-forget thread that monotonically
    raises `measurements.<device>.peak_bytes` toward reality.

    The director swallows IO errors so a corrupted catalog or transient
    filesystem failure does not surface as a 500 to the user; this test
    suite verifies the full happy-path + the swallow-and-log paths.
    """

    def test_writes_back_when_observed_exceeds_recorded(self, catalog_dir):
        # Recorded peak: 1 GB. Observed: 2 GB. Writeback must replace it.
        _seed_catalog("fake-model", device="cuda", peak_bytes=1 * 1024**3)

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        thread = director.observed_peak(
            "fake-model",
            observed_peak_bytes=2 * 1024**3,
            device="cuda",
        )
        assert thread is not None
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "writeback thread did not complete in 2s"

        catalog = _read_catalog()
        m = catalog["fake-model"]["measurements"]["cuda"]
        assert m["peak_bytes"] == 2 * 1024**3
        # observed_at + source must be set on this writeback path.
        assert m["source"] == "lazy_load_observation"
        assert "observed_at" in m
        # ISO 8601 UTC sanity (e.g. "2026-05-05T..."): contains "T" + ends "+00:00" or "Z".
        assert "T" in m["observed_at"]

    def test_does_not_write_back_when_observed_lte_recorded(self, catalog_dir):
        # Estimate is monotonically upward only; a smaller observed value
        # must NOT erase a larger recorded one.
        _seed_catalog("fake-model", device="cuda", peak_bytes=5 * 1024**3)

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        thread = director.observed_peak(
            "fake-model",
            observed_peak_bytes=2 * 1024**3,
            device="cuda",
        )
        if thread is not None:
            thread.join(timeout=2.0)

        catalog = _read_catalog()
        m = catalog["fake-model"]["measurements"]["cuda"]
        assert m["peak_bytes"] == 5 * 1024**3
        # No writeback happened: source field stays absent (the seed didn't set one).
        assert "source" not in m
        assert "observed_at" not in m

    def test_writes_back_when_no_recorded_value_exists(self, catalog_dir):
        # First cold load: catalog entry exists but has no `measurements`
        # block at all. Writeback must create the bucket and populate it.
        _seed_catalog("fake-model", device="cuda", peak_bytes=None)

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        thread = director.observed_peak(
            "fake-model",
            observed_peak_bytes=3 * 1024**3,
            device="cuda",
        )
        assert thread is not None
        thread.join(timeout=2.0)

        catalog = _read_catalog()
        m = catalog["fake-model"]["measurements"]["cuda"]
        assert m["peak_bytes"] == 3 * 1024**3
        assert m["source"] == "lazy_load_observation"
        assert "observed_at" in m

    def test_swallows_ioerror_during_write(self, catalog_dir, monkeypatch, caplog):
        # If the catalog write fails (disk full, permissions, etc.), the
        # writeback path must log a warning and swallow the error so the
        # request hot path is unaffected. The caller has no observable
        # failure: thread completes cleanly.
        _seed_catalog("fake-model", device="cuda", peak_bytes=1 * 1024**3)

        # Patch the writeback's _write_catalog reference (LoadDirector
        # imports it locally inside the thread so we patch by module path).
        import muse.cli_impl.load_director as ld_mod

        def boom(_data: dict) -> None:
            raise IOError("disk full")

        monkeypatch.setattr(ld_mod, "_write_catalog", boom)

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        with caplog.at_level("WARNING", logger="muse.cli_impl.load_director"):
            thread = director.observed_peak(
                "fake-model",
                observed_peak_bytes=10 * 1024**3,
                device="cuda",
            )
            assert thread is not None
            thread.join(timeout=2.0)
            assert not thread.is_alive()

        # The seed value must remain because the write was rejected.
        catalog = _read_catalog()
        assert catalog["fake-model"]["measurements"]["cuda"]["peak_bytes"] == 1 * 1024**3
        # And the logger captured the failure (admin can debug it later).
        warned = any(
            "observed_peak" in rec.getMessage().lower()
            or "writeback" in rec.getMessage().lower()
            or "disk full" in rec.getMessage().lower()
            for rec in caplog.records
            if rec.levelname == "WARNING"
        )
        assert warned, "expected a WARNING log when writeback fails"

    def test_swallows_oserror_during_read(self, catalog_dir, monkeypatch, caplog):
        # If the catalog read fails, the writeback path must also swallow
        # the error gracefully. Same contract as IO during write.
        _seed_catalog("fake-model", device="cuda", peak_bytes=1 * 1024**3)

        import muse.cli_impl.load_director as ld_mod

        def boom() -> dict:
            raise OSError("filesystem error")

        monkeypatch.setattr(ld_mod, "_read_catalog", boom)

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        with caplog.at_level("WARNING", logger="muse.cli_impl.load_director"):
            thread = director.observed_peak(
                "fake-model",
                observed_peak_bytes=10 * 1024**3,
                device="cuda",
            )
            assert thread is not None
            thread.join(timeout=2.0)
            assert not thread.is_alive()

    def test_unknown_model_in_catalog_is_swallowed(self, catalog_dir):
        # If the model is missing from the catalog entirely (race with
        # `muse models remove`), writeback must not crash. No catalog
        # mutation should result.
        # No seed: catalog stays empty.
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        thread = director.observed_peak(
            "ghost-model",
            observed_peak_bytes=2 * 1024**3,
            device="cuda",
        )
        if thread is not None:
            thread.join(timeout=2.0)

        catalog = _read_catalog()
        # Either: catalog stayed empty, OR the writeback decided not to
        # auto-create entries for unknown models. Both are acceptable;
        # the contract is "don't crash". Verify no crash.
        assert "ghost-model" not in catalog or catalog["ghost-model"].get(
            "measurements"
        ) is None

    def test_zero_or_negative_observed_is_no_writeback(self, catalog_dir):
        # A negative or zero observed peak indicates the load actually
        # FREED memory (other process released VRAM during our load
        # window), or the measurement was meaningless. Either way, do
        # not corrupt the recorded peak with a 0 value.
        _seed_catalog("fake-model", device="cuda", peak_bytes=2 * 1024**3)

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        thread = director.observed_peak(
            "fake-model",
            observed_peak_bytes=0,
            device="cuda",
        )
        if thread is not None:
            thread.join(timeout=2.0)

        catalog = _read_catalog()
        m = catalog["fake-model"]["measurements"]["cuda"]
        assert m["peak_bytes"] == 2 * 1024**3
        assert "source" not in m

    def test_auto_device_resolves_to_cuda_when_gpu_available(self, catalog_dir):
        """Manifests with `device: "auto"` must NOT split-brain against
        probe records that always write the resolved device name.
        When pynvml reports a free-VRAM number, "auto" -> "cuda"."""
        _seed_catalog("fake-model", device="cuda", peak_bytes=None)

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(gpu_free=32.0, cpu_free=64.0),
        )

        thread = director.observed_peak(
            "fake-model",
            observed_peak_bytes=3 * 1024**3,
            device="auto",
        )
        assert thread is not None
        thread.join(timeout=2.0)

        catalog = _read_catalog()
        # Bucket key must be "cuda", NOT "auto".
        assert "cuda" in catalog["fake-model"]["measurements"]
        assert "auto" not in catalog["fake-model"]["measurements"]
        assert catalog["fake-model"]["measurements"]["cuda"]["peak_bytes"] == 3 * 1024**3

    def test_auto_device_resolves_to_cpu_when_no_gpu(self, catalog_dir):
        """When pynvml is unavailable (gpu_free_gb returns None),
        "auto" -> "cpu". Mirrors the runtime's actual fallback."""
        _seed_catalog("fake-model", device="cpu", peak_bytes=None)

        # gpu_free=None simulates pynvml-missing / CPU-only host.
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 64.0
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
        )

        thread = director.observed_peak(
            "fake-model",
            observed_peak_bytes=2 * 1024**3,
            device="auto",
        )
        assert thread is not None
        thread.join(timeout=2.0)

        catalog = _read_catalog()
        assert "cpu" in catalog["fake-model"]["measurements"]
        assert "auto" not in catalog["fake-model"]["measurements"]
        assert catalog["fake-model"]["measurements"]["cpu"]["peak_bytes"] == 2 * 1024**3


class TestColdLoadCommitFiresWriteback:
    """The wired callsite: a successful cold load fires the writeback
    thread automatically, with the observed delta computed from the
    free_before / free_after captured in `_load_and_commit`.
    """

    def test_cold_load_fires_writeback_after_commit(self, catalog_dir):
        # Probe simulates 4 GB consumed: pre=10 GB free, post=6 GB free.
        # Observed peak bytes = 4 GB. Recorded = 1 GB. Writeback raises.
        _seed_catalog("fake-model", device="cpu", peak_bytes=1 * 1024**3)

        # cpu_free_gb is called twice during _load_and_commit (free_before
        # and free_after) plus possibly once or twice for decision/eviction.
        # Use side_effect with a long tail so all calls succeed.
        free_seq = iter([10.0, 10.0, 6.0] + [6.0] * 50)
        probe = MagicMock()
        probe.gpu_free_gb.return_value = 32.0
        probe.cpu_free_gb.side_effect = lambda: next(free_seq)

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
        )

        # Capture the writeback thread by patching threading.Thread on
        # the module so we can join it deterministically.
        spawned: list[threading.Thread] = []
        import muse.cli_impl.load_director as ld_mod
        orig_thread = ld_mod.threading.Thread

        def capture_thread(*args, **kwargs):
            t = orig_thread(*args, **kwargs)
            spawned.append(t)
            return t

        # Patch the load_director's threading.Thread so we capture only
        # our writeback thread, not arbitrary ones from other code paths.
        import unittest.mock as um
        with um.patch.object(ld_mod.threading, "Thread", side_effect=capture_thread):
            director.acquire(
                "fake-model",
                manifest=_manifest(memory_gb=1.0, device="cpu"),
            )

        # One writeback thread must have been started; join + verify catalog.
        assert len(spawned) >= 1, "no writeback thread was spawned during cold load"
        for t in spawned:
            t.join(timeout=2.0)

        catalog = _read_catalog()
        m = catalog["fake-model"]["measurements"]["cpu"]
        # 10.0 - 6.0 = 4.0 GB observed, > 1 GB recorded; writeback wins.
        assert m["peak_bytes"] == int(4.0 * 1024**3)
        assert m["source"] == "lazy_load_observation"

    def test_cold_load_with_no_observed_consumption_does_not_overwrite(self, catalog_dir):
        # Probe shows free_before = free_after (no consumption observed,
        # e.g. enable_fn was a no-op or memory was already accounted for).
        # Writeback must not reduce the recorded peak.
        _seed_catalog("fake-model", device="cpu", peak_bytes=2 * 1024**3)

        probe = MagicMock()
        probe.gpu_free_gb.return_value = 32.0
        probe.cpu_free_gb.return_value = 8.0

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
        )

        # Run a cold acquire; the writeback should fire but observe a 0-byte
        # delta and skip the catalog mutation.
        director.acquire(
            "fake-model",
            manifest=_manifest(memory_gb=0.5, device="cpu"),
        )
        # Give any writeback thread time to complete (worst case).
        time.sleep(0.1)

        catalog = _read_catalog()
        m = catalog["fake-model"]["measurements"]["cpu"]
        assert m["peak_bytes"] == 2 * 1024**3


# ----------------------------------------------------------------------
# Task G: warmup
# ----------------------------------------------------------------------

class TestWarmup:
    """LoadDirector.warmup: like acquire but does NOT increment refcount.

    Spec semantic: "load this model now without serving a request" so
    subsequent requests are hot. The loaded entry has refcount=0 from the
    start, making it immediately eligible for LRU eviction if pressure
    arises before the first real request.
    """

    def test_warmup_cold_load_returns_port(self):
        enable_fn = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        port = director.warmup("fake-model", manifest=_manifest())

        assert port == 9001
        enable_fn.assert_called_once_with("fake-model")

    def test_warmup_does_not_increment_refcount(self):
        # The defining characteristic: warmup leaves refcount at 0 so
        # the entry is immediately evictable. (acquire would set it to 1.)
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.warmup("fake-model", manifest=_manifest())

        snapshot = director.status()
        assert "fake-model" in snapshot
        assert snapshot["fake-model"]["refcount"] == 0
        assert snapshot["fake-model"]["loaded"] is True

    def test_warmup_when_already_loaded_returns_port_no_op(self):
        # Hot warmup: model is already loaded. Don't double-load,
        # don't bump refcount, just return the existing port.
        enable_fn = MagicMock(return_value=9001)
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        director.release("fake-model")
        assert director.status()["fake-model"]["refcount"] == 0
        enable_fn.reset_mock()

        port = director.warmup("fake-model", manifest=_manifest())

        assert port == 9001
        enable_fn.assert_not_called()
        # Refcount must still be 0 (warmup must not bump).
        assert director.status()["fake-model"]["refcount"] == 0

    def test_warmup_when_already_loaded_with_inflight_request(self):
        # If the model is loaded with refcount > 0 (a live request),
        # warmup must not interfere: just return the port.
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.acquire("fake-model", manifest=_manifest())
        assert director.status()["fake-model"]["refcount"] == 1

        port = director.warmup("fake-model", manifest=_manifest())

        assert port == 9001
        # refcount unchanged (still 1 from the live acquire).
        assert director.status()["fake-model"]["refcount"] == 1

    def test_warmup_records_decision_log_on_load(self):
        # Warmup should still be observable via recent_decisions.
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )

        director.warmup("fake-model", manifest=_manifest())

        decisions = list(director.recent_decisions)
        assert len(decisions) == 1
        assert decisions[0].model_id == "fake-model"
        assert decisions[0].action == "load"

    def test_warmup_evicts_lru_when_no_room(self):
        # Same eviction loop as cold acquire. The newcomer's warmup
        # forces an LRU eviction since memory is tight.
        cpu_free_state = {"value": 10.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = 32.0
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        def enable_side_effect(model_id: str) -> int:
            if model_id == "victim":
                cpu_free_state["value"] = 4.0
                return 9001
            if model_id == "newcomer":
                return 9002
            raise AssertionError(f"unexpected enable: {model_id}")

        enable_fn = MagicMock(side_effect=enable_side_effect)

        def disable_side_effect(model_id: str) -> None:
            cpu_free_state["value"] = 9.0

        disable_fn = MagicMock(side_effect=disable_side_effect)

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=2.0,
        )

        director.acquire("victim", manifest=_manifest(memory_gb=5.0, device="cpu"))
        director.release("victim")

        port = director.warmup("newcomer", manifest=_manifest(memory_gb=6.0, device="cpu"))

        assert port == 9002
        disable_fn.assert_called_once_with("victim")
        snapshot = director.status()
        assert "newcomer" in snapshot
        # Newcomer at refcount=0 (warmup default).
        assert snapshot["newcomer"]["refcount"] == 0
        assert "victim" not in snapshot


class TestAutoDevicePoolSelection:
    """Regression for the v0.48.0 control-plane bug.

    A model declaring `device: "auto"` (the new default for CUDA-safe
    bundled models) loads on the GPU when one is present. The director
    must therefore SIZE / ADMIT / EVICT it against the VRAM pool, not
    host RAM. Before the fix, `_free_for_device` / `_available_for_device`
    fell through their `else` branch and accounted "auto" as CPU, so an
    auto model could "fit" against terabytes of host RAM while the GPU
    OOM'd -- the director would never evict resident GPU models to make
    room. The resolution mirrors `observed_peak`'s existing auto handling:
    gpu_free_gb() is not None -> "cuda", else "cpu".
    """

    def test_free_for_device_auto_reads_gpu_pool_when_gpu_present(self):
        probe = _make_probe(gpu_free=7.0, cpu_free=512.0)
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
        )
        # auto resolves to the GPU pool (7.0), NOT host RAM (512.0).
        assert director._free_for_device("auto") == 7.0

    def test_free_for_device_auto_reads_cpu_pool_when_no_gpu(self):
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None  # pynvml absent / CPU-only host
        probe.cpu_free_gb.return_value = 512.0
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
        )
        # No GPU: auto falls back to host RAM, exactly as runtime select_device does.
        assert director._free_for_device("auto") == 512.0

    def test_available_for_device_auto_uses_gpu_headroom_when_gpu_present(self):
        probe = _make_probe(gpu_free=7.0, cpu_free=512.0)
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
            gpu_headroom_gb=1.0,
            cpu_headroom_gb=2.0,
        )
        free, available = director._available_for_device("auto")
        assert free == 7.0
        # GPU headroom (1.0) must apply, NOT CPU headroom (2.0).
        assert available == 6.0

    def test_explicit_cuda_uses_budget_when_nvml_is_unavailable(self):
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 512.0
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
            gpu_budget_gb=8.0,
            gpu_headroom_gb=1.0,
        )

        assert director._available_for_device("cuda") == (8.0, 7.0)

    def test_rocm_like_auto_uses_static_gpu_budget(self):
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 512.0
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
            gpu_budget_gb=8.0,
            gpu_headroom_gb=1.0,
            cuda_available_fn=lambda: True,
        )

        assert director._resolve_pool_device("auto") == "cuda"
        assert director._available_for_device("auto") == (8.0, 7.0)

    def test_static_gpu_budget_debits_resident_models(self):
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 512.0
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9002),
            disable_fn=MagicMock(),
            memory_probe=probe,
            gpu_budget_gb=8.0,
            gpu_headroom_gb=1.0,
        )
        now = time.monotonic()
        director.loaded["resident"] = LoadEntry(
            model_id="resident",
            worker_port=9001,
            memory_gb=4.0,
            refcount=0,
            last_touched_at=now,
            loaded_at=now,
            pool="cuda",
        )

        assert director._available_for_device("cuda") == (4.0, 3.0)

        # A 4 GB newcomer needs the resident evicted. Static accounting
        # observes the declared 4 GB release without NVML polling data.
        assert director.acquire(
            "newcomer", manifest=_manifest(memory_gb=4.0, device="cuda"),
        ) == 9002
        director.disable_fn.assert_called_once_with("resident")

    def test_static_budget_stays_reserved_until_disable_finishes(self):
        """The pop-before-shutdown window must not over-admit another load."""
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 512.0
        observed: list[tuple[float, float]] = []
        raced_decisions: list[tuple] = []
        holder = {}

        def disable(_model_id):
            director = holder["director"]
            observed.append(director._available_for_device("cuda"))
            raced_decisions.append(director._decide(
                "racer",
                manifest=_manifest(memory_gb=4.0, device="cuda"),
            ))

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9002),
            disable_fn=disable,
            memory_probe=probe,
            gpu_budget_gb=8.0,
            gpu_headroom_gb=1.0,
        )
        holder["director"] = director
        now = time.monotonic()
        director.loaded["resident"] = LoadEntry(
            model_id="resident", worker_port=9001, memory_gb=4.0,
            refcount=0, last_touched_at=now, loaded_at=now, pool="cuda",
        )

        assert director.acquire(
            "newcomer", manifest=_manifest(memory_gb=4.0, device="cuda"),
        ) == 9002

        assert observed == [(4.0, 3.0)]
        assert raced_decisions[0][0] == "evict_and_retry"
        assert "racer" not in director.in_flight_loads
        assert director._evicting_memory_gb == {}

    def test_live_cpu_budget_debits_resident_models(self):
        """A CPU budget caps the whole Muse working set, not each load."""
        probe = _make_probe(gpu_free=None, cpu_free=512.0)
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9002),
            disable_fn=MagicMock(),
            memory_probe=probe,
            cpu_budget_gb=8.0,
            cpu_headroom_gb=1.0,
        )
        now = time.monotonic()
        director.loaded["resident"] = LoadEntry(
            model_id="resident", worker_port=9001, memory_gb=4.0,
            refcount=0, last_touched_at=now, loaded_at=now, pool="cpu",
        )

        # min(512 GiB live, 8 - 4 GiB remaining budget) - 1 headroom.
        assert director._available_for_device("cpu") == (512.0, 3.0)

        # The budget-bound eviction does not wait for psutil's noisy host
        # counter to increase: disable completion plus updated bookkeeping is
        # sufficient, and the newcomer then fits the emptied 7 GiB pool.
        director._wait_for_memory_release = MagicMock(
            side_effect=AssertionError("physical poll should be unnecessary"),
        )
        assert director.acquire(
            "newcomer", manifest=_manifest(memory_gb=4.0, device="cpu"),
        ) == 9002
        director.disable_fn.assert_called_once_with("resident")
        director._wait_for_memory_release.assert_not_called()

    def test_live_gpu_budget_debits_resident_models(self):
        probe = _make_probe(gpu_free=100.0, cpu_free=512.0)
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9002),
            disable_fn=MagicMock(),
            memory_probe=probe,
            gpu_budget_gb=8.0,
            gpu_headroom_gb=1.0,
        )
        now = time.monotonic()
        director.loaded["resident"] = LoadEntry(
            model_id="resident", worker_port=9001, memory_gb=4.0,
            refcount=0, last_touched_at=now, loaded_at=now, pool="cuda",
        )

        assert director._available_for_device("cuda") == (100.0, 3.0)

    def test_eviction_never_sacrifices_an_unrelated_pool(self):
        probe = _make_probe(gpu_free=1.0, cpu_free=512.0)
        disable = MagicMock()
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9002),
            disable_fn=disable,
            memory_probe=probe,
            gpu_headroom_gb=0.0,
        )
        now = time.monotonic()
        director.loaded["cpu-worker"] = LoadEntry(
            model_id="cpu-worker", worker_port=9001, memory_gb=20.0,
            refcount=0, last_touched_at=now, loaded_at=now, pool="cpu",
        )

        with pytest.raises(OperationError) as exc_info:
            director.acquire(
                "gpu-newcomer",
                manifest=_manifest(memory_gb=4.0, device="cuda"),
            )

        assert exc_info.value.retryable is False
        disable.assert_not_called()
        assert "cpu-worker" in director.loaded


class TestHardCapacityBackstop:
    class _GpuProbe:
        def __init__(self, free_gb=2.0, total_gb=8.0):
            self.free_gb = free_gb
            self.total_gb = total_gb

        def gpu_free_gb(self):
            return self.free_gb

        def gpu_total_gb(self):
            return self.total_gb

        def cpu_free_gb(self):
            return 64.0

        def cpu_total_gb(self):
            return 64.0

    def _idle_victim(self):
        now = time.monotonic()
        return LoadEntry(
            model_id="victim", worker_port=9001, memory_gb=5.0,
            refcount=0, last_touched_at=now, loaded_at=now, pool="cuda",
        )

    def test_oversized_model_never_evicts_before_failing(self):
        probe = self._GpuProbe(free_gb=2.0, total_gb=8.0)
        disable = MagicMock()
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9002), disable_fn=disable,
            memory_probe=probe, gpu_headroom_gb=1.0,
        )
        director.loaded["victim"] = self._idle_victim()

        with pytest.raises(OperationError) as exc_info:
            director.acquire(
                "impossible", manifest=_manifest(memory_gb=10.0, device="cuda"),
            )

        assert exc_info.value.retryable is False
        assert "exceeds 7.00 GB capacity" in exc_info.value.message
        disable.assert_not_called()
        assert "victim" in director.loaded

    def test_transient_pressure_still_evicts_when_model_fits_total(self):
        probe = self._GpuProbe(free_gb=2.0, total_gb=8.0)

        def disable(_model_id):
            probe.free_gb = 8.0

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9002), disable_fn=disable,
            memory_probe=probe, gpu_headroom_gb=1.0,
        )
        director.loaded["victim"] = self._idle_victim()

        assert director.acquire(
            "fits-empty", manifest=_manifest(memory_gb=6.0, device="cuda"),
        ) == 9002
        assert "victim" not in director.loaded

    def test_static_budget_is_also_a_hard_ceiling(self):
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 64.0
        disable = MagicMock()
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9002), disable_fn=disable,
            memory_probe=probe, gpu_budget_gb=8.0, gpu_headroom_gb=1.0,
        )
        director.loaded["victim"] = self._idle_victim()

        with pytest.raises(OperationError):
            director.warmup(
                "impossible", manifest=_manifest(memory_gb=8.0, device="cuda"),
            )

        disable.assert_not_called()


class TestAutoDeviceLiveAccounting:
    def test_live_nvidia_reading_remains_authoritative(self):
        probe = _make_probe(gpu_free=6.0, cpu_free=512.0)
        cuda_checker = MagicMock(return_value=False)
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
            gpu_budget_gb=8.0,
            gpu_headroom_gb=1.0,
            cuda_available_fn=cuda_checker,
        )

        assert director._available_for_device("auto") == (6.0, 5.0)
        cuda_checker.assert_not_called()

    def test_auto_model_evicts_idle_gpu_victim_to_fit_vram(self, catalog_dir):
        # GPU host. A 6 GB idle "auto" victim is resident; loading it drops
        # the GPU to 2 GB free. A 5 GB "auto" newcomer cannot fit live VRAM
        # and must evict the victim. With the pre-fix bug (auto sized against
        # the 512 GB CPU pool), the director would believe the newcomer fits
        # without eviction and never call disable_fn -> GPU OOM. The fix
        # sizes auto against the VRAM pool, so disable_fn fires exactly once.
        gpu_free_state = {"value": 8.0}
        probe = MagicMock()
        probe.gpu_free_gb.side_effect = lambda: gpu_free_state["value"]
        probe.cpu_free_gb.return_value = 512.0

        def enable_side_effect(model_id: str) -> int:
            if model_id == "victim":
                gpu_free_state["value"] = 2.0
                return 9001
            if model_id == "newcomer":
                return 9002
            raise AssertionError(f"unexpected enable: {model_id}")

        def disable_side_effect(model_id: str) -> None:
            gpu_free_state["value"] = 8.0

        enable_fn = MagicMock(side_effect=enable_side_effect)
        disable_fn = MagicMock(side_effect=disable_side_effect)

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            gpu_headroom_gb=1.0,
        )

        # Pre-load victim: gpu_free=8, headroom=1, available=7 >= 6, fits.
        director.acquire("victim", manifest=_manifest(memory_gb=6.0, device="auto"))
        director.release("victim")
        assert director.status()["victim"]["refcount"] == 0

        # Newcomer at 5 GB: gpu_free=2, available=1, shortfall -> evict victim.
        port = director.acquire("newcomer", manifest=_manifest(memory_gb=5.0, device="auto"))
        assert port == 9002
        disable_fn.assert_called_once_with("victim")
        snapshot = director.status()
        assert "newcomer" in snapshot
        assert "victim" not in snapshot


# ----------------------------------------------------------------------
# v0.50.1: absent capabilities.device must pool as "auto", not "cpu"
# ----------------------------------------------------------------------

class TestAbsentDeviceDefaultsToAutoPool:
    """A manifest WITHOUT capabilities.device (every resolver-pulled model:
    the HF plugins never set the key) must be sized against the pool that
    "auto" resolves to on this host, because the worker it spawns runs
    with `--device auto` and loads the GPU when one exists.

    Regression: defaulting the absent key to "cpu" sized GPU loads against
    host RAM, so an 8 GB model always "fit" a 64 GB-RAM box, the director
    never evicted the resident model, and the spawned worker OOM'd on the
    real GPU (live incident on the 12 GB box, v0.50.0)."""

    def _manifest_without_device(self, memory_gb: float = 8.0) -> dict:
        return {
            "model_id": "sdxl-turbo",
            "modality": "image/generation",
            "capabilities": {"memory_gb": memory_gb},
        }

    def test_decide_routes_to_eviction_when_vram_short(self):
        # GPU visible (gpu_free_gb is not None) but nearly full; host RAM
        # huge. The old cpu default saw 64 GB free and returned "load".
        probe = _make_probe(gpu_free=0.5, cpu_free=64.0)
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
        )
        decision = director._decide(
            "sdxl-turbo", manifest=self._manifest_without_device(),
        )
        assert decision[0] == "evict_and_retry"

    def test_decide_loads_when_vram_roomy(self):
        probe = _make_probe(gpu_free=20.0, cpu_free=64.0)
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
        )
        decision = director._decide(
            "sdxl-turbo", manifest=self._manifest_without_device(),
        )
        assert decision[0] == "load"

    def test_absent_device_on_cpu_only_host_pools_against_ram(self):
        # No GPU visible: auto degrades to the cpu pool, exactly like the
        # worker's own select_device("auto") would.
        probe = _make_probe(gpu_free=None, cpu_free=64.0)
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9001),
            disable_fn=MagicMock(),
            memory_probe=probe,
        )
        decision = director._decide(
            "sdxl-turbo", manifest=self._manifest_without_device(),
        )
        assert decision[0] == "load"


# ----------------------------------------------------------------------
# v0.50.1: worker spawn failure surfaces as OperationError(503), not a
# raw exception that escapes the gateway as a bare 500
# ----------------------------------------------------------------------

class TestLoadFailureSurfacesAsOperationError:
    def test_spawn_failure_raises_operation_error_503(self):
        enable_fn = MagicMock(
            side_effect=RuntimeError("worker failed to become healthy"),
        )
        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )
        with pytest.raises(OperationError) as ei:
            director.acquire("fake-model", manifest=_manifest())
        assert ei.value.status == 503
        assert ei.value.code == "model_load_failed"
        assert "worker failed to become healthy" in ei.value.message
        # Cleanup ran: no stale LoadEntry, no stranded in-flight slot.
        assert director.status() == {}
        assert director.in_flight_loads == {}

    def test_operation_error_from_load_passes_through_unwrapped(self):
        original = OperationError("model_disabled", "disabled by operator", status=503)
        director = LoadDirector(
            enable_fn=MagicMock(side_effect=original),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(),
        )
        with pytest.raises(OperationError) as ei:
            director.acquire("fake-model", manifest=_manifest())
        assert ei.value is original


# ----------------------------------------------------------------------
# Spec 2026-07-08: retryable flag + capacity_listener hooks
# ----------------------------------------------------------------------


def test_same_pool_pressure_serializes_and_avoids_second_eviction():
    first_disable_started = threading.Event()
    release_first_disable = threading.Event()
    disabled: list[str] = []

    def disable(model_id):
        disabled.append(model_id)
        if len(disabled) == 1:
            first_disable_started.set()
            assert release_first_disable.wait(timeout=5.0)

    director = LoadDirector(
        enable_fn=MagicMock(return_value=9001),
        disable_fn=disable,
        memory_probe=_make_probe(cpu_free=100.0),
        cpu_budget_gb=10.0,
        cpu_headroom_gb=0.0,
    )
    now = time.monotonic()
    director.loaded = {
        "a": LoadEntry("a", 9001, 4.0, 0, now - 2, now - 2, "cpu"),
        "b": LoadEntry("b", 9002, 4.0, 0, now - 1, now - 1, "cpu"),
    }
    errors: list[BaseException] = []

    def evict_for(model_id):
        try:
            director._evict_lru_until_fits(
                model_id=model_id,
                shortfall_gb=4.0,
                device="cpu",
                required_gb=6.0,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    first = threading.Thread(target=evict_for, args=("incoming-a",))
    second = threading.Thread(target=evict_for, args=("incoming-b",))
    first.start()
    assert first_disable_started.wait(timeout=5.0)
    second.start()
    time.sleep(0.05)
    assert disabled == ["a"]

    release_first_disable.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert disabled == ["a"]
    assert set(director.loaded) == {"b"}


def test_partial_eviction_notifies_only_after_releasing_director_lock():
    director = LoadDirector(
        enable_fn=MagicMock(return_value=9001),
        disable_fn=MagicMock(),
        memory_probe=_make_probe(cpu_free=100.0),
        cpu_budget_gb=8.0,
        cpu_headroom_gb=0.0,
    )
    now = time.monotonic()
    director.loaded["victim"] = LoadEntry(
        "victim", 9001, 2.0, 0, now, now, "cpu",
    )
    director._wait_for_memory_release = MagicMock(return_value=2.0)
    ownership: list[bool] = []
    director.capacity_listener = lambda: ownership.append(
        director.lock._is_owned()
    )

    with pytest.raises(OperationError, match="no evictable"):
        director._evict_lru_until_fits(
            model_id="too-large",
            shortfall_gb=4.0,
            device="cpu",
            required_gb=10.0,
        )

    assert ownership == [False]

class TestCapacitySignal:
    """Spec 2026-07-08: retryable flag + capacity_listener hooks."""

    def _director(self, *, gpu_free=1.0):
        from muse.cli_impl.load_director import LoadDirector
        return LoadDirector(
            enable_fn=lambda mid: 9001,
            disable_fn=lambda mid: None,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: gpu_free),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )

    def _entry(self, model_id, refcount, *, pool="cuda"):
        import time
        from muse.cli_impl.load_director import LoadEntry
        now = time.monotonic()
        return LoadEntry(model_id=model_id, worker_port=9001, memory_gb=4.0,
                         refcount=refcount, last_touched_at=now, loaded_at=now,
                         pool=pool)

    def test_release_to_zero_fires_listener(self):
        d = self._director()
        fired = []
        d.capacity_listener = lambda: fired.append(1)
        d.loaded["m"] = self._entry("m", refcount=1)
        d.release("m")
        assert fired == [1]

    def test_release_above_zero_does_not_fire(self):
        d = self._director()
        fired = []
        d.capacity_listener = lambda: fired.append(1)
        d.loaded["m"] = self._entry("m", refcount=2)
        d.release("m")
        assert fired == []

    def test_listener_exception_swallowed(self):
        d = self._director()
        def boom():
            raise RuntimeError("listener broke")
        d.capacity_listener = boom
        d.loaded["m"] = self._entry("m", refcount=1)
        d.release("m")  # must not raise
        assert d.loaded["m"].refcount == 0

    def test_capacity_503_retryable_when_inuse_models_block(self):
        from muse.admin.operations import OperationError
        d = self._director(gpu_free=1.0)  # 1 GB free, need 8
        d.loaded["busy"] = self._entry("busy", refcount=1)  # in-use: not evictable
        with pytest.raises(OperationError) as exc:
            d._evict_lru_until_fits(
                model_id="new", shortfall_gb=7.0, device="cuda", required_gb=8.0,
            )
        assert exc.value.retryable is True

    def test_capacity_503_not_retryable_when_nothing_will_free(self):
        from muse.admin.operations import OperationError
        d = self._director(gpu_free=1.0)
        # No loaded models at all: nothing will ever release capacity.
        with pytest.raises(OperationError) as exc:
            d._evict_lru_until_fits(
                model_id="new", shortfall_gb=7.0, device="cuda", required_gb=8.0,
            )
        assert exc.value.retryable is False

    def test_no_candidate_eviction_preserves_another_load_claim_on_503(self):
        import threading

        from muse.admin.operations import OperationError
        from muse.cli_impl.load_director import InFlightLoad

        d = self._director(gpu_free=1.0)
        claim = InFlightLoad(
            event=threading.Event(), memory_gb=8.0, pool="cuda",
        )
        d.in_flight_loads["new"] = claim

        with pytest.raises(OperationError):
            d._evict_lru_until_fits(
                model_id="new", shortfall_gb=7.0, device="cuda",
                required_gb=8.0,
            )

        assert d.in_flight_loads["new"] is claim
        assert not claim.event.is_set()

    def test_no_candidate_fit_recheck_preserves_another_load_claim(self):
        import threading

        from muse.cli_impl.load_director import InFlightLoad

        available = iter((2.0, 10.0))
        probe = type("P", (), {
            "gpu_free_gb": staticmethod(lambda: None),
            "cpu_free_gb": staticmethod(lambda: next(available)),
        })()
        d = LoadDirector(
            enable_fn=lambda mid: 9001,
            disable_fn=lambda mid: None,
            memory_probe=probe,
            cpu_headroom_gb=1.0,
        )
        claim = InFlightLoad(
            event=threading.Event(), memory_gb=4.0, pool="cpu",
        )
        d.in_flight_loads["new"] = claim

        d._evict_lru_until_fits(
            model_id="new", shortfall_gb=3.0, device="cpu",
            required_gb=4.0,
        )

        assert d.in_flight_loads["new"] is claim
        assert not claim.event.is_set()

    def test_operation_error_default_not_retryable(self):
        from muse.admin.operations import OperationError
        assert OperationError("x", "y").retryable is False

    def test_eviction_success_fires_listener_once(self):
        """An eviction that frees enough memory fires capacity_listener
        exactly once (the loop-exited-normally path in
        _evict_lru_until_fits), so parked gateway capacity-waiters wake."""
        cpu_free = {"v": 4.0}
        probe = type("P", (), {
            "gpu_free_gb": staticmethod(lambda: None),  # no GPU -> cpu pool
            "cpu_free_gb": staticmethod(lambda: cpu_free["v"]),
        })()

        def disable(mid):
            cpu_free["v"] = 12.0  # eviction releases memory

        fired = []
        d = LoadDirector(enable_fn=lambda mid: 9001, disable_fn=disable,
                         memory_probe=probe, cpu_headroom_gb=1.0)
        d.capacity_listener = lambda: fired.append(1)
        d.loaded["victim"] = self._entry(
            "victim", refcount=0, pool="cpu",
        )  # evictable

        d._evict_lru_until_fits(
            model_id="new", shortfall_gb=4.0, device="cpu", required_gb=6.0,
        )

        assert fired == [1]
        assert "victim" not in d.loaded

    def test_fits_now_recheck_fires_listener_once(self):
        """The no-candidates-but-live-fit re-check path also fires
        capacity_listener exactly once before returning to let acquire
        re-decide (-> load)."""
        fired = []
        probe = type("P", (), {
            "gpu_free_gb": staticmethod(lambda: None),
            "cpu_free_gb": staticmethod(lambda: 10.0),  # 10 - 1 headroom = 9
        })()
        d = LoadDirector(enable_fn=lambda mid: 9001, disable_fn=lambda mid: None,
                         memory_probe=probe, cpu_headroom_gb=1.0)
        d.capacity_listener = lambda: fired.append(1)

        # No loaded candidates, but live memory now fits (9 >= required 4).
        d._evict_lru_until_fits(
            model_id="new", shortfall_gb=4.0, device="cpu", required_gb=4.0,
        )

        assert fired == [1]
        assert d.in_flight_loads == {}


class TestStressFindings:
    """v0.57.1: gaps found by the 2026-07-08 live stress test on .204.

    1. A capacity 503 raised while OTHER cold loads are in flight must be
       retryable=True: those loads hold the reservations that made the fit
       fail, and they WILL complete (freeing reservation into evictable
       residency) or abort (freeing it outright). The v0.55.0 check only
       looked at loaded refcounts, so a cold-start burst on an empty card
       mass-503'd instantly instead of queueing behind the first load.
    2. The capacity listener must fire on load COMMIT and ABORT (not just
       release-to-zero and eviction) so parked capacity-waiters re-decide
       when a reservation resolves.
    """

    def _director(self, *, gpu_free=1.0):
        from muse.cli_impl.load_director import LoadDirector
        return LoadDirector(
            enable_fn=lambda mid: 9001,
            disable_fn=lambda mid: None,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: gpu_free),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )

    def test_capacity_503_retryable_when_other_load_in_flight(self):
        import threading
        from muse.admin.operations import OperationError
        from muse.cli_impl.load_director import InFlightLoad
        d = self._director(gpu_free=1.0)
        # No loaded models at all, but ANOTHER model's cold load is in
        # flight holding a reservation.
        d.in_flight_loads["other-model"] = InFlightLoad(
            event=threading.Event(), memory_gb=6.0, pool="cuda",
        )
        with pytest.raises(OperationError) as exc:
            d._evict_lru_until_fits(
                model_id="new", shortfall_gb=7.0, device="cuda",
                required_gb=8.0,
            )
        assert exc.value.retryable is True

    def test_capacity_503_still_not_retryable_when_truly_empty(self):
        from muse.admin.operations import OperationError
        d = self._director(gpu_free=1.0)
        with pytest.raises(OperationError) as exc:
            d._evict_lru_until_fits(
                model_id="new", shortfall_gb=7.0, device="cuda",
                required_gb=8.0,
            )
        assert exc.value.retryable is False

    def test_abort_fires_capacity_listener(self):
        import threading
        from muse.cli_impl.load_director import InFlightLoad
        d = self._director()
        fired = []
        d.capacity_listener = lambda: fired.append(1)
        d.in_flight_loads["m"] = InFlightLoad(
            event=threading.Event(), memory_gb=2.0, pool="cuda",
        )
        d._abort("m")
        assert fired == [1]

    def test_commit_fires_capacity_listener(self):
        d = self._director(gpu_free=32.0)
        fired = []
        d.capacity_listener = lambda: fired.append(1)
        manifest = {"capabilities": {"memory_gb": 1.0, "device": "cuda"}}
        port = d.acquire("m", manifest=manifest)
        assert port == 9001
        assert fired  # commit fired the listener at least once


class TestEvictionWindowRetryable:
    """v0.57.2: residual gap found by re-running the stress burst on .204
    AFTER the v0.57.1 fix shipped.

    An eviction is a multi-second out-of-lock operation: the victim is
    popped from `loaded` under lock, then disable_fn (SIGTERM + worker
    shutdown) runs with the lock released, and the evictor registers its
    own in-flight slot only afterwards. During that window the director
    looks completely idle (loaded empty, in_flight_loads empty) while the
    victim's VRAM is still physically occupied, so a concurrent burst of
    decides fails the fit check against stale free memory AND raises with
    retryable=False -- mass instant-503s exactly like the pre-v0.57.1
    behavior. Observed live: six 503s at ~0.5s during qwen's eviction of
    ace-step (decision log 03:19:09Z, 2026-07-09).

    Fix: track evictions-in-progress in `self.evicting`; include it in
    the retryable condition; fire the capacity listener when an eviction
    completes so parked waiters re-decide against the real freed memory.
    """

    def _director(self, *, gpu_free=1.0):
        return LoadDirector(
            enable_fn=lambda mid: 9001,
            disable_fn=lambda mid: None,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: gpu_free),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )

    def test_capacity_503_retryable_during_inflight_eviction(self):
        d = self._director(gpu_free=1.0)
        # Mid-eviction window: victim already popped from loaded,
        # disable_fn still shutting the worker down, evictor not yet
        # registered in in_flight_loads. Director looks idle but capacity
        # WILL free when the eviction completes.
        d.evicting.add("victim-model")
        with pytest.raises(OperationError) as exc:
            d._evict_lru_until_fits(
                model_id="new", shortfall_gb=7.0, device="cuda",
                required_gb=8.0,
            )
        assert exc.value.retryable is True

    def test_eviction_fires_listener_before_new_load_starts(self):
        # The capacity listener must fire when disable_fn completes (VRAM
        # actually freed), BEFORE the evictor's own cold load runs -- so
        # capacity-waiters parked on the notifier re-decide as soon as
        # room exists instead of waiting for the evictor's commit.
        free = {"gb": 1.0}
        events = []

        def disable_fn(mid):
            events.append(("disable", mid))
            free["gb"] = 12.0

        def enable_fn(mid):
            events.append(("enable", mid))
            return 9002

        d = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: free["gb"]),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )
        d.capacity_listener = lambda: events.append(("fire",))
        now = time.monotonic()
        d.loaded["victim"] = LoadEntry(
            model_id="victim", worker_port=9001, memory_gb=8.0,
            refcount=0, last_touched_at=now, loaded_at=now,
            pool="cuda",
        )
        port = d.acquire(
            "new", manifest={"capabilities": {"memory_gb": 6.0, "device": "cuda"}},
        )
        assert port == 9002
        assert ("fire",) in events
        assert events.index(("fire",)) < events.index(("enable", "new"))
        assert d.evicting == set()

    def test_evicting_cleared_when_memory_poll_raises(self):
        # v0.57.3 review finding: the marker must clear even when the
        # out-of-lock post-eviction memory poll raises (e.g. a transient
        # psutil/probe error). A leaked marker makes every future capacity
        # 503 spuriously retryable forever.
        free = {"gb": 1.0}
        d = LoadDirector(
            enable_fn=lambda mid: 9002,
            disable_fn=lambda mid: None,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: free["gb"]),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )

        def poll_raises(**kwargs):
            raise RuntimeError("probe hiccup")

        d._wait_for_memory_release = poll_raises
        now = time.monotonic()
        d.loaded["victim"] = LoadEntry(
            model_id="victim", worker_port=9001, memory_gb=8.0,
            refcount=0, last_touched_at=now, loaded_at=now,
            pool="cuda",
        )
        with pytest.raises(RuntimeError, match="probe hiccup"):
            d.acquire(
                "new",
                manifest={"capabilities": {"memory_gb": 6.0, "device": "cuda"}},
            )
        assert d.evicting == set()

    def test_partial_eviction_before_503_fires_listener(self):
        # v0.57.3 review finding: when an eviction frees SOME memory but the
        # request still 503s (no candidates left for the remaining
        # shortfall), the freed memory is a capacity event for OTHER parked
        # models -- the listener must fire so they re-decide instead of
        # sleeping to their deadline.
        free = {"gb": 1.0}
        fired = []

        def disable_fn(mid):
            free["gb"] = 5.0  # victim freed 4 GB, still short of 10

        d = LoadDirector(
            enable_fn=lambda mid: 9002,
            disable_fn=disable_fn,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: free["gb"]),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )
        d.capacity_listener = lambda: fired.append(1)
        now = time.monotonic()
        d.loaded["victim"] = LoadEntry(
            model_id="victim", worker_port=9001, memory_gb=4.0,
            refcount=0, last_touched_at=now, loaded_at=now,
            pool="cuda",
        )
        with pytest.raises(OperationError):
            d.acquire(
                "huge",
                manifest={"capabilities": {"memory_gb": 10.0, "device": "cuda"}},
            )
        assert fired  # the partial free was signalled despite the 503

    def test_evicting_cleared_when_disable_fn_raises(self):
        def disable_fn(mid):
            raise RuntimeError("worker refused to die")

        d = LoadDirector(
            enable_fn=lambda mid: 9001,
            disable_fn=disable_fn,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: 1.0),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )
        now = time.monotonic()
        d.loaded["victim"] = LoadEntry(
            model_id="victim", worker_port=9001, memory_gb=8.0,
            refcount=0, last_touched_at=now, loaded_at=now,
            pool="cuda",
        )
        with pytest.raises(OperationError):
            d.acquire(
                "new",
                manifest={"capabilities": {"memory_gb": 6.0, "device": "cuda"}},
            )
        # The failed eviction re-inserted the victim; the evicting marker
        # must not leak (a leaked marker would make every future capacity
        # 503 spuriously retryable forever).
        assert d.evicting == set()
