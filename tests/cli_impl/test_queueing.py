"""Unit tests for the gateway queueing primitives (spec 2026-07-08)."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from muse.cli_impl.queueing import (
    CapacityNotifier, ConcurrencyGate, QueueFull, QueueTimeout,
)


def _deadline(seconds: float) -> float:
    return time.monotonic() + seconds


class TestConcurrencyGate:
    async def test_unlimited_cap_is_noop(self):
        gate = ConcurrencyGate()
        async with gate.slot("m", None, deadline=_deadline(1)):
            assert gate.depth("m") == 0  # unlimited: not even tracked
        async with gate.slot("m", 0, deadline=_deadline(1)):
            pass  # 0 also means unlimited

    async def test_cap_serializes_and_wakes_fifo(self):
        gate = ConcurrencyGate()
        order: list[int] = []
        release_first = asyncio.Event()

        async def holder():
            async with gate.slot("m", 1, deadline=_deadline(5)):
                order.append(0)
                await release_first.wait()

        async def waiter(i: int):
            async with gate.slot("m", 1, deadline=_deadline(5)):
                order.append(i)

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.01)  # holder owns the slot
        w1 = asyncio.create_task(waiter(1))
        await asyncio.sleep(0.01)
        w2 = asyncio.create_task(waiter(2))
        await asyncio.sleep(0.01)
        assert gate.depth("m") == 2  # two parked waiters
        release_first.set()
        await asyncio.gather(h, w1, w2)
        assert order == [0, 1, 2]  # FIFO
        assert gate.depth("m") == 0

    async def test_timeout_raises_queue_timeout(self):
        gate = ConcurrencyGate()
        started = asyncio.Event()

        async def holder():
            async with gate.slot("m", 1, deadline=_deadline(5)):
                started.set()
                await asyncio.sleep(0.5)

        h = asyncio.create_task(holder())
        await started.wait()
        with pytest.raises(QueueTimeout) as exc:
            async with gate.slot("m", 1, deadline=_deadline(0.05)):
                pass
        assert exc.value.model_id == "m"
        h.cancel()

    async def test_depth_bound_raises_queue_full(self, monkeypatch):
        monkeypatch.setenv("MUSE_MAX_QUEUE_DEPTH", "1")
        gate = ConcurrencyGate()
        started = asyncio.Event()

        async def holder():
            async with gate.slot("m", 1, deadline=_deadline(5)):
                started.set()
                await asyncio.sleep(0.5)

        h = asyncio.create_task(holder())
        await started.wait()
        w = asyncio.create_task(gate.slot("m", 1, deadline=_deadline(5)).__aenter__())
        await asyncio.sleep(0.01)  # w is parked; depth == 1 == bound
        with pytest.raises(QueueFull):
            async with gate.slot("m", 1, deadline=_deadline(5)):
                pass
        h.cancel(); w.cancel()

    async def test_slot_released_on_exception(self):
        gate = ConcurrencyGate()
        with pytest.raises(RuntimeError):
            async with gate.slot("m", 1, deadline=_deadline(1)):
                raise RuntimeError("boom")
        # slot free again: immediate re-acquire succeeds
        async with gate.slot("m", 1, deadline=_deadline(1)):
            pass

    async def test_burst_arrival_enforces_queue_full(self, monkeypatch):
        # Regression test for a burst-arrival race: asyncio.wait_for wraps
        # sem.acquire() in its own sub-Task, so EVERY slot() call runs its
        # synchronous entry-check prefix before any acquire sub-Task has a
        # chance to run. A check based on sem.locked() is therefore stale
        # for concurrent arrivals with no sleeps between them; the entry
        # check must be a single synchronous compare-and-increment.
        monkeypatch.setenv("MUSE_MAX_QUEUE_DEPTH", "1")
        gate = ConcurrencyGate()
        ok: list[int] = []
        full: list[int] = []

        async def worker(i: int):
            try:
                async with gate.slot("m", 1, deadline=_deadline(5)):
                    ok.append(i)
                    await asyncio.sleep(0.05)  # hold the slot briefly
            except QueueFull:
                full.append(i)
            except QueueTimeout:
                pass

        # No sleeps between task creation: all 10 workers are scheduled
        # in the same burst.
        await asyncio.gather(*[worker(i) for i in range(10)])
        assert len(ok) <= 2  # 1 holding + 1 queued (cap=1, max_depth=1)
        assert len(full) >= 8

    async def test_zero_timeout_free_slot_grants_immediately(self):
        # A zero/expired wait budget means "do not WAIT", not "do not TRY":
        # a genuinely free slot must be granted even with an already-expired
        # deadline.
        gate = ConcurrencyGate()
        start = time.monotonic()
        async with gate.slot("m", 2, deadline=time.monotonic()):
            assert gate.depth("m") == 0
        elapsed = time.monotonic() - start
        assert elapsed < 0.05
        assert gate.depth("m") == 0
        assert gate._entered.get("m", 0) == 0

    async def test_zero_timeout_occupied_slot_fails_fast(self):
        gate = ConcurrencyGate()
        started = asyncio.Event()

        async def holder():
            async with gate.slot("m", 1, deadline=_deadline(5)):
                started.set()
                await asyncio.sleep(0.5)

        h = asyncio.create_task(holder())
        await started.wait()
        start = time.monotonic()
        with pytest.raises(QueueTimeout):
            async with gate.slot("m", 1, deadline=time.monotonic()):
                pass
        elapsed = time.monotonic() - start
        assert elapsed < 0.05
        h.cancel()

    async def test_burst_depth_counts_only_excess(self):
        # depth() must report entered-minus-cap ("actually queued"), not
        # a raw attempt/entry count, even when a burst of concurrent
        # entries transiently inflates the entered counter before their
        # acquires settle.
        gate = ConcurrencyGate()
        samples: list[int] = []

        async def worker(i: int):
            async with gate.slot("m", 5, deadline=_deadline(5)):
                samples.append(gate.depth("m"))
                await asyncio.sleep(0.02)

        # 20 concurrent acquires against cap=5: entered can transiently
        # reach 20 (every task runs its synchronous entry-check prefix
        # before any acquire sub-Task completes -- see
        # test_burst_arrival_enforces_queue_full), so depth must never
        # exceed entered - cap == 15, and must settle back to 0.
        await asyncio.gather(*[worker(i) for i in range(20)])
        assert all(d <= 15 for d in samples)
        assert gate.depth("m") == 0

        # No contention (concurrency == cap): depth stays 0 throughout,
        # not just after the burst settles, because excess never goes
        # positive when entered never exceeds cap.
        samples.clear()

        async def worker_no_contention(i: int):
            async with gate.slot("m2", 5, deadline=_deadline(5)):
                samples.append(gate.depth("m2"))
                await asyncio.sleep(0.01)

        await asyncio.gather(*[worker_no_contention(i) for i in range(5)])
        assert all(d == 0 for d in samples)
        assert gate.depth("m2") == 0


class TestAcquireReleaseSlot:
    """Split acquire/release variant used by the gateway request path.

    Same entry semantics as slot(), but the held region spans a non-lexical
    scope: acquire_slot returns holding the slot; release_slot returns it
    (threadsafe, over-release-safe).
    """

    async def test_unlimited_cap_is_noop(self):
        gate = ConcurrencyGate()
        await gate.acquire_slot("m", None, deadline=_deadline(1))
        assert gate.depth("m") == 0
        gate.release_slot("m")  # no-op: nothing was reserved
        assert gate._entered.get("m", 0) == 0

    async def test_acquire_release_balances_on_success(self):
        gate = ConcurrencyGate()
        await gate.acquire_slot("m", 2, deadline=_deadline(1))
        assert gate.depth("m") == 0  # entered 1, cap 2 -> excess 0
        gate.release_slot("m")
        assert gate.depth("m") == 0
        assert gate._entered.get("m", 0) == 0  # fully drained
        # slot free again: an immediate re-acquire succeeds
        await gate.acquire_slot("m", 2, deadline=_deadline(1))
        gate.release_slot("m")

    async def test_acquire_slot_reports_whether_slot_taken(self):
        # #332: the caller must know whether a slot was actually reserved,
        # so its release can be conditional instead of re-deriving the cap
        # (which can change live via server.default_max_concurrency).
        gate = ConcurrencyGate()
        assert await gate.acquire_slot("m", 2, deadline=_deadline(1)) is True
        gate.release_slot("m")
        assert await gate.acquire_slot("m", None, deadline=_deadline(1)) is False
        assert await gate.acquire_slot("m", 0, deadline=_deadline(1)) is False

    async def test_uncapped_request_cannot_over_release_capped_holder(self):
        # #332 failure scenario: a semaphore exists from an earlier capped
        # era; the operator sets default_max_concurrency to 0 live; a new
        # request resolves cap=None and reserves nothing. Its release --
        # guarded by acquire_slot's return value -- must not decrement the
        # concurrently-held capped request's accounting.
        gate = ConcurrencyGate()
        took_capped = await gate.acquire_slot("m", 1, deadline=_deadline(1))
        assert took_capped is True
        took_uncapped = await gate.acquire_slot("m", None, deadline=_deadline(1))
        if took_uncapped:  # the gateway's guarded-release contract
            gate.release_slot("m")
        # The capped holder's slot is still accounted: cap 1 is still full.
        assert gate._entered.get("m", 0) == 1
        with pytest.raises(QueueTimeout):
            await gate.acquire_slot("m", 1, deadline=_deadline(0.05))
        gate.release_slot("m")

    def test_release_slot_after_loop_closed_does_not_raise(self):
        # v0.57.3 review finding: like CapacityNotifier.notify, the
        # threadsafe branch must swallow the loop-closed RuntimeError
        # during shutdown instead of propagating out of a release site.
        gate = ConcurrencyGate()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                gate.acquire_slot("m", 2, deadline=_deadline(1)))
        finally:
            loop.close()
        # No running loop here, captured loop is CLOSED: must not raise.
        gate.release_slot("m")

    async def test_queue_timeout_undoes_entered_increment(self):
        gate = ConcurrencyGate()
        started = asyncio.Event()

        async def holder():
            await gate.acquire_slot("m", 1, deadline=_deadline(5))
            started.set()
            await asyncio.sleep(0.3)
            gate.release_slot("m")

        h = asyncio.create_task(holder())
        await started.wait()
        with pytest.raises(QueueTimeout):
            await gate.acquire_slot("m", 1, deadline=_deadline(0.05))
        # the timed-out acquirer's increment was undone: only the holder
        # remains, so no residual queue depth.
        assert gate.depth("m") == 0
        await h
        assert gate._entered.get("m", 0) == 0

    async def test_queue_full_does_not_leak_entered(self, monkeypatch):
        monkeypatch.setenv("MUSE_MAX_QUEUE_DEPTH", "1")
        gate = ConcurrencyGate()
        started = asyncio.Event()

        async def holder():
            await gate.acquire_slot("m", 1, deadline=_deadline(5))
            started.set()
            await asyncio.sleep(0.3)
            gate.release_slot("m")

        h = asyncio.create_task(holder())
        await started.wait()
        w = asyncio.create_task(gate.acquire_slot("m", 1, deadline=_deadline(5)))
        await asyncio.sleep(0.01)  # w parks; depth == 1 == bound
        assert gate.depth("m") == 1
        with pytest.raises(QueueFull):
            await gate.acquire_slot("m", 1, deadline=_deadline(5))
        # QueueFull rejects BEFORE the increment -> depth unchanged
        assert gate.depth("m") == 1
        h.cancel()
        w.cancel()

    async def test_release_from_worker_thread_is_threadsafe(self):
        gate = ConcurrencyGate()
        await gate.acquire_slot("m", 1, deadline=_deadline(1))
        # release from OFF the loop: scheduled on the captured loop.
        await asyncio.to_thread(gate.release_slot, "m")
        await asyncio.sleep(0.01)  # let the scheduled release run
        assert gate.depth("m") == 0
        assert gate._entered.get("m", 0) == 0
        # slot is free again
        async with gate.slot("m", 1, deadline=_deadline(1)):
            pass

    async def test_double_release_is_safe(self):
        gate = ConcurrencyGate()
        await gate.acquire_slot("m", 1, deadline=_deadline(1))
        gate.release_slot("m")
        gate.release_slot("m")  # extra release: must not corrupt state
        assert gate.depth("m") == 0
        assert gate._entered.get("m", 0) == 0  # never negative
        # the semaphore was not over-released: exactly one slot is free
        await gate.acquire_slot("m", 1, deadline=_deadline(1))
        with pytest.raises(QueueTimeout):
            # a second concurrent acquire must still block (cap==1 intact)
            await gate.acquire_slot("m", 1, deadline=_deadline(0.05))
        gate.release_slot("m")

    async def test_release_before_any_acquire_is_noop(self):
        gate = ConcurrencyGate()
        gate.release_slot("never-acquired")  # must not raise / corrupt
        assert gate.depth("never-acquired") == 0

    async def test_zero_timeout_free_slot_grants_immediately(self):
        # Same fix, split acquire/release variant: a zero/expired deadline
        # against a free slot must succeed immediately, not QueueTimeout.
        gate = ConcurrencyGate()
        start = time.monotonic()
        await gate.acquire_slot("m", 2, deadline=time.monotonic())
        elapsed = time.monotonic() - start
        assert elapsed < 0.05
        assert gate.depth("m") == 0
        gate.release_slot("m")
        assert gate._entered.get("m", 0) == 0

    async def test_zero_timeout_occupied_slot_fails_fast(self):
        gate = ConcurrencyGate()
        started = asyncio.Event()

        async def holder():
            await gate.acquire_slot("m", 1, deadline=_deadline(5))
            started.set()
            await asyncio.sleep(0.5)
            gate.release_slot("m")

        h = asyncio.create_task(holder())
        await started.wait()
        start = time.monotonic()
        with pytest.raises(QueueTimeout):
            await gate.acquire_slot("m", 1, deadline=time.monotonic())
        elapsed = time.monotonic() - start
        assert elapsed < 0.05
        # timed-out acquirer's increment was undone
        assert gate.depth("m") == 0
        h.cancel()


class TestConcurrencyGateCrossThreadReads:
    """Regression coverage for the depths()/depth() cross-thread race.

    /v1/admin/memory and /v1/telemetry/summary are SYNC route handlers;
    Starlette dispatches sync handlers via run_in_threadpool, a DIFFERENT
    OS thread than the gateway's single event-loop thread. That loop thread
    mutates gate._entered on every acquire_slot/release_slot (key adds when
    a model_id is first seen, key pops when its count drops to 0). Before
    the fix, depths() did `for model_id in self._entered` with no
    synchronization, so a concurrent add/pop from the loop thread could
    raise "RuntimeError: dictionary changed size during iteration" mid-scan
    -- surfacing as a 500 on exactly the endpoints an operator polls while
    the system is busy queueing requests.
    """

    async def test_depths_survive_concurrent_rotating_mutation(self):
        gate = ConcurrencyGate()
        errors: list[BaseException] = []
        stop = threading.Event()

        def reader() -> None:
            # Tight loop: no sleeps, maximizing overlap with the mutator's
            # dict adds/pops on the loop thread.
            while not stop.is_set():
                try:
                    gate.depths()
                    for i in range(8):
                        gate.depth(f"m{i}")
                except BaseException as exc:  # capture ANY exception, not just RuntimeError
                    errors.append(exc)
                    return

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        async def mutator(offset: int) -> None:
            # Rotates through 8 model ids so keys are repeatedly ADDED
            # (first acquire for that id) and POPPED (count back to 0),
            # changing dict SIZE, not just values -- the precondition for
            # "dictionary changed size during iteration".
            end = time.monotonic() + 0.5
            i = offset
            while time.monotonic() < end:
                model_id = f"m{i % 8}"
                i += 1
                try:
                    await gate.acquire_slot(model_id, 1, deadline=_deadline(0.02))
                except QueueTimeout:
                    continue
                await asyncio.sleep(0)  # yield once while the slot is held
                gate.release_slot(model_id)

        # Several concurrent mutator tasks so multiple model_ids exist in
        # _entered simultaneously, maximizing add/pop churn during the
        # reader thread's scan window.
        await asyncio.gather(*[mutator(k) for k in range(4)])

        stop.set()
        reader_thread.join(timeout=2)
        assert not reader_thread.is_alive(), "reader thread did not exit in time"
        assert errors == [], f"depths()/depth() raised under concurrent mutation: {errors!r}"


class TestCapacityNotifier:
    async def test_snapshot_then_notify_wakes(self):
        n = CapacityNotifier()
        ev = n.snapshot()
        assert not ev.is_set()
        n.notify()  # threadsafe path, same loop here
        await asyncio.wait_for(ev.wait(), timeout=1)

    async def test_notify_before_any_snapshot_is_noop(self):
        n = CapacityNotifier()
        n.notify()  # must not raise (loop not yet captured)

    async def test_generation_semantics_no_missed_wakeup(self):
        n = CapacityNotifier()
        ev1 = n.snapshot()
        n.notify()
        await asyncio.wait_for(ev1.wait(), timeout=1)
        ev2 = n.snapshot()
        assert ev2 is not ev1 and not ev2.is_set()  # fresh generation

    async def test_notify_from_thread(self):
        n = CapacityNotifier()
        ev = n.snapshot()
        await asyncio.to_thread(n.notify)
        await asyncio.wait_for(ev.wait(), timeout=1)


class TestConfigRows:
    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("MUSE_CONFIG", "/nonexistent-config.yaml")
        from muse.core import config
        assert config.get("server.default_max_concurrency") == 0
        assert config.get("server.queue_timeout_seconds") == 300.0
        assert config.get("server.max_queue_depth") == 256
