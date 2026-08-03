"""Concurrency-safety tests for LoadDirector under off-loop (multi-thread)
acquire.

Motivation: v0.50.x moves gateway `director.acquire` off the single event
loop via asyncio.to_thread, so request-path acquires now run CONCURRENTLY in
worker threads instead of serialized on one loop thread. Two hazards that the
serialized path masked must be closed:

  1. Over-admission: the fit check sized against LIVE free VRAM but never
     debited a claimed-but-not-yet-resident load, so two concurrent cold
     loads for different models could both "fit" against the same free
     reading and over-commit the device (OOM). The director must RESERVE an
     in-flight load's memory at decision time.
  2. Observed-peak pollution: the self-heal writeback infers a model's peak
     from the GLOBAL free-memory delta across its load. When another load
     overlaps that window the delta is polluted, and the writeback (which
     only ever raises the recorded value) would permanently inflate the
     estimate. The writeback must be skipped when any other load overlaps.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from muse.admin.operations import OperationError
from muse.cli_impl.load_director import LoadDirector


def _make_probe(gpu_free: float = 32.0, cpu_free: float = 64.0) -> MagicMock:
    probe = MagicMock()
    probe.gpu_free_gb.return_value = gpu_free
    probe.cpu_free_gb.return_value = cpu_free
    return probe


def _manifest(memory_gb: float, device: str = "cuda") -> dict:
    return {
        "model_id": "fake",
        "modality": "image/generation",
        "capabilities": {"memory_gb": memory_gb, "device": device},
    }


class TestInFlightReservation:
    def test_concurrent_cold_loads_do_not_over_admit(self):
        """10 GB free, 1 GB headroom -> 9 GB usable. Two 7 GB models fit
        alone but not together. Concurrent acquires must admit exactly one;
        the second gets a clean 503 because the first's in-flight memory is
        reserved (not visible in the live probe until its worker allocates).
        """
        probe = _make_probe(gpu_free=10.0)
        gate = threading.Event()
        started: list[str] = []
        started_lock = threading.Lock()

        def enable_fn(mid: str) -> int:
            with started_lock:
                started.append(mid)
            gate.wait(timeout=5)  # stay in-flight so the reservation is live
            return 9000

        director = LoadDirector(
            enable_fn=enable_fn, disable_fn=MagicMock(),
            memory_probe=probe, gpu_headroom_gb=1.0,
        )
        man = _manifest(memory_gb=7.0, device="cuda")
        results: dict[str, tuple] = {}

        def worker(mid: str) -> None:
            try:
                results[mid] = ("ok", director.acquire(mid, manifest=man))
            except OperationError as e:
                results[mid] = ("err", e.code)

        tA = threading.Thread(target=worker, args=("A",))
        tA.start()
        while not started:  # wait until A reserved + entered its blocking load
            time.sleep(0.005)
        tB = threading.Thread(target=worker, args=("B",))
        tB.start()
        tB.join(timeout=5)  # B should 503 fast (nothing evictable)

        assert results.get("B") == ("err", "model_too_large_for_device")
        gate.set()
        tA.join(timeout=5)
        assert results["A"][0] == "ok"
        assert started == ["A"]  # B never reached the load phase


class TestObservedPeakOverlapGate:
    def test_observed_peak_skipped_when_loads_overlap(self):
        """When a second load overlaps the first's free_before..free_after
        window, the global-delta measurement is polluted, so the observed
        peak writeback must be skipped for BOTH loads.
        """
        probe = _make_probe(gpu_free=100.0)  # tons of room, no eviction
        a_in_enable = threading.Event()
        release_a = threading.Event()

        def enable_fn(mid: str) -> int:
            if mid == "A":
                a_in_enable.set()
                release_a.wait(timeout=5)  # keep A in-flight while B loads
            return 9000

        director = LoadDirector(
            enable_fn=enable_fn, disable_fn=MagicMock(), memory_probe=probe,
        )
        director.observed_peak = MagicMock()  # spy on the writeback scheduler
        man = _manifest(memory_gb=1.0, device="cuda")

        tA = threading.Thread(target=lambda: director.acquire("A", manifest=man))
        tA.start()
        a_in_enable.wait(timeout=5)
        director.acquire("B", manifest=man)  # loads fully while A in-flight
        release_a.set()
        tA.join(timeout=5)

        called_models = [c.args[0] for c in director.observed_peak.call_args_list]
        assert "A" not in called_models  # A's window overlapped B
        assert "B" not in called_models  # B loaded while A in-flight

    def test_observed_peak_written_for_a_solo_load(self):
        """The common case (no overlap) still writes the observed peak."""
        director = LoadDirector(
            enable_fn=MagicMock(return_value=9000),
            disable_fn=MagicMock(),
            memory_probe=_make_probe(gpu_free=100.0),
        )
        director.observed_peak = MagicMock()
        director.acquire("solo", manifest=_manifest(memory_gb=1.0, device="cuda"))
        assert director.observed_peak.call_count == 1
        call = director.observed_peak.call_args
        assert call.args[0] == "solo"
        assert call.kwargs.get("device") == "cuda"
        assert "observed_peak_bytes" in call.kwargs


class TestEvictionEpoch:
    def test_eviction_bumps_epoch_so_overlapping_load_skips_writeback(self):
        """An eviction frees VRAM and pollutes a concurrent load's
        free_before..free_after delta, so it must bump the pollution epoch
        (like a concurrent load). Otherwise the overlapping load would record
        an under-estimated peak from a delta the eviction shrank.
        """
        free = {"v": 10.0}
        probe = MagicMock()
        probe.gpu_free_gb.side_effect = lambda: free["v"]
        probe.cpu_free_gb.return_value = 64.0

        def disable(_mid):
            free["v"] = 20.0  # the evicted victim released memory

        director = LoadDirector(
            enable_fn=MagicMock(return_value=9000), disable_fn=disable,
            memory_probe=probe, gpu_headroom_gb=1.0,
        )
        # Load an idle victim Z (refcount 0 after release) so eviction has a
        # candidate.
        director.acquire("Z", manifest=_manifest(memory_gb=2.0, device="cuda"))
        director.release("Z")
        free["v"] = 1.0  # force a real eviction instead of the live-fit fast path

        epoch_before = director._inflight_epoch
        director._evict_lru_until_fits(
            model_id="M", shortfall_gb=1.0, device="cuda", required_gb=1.0,
        )
        assert director._inflight_epoch > epoch_before
