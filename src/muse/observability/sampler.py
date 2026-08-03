"""Periodic sampler: free VRAM/RAM + loaded/in-flight model counts.

`Sampler` runs a daemon thread that periodically records a `sample`
telemetry event via the shared `record` function (Task 4's recorder).
Each sample captures a point-in-time view of resource pressure:

- `free_vram_gb`: live free VRAM via `gpu_free_gb()` (None on a CPU-only
  host or when pynvml is unavailable; passed through as-is, never
  coerced to 0.0, since that would fabricate data the nullable store
  column is designed to represent honestly).
- `free_ram_gb`: live free host RAM via `cpu_free_gb()`.
- `loaded_count`: number of currently loaded models (`len(loaded_fn())`).
- `in_flight_count`: number of in-flight requests (`inflight_fn()`).

`gpu_free_gb` and `cpu_free_gb` are imported at module top (not
called through an indirection layer) so tests can monkeypatch this
module's globals directly; `sample_once` must reference the bare
names so patched globals are actually observed at call time.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any, Callable

from muse.core.memory_probe import gpu_free_gb, cpu_free_gb
from muse.observability.recorder import record

logger = logging.getLogger(__name__)

_DEFAULT_STOP_TIMEOUT = 5.0


class Sampler:
    """Background daemon that periodically records a `sample` event.

    `stop_event` is an optional external `threading.Event`. Pass the
    supervisor-wide `state.stop_event` so a single Ctrl+C / SIGTERM
    unblocks this sampler's loop along with every other supervisor-owned
    daemon thread (mirrors `IdleSweeper`'s `stop_event` parameter). If
    omitted, the sampler creates its own private Event (unchanged
    behavior for existing callers/tests).
    """

    def __init__(
        self,
        *,
        interval: float,
        loaded_fn: Callable[[], dict[str, Any]],
        inflight_fn: Callable[[], int],
        record_fn: Callable[..., None] = record,
        stop_event: threading.Event | None = None,
        stop_timeout: float = _DEFAULT_STOP_TIMEOUT,
    ) -> None:
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError("interval must be a positive finite number")
        if (
            isinstance(stop_timeout, bool)
            or not isinstance(stop_timeout, (int, float))
            or not math.isfinite(stop_timeout)
            or stop_timeout <= 0
        ):
            raise ValueError("stop_timeout must be a positive finite number")
        self.interval = float(interval)
        self.loaded_fn = loaded_fn
        self.inflight_fn = inflight_fn
        self.record_fn = record_fn
        self._owns_stop = stop_event is None
        self._stop = stop_event if stop_event is not None else threading.Event()
        self._stop_timeout = float(stop_timeout)
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()

    def sample_once(self) -> None:
        free_vram_gb = gpu_free_gb()
        free_ram_gb = cpu_free_gb()
        loaded_count = len(self.loaded_fn())
        in_flight_count = self.inflight_fn()
        self.record_fn(
            "sample",
            free_vram_gb=free_vram_gb,
            free_ram_gb=free_ram_gb,
            loaded_count=loaded_count,
            in_flight_count=in_flight_count,
        )

    def start(self) -> bool:
        """Start sampling unless an externally-owned stop is already set."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive() is True:
                # A thread still unwinding after a timed-out stop is not a
                # successful restart: it will exit as soon as its current
                # sample settles because the stop Event remains set.
                return not self._stop.is_set()
            self._thread = None
            if self._owns_stop:
                self._stop.clear()
            elif self._stop.is_set():
                # Clearing a supervisor-owned Event here could resurrect its
                # monitor/sweeper after SIGINT raced telemetry initialization.
                return False
            thread = threading.Thread(
                target=self._run, name="telemetry-sampler", daemon=True
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                raise
            return True

    def stop(self) -> bool:
        """Stop within a fixed bound, retaining a live handle on timeout."""
        with self._lifecycle_lock:
            self._stop.set()
            thread = self._thread
            if thread is threading.current_thread():
                logger.warning("telemetry sampler cannot join its own thread")
                return False
            if thread is not None:
                try:
                    thread.join(timeout=self._stop_timeout)
                except RuntimeError:
                    if thread.is_alive() is True:
                        logger.warning(
                            "could not join telemetry sampler thread",
                            exc_info=True,
                        )
                        return False
                if thread.is_alive() is True:
                    logger.warning(
                        "telemetry sampler thread did not stop within %.1fs",
                        self._stop_timeout,
                    )
                    return False
                if self._thread is thread:
                    self._thread = None
            return True

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample_once()
            except Exception:
                logger.warning("sampler: sample_once failed", exc_info=True)
