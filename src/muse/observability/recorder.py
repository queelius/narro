"""Fire-and-forget telemetry recorder.

`record(type, **fields)` is meant to be called from hot request-handling
paths, so it must never block and must never raise. Events are enqueued
onto a bounded queue and drained by a background daemon thread that
batches writes into the TelemetryStore. When the queue is full the event
is silently dropped and `dropped` is incremented, rather than blocking
the caller or losing more than the one event.
"""
from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Any

from muse.observability.events import event_to_row
from muse.observability.store import TelemetryStore

logger = logging.getLogger(__name__)

_DEFAULT_STOP_TIMEOUT = 5.0


class TelemetryRecorder:
    """Background-flushing telemetry recorder backed by a TelemetryStore."""

    def __init__(
        self,
        store: TelemetryStore,
        *,
        max_queue: int = 10000,
        flush_interval: float = 0.5,
        stop_timeout: float = _DEFAULT_STOP_TIMEOUT,
    ) -> None:
        if isinstance(max_queue, bool) or not isinstance(max_queue, int) or max_queue <= 0:
            raise ValueError("max_queue must be a positive integer")
        if (
            isinstance(flush_interval, bool)
            or not isinstance(flush_interval, (int, float))
            or not math.isfinite(flush_interval)
            or flush_interval <= 0
        ):
            raise ValueError("flush_interval must be a positive finite number")
        if (
            isinstance(stop_timeout, bool)
            or not isinstance(stop_timeout, (int, float))
            or not math.isfinite(stop_timeout)
            or stop_timeout <= 0
        ):
            raise ValueError("stop_timeout must be a positive finite number")
        self._store = store
        self._flush_interval = float(flush_interval)
        self._stop_timeout = float(stop_timeout)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        # Guards `dropped` increments: record() is called from arbitrary
        # request-handling threads across the process, so a bare `+= 1`
        # is a cross-thread read-modify-write that can under-report the
        # true drop count under concurrent callers.
        self._dropped_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Lifecycle calls can originate from supervisor cleanup, tests, or
        # same-interpreter reinitialization. Serializing start/stop prevents
        # a new flush thread from racing a timed join of its predecessor.
        self._lifecycle_lock = threading.Lock()
        # Setting the stopped state under the same short-held lock used by
        # record() closes the enqueue-after-final-flush race without ever
        # holding request threads behind store I/O or a thread join.
        self._record_lock = threading.Lock()
        # Public flush() and the background loop may be invoked at the same
        # time. Only one owner may drain the queue into the store at once.
        self._flush_lock = threading.Lock()

    def _mark_dropped(self) -> None:
        with self._dropped_lock:
            self.dropped += 1

    def record(self, type: str, **fields: Any) -> None:
        # event_to_row() raises ValueError on an unknown field name (a
        # typo'd kwarg). record() must never raise -- treat a bad field
        # the same as a dropped event rather than letting it escape into
        # a hot request-handling path.
        try:
            row = event_to_row(type, time.time(), **fields)
        except ValueError:
            logger.warning(
                "telemetry recorder: dropping event with unknown field(s) "
                "type=%r fields=%r", type, sorted(fields), exc_info=True,
            )
            self._mark_dropped()
            return
        with self._record_lock:
            if self._stop_event.is_set():
                self._mark_dropped()
                return
            try:
                self._queue.put_nowait(row)
            except queue.Full:
                self._mark_dropped()

    def flush(self) -> None:
        with self._flush_lock:
            rows: list[dict[str, Any]] = []
            while True:
                try:
                    rows.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if rows:
                self._store.insert_many(rows)

    def start(self) -> bool:
        """Start the flush thread; return whether it is running."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive() is True:
                # A bounded stop may return while a flush is still wedged.
                # That live thread is already committed to exit once the
                # store call settles; reporting a successful restart here
                # would leave the caller with no eventual recorder thread.
                return not self._stop_event.is_set()
            self._thread = None
            with self._record_lock:
                self._stop_event.clear()
            thread = threading.Thread(
                target=self._run, name="telemetry-recorder-flush", daemon=True
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                with self._record_lock:
                    self._stop_event.set()
                raise
            return True

    def stop(self) -> bool:
        """Stop and drain within a fixed bound.

        Returns false if a store call has wedged the flush thread past the
        deadline. In that case the live thread handle is deliberately kept,
        so a later cleanup attempt can retry and diagnostics never mistake a
        still-running thread for a completed shutdown.
        """
        with self._lifecycle_lock:
            with self._record_lock:
                self._stop_event.set()
            thread = self._thread
            if thread is threading.current_thread():
                logger.warning("telemetry recorder cannot join its own flush thread")
                return False
            if thread is not None:
                try:
                    thread.join(timeout=self._stop_timeout)
                except RuntimeError:
                    if thread.is_alive() is True:
                        logger.warning(
                            "could not join telemetry recorder flush thread",
                            exc_info=True,
                        )
                        return False
                if thread.is_alive() is True:
                    logger.warning(
                        "telemetry recorder flush thread did not stop within %.1fs",
                        self._stop_timeout,
                    )
                    return False
                if self._thread is thread:
                    self._thread = None
            # Final drain so nothing queued is silently lost on shutdown.
            try:
                self.flush()
            except Exception:
                logger.warning("telemetry recorder: final flush failed", exc_info=True)
            return True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._flush_interval)
            try:
                self.flush()
            except Exception:
                logger.warning("telemetry recorder: flush failed", exc_info=True)


class _NoopRecorder:
    """Silent stand-in used when telemetry is disabled or uninitialized."""

    dropped = 0

    def record(self, *a: Any, **k: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        return True


_NOOP = _NoopRecorder()
_RECORDER: TelemetryRecorder | _NoopRecorder | None = None
_RECORDER_LOCK = threading.Lock()


def init_recorder(
    store: TelemetryStore, *, enabled: bool = True,
) -> TelemetryRecorder | _NoopRecorder:
    """Install a recorder, stopping the prior global instance first."""
    global _RECORDER
    candidate: TelemetryRecorder | _NoopRecorder
    candidate = TelemetryRecorder(store) if enabled else _NoopRecorder()
    candidate.start()
    with _RECORDER_LOCK:
        previous = _RECORDER
        if previous is not None and not previous.stop():
            candidate.stop()
            raise RuntimeError("previous telemetry recorder did not stop")
        _RECORDER = candidate
    return candidate


def get_recorder() -> TelemetryRecorder | _NoopRecorder:
    if _RECORDER is None:
        return _NOOP
    return _RECORDER


def record(type: str, **fields: Any) -> None:
    get_recorder().record(type, **fields)


def reset_recorder(
    expected: TelemetryRecorder | _NoopRecorder | None = None,
) -> bool:
    """Stop and clear the global recorder.

    When ``expected`` is supplied, only that exact lifecycle owner may be
    reset. This prevents cleanup from an older supervisor instance stopping
    a recorder installed by a newer same-interpreter run.
    """
    global _RECORDER
    with _RECORDER_LOCK:
        current = _RECORDER
        if expected is not None and current is not expected:
            return False
        if current is None:
            return True
        _RECORDER = None
        if current.stop():
            return True
        # Keep a still-live recorder reachable for a later cleanup retry,
        # unless another initializer installed a replacement (the lock makes
        # that impossible in this block, but the guard documents ownership).
        if _RECORDER is None:
            _RECORDER = current
        return False
