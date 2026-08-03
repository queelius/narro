"""Per-model log hub: byte-bounded ring buffer plus pub/sub for live tails.

Stdlib-only. Does not import torch, fastapi, or any other observability
module. One lock guards all mutation (buffers, byte counts, subscriber
sets) so snapshot/subscribe/unsubscribe/drop are all safe against a
concurrent append from another thread.
"""

from __future__ import annotations

import collections
import queue
import threading

# Bounds a stalled subscriber's queue to this many buffered lines. A
# healthy consumer (the SSE tail polls hub.subscribe() every 250ms, see
# dashboard.py) drains far faster than workers emit, so this only bites
# a disconnected-but-not-yet-unsubscribed or genuinely slow client, and
# caps its memory growth instead of letting it accumulate for the life
# of the connection.
SUBSCRIBER_QUEUE_MAXSIZE = 1024
_TRUNCATION_MARKER = b"...[truncated]"


def _truncate_line(line: str, max_bytes: int) -> str:
    """Fit one line into ``max_bytes`` without emitting invalid UTF-8."""
    raw = line.encode("utf-8")
    if len(raw) <= max_bytes:
        return line
    if max_bytes <= len(_TRUNCATION_MARKER):
        return _TRUNCATION_MARKER[:max_bytes].decode("ascii")
    prefix = raw[: max_bytes - len(_TRUNCATION_MARKER)].decode(
        "utf-8", errors="ignore",
    )
    return prefix + _TRUNCATION_MARKER.decode("ascii")


class LogHub:
    """Buffers recent log lines per model_id and fans them out to subscribers.

    - append(model_id, line): buffer the line (evicting oldest lines once the
      running byte count exceeds buffer_bytes) then publish it to every
      subscriber for that model_id.
    - snapshot(model_id): a point-in-time copy of the current buffer.
    - subscribe(model_id)/unsubscribe(model_id, q): live-tail registration.
    - drop(model_id): remove a model's buffer and all its subscribers.
    """

    def __init__(self, *, buffer_bytes: int = 65536) -> None:
        if (
            isinstance(buffer_bytes, bool)
            or not isinstance(buffer_bytes, int)
            or buffer_bytes <= 0
        ):
            raise ValueError("buffer_bytes must be a positive integer")
        self._buffer_bytes = buffer_bytes
        self._lock = threading.Lock()
        self._buffers: dict[str, collections.deque] = {}
        self._byte_counts: dict[str, int] = {}
        self._subscribers: dict[str, set[queue.Queue]] = {}

    def append(self, model_id: str, line: str) -> None:
        line = _truncate_line(line, self._buffer_bytes)
        with self._lock:
            buf = self._buffers.setdefault(model_id, collections.deque())
            buf.append(line)
            self._byte_counts[model_id] = self._byte_counts.get(model_id, 0) + len(
                line.encode("utf-8")
            )

            # Every individual line is already capped, so evicting older
            # lines makes the advertised byte bound unconditional.
            while self._byte_counts[model_id] > self._buffer_bytes:
                oldest = buf.popleft()
                self._byte_counts[model_id] -= len(oldest.encode("utf-8"))

            subscribers = self._subscribers.get(model_id, ())
            for q in subscribers:
                try:
                    q.put_nowait(line)
                except queue.Full:
                    # A slow/full subscriber must not block the reader
                    # thread or the append; drop the line for that
                    # subscriber only.
                    pass

    def snapshot(self, model_id: str) -> list[str]:
        with self._lock:
            buf = self._buffers.get(model_id)
            return list(buf) if buf is not None else []

    def subscribe(self, model_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE_MAXSIZE)
        with self._lock:
            self._subscribers.setdefault(model_id, set()).add(q)
        return q

    def subscribe_with_snapshot(
        self, model_id: str,
    ) -> tuple[list[str], queue.Queue]:
        """Atomically subscribe and snapshot buffered history.

        The lock gives each append exactly one side of the handoff: an append
        that wins the lock first appears in ``history``; one that follows the
        subscription appears in the queue. This removes the snapshot-then-
        subscribe gap without duplicating a line in both channels.
        """
        q: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE_MAXSIZE)
        with self._lock:
            self._subscribers.setdefault(model_id, set()).add(q)
            buf = self._buffers.get(model_id)
            history = list(buf) if buf is not None else []
        return history, q

    def unsubscribe(self, model_id: str, q: queue.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(model_id)
            if subscribers is not None:
                subscribers.discard(q)
                if not subscribers:
                    self._subscribers.pop(model_id, None)

    def drop(self, model_id: str) -> None:
        with self._lock:
            self._buffers.pop(model_id, None)
            self._byte_counts.pop(model_id, None)
            self._subscribers.pop(model_id, None)
