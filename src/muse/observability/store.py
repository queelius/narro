from __future__ import annotations

import math
import pathlib
import sqlite3
import threading

from muse.observability.events import EVENT_COLUMNS

_COLUMN_LIST = ", ".join(EVENT_COLUMNS)
_PLACEHOLDER_LIST = ", ".join(f":{c}" for c in EVENT_COLUMNS)

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS events (
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    model_id TEXT,
    pool TEXT,
    gb REAL,
    latency_ms REAL,
    queued_ms REAL,
    status INTEGER,
    reason TEXT,
    cold_load_seconds REAL,
    stream INTEGER,
    free_vram_gb REAL,
    free_ram_gb REAL,
    gpu_used_gb REAL,
    loaded_count INTEGER,
    in_flight_count INTEGER,
    modality TEXT
)
"""

_CREATE_IDX_TS_SQL = "CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts)"
_CREATE_IDX_TYPE_SQL = "CREATE INDEX IF NOT EXISTS idx_events_type ON events (type)"

_INSERT_SQL = f"INSERT INTO events ({_COLUMN_LIST}) VALUES ({_PLACEHOLDER_LIST})"

# Bucket label is the bucket END: CAST(ts/:b AS INT)*:b + :b
_BUCKET_EXPR = "CAST(ts / :bucket AS INT) * :bucket + :bucket"

_SERIES_SQL = {
    "request_rate": f"""
        SELECT {_BUCKET_EXPR} AS t, COUNT(*) AS count
        FROM events
        WHERE type = 'request' AND ts >= :since
        GROUP BY t
        ORDER BY t
    """,
    "latency": f"""
        SELECT {_BUCKET_EXPR} AS t, AVG(latency_ms) AS avg, MAX(latency_ms) AS max
        FROM events
        WHERE type = 'request' AND ts >= :since
        GROUP BY t
        ORDER BY t
    """,
    "vram": f"""
        SELECT {_BUCKET_EXPR} AS t, AVG(free_vram_gb) AS avg
        FROM events
        WHERE type = 'sample' AND ts >= :since
        GROUP BY t
        ORDER BY t
    """,
    "ram": f"""
        SELECT {_BUCKET_EXPR} AS t, AVG(free_ram_gb) AS avg
        FROM events
        WHERE type = 'sample' AND ts >= :since
        GROUP BY t
        ORDER BY t
    """,
    "load_evict": f"""
        SELECT
            {_BUCKET_EXPR} AS t,
            SUM(CASE WHEN type = 'model_load' THEN 1 ELSE 0 END) AS loads,
            SUM(CASE WHEN type = 'model_evict' THEN 1 ELSE 0 END) AS evicts
        FROM events
        WHERE type IN ('model_load', 'model_evict') AND ts >= :since
        GROUP BY t
        ORDER BY t
    """,
}


class TelemetryStore:
    """SQLite-backed telemetry event store.

    Opened in WAL mode with check_same_thread=False since the recorder
    flush thread writes while request-handling threads read. Writes are
    guarded by an internal lock; sqlite3's own serialization does not
    protect multi-step operations (e.g. delete + changes count) from
    interleaving across threads.
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._closed = False
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            with self._lock:
                self._conn.execute(_CREATE_TABLE_SQL)
                self._conn.execute(_CREATE_IDX_TS_SQL)
                self._conn.execute(_CREATE_IDX_TYPE_SQL)
                # Migrate older DBs in place: add any EVENT_COLUMNS the existing
                # table lacks (new columns are always nullable in the sparse model,
                # so ALTER TABLE ADD COLUMN is safe and idempotent).
                have = {
                    row[1]
                    for row in self._conn.execute("PRAGMA table_info(events)")
                }
                for col in EVENT_COLUMNS:
                    if col not in have:
                        self._conn.execute(f"ALTER TABLE events ADD COLUMN {col}")
                self._conn.commit()
        except BaseException:
            self._closed = True
            self._conn.close()
            raise

    def _require_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("telemetry store is closed")

    def insert_many(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self._lock:
            self._require_open_locked()
            self._conn.executemany(_INSERT_SQL, rows)
            self._conn.commit()

    def prune(self, older_than_ts: float) -> int:
        try:
            cutoff = float(older_than_ts)
        except (TypeError, ValueError) as exc:
            raise ValueError("older_than_ts must be finite") from exc
        if not math.isfinite(cutoff):
            raise ValueError("older_than_ts must be finite")
        with self._lock:
            self._require_open_locked()
            cur = self._conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            self._conn.commit()
            return cur.rowcount

    def series(self, metric: str, since_ts: float, bucket_seconds: float) -> dict:
        sql = _SERIES_SQL.get(metric)
        if sql is None:
            raise ValueError(f"unknown telemetry metric: {metric!r}")
        try:
            since_value = float(since_ts)
            bucket_value = float(bucket_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("series bounds must be numeric") from exc
        if not math.isfinite(since_value):
            raise ValueError("since_ts must be finite")
        if not math.isfinite(bucket_value) or bucket_value <= 0:
            raise ValueError("bucket_seconds must be positive and finite")
        with self._lock:
            self._require_open_locked()
            cur = self._conn.execute(
                sql, {"bucket": bucket_value, "since": since_value},
            )
            columns = [d[0] for d in cur.description]
            points = [dict(zip(columns, row)) for row in cur.fetchall()]
        return {"metric": metric, "points": points}

    def summary_counts(self) -> dict:
        with self._lock:
            self._require_open_locked()
            (total,) = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return {"total": total}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._conn.close()
            finally:
                self._closed = True
