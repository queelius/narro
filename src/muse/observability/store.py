from __future__ import annotations

import json
import math
import pathlib
import sqlite3
import threading
from typing import Any

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
    modality TEXT,
    request_id TEXT,
    cold INTEGER,
    load_ms REAL,
    forward_ms REAL,
    peak_vram_gb REAL,
    evicted_models TEXT
)
"""

_CREATE_IDX_TS_SQL = "CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts)"
_CREATE_IDX_TYPE_SQL = "CREATE INDEX IF NOT EXISTS idx_events_type ON events (type)"
_CREATE_IDX_REQUEST_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_events_request_ts ON events (type, ts DESC)"
)

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
        SELECT {_BUCKET_EXPR} AS t,
               AVG(free_vram_gb) AS avg,
               AVG(free_vram_gb) AS free,
               AVG(COALESCE(
                   gpu_used_gb,
                   (SELECT MAX(gpu_used_gb + free_vram_gb)
                    FROM events
                    WHERE type = 'sample' AND gpu_used_gb IS NOT NULL)
                   - free_vram_gb
               )) AS used,
               MAX(COALESCE(
                   gpu_used_gb,
                   (SELECT MAX(gpu_used_gb + free_vram_gb)
                    FROM events
                    WHERE type = 'sample' AND gpu_used_gb IS NOT NULL)
                   - free_vram_gb
               )) AS peak
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
                self._conn.execute(_CREATE_IDX_REQUEST_SQL)
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

    def status(self) -> dict[str, Any]:
        """Return compact database bounds and per-event counts."""
        with self._lock:
            self._require_open_locked()
            total, first_ts, last_ts = self._conn.execute(
                "SELECT COUNT(*), MIN(ts), MAX(ts) FROM events"
            ).fetchone()
            counts = {
                row[0]: row[1]
                for row in self._conn.execute(
                    "SELECT type, COUNT(*) FROM events GROUP BY type ORDER BY type"
                )
            }
        return {
            "path": self._path,
            "total": total,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "counts": counts,
        }

    def count_before(self, cutoff_ts: float) -> int:
        cutoff = _finite_float(cutoff_ts, "cutoff_ts")
        with self._lock:
            self._require_open_locked()
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE ts < ?", (cutoff,),
            ).fetchone()
        return int(count)

    def recent_requests(
        self,
        *,
        since_ts: float,
        limit: int = 100,
        model_id: str | None = None,
        modality: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return newest request traces with optional exact filters."""
        since = _finite_float(since_ts, "since_ts")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        clauses = ["type = 'request'", "ts >= ?"]
        params: list[Any] = [since]
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if modality is not None:
            clauses.append("modality = ?")
            params.append(modality)
        params.append(limit)
        sql = f"""
            SELECT ts, request_id, modality, model_id, cold, latency_ms,
                   load_ms, forward_ms, queued_ms, peak_vram_gb,
                   evicted_models, status, stream
            FROM events
            WHERE {' AND '.join(clauses)}
            ORDER BY ts DESC, rowid DESC
            LIMIT ?
        """
        with self._lock:
            self._require_open_locked()
            cur = self._conn.execute(sql, params)
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def request_report(self, *, since_ts: float) -> list[dict[str, Any]]:
        """Aggregate the blog-ready cold/hot report by request + model."""
        since = _finite_float(since_ts, "since_ts")
        sql = """
            SELECT modality, model_id,
                   SUM(CASE WHEN cold = 1 THEN 1 ELSE 0 END) AS cold_count,
                   SUM(CASE WHEN cold = 0 THEN 1 ELSE 0 END) AS hot_count,
                   AVG(CASE WHEN cold = 1 THEN latency_ms END) AS cold_latency_ms,
                   AVG(CASE WHEN cold = 0 THEN latency_ms END) AS hot_latency_ms,
                   MAX(peak_vram_gb) AS peak_vram_gb,
                   SUM(CASE WHEN cold IS NULL THEN 1 ELSE 0 END) AS legacy_count,
                   AVG(CASE WHEN cold IS NULL THEN latency_ms END) AS legacy_latency_ms,
                   COUNT(*) AS request_count
            FROM events
            WHERE type = 'request' AND ts >= ?
            GROUP BY modality, model_id
            ORDER BY modality, model_id
        """
        evicted_sql = """
            SELECT modality, model_id, evicted_models
            FROM events
            WHERE type = 'request' AND ts >= ?
              AND evicted_models IS NOT NULL
              AND evicted_models NOT IN ('', '[]')
            ORDER BY ts DESC, rowid DESC
        """
        load_sql = """
            SELECT model_id, COUNT(*) AS load_count,
                   AVG(cold_load_seconds) AS avg_load_seconds
            FROM events
            WHERE type = 'model_load' AND ts >= ?
            GROUP BY model_id
        """
        legacy_evict_sql = """
            SELECT model_id, reason
            FROM events
            WHERE type = 'model_evict' AND ts >= ?
              AND reason LIKE 'evicted_for_%'
            ORDER BY ts DESC, rowid DESC
        """
        with self._lock:
            self._require_open_locked()
            cur = self._conn.execute(sql, (since,))
            columns = [d[0] for d in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
            last_evicted: dict[tuple[Any, Any], Any] = {}
            for modality, model_id, evicted in self._conn.execute(
                evicted_sql, (since,),
            ):
                last_evicted.setdefault((modality, model_id), evicted)
            loads = {
                model_id: (load_count, avg_load_seconds)
                for model_id, load_count, avg_load_seconds in self._conn.execute(
                    load_sql, (since,),
                )
            }
            legacy_evicted: dict[str, list[str]] = {}
            for victim, reason in self._conn.execute(legacy_evict_sql, (since,)):
                target = str(reason).removeprefix("evicted_for_")
                victims = legacy_evicted.setdefault(target, [])
                if victim not in victims:
                    victims.append(victim)
        for row in rows:
            row["evicted_models"] = last_evicted.get(
                (row["modality"], row["model_id"]),
            )
            row["estimated"] = False
            legacy_count = row.pop("legacy_count")
            legacy_latency_ms = row.pop("legacy_latency_ms")
            measured_count = row["hot_count"] + row["cold_count"]
            if legacy_count:
                load_count, avg_load_seconds = loads.get(
                    row["model_id"], (0, None),
                )
                if row["hot_latency_ms"] is None:
                    row["hot_latency_ms"] = legacy_latency_ms
                    row["estimated"] = True
                if (
                    row["cold_latency_ms"] is None
                    and legacy_latency_ms is not None
                    and avg_load_seconds is not None
                ):
                    row["cold_latency_ms"] = (
                        legacy_latency_ms + avg_load_seconds * 1000.0
                    )
                    row["cold_count"] = load_count
                    row["estimated"] = True
                if row["evicted_models"] is None:
                    victims = legacy_evicted.get(row["model_id"], [])
                    if victims:
                        row["evicted_models"] = json.dumps(
                            victims, separators=(",", ":"),
                        )
                        row["estimated"] = True
            row["legacy_count"] = legacy_count
            if legacy_count and measured_count:
                row["basis"] = "mixed"
            elif row["estimated"]:
                row["basis"] = "estimated"
            else:
                row["basis"] = "measured"
        return rows

    def samples(self, *, since_ts: float, limit: int = 5000) -> list[dict[str, Any]]:
        """Return chronological resource samples for exports and graphs."""
        since = _finite_float(since_ts, "since_ts")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        with self._lock:
            self._require_open_locked()
            cur = self._conn.execute(
                """
                SELECT ts, free_vram_gb, gpu_used_gb, free_ram_gb,
                       loaded_count, in_flight_count
                FROM events
                WHERE type = 'sample' AND ts >= ?
                ORDER BY ts DESC, rowid DESC
                LIMIT ?
                """,
                (since, limit),
            )
            columns = [d[0] for d in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        rows.reverse()
        totals = [
            row["gpu_used_gb"] + row["free_vram_gb"]
            for row in rows
            if row["gpu_used_gb"] is not None and row["free_vram_gb"] is not None
        ]
        if totals:
            total_vram_gb = max(totals)
            for row in rows:
                if row["gpu_used_gb"] is None and row["free_vram_gb"] is not None:
                    row["gpu_used_gb"] = max(
                        0.0, total_vram_gb - row["free_vram_gb"],
                    )
        return rows

    def export_events(self, *, since_ts: float) -> list[dict[str, Any]]:
        """Return all retained rows in stable chronological order."""
        since = _finite_float(since_ts, "since_ts")
        with self._lock:
            self._require_open_locked()
            cur = self._conn.execute(
                f"SELECT {_COLUMN_LIST} FROM events WHERE ts >= ? "
                "ORDER BY ts, rowid",
                (since,),
            )
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._conn.close()
            finally:
                self._closed = True


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
