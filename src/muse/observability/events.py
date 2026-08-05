from __future__ import annotations
from typing import Any

EVENT_COLUMNS: tuple[str, ...] = (
    "ts", "type", "model_id", "pool", "gb", "latency_ms", "queued_ms",
    "status", "reason", "cold_load_seconds", "stream", "free_vram_gb",
    "free_ram_gb", "gpu_used_gb", "loaded_count", "in_flight_count", "modality",
    "request_id", "cold", "load_ms", "forward_ms", "peak_vram_gb",
    "evicted_models",
)
_FIELD_COLUMNS = frozenset(EVENT_COLUMNS) - {"ts", "type"}


def event_to_row(type: str, ts: float, **fields: Any) -> dict[str, Any]:
    """Build a full sparse row dict (every column present, None where unset)."""
    unknown = set(fields) - _FIELD_COLUMNS
    if unknown:
        raise ValueError(f"unknown telemetry field(s): {sorted(unknown)}")
    row: dict[str, Any] = {c: None for c in EVENT_COLUMNS}
    row["ts"] = ts
    row["type"] = type
    row.update(fields)
    return row
