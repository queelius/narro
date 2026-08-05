"""Implementation for the local ``muse telemetry`` command group."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from muse.core import config
from muse.observability.store import TelemetryStore


_DURATION_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}
_REQUEST_NAMES = {
    "audio/speech": "Generate speech",
    "audio/transcriptions": "Transcribe audio",
    "audio/transcription": "Transcribe audio",
    "images/segmentations": "Segment image",
    "images/segmentation": "Segment image",
    "images/segment": "Segment image",
    "images/generations": "Generate image",
    "chat/completions": "Chat completion",
    "embeddings": "Create embedding",
}


def parse_duration(value: str) -> float:
    """Parse compact durations such as 90s, 6h, 7d, or raw seconds."""
    raw = str(value).strip().lower()
    if not raw:
        raise ValueError("duration must not be empty")
    unit = raw[-1]
    multiplier = _DURATION_UNITS.get(unit, 1.0)
    number = raw[:-1] if unit in _DURATION_UNITS else raw
    try:
        result = float(number) * multiplier
    except ValueError as exc:
        raise ValueError(f"invalid duration {value!r}; use forms like 1h or 7d") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("duration must be positive and finite")
    return result


def telemetry_path() -> Path:
    return Path(config.get("paths.catalog_dir")).expanduser() / "telemetry.db"


def _open_existing() -> TelemetryStore | None:
    path = telemetry_path()
    return TelemetryStore(path) if path.is_file() else None


def _request_name(modality: Any) -> str:
    value = str(modality or "Unknown")
    return _REQUEST_NAMES.get(value, value.replace("_", " ").replace("/", " ").title())


def _fmt_ms(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "--"
    return f"{value:.1f}" if value < 100 else f"{value:.0f}"


def _fmt_gb(value: Any) -> str:
    return "--" if not isinstance(value, (int, float)) else f"{value:.2f}"


def _decode_evictions(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return [str(raw)]
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def _print_table(
    headers: list[str], rows: list[list[Any]], out: TextIO | None = None,
) -> None:
    out = out or sys.stdout
    rendered = [["--" if value is None else str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), file=out)
    print("  ".join("-" * width for width in widths), file=out)
    for row in rendered:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)), file=out)


def run_status(*, as_json: bool = False) -> int:
    store = _open_existing()
    if store is None:
        payload = {"path": str(telemetry_path()), "exists": False, "total": 0, "counts": {}}
    else:
        try:
            payload = {"exists": True, **store.status()}
        finally:
            store.close()
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Telemetry DB: {payload['path']}")
        print(f"Events: {payload['total']}")
        for event_type, count in payload.get("counts", {}).items():
            print(f"  {event_type}: {count}")
        if not payload["exists"]:
            print("No telemetry has been recorded yet. Start `muse serve` with telemetry enabled.")
    return 0


def run_summary(*, since: str = "24h", as_json: bool = False) -> int:
    seconds = parse_duration(since)
    store = _open_existing()
    rows: list[dict[str, Any]] = []
    if store is not None:
        try:
            rows = store.request_report(since_ts=time.time() - seconds)
        finally:
            store.close()
    for row in rows:
        row["evicted_models"] = _decode_evictions(row.get("evicted_models"))
    if as_json:
        print(json.dumps({"window_seconds": seconds, "rows": rows}, indent=2))
        return 0
    if not rows:
        print(f"No request traces in the last {since}.")
        return 0
    table = []
    for row in rows:
        table.append([
            _request_name(row["modality"]), row["model_id"],
            _fmt_ms(row["cold_latency_ms"]), _fmt_ms(row["hot_latency_ms"]),
            _fmt_gb(row["peak_vram_gb"]),
            ", ".join(row["evicted_models"]) or "--", row["request_count"],
            row.get("basis", "measured"),
        ])
    _print_table(
        ["Request", "Model", "Cold ms", "Hot ms", "Peak VRAM GB", "Evicted model", "Samples", "Basis"],
        table,
    )
    return 0


def run_traces(
    *,
    since: str = "24h",
    limit: int = 50,
    model_id: str | None = None,
    modality: str | None = None,
    as_json: bool = False,
) -> int:
    seconds = parse_duration(since)
    store = _open_existing()
    rows: list[dict[str, Any]] = []
    if store is not None:
        try:
            rows = store.recent_requests(
                since_ts=time.time() - seconds,
                limit=limit,
                model_id=model_id,
                modality=modality,
            )
        finally:
            store.close()
    for row in rows:
        row["evicted_models"] = _decode_evictions(row.get("evicted_models"))
        if row.get("cold") is not None:
            row["cold"] = bool(row["cold"])
    if as_json:
        print(json.dumps({"window_seconds": seconds, "rows": rows}, indent=2))
        return 0
    if not rows:
        print(f"No request traces in the last {since}.")
        return 0
    table = []
    for row in rows:
        stamp = datetime.fromtimestamp(row["ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        state = "cold" if row.get("cold") else "hot" if row.get("cold") is not None else "legacy"
        table.append([
            stamp, _request_name(row["modality"]), row["model_id"], state,
            _fmt_ms(row["latency_ms"]), _fmt_ms(row["load_ms"]),
            _fmt_ms(row["forward_ms"]), _fmt_gb(row["peak_vram_gb"]),
            ", ".join(row["evicted_models"]) or "--", row["status"],
        ])
    _print_table(
        ["UTC", "Request", "Model", "State", "Total ms", "Load ms", "Forward ms", "Peak GB", "Evicted", "Status"],
        table,
    )
    return 0


def _downsample(values: list[float], width: int) -> list[float]:
    if len(values) <= width:
        return values
    result = []
    for index in range(width):
        start = index * len(values) // width
        end = max(start + 1, (index + 1) * len(values) // width)
        result.append(max(values[start:end]))
    return result


def run_vram(*, since: str = "1h", width: int = 72, as_json: bool = False) -> int:
    seconds = parse_duration(since)
    store = _open_existing()
    points: list[dict[str, Any]] = []
    if store is not None:
        try:
            points = store.series(
                "vram",
                since_ts=time.time() - seconds,
                bucket_seconds=max(seconds / width, 0.1),
            )["points"]
        finally:
            store.close()
    if as_json:
        print(json.dumps({"window_seconds": seconds, "points": points}, indent=2))
        return 0
    values = [
        float(row["peak"] if row["peak"] is not None else row["used"])
        for row in points
        if row["peak"] is not None or row["used"] is not None
    ]
    if not values:
        print(f"No GPU VRAM samples in the last {since}.")
        return 0
    values = _downsample(values, width)
    high = max(values)
    rows = 10
    print(f"GPU working set over {since} (peak {high:.2f} GB)")
    for level in range(rows, 0, -1):
        threshold = high * level / rows
        line = "".join("#" if value >= threshold else " " for value in values)
        print(f"{threshold:7.2f} |{line}")
    print("        +" + "-" * len(values))
    print(f"         oldest{' ' * max(1, len(values) - 12)}newest")
    return 0


def run_export(*, since: str, output: Path, format: str, force: bool = False) -> int:
    seconds = parse_duration(since)
    store = _open_existing()
    rows: list[dict[str, Any]] = []
    if store is not None:
        try:
            rows = store.export_events(since_ts=time.time() - seconds)
        finally:
            store.close()
    output = output.expanduser()
    if output.exists() and not force:
        raise ValueError(f"refusing to overwrite {output}; pass --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    elif format == "csv":
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
    else:
        raise ValueError("format must be json or csv")
    print(f"Exported {len(rows)} events to {output}")
    return 0


def run_prune(*, older_than: str, dry_run: bool = False) -> int:
    seconds = parse_duration(older_than)
    cutoff = time.time() - seconds
    store = _open_existing()
    if store is None:
        print("No telemetry database exists; nothing to prune.")
        return 0
    try:
        count = store.count_before(cutoff)
        if not dry_run:
            count = store.prune(cutoff)
    finally:
        store.close()
    action = "Would prune" if dry_run else "Pruned"
    print(f"{action} {count} events older than {older_than}.")
    return 0
