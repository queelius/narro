import pytest

from muse.observability.store import TelemetryStore
from muse.observability.events import event_to_row


@pytest.fixture
def store(tmp_path):
    s = TelemetryStore(tmp_path / "t.db")
    yield s
    s.close()


def test_insert_and_request_rate_bucketing(store):
    rows = [event_to_row("request", ts, model_id="m", latency_ms=10.0, status=200)
            for ts in (100.0, 101.0, 102.0, 160.0)]
    store.insert_many(rows)
    out = store.series("request_rate", since_ts=0.0, bucket_seconds=60.0)
    assert out["metric"] == "request_rate"
    # bucket [60,120) has 3 requests, [120,180) has 1
    counts = {p["t"]: p["count"] for p in out["points"]}
    assert counts[120.0] == 3 and counts[180.0] == 1


def test_latency_series(store):
    store.insert_many([event_to_row("request", 61.0, model_id="m", latency_ms=x, status=200)
                       for x in (10.0, 20.0, 30.0)])
    out = store.series("latency", since_ts=0.0, bucket_seconds=60.0)
    p = out["points"][0]
    assert p["avg"] == 20.0 and p["max"] == 30.0


def test_prune(store):
    store.insert_many([event_to_row("sample", ts, free_vram_gb=1.0) for ts in (10.0, 5000.0)])
    removed = store.prune(older_than_ts=100.0)
    assert removed == 1
    assert store.summary_counts()["total"] == 1


@pytest.mark.parametrize("cutoff", [float("nan"), float("inf"), "invalid"])
def test_prune_rejects_nonfinite_or_nonnumeric_cutoff(store, cutoff):
    with pytest.raises(ValueError, match="older_than_ts must be finite"):
        store.prune(cutoff)


def test_vram_series(store):
    store.insert_many([
        event_to_row("sample", 61.0, free_vram_gb=4.0),
        event_to_row("sample", 65.0, free_vram_gb=6.0),
    ])
    out = store.series("vram", since_ts=0.0, bucket_seconds=60.0)
    assert out["metric"] == "vram"
    p = out["points"][0]
    # bucket [60,120) labeled by its END, per the bucketing convention
    # exercised in test_insert_and_request_rate_bucketing.
    assert p["t"] == 120.0
    assert p["avg"] == 5.0


def test_vram_series_recovers_legacy_used_values_after_one_new_sample(store):
    store.insert_many([
        event_to_row("sample", 61.0, free_vram_gb=8.0),
        event_to_row("sample", 65.0, free_vram_gb=6.0, gpu_used_gb=6.0),
    ])
    point = store.series("vram", since_ts=0, bucket_seconds=60)["points"][0]
    assert point["used"] == 5.0
    samples = store.samples(since_ts=0)
    assert samples[0]["gpu_used_gb"] == 4.0


def test_ram_series(store):
    store.insert_many([
        event_to_row("sample", 61.0, free_ram_gb=8.0),
        event_to_row("sample", 65.0, free_ram_gb=12.0),
    ])
    out = store.series("ram", since_ts=0.0, bucket_seconds=60.0)
    assert out["metric"] == "ram"
    p = out["points"][0]
    assert p["t"] == 120.0
    assert p["avg"] == 10.0


def test_load_evict_series(store):
    store.insert_many([
        event_to_row("model_load", 61.0, model_id="m1"),
        event_to_row("model_load", 65.0, model_id="m2"),
        event_to_row("model_evict", 70.0, model_id="m1"),
    ])
    out = store.series("load_evict", since_ts=0.0, bucket_seconds=60.0)
    assert out["metric"] == "load_evict"
    p = out["points"][0]
    assert p["t"] == 120.0
    assert p["loads"] == 2
    assert p["evicts"] == 1


def test_series_unknown_metric_raises(store):
    with pytest.raises(ValueError):
        store.series("bogus", since_ts=0.0, bucket_seconds=60.0)


@pytest.mark.parametrize(
    "since,bucket",
    [
        (float("nan"), 60.0),
        (float("inf"), 60.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, float("inf")),
    ],
)
def test_series_rejects_invalid_numeric_bounds(store, since, bucket):
    with pytest.raises(ValueError):
        store.series("request_rate", since_ts=since, bucket_seconds=bucket)


def test_close_is_idempotent_and_later_operations_fail_clearly(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")

    store.close()
    store.close()

    with pytest.raises(RuntimeError, match="telemetry store is closed"):
        store.summary_counts()
    with pytest.raises(RuntimeError, match="telemetry store is closed"):
        store.insert_many([event_to_row("request", 1.0, model_id="m")])


def test_request_report_and_recent_traces(store):
    store.insert_many([
        event_to_row(
            "request", 100.0, request_id="cold-1", model_id="tts",
            modality="audio/speech", cold=True, latency_ms=1200.0,
            load_ms=900.0, forward_ms=300.0, peak_vram_gb=4.25,
            evicted_models='["whisper"]', status=200,
        ),
        event_to_row(
            "request", 110.0, request_id="hot-1", model_id="tts",
            modality="audio/speech", cold=False, latency_ms=250.0,
            load_ms=0.0, forward_ms=240.0, peak_vram_gb=4.0,
            evicted_models="[]", status=200,
        ),
    ])

    report = store.request_report(since_ts=0.0)
    assert report == [{
        "modality": "audio/speech",
        "model_id": "tts",
        "cold_count": 1,
        "hot_count": 1,
        "cold_latency_ms": 1200.0,
        "hot_latency_ms": 250.0,
        "peak_vram_gb": 4.25,
        "request_count": 2,
        "evicted_models": '["whisper"]',
        "estimated": False,
        "legacy_count": 0,
        "basis": "measured",
    }]
    traces = store.recent_requests(since_ts=0.0, limit=1)
    assert traces[0]["request_id"] == "hot-1"


def test_store_migrates_request_trace_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE events (ts REAL NOT NULL, type TEXT NOT NULL)")
    conn.commit()
    conn.close()

    migrated = TelemetryStore(path)
    try:
        migrated.insert_many([
            event_to_row("request", 1.0, request_id="r", cold=True),
        ])
        assert migrated.recent_requests(since_ts=0, limit=10)[0]["request_id"] == "r"
    finally:
        migrated.close()


def test_request_report_estimates_legacy_cold_hot_and_eviction(store):
    store.insert_many([
        event_to_row(
            "request", 100.0, model_id="new", modality="audio/speech",
            latency_ms=200.0, status=200,
        ),
        event_to_row(
            "model_load", 90.0, model_id="new", cold_load_seconds=1.0,
        ),
        event_to_row(
            "model_evict", 89.0, model_id="old", reason="evicted_for_new",
        ),
    ])

    row = store.request_report(since_ts=0)[0]

    assert row["cold_latency_ms"] == 1200.0
    assert row["hot_latency_ms"] == 200.0
    assert row["cold_count"] == 1
    assert row["legacy_count"] == 1
    assert row["evicted_models"] == '["old"]'
    assert row["basis"] == "estimated"
