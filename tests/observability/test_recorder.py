import threading
from unittest.mock import MagicMock

import pytest

from muse.observability.store import TelemetryStore
from muse.observability import recorder as rec


@pytest.fixture(autouse=True)
def _reset():
    rec.reset_recorder(); yield; rec.reset_recorder()


def test_record_enqueues_and_flush_writes(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store, flush_interval=0.05)
    r.record("request", model_id="m", latency_ms=5.0, status=200)
    r.flush()
    assert store.summary_counts()["total"] == 1
    r.stop(); store.close()


def test_overflow_drops_not_raises(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store, max_queue=2)
    for _ in range(10):
        r.record("sample", free_vram_gb=1.0)   # must never raise
    assert r.dropped >= 1
    r.stop(); store.close()


def test_module_record_is_noop_until_init(tmp_path):
    rec.record("request", model_id="m")   # no recorder yet -> silent no-op, no raise
    assert rec.get_recorder().dropped == 0


def test_disabled_init_is_noop(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    rec.init_recorder(store, enabled=False)
    rec.record("request", model_id="m")
    assert store.summary_counts()["total"] == 0


def test_record_with_unknown_field_does_not_raise_and_counts_as_dropped(tmp_path):
    """Regression: event_to_row() ran BEFORE the try/except in record(),
    so an unknown kwarg (e.g. a typo'd field name) raised ValueError
    straight out of record(), violating its never-raises contract.
    """
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store)
    r.record("model_load", bogus_field=1)  # must not raise
    assert r.dropped == 1
    r.stop(); store.close()


def test_dropped_counter_increments_are_lock_protected(tmp_path):
    """Sanity check that the dropped counter still increments correctly
    under normal (single-threaded) use across both drop sites: queue
    overflow and a bad field name.
    """
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store, max_queue=1)
    assert hasattr(r, "_dropped_lock")
    r.record("sample", free_vram_gb=1.0)
    r.record("sample", free_vram_gb=1.0)  # queue full -> dropped
    r.record("model_load", bogus_field=1)  # bad field -> dropped
    assert r.dropped == 2
    r.stop(); store.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_queue": 0}, "max_queue"),
        ({"max_queue": -1}, "max_queue"),
        ({"flush_interval": 0}, "flush_interval"),
        ({"flush_interval": float("inf")}, "flush_interval"),
        ({"flush_interval": float("nan")}, "flush_interval"),
        ({"stop_timeout": 0}, "stop_timeout"),
    ],
)
def test_constructor_rejects_unbounded_or_busy_loop_values(tmp_path, kwargs, message):
    store = TelemetryStore(tmp_path / "t.db")
    try:
        with pytest.raises(ValueError, match=message):
            rec.TelemetryRecorder(store, **kwargs)
    finally:
        store.close()


def test_stop_rejects_late_records_and_drains_existing_rows(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store)
    r.record("request", model_id="before")

    assert r.stop() is True
    r.record("request", model_id="after")

    assert store.summary_counts()["total"] == 1
    assert r.dropped == 1
    assert r._queue.empty()
    store.close()


def test_stop_retains_thread_handle_when_bounded_join_times_out(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store, stop_timeout=0.01)
    wedged = MagicMock()
    wedged.is_alive.return_value = True
    r._thread = wedged

    assert r.stop() is False
    assert r._thread is wedged
    wedged.join.assert_called_once_with(timeout=0.01)
    store.close()


def test_start_does_not_claim_restart_while_timed_out_stop_is_unwinding(
    tmp_path,
):
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store, stop_timeout=0.01)
    wedged = MagicMock()
    wedged.is_alive.return_value = True
    r._thread = wedged

    assert r.stop() is False
    assert r._stop_event.is_set()
    assert r.start() is False
    assert r._thread is wedged

    store.close()


def test_stop_clears_an_inert_thread_whose_start_never_completed(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    r = rec.TelemetryRecorder(store)
    r._thread = threading.Thread(target=lambda: None)

    assert r.stop() is True
    assert r._thread is None
    store.close()


def test_init_replaces_and_stops_previous_recorder(tmp_path):
    first_store = TelemetryStore(tmp_path / "first.db")
    second_store = TelemetryStore(tmp_path / "second.db")
    try:
        first = rec.init_recorder(first_store)
        first_thread = first._thread

        second = rec.init_recorder(second_store)

        assert rec.get_recorder() is second
        assert first_thread is not None and not first_thread.is_alive()
        assert first._thread is None
    finally:
        rec.reset_recorder()
        first_store.close()
        second_store.close()


def test_expected_reset_cannot_stop_newer_owner(tmp_path):
    first_store = TelemetryStore(tmp_path / "first.db")
    second_store = TelemetryStore(tmp_path / "second.db")
    try:
        first = rec.init_recorder(first_store)
        second = rec.init_recorder(second_store)

        assert rec.reset_recorder(expected=first) is False
        assert rec.get_recorder() is second
        assert second._thread is not None and second._thread.is_alive()
    finally:
        rec.reset_recorder()
        first_store.close()
        second_store.close()
