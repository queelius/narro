import threading
from unittest.mock import MagicMock

import pytest

from muse.observability.sampler import Sampler, VramTracker


def test_sample_once_records(monkeypatch):
    import muse.observability.sampler as smod
    monkeypatch.setattr(smod, "gpu_free_gb", lambda: 3.0)
    monkeypatch.setattr(smod, "cpu_free_gb", lambda: 20.0)
    seen = []
    s = Sampler(interval=999, loaded_fn=lambda: {"m": object()},
                inflight_fn=lambda: 2, record_fn=lambda t, **k: seen.append((t, k)))
    s.sample_once()
    assert seen[0][0] == "sample"
    k = seen[0][1]
    assert k["free_vram_gb"] == 3.0 and k["loaded_count"] == 1 and k["in_flight_count"] == 2


def test_shared_stop_event_stops_the_loop(monkeypatch):
    # Constructing with an external stop_event (the shape run_supervisor's
    # _init_telemetry uses, passing state.stop_event) must let that shared
    # event unblock the sampler loop, same as IdleSweeper's stop_event param.
    import muse.observability.sampler as smod
    monkeypatch.setattr(smod, "gpu_free_gb", lambda: 1.0)
    monkeypatch.setattr(smod, "cpu_free_gb", lambda: 1.0)

    shared = threading.Event()
    s = Sampler(
        interval=0.01,
        loaded_fn=lambda: {},
        inflight_fn=lambda: 0,
        record_fn=lambda t, **k: None,
        stop_event=shared,
    )
    assert s._stop is shared

    s.start()
    shared.set()
    s._thread.join(timeout=2.0)
    assert not s._thread.is_alive()


def test_no_arg_construction_still_works():
    # Backward compatibility: omitting stop_event still gives the sampler
    # its own private Event, and start/stop still function.
    s = Sampler(
        interval=0.01,
        loaded_fn=lambda: {},
        inflight_fn=lambda: 0,
        record_fn=lambda t, **k: None,
    )
    assert isinstance(s._stop, threading.Event)

    s.start()
    s.stop()
    assert s._thread is None


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_rejects_busy_loop_or_unbounded_interval(interval):
    with pytest.raises(ValueError, match="interval"):
        Sampler(
            interval=interval,
            loaded_fn=lambda: {},
            inflight_fn=lambda: 0,
        )


def test_start_never_clears_an_external_shutdown_event():
    shared = threading.Event()
    shared.set()
    s = Sampler(
        interval=1,
        loaded_fn=lambda: {},
        inflight_fn=lambda: 0,
        stop_event=shared,
    )

    assert s.start() is False
    assert shared.is_set()
    assert s._thread is None


def test_private_sampler_can_restart_after_stop():
    s = Sampler(
        interval=0.01,
        loaded_fn=lambda: {},
        inflight_fn=lambda: 0,
    )

    assert s.start() is True
    assert s.stop() is True
    assert s.start() is True
    assert s.stop() is True


def test_stop_retains_thread_handle_when_bounded_join_times_out():
    s = Sampler(
        interval=1,
        loaded_fn=lambda: {},
        inflight_fn=lambda: 0,
        stop_timeout=0.01,
    )
    wedged = MagicMock()
    wedged.is_alive.return_value = True
    s._thread = wedged

    assert s.stop() is False
    assert s._thread is wedged
    wedged.join.assert_called_once_with(timeout=0.01)


def test_start_does_not_claim_restart_while_timed_out_stop_is_unwinding():
    s = Sampler(
        interval=1,
        loaded_fn=lambda: {},
        inflight_fn=lambda: 0,
        stop_timeout=0.01,
    )
    wedged = MagicMock()
    wedged.is_alive.return_value = True
    s._thread = wedged

    assert s.stop() is False
    assert s._stop.is_set()
    assert s.start() is False
    assert s._thread is wedged


def test_stop_clears_an_inert_thread_whose_start_never_completed():
    s = Sampler(
        interval=1,
        loaded_fn=lambda: {},
        inflight_fn=lambda: 0,
    )
    s._thread = threading.Thread(target=lambda: None)

    assert s.stop() is True
    assert s._thread is None


def test_vram_tracker_captures_peak_for_each_active_request(monkeypatch):
    import muse.observability.sampler as smod
    monkeypatch.setattr(smod, "gpu_free_gb", lambda: 4.0)
    monkeypatch.setattr(smod, "gpu_total_gb", lambda: 12.0)
    monkeypatch.setattr(smod, "cpu_free_gb", lambda: 20.0)
    tracker = VramTracker()
    seen = []
    sampler = Sampler(
        interval=10,
        active_interval=0.01,
        loaded_fn=lambda: {},
        inflight_fn=lambda: 0,
        record_fn=lambda event, **fields: seen.append((event, fields)),
        vram_tracker=tracker,
    )

    tracker.begin("request-1")
    sampler.sample_once()

    assert tracker.finish("request-1") == 8.0
    assert seen[0][1]["gpu_used_gb"] == 8.0
