from unittest.mock import MagicMock

from muse.cli_impl.load_director import LoadDirector
from muse.observability.traces import begin_request_trace, reset_request_trace


def _manifest(memory_gb=1.0):
    return {
        "model_id": "new",
        "capabilities": {"memory_gb": memory_gb, "device": "cpu"},
    }


def test_director_annotates_exact_request_cold_load_and_eviction(monkeypatch):
    events = []
    monkeypatch.setattr(
        "muse.cli_impl.load_director.record",
        lambda event, **fields: events.append((event, fields)),
    )
    free = {"value": 10.0}
    probe = MagicMock()
    probe.gpu_free_gb.return_value = 32.0
    probe.cpu_free_gb.side_effect = lambda: free["value"]

    def enable(model_id):
        if model_id == "victim":
            free["value"] = 4.0
            return 9001
        return 9002

    def disable(_model_id):
        free["value"] = 9.0

    director = LoadDirector(
        enable_fn=enable, disable_fn=disable, memory_probe=probe,
        cpu_headroom_gb=2.0,
    )
    director.acquire("victim", manifest=_manifest(5.0))
    director.release("victim")
    events.clear()

    trace, token = begin_request_trace("new", "audio/speech")
    try:
        director.acquire("new", manifest=_manifest(6.0))
        snapshot = trace.snapshot()
    finally:
        reset_request_trace(token)

    assert snapshot["cold"] is True
    assert snapshot["load_ms"] >= 0
    assert snapshot["evicted_models"] == ["victim"]
    assert {
        fields["request_id"] for _, fields in events
        if fields.get("request_id") is not None
    } == {trace.request_id}
