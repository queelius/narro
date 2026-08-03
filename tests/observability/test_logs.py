import queue

import pytest

from muse.observability.logs import LogHub, SUBSCRIBER_QUEUE_MAXSIZE


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "32"])
def test_buffer_bytes_requires_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        LogHub(buffer_bytes=value)

def test_snapshot_and_byte_bound():
    hub = LogHub(buffer_bytes=20)
    for i in range(10):
        hub.append("m", f"line{i}")   # each ~5-6 bytes; only most-recent fit
    snap = hub.snapshot("m")
    assert snap and snap[-1] == "line9"
    assert sum(len(s) for s in snap) <= 20

def test_pubsub_delivers_new_lines():
    hub = LogHub()
    q = hub.subscribe("m")
    hub.append("m", "hello")
    assert q.get_nowait() == "hello"
    hub.unsubscribe("m", q)
    hub.append("m", "after")
    assert q.qsize() == 0  # unsubscribed -> no more


def test_subscribe_with_snapshot_has_gap_free_nonduplicating_handoff():
    hub = LogHub()
    hub.append("m", "before")

    history, q = hub.subscribe_with_snapshot("m")
    hub.append("m", "after")

    assert history == ["before"]
    assert q.get_nowait() == "after"
    assert q.empty()
    hub.unsubscribe("m", q)

def test_eviction_counts_utf8_bytes_not_chars():
    # 3 emoji chars, but each is a 4-byte UTF-8 sequence -> 12 bytes total.
    # Under (buggy) char-count accounting this line would measure as 3
    # bytes and comfortably fit inside buffer_bytes=8 alongside a second
    # short line. Under correct byte-count accounting it measures as 12
    # bytes, already over the bound on its own.
    emoji_line = "\U0001F600\U0001F600\U0001F600"
    assert len(emoji_line) == 3
    assert len(emoji_line.encode("utf-8")) == 12

    hub = LogHub(buffer_bytes=8)
    hub.append("m", emoji_line)

    # A single line can no longer bypass the total byte bound. The UTF-8
    # sequence is replaced with an ASCII truncation marker that fits exactly.
    snap = hub.snapshot("m")
    assert len(snap) == 1
    assert len(snap[0].encode("utf-8")) <= 8
    assert emoji_line not in snap

    # A second short line evicts the capped predecessor as needed.
    hub.append("m", "hi")
    snap = hub.snapshot("m")
    assert emoji_line not in snap
    assert snap == ["hi"]


def test_oversized_line_is_capped_before_subscriber_fanout():
    hub = LogHub(buffer_bytes=32)
    q = hub.subscribe("m")

    hub.append("m", "x" * 10_000)

    buffered = hub.snapshot("m")[0]
    published = q.get_nowait()
    assert buffered == published
    assert len(buffered.encode("utf-8")) <= 32
    assert buffered.endswith("...[truncated]")


def test_subscriber_queue_is_bounded_and_does_not_block_or_raise():
    """A stalled-but-connected subscriber must not grow memory unboundedly.

    Regression: subscribe() previously returned an unbounded queue.Queue(),
    so append()'s `except queue.Full: pass` guard was dead code -- a slow
    SSE client that never drains its queue accumulated every line forever.
    """
    hub = LogHub(buffer_bytes=10_000_000)
    q = hub.subscribe("m")

    for i in range(SUBSCRIBER_QUEUE_MAXSIZE + 50):
        hub.append("m", f"line{i}")  # must never raise / block

    assert q.qsize() == SUBSCRIBER_QUEUE_MAXSIZE


def test_fresh_subscriber_after_a_stalled_one_still_receives_new_lines():
    hub = LogHub(buffer_bytes=10_000_000)
    stalled = hub.subscribe("m")

    for i in range(SUBSCRIBER_QUEUE_MAXSIZE + 50):
        hub.append("m", f"line{i}")

    fresh = hub.subscribe("m")
    hub.append("m", "fresh-line")
    assert fresh.get_nowait() == "fresh-line"
    # The stalled subscriber's queue is still capped, not raising/blocking.
    assert stalled.qsize() == SUBSCRIBER_QUEUE_MAXSIZE


def test_unsubscribe_removes_empty_model_subscription_bucket():
    hub = LogHub()
    q = hub.subscribe("transient-model")

    hub.unsubscribe("transient-model", q)

    assert "transient-model" not in hub._subscribers
