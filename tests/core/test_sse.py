"""Tests for dependency-free Server-Sent Events parsing."""

from muse.core.sse import iter_sse_events


def test_accepts_spaced_and_compact_fields():
    lines = [
        "data: first",
        "",
        "data:second",
        "event:error",
        "",
    ]

    assert list(iter_sse_events(lines)) == [
        (None, "first"),
        ("error", "second"),
    ]


def test_ignores_one_initial_utf8_bom():
    lines = ["\ufeffdata: value", ""]

    assert list(iter_sse_events(lines)) == [(None, "value")]


def test_joins_multiline_data_and_removes_only_one_optional_space():
    lines = ["data: first", "data:  second", "data:third", ""]

    assert list(iter_sse_events(lines)) == [
        (None, "first\n second\nthird"),
    ]


def test_ignores_comments_and_unknown_fields_and_discards_unterminated_event():
    lines = [
        ": keepalive",
        "id: 7",
        "retry: 10",
        "data: value",
        "",
        "data: incomplete",
    ]

    assert list(iter_sse_events(lines)) == [(None, "value")]
