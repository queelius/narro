"""Dependency-free parsing helpers for Server-Sent Events."""
from __future__ import annotations

from collections.abc import Iterable, Iterator


def iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str | None, str]]:
    """Yield ``(event_type, data)`` pairs from decoded SSE lines.

    SSE removes at most one ASCII space after a field's colon and joins
    repeated ``data`` fields with newlines.  Accepting both ``data:value``
    and ``data: value`` matters because field spacing belongs to the server
    implementation, not the client contract.
    """
    event_type: str | None = None
    data_lines: list[str] = []
    first_line = True

    def dispatch() -> tuple[str | None, str] | None:
        if not data_lines:
            return None
        return event_type, "\n".join(data_lines)

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if first_line:
            if line.startswith("\ufeff"):
                line = line[1:]
            first_line = False
        if not line:
            event = dispatch()
            if event is not None:
                yield event
            event_type = None
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]

        if field == "event":
            event_type = value or None
        elif field == "data":
            data_lines.append(value)
