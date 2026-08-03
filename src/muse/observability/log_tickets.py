"""Short-lived, reusable tickets that gate the SSE log-tail endpoint.

Why a ticket at all: the admin token is header-only everywhere else in
this package (see dashboard_auth.py), but `EventSource` (the browser API
behind Server-Sent Events) cannot set custom request headers, so it has
no way to present `Authorization: Bearer <token>`. Rather than fall back
to putting the long-lived admin token in the URL (which lands in access
logs, proxy logs, and browser history), the dashboard exchanges the real
token for a short-lived, random ticket via a header-gated mint endpoint
(`POST /v1/telemetry/logs-ticket`), then opens the `EventSource` with
that ticket in the query string instead. A leaked ticket in a log line
is useless once its TTL elapses.

The ticket is REUSABLE within its TTL (not single-use): `EventSource`
auto-reconnects on transient network blips, and a single-use ticket
would force a re-mint on every reconnect, which is unnecessary friction
for a credential that already expires quickly on its own. It is also
unscoped -- it authorizes the logs SSE surface generally, not one
specific model_id -- because the admin token itself is all-or-nothing,
so a per-model ticket would not narrow the actual privilege boundary.

Stdlib only (`secrets`, `time`, `threading`): this module must stay
import-light so it can be constructed without pulling in fastapi.
"""
from __future__ import annotations

import secrets
import math
import threading
import time

DEFAULT_MAX_TICKETS = 4096


class LogTicketStore:
    """In-memory store of ticket -> expiry timestamp.

    Uses `time.monotonic()` internally (not wall-clock), so it is
    immune to system clock adjustments. Thread-safe via a single lock;
    lazily prunes expired entries on every `validate()` call rather
    than running a background sweep thread.
    """

    def __init__(
        self,
        ttl_seconds: float,
        *,
        max_tickets: int = DEFAULT_MAX_TICKETS,
    ) -> None:
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds < 0
        ):
            raise ValueError("ttl_seconds must be a finite non-negative number")
        if (
            isinstance(max_tickets, bool)
            or not isinstance(max_tickets, int)
            or max_tickets <= 0
        ):
            raise ValueError("max_tickets must be a positive integer")
        self._ttl_seconds = float(ttl_seconds)
        self._max_tickets = max_tickets
        self._tickets: dict[str, float] = {}
        self._lock = threading.Lock()

    def _prune_expired_locked(self, now: float) -> None:
        expired = [t for t, expiry in self._tickets.items() if expiry <= now]
        for ticket in expired:
            del self._tickets[ticket]

    def mint(self) -> tuple[str, int]:
        """Create a new ticket. Returns (ticket, expires_in_seconds)."""
        ticket = secrets.token_urlsafe(32)
        now = time.monotonic()
        expiry = now + self._ttl_seconds
        with self._lock:
            self._prune_expired_locked(now)
            while len(self._tickets) >= self._max_tickets:
                # Dicts preserve insertion order; discard the oldest ticket
                # rather than allowing authenticated mint-only traffic to
                # grow process memory without bound.
                self._tickets.pop(next(iter(self._tickets)))
            self._tickets[ticket] = expiry
        return ticket, int(self._ttl_seconds)

    def validate(self, ticket: str | None) -> bool:
        """True iff `ticket` exists and has not expired.

        Lazily prunes: EVERY expired entry in the store (not just the
        ticket being checked) is dropped as a side effect of calling
        this method. Without this, tickets that are minted but never
        re-validated (a client that mints and then never opens the SSE
        connection, or reconnects with a fresh ticket) would sit in
        `_tickets` for the lifetime of the process. A non-existent,
        empty, or None ticket returns False without raising.
        """
        if not ticket:
            return False
        now = time.monotonic()
        with self._lock:
            self._prune_expired_locked(now)
            expiry = self._tickets.get(ticket)
            if expiry is None:
                return False
            return True
