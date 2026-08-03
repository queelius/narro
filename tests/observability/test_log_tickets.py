"""Tests for muse.observability.log_tickets.LogTicketStore.

Stdlib-only, no fastapi. A ticket authorizes the SSE logs surface for a
short, configurable TTL; it is reusable within that window (not
single-use), so EventSource auto-reconnect works without re-minting.
"""
from __future__ import annotations

import threading

import pytest

from muse.observability.log_tickets import LogTicketStore


def test_mint_returns_nonempty_urlsafe_ticket_and_positive_expires_in():
    store = LogTicketStore(60.0)
    ticket, expires_in = store.mint()
    assert isinstance(ticket, str) and len(ticket) > 0
    assert isinstance(expires_in, int)
    assert expires_in > 0


def test_validate_minted_ticket_returns_true():
    store = LogTicketStore(60.0)
    ticket, _ = store.mint()
    assert store.validate(ticket) is True


def test_validate_unknown_ticket_returns_false():
    store = LogTicketStore(60.0)
    assert store.validate("nope") is False


def test_validate_empty_or_none_returns_false():
    store = LogTicketStore(60.0)
    assert store.validate("") is False
    assert store.validate(None) is False


def test_expired_ticket_validates_false_and_is_pruned():
    store = LogTicketStore(0.0)
    ticket, _ = store.mint()
    # A 0.0 TTL means the ticket is already expired by the time we check.
    assert store.validate(ticket) is False
    # Pruned: a second validate call must not find it either (no crash,
    # and it's genuinely gone from internal state).
    assert ticket not in store._tickets


def test_reusable_within_ttl_not_single_use():
    store = LogTicketStore(60.0)
    ticket, _ = store.mint()
    assert store.validate(ticket) is True
    # Validating again must still succeed -- NOT single-use.
    assert store.validate(ticket) is True
    assert store.validate(ticket) is True


def test_validate_prunes_all_expired_tickets_not_just_the_one_checked():
    """Regression: validate() only pruned the ticket it was checking, so
    abandoned expired tickets accumulated in the store for process
    lifetime. validate() must sweep every expired entry on each call.
    """
    store = LogTicketStore(0.0)
    t1, _ = store.mint()
    t2, _ = store.mint()
    # Both already expired (0.0 TTL). Validate an unrelated third token.
    assert store.validate("some-other-unknown-token") is False
    assert t1 not in store._tickets
    assert t2 not in store._tickets


def test_two_mints_are_distinct():
    store = LogTicketStore(60.0)
    t1, _ = store.mint()
    t2, _ = store.mint()
    assert t1 != t2


def test_mint_capacity_evicts_oldest_ticket():
    store = LogTicketStore(60.0, max_tickets=2)
    first, _ = store.mint()
    second, _ = store.mint()
    third, _ = store.mint()

    assert len(store._tickets) == 2
    assert store.validate(first) is False
    assert store.validate(second) is True
    assert store.validate(third) is True


def test_nonpositive_ticket_capacity_rejected():
    with pytest.raises(ValueError, match="positive"):
        LogTicketStore(60.0, max_tickets=0)


@pytest.mark.parametrize("ttl", [-1, float("nan"), float("inf"), True, "60"])
def test_invalid_ticket_ttl_rejected(ttl):
    with pytest.raises(ValueError, match="finite non-negative"):
        LogTicketStore(ttl)


@pytest.mark.parametrize("capacity", [-1, True, 1.5, "2"])
def test_invalid_ticket_capacity_type_rejected(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        LogTicketStore(60.0, max_tickets=capacity)


def test_concurrent_mint_and_validate_does_not_raise():
    store = LogTicketStore(60.0)
    errors = []

    def worker():
        try:
            for _ in range(50):
                ticket, _ = store.mint()
                store.validate(ticket)
                store.validate("garbage")
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
