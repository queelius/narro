import asyncio

import httpx

from muse.federation.nodes import NodeSpec
from muse.federation.registry import NodeRegistry, _get_json


async def test_get_json_rejects_declared_oversized_body(monkeypatch):
    monkeypatch.setattr("muse.federation.registry._MAX_FETCH_BODY_BYTES", 8)

    async def handler(_request):
        return httpx.Response(200, content=b'{"data": []}')

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as client:
        assert await _get_json(client, "http://node/v1/models") is None


async def test_get_json_parses_body_within_limit(monkeypatch):
    monkeypatch.setattr("muse.federation.registry._MAX_FETCH_BODY_BYTES", 64)

    async def handler(_request):
        return httpx.Response(200, json={"status": "ok"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as client:
        assert await _get_json(client, "http://node/health") == {
            "status": "ok",
        }


async def test_refresh_once_builds_snapshot():
    specs = [NodeSpec(url="http://a:8000", name="a"), NodeSpec(url="http://b:8000", name="b")]

    async def fake_fetch(url, token):
        if "a:" in url:
            return ({"data": [{"id": "m1", "loaded": True}]}, {"status": "ok"}, {"in_flight": 0})
        return (None, None, None)  # b unreachable

    reg = NodeRegistry(specs, refresh_interval=999, fetch=fake_fetch)
    await reg.refresh_once()
    snap = {s.spec.name: s for s in reg.snapshot()}
    assert snap["a"].reachable and snap["a"].models["m1"].loaded and snap["a"].in_flight == 0
    assert snap["b"].reachable is False


async def test_refresh_once_isolates_node_that_raises():
    """One node's fetch RAISES (proves gather-level per-node isolation:
    a bad node must not abort the refresh for a healthy sibling)."""
    specs = [NodeSpec(url="http://good:8000", name="good"), NodeSpec(url="http://bad:8000", name="bad")]

    async def fake_fetch(url, token):
        if "bad:" in url:
            raise RuntimeError("boom: differently-shaped /v1/models response")
        return ({"data": [{"id": "m1", "loaded": True}]}, {"status": "ok"}, {"in_flight": 0})

    reg = NodeRegistry(specs, refresh_interval=999, fetch=fake_fetch)
    await reg.refresh_once()  # must not raise/abort
    snap = {s.spec.name: s for s in reg.snapshot()}

    # healthy node's state is present, reachable, and has its model
    assert snap["good"].reachable is True
    assert snap["good"].models["m1"].loaded is True

    # bad node degrades to an unreachable NodeState instead of aborting
    assert snap["bad"].reachable is False
    assert snap["bad"].models == {}
    assert snap["bad"].in_flight is None


async def test_refresh_once_skips_id_less_entry_but_stays_reachable():
    """A node returns a well-formed 200 whose /v1/models entries lack an
    "id" key (a differently-shaped/older muse). The node should stay
    reachable with the malformed entry skipped, not raise and abort."""
    specs = [NodeSpec(url="http://a:8000", name="a")]

    async def fake_fetch(url, token):
        return ({"data": [{"name": "x"}]}, {"status": "ok"}, None)

    reg = NodeRegistry(specs, refresh_interval=999, fetch=fake_fetch)
    await reg.refresh_once()
    snap = {s.spec.name: s for s in reg.snapshot()}

    assert snap["a"].reachable is True
    assert snap["a"].models == {}


def test_poll_timeout_binds_into_default_fetch():
    """NodeRegistry(poll_timeout=X) binds X into the default httpx fetch."""
    from functools import partial
    from muse.federation.nodes import NodeSpec
    from muse.federation.registry import NodeRegistry, _default_fetch
    reg = NodeRegistry(
        [NodeSpec(url="http://a:8000", name="a")],
        refresh_interval=3.0,
        poll_timeout=7.5,
    )
    assert isinstance(reg._fetch, partial)
    assert reg._fetch.func is _default_fetch
    assert reg._fetch.keywords["timeout"] == 7.5


def test_poll_timeout_defaults_when_unset():
    """No poll_timeout -> falls back to the module default (>= node aggregation timeout)."""
    from functools import partial
    from muse.federation.nodes import NodeSpec
    from muse.federation.registry import NodeRegistry, _FETCH_TIMEOUT_SECONDS
    reg = NodeRegistry([NodeSpec(url="http://a:8000", name="a")], refresh_interval=3.0)
    assert isinstance(reg._fetch, partial)
    assert reg._fetch.keywords["timeout"] == _FETCH_TIMEOUT_SECONDS
    assert _FETCH_TIMEOUT_SECONDS >= 5.0  # above the 5s node aggregation timeout


def test_injected_fetch_is_used_as_is():
    """An injected fetch is NOT wrapped (preserves the test-injection contract)."""
    from muse.federation.nodes import NodeSpec
    from muse.federation.registry import NodeRegistry
    async def fake(url, token): return (None, None, None)
    reg = NodeRegistry([NodeSpec(url="http://a:8000", name="a")], refresh_interval=3.0, fetch=fake)
    assert reg._fetch is fake


def test_federation_poll_timeout_config_default():
    from muse.core import config
    assert config.get("federation.poll_timeout_seconds") == 10.0


async def test_start_is_idempotent_and_close_releases_the_only_task():
    calls = 0
    fetched = asyncio.Event()

    async def fake_fetch(url, token):
        nonlocal calls
        calls += 1
        fetched.set()
        return ({"data": []}, {"status": "ok"}, None)

    reg = NodeRegistry(
        [NodeSpec(url="http://a:8000", name="a")],
        refresh_interval=999,
        fetch=fake_fetch,
    )

    assert reg.start() is True
    first = reg._task
    assert first is not None
    assert reg.start() is True
    assert reg._task is first
    await asyncio.wait_for(fetched.wait(), timeout=1)

    await reg.aclose()

    assert calls == 1
    assert first.done()
    assert reg._task is None


async def test_start_returns_false_while_close_is_in_progress():
    reg = NodeRegistry(
        [NodeSpec(url="http://a:8000", name="a")],
        refresh_interval=999,
        fetch=lambda *_args: None,
    )
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def slow_cancel() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
            raise

    owned_task = asyncio.create_task(slow_cancel())
    reg._task = owned_task

    close_task = asyncio.create_task(reg.aclose())
    await cancelling.wait()

    assert reg._closing is True
    assert reg.start() is False

    release.set()
    await close_task
    assert owned_task.done()
    assert reg._task is None


async def test_concurrent_close_callers_wait_for_same_teardown():
    reg = NodeRegistry(
        [NodeSpec(url="http://a:8000", name="a")],
        refresh_interval=999,
        fetch=lambda *_args: None,
    )
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def slow_cancel() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
            raise

    owned_task = asyncio.create_task(slow_cancel())
    reg._task = owned_task

    first = asyncio.create_task(reg.aclose())
    await cancelling.wait()
    shared_close = reg._close_task
    second = asyncio.create_task(reg.aclose())
    await asyncio.sleep(0)

    assert shared_close is not None
    assert reg._close_task is shared_close
    assert not first.done()
    assert not second.done()

    release.set()
    await asyncio.gather(first, second)
    assert owned_task.done()
    assert reg._task is None
    assert reg._close_task is None
    assert reg._closing is False


async def test_cancelled_close_waiter_does_not_cancel_shared_teardown():
    reg = NodeRegistry(
        [NodeSpec(url="http://a:8000", name="a")],
        refresh_interval=999,
        fetch=lambda *_args: None,
    )
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def slow_cancel() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
            raise

    owned_task = asyncio.create_task(slow_cancel())
    reg._task = owned_task
    cancelled_waiter = asyncio.create_task(reg.aclose())
    await cancelling.wait()
    surviving_waiter = asyncio.create_task(reg.aclose())
    await asyncio.sleep(0)

    cancelled_waiter.cancel("caller stopped waiting")
    result = await asyncio.gather(cancelled_waiter, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)
    assert reg._closing is True
    assert reg._close_task is not None
    assert not surviving_waiter.done()

    release.set()
    await surviving_waiter
    assert owned_task.done()
    assert reg._task is None
    assert reg._close_task is None
    assert reg._closing is False


async def test_close_failure_reaches_every_waiter_and_resets_lifecycle_state():
    reg = NodeRegistry(
        [NodeSpec(url="http://a:8000", name="a")],
        refresh_interval=999,
        fetch=lambda *_args: None,
    )
    cancelling = asyncio.Event()
    release = asyncio.Event()
    failure = RuntimeError("refresh teardown failed")

    async def failed_cancel() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
            raise failure

    owned_task = asyncio.create_task(failed_cancel())
    reg._task = owned_task
    first = asyncio.create_task(reg.aclose())
    await cancelling.wait()
    second = asyncio.create_task(reg.aclose())
    await asyncio.sleep(0)

    release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert results == [failure, failure]
    assert reg._task is None
    assert reg._close_task is None
    assert reg._closing is False
