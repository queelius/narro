"""Async node registry for the muse federation coordinator.

`NodeRegistry` polls every configured node concurrently, folds each
node's payloads into a `NodeState` via the pure reducer in
`muse.federation.state`, and caches the resulting snapshot list. Route
handlers (sync) read the cache via `snapshot()`; a background asyncio
task refreshes it on an interval via `start()` / `aclose()`.

The fetch function is injectable so tests never touch the network or
depend on wall-clock sleeps: `fetch(url, token) -> (models_payload,
health_payload, summary_payload)`, each `None` on any per-call failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from functools import partial

import httpx

from muse.federation.nodes import NodeSpec
from muse.federation.state import NodeState, build_node_state

logger = logging.getLogger(__name__)

# Default per-node poll timeout. Chosen ABOVE a node's own
# server.aggregation_timeout_seconds (default 5s) so a node that is briefly
# slow to aggregate its workers' /v1/models (e.g. a worker mid-generation on
# a CPU box) is NOT falsely marked unreachable and dropped from routing. The
# coordinator overrides this from federation.poll_timeout_seconds.
_FETCH_TIMEOUT_SECONDS = 10.0
_MAX_FETCH_BODY_BYTES = 4 * 1024 * 1024


def _consume_task_result(task: asyncio.Task) -> None:
    """Retrieve a detached lifecycle task result to avoid loop warnings."""
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


async def _get_json(client: httpx.AsyncClient, url: str, **kwargs) -> dict | None:
    """GET `url` and return parsed JSON, or None on any error (connect,
    timeout, non-200 status, oversized body, or invalid JSON)."""
    try:
        async with client.stream("GET", url, **kwargs) as response:
            response.raise_for_status()
            raw_length = response.headers.get("content-length")
            if raw_length is not None:
                try:
                    declared_length = int(raw_length)
                except ValueError:
                    return None
                if declared_length < 0 or declared_length > _MAX_FETCH_BODY_BYTES:
                    return None

            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > _MAX_FETCH_BODY_BYTES:
                    return None
                body.extend(chunk)
        return json.loads(body)
    except (httpx.HTTPError, ValueError, UnicodeError):
        return None


async def _default_fetch(
    url: str, token: str | None, *, timeout: float = _FETCH_TIMEOUT_SECONDS
) -> tuple[dict | None, dict | None, dict | None]:
    """Default httpx-based fetch: GET /v1/models and /health always;
    GET /v1/telemetry/summary (bearer-authed) only when `token` is
    truthy. Every GET is isolated: a failure on one payload never
    prevents the others from being fetched."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        models_payload = await _get_json(client, f"{url}/v1/models")
        health_payload = await _get_json(client, f"{url}/health")
        summary_payload = None
        if token:
            summary_payload = await _get_json(
                client,
                f"{url}/v1/telemetry/summary",
                headers={"Authorization": f"Bearer {token}"},
            )
    return models_payload, health_payload, summary_payload


class NodeRegistry:
    """Polls a fixed set of nodes and caches their `NodeState` snapshot."""

    def __init__(
        self,
        nodes: list[NodeSpec],
        *,
        refresh_interval: float,
        clock=time.monotonic,
        fetch=None,
        poll_timeout: float | None = None,
    ) -> None:
        self._nodes = list(nodes)
        self._refresh_interval = refresh_interval
        self._clock = clock
        if fetch is not None:
            self._fetch = fetch
        else:
            # Bind the configured per-node poll timeout into the default fetch,
            # preserving the (url, token) fetch protocol that injected fetches use.
            _timeout = poll_timeout if poll_timeout is not None else _FETCH_TIMEOUT_SECONDS
            self._fetch = partial(_default_fetch, timeout=_timeout)
        self._lock = threading.Lock()
        self._states: list[NodeState] = []
        self._task: asyncio.Task | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False

    async def _fetch_one(self, spec: NodeSpec) -> NodeState:
        models_payload, health_payload, summary_payload = await self._fetch(
            spec.url, spec.token
        )
        return build_node_state(
            spec,
            models_payload=models_payload,
            health_payload=health_payload,
            summary_payload=summary_payload,
            now=self._clock(),
        )

    async def _fetch_one_isolated(self, spec: NodeSpec) -> NodeState:
        """Wrap `_fetch_one` so a raised exception for THIS node degrades
        to an unreachable NodeState rather than propagating. This is what
        gives `refresh_once` per-node isolation: one bad node (raised
        exception anywhere in fetch-or-reduce) never prevents the other
        nodes' states from being built."""
        try:
            return await self._fetch_one(spec)
        except Exception:
            logger.warning(
                "federation: refresh failed for node %r, marking unreachable",
                spec.name,
            )
            return NodeState(
                spec=spec,
                reachable=False,
                models={},
                in_flight=None,
                last_poll_ts=self._clock(),
            )

    async def refresh_once(self) -> None:
        """Concurrently poll every node and atomically replace the cached
        snapshot list. Per-node failures are isolated: a raised exception
        while fetching/reducing one node degrades only that node's state
        to unreachable, and never aborts the refresh for the others."""
        results = await asyncio.gather(
            *(self._fetch_one_isolated(spec) for spec in self._nodes),
            return_exceptions=True,
        )
        states: list[NodeState] = []
        for spec, result in zip(self._nodes, results):
            if isinstance(result, BaseException):
                # Belt-and-suspenders backstop: _fetch_one_isolated already
                # catches Exception, so this only fires for something it
                # didn't (e.g. a BaseException subclass slipping through).
                logger.warning(
                    "federation: unexpected error refreshing node %r, "
                    "marking unreachable",
                    spec.name,
                )
                states.append(
                    NodeState(
                        spec=spec,
                        reachable=False,
                        models={},
                        in_flight=None,
                        last_poll_ts=self._clock(),
                    )
                )
            else:
                states.append(result)
        with self._lock:
            self._states = states

    def snapshot(self) -> list[NodeState]:
        """Return the cached states (empty list before the first refresh).
        Returns a fresh list copy so callers cannot mutate the internal
        cache."""
        with self._lock:
            return list(self._states)

    def node_by_url(self, url: str) -> NodeSpec | None:
        for spec in self._nodes:
            if spec.url == url:
                return spec
        return None

    def start(self) -> bool:
        """Launch the background refresh loop once. One
        failed refresh is logged-and-swallowed (via try/except) rather
        than killing the loop. Repeated calls are idempotent.

        Returns false only while an overlapping ``aclose`` owns teardown.
        """

        if self._closing:
            return False
        if self._task is not None and not self._task.done():
            return True

        async def _loop() -> None:
            while True:
                try:
                    await self.refresh_once()
                except Exception:
                    pass
                await asyncio.sleep(self._refresh_interval)

        self._task = asyncio.create_task(_loop())
        return True

    async def _close_background_task(self, task: asyncio.Task) -> None:
        """Own cancellation of one refresh task until it actually settles."""
        try:
            task.cancel()
            try:
                # Caller cancellation must not cancel the shared refresh-task
                # teardown. It can be retried only when this dedicated owner
                # itself is cancelled (for example during loop shutdown).
                await asyncio.shield(task)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                if not task.done():
                    raise
        except Exception:
            logger.warning("federation registry close failed", exc_info=True)
            raise
        finally:
            if task.done() and self._task is task:
                self._task = None
            current = asyncio.current_task()
            if self._close_task is current:
                self._close_task = None
            self._closing = False

    async def aclose(self) -> None:
        """Cancel the refresh loop, sharing one teardown across all callers.

        Every overlapping caller awaits the same shielded lifecycle task. A
        cancelled waiter therefore cannot strand the registry half-closed or
        make another waiter return before the refresh task has settled.
        """
        close_task = self._close_task
        if close_task is None:
            task = self._task
            if task is None:
                return
            self._closing = True
            try:
                close_task = asyncio.create_task(
                    self._close_background_task(task),
                )
            except BaseException:
                self._closing = False
                raise
            self._close_task = close_task
            close_task.add_done_callback(_consume_task_result)
            # Be correct under an eager task factory: teardown may have
            # completed synchronously before `_close_task` was assigned.
            if close_task.done() and self._close_task is close_task:
                self._close_task = None
        await asyncio.shield(close_task)
