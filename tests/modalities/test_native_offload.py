"""Deterministic ownership tests for native modality offloads."""
from __future__ import annotations

import asyncio
import sys
import threading

import pytest

from muse.modalities import _native_offload as offload


class _Closable:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


@pytest.mark.asyncio
async def test_normal_settlement_leaves_resources_with_caller():
    input_resource = _Closable()
    output_resource = _Closable()

    result = await offload.run_native_offload(
        lambda: output_resource,
        cleanup_abandoned=lambda value: (
            value.close(), input_resource.close()
        ),
    )

    assert result is output_resource
    assert input_resource.close_calls == 0
    assert output_resource.close_calls == 0


@pytest.mark.asyncio
async def test_normal_failure_leaves_inputs_with_caller():
    input_resource = _Closable()
    cleanup_results: list[object] = []
    failure = RuntimeError("backend failed")

    def _fail():
        raise failure

    with pytest.raises(RuntimeError) as caught:
        await offload.run_native_offload(
            _fail,
            cleanup_abandoned=cleanup_results.append,
        )

    assert caught.value is failure
    assert input_resource.close_calls == 0
    assert cleanup_results == []


@pytest.mark.asyncio
async def test_cancelled_success_keeps_input_open_and_closes_eventual_result():
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    cleanup_done = asyncio.Event()
    release = threading.Event()
    backend_exited = threading.Event()
    input_resource = _Closable()
    output_resource = _Closable()

    def _call():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        assert not input_resource.closed
        backend_exited.set()
        return output_resource

    def _cleanup(result) -> None:
        assert backend_exited.is_set()
        result.close()
        input_resource.close()
        loop.call_soon_threadsafe(cleanup_done.set)

    task = asyncio.create_task(offload.run_native_offload(
        _call,
        cleanup_abandoned=_cleanup,
    ))
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel("client disconnected")
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    # Python 3.11 began propagating Task.cancel(msg) to awaiters.
    expected_args = ("client disconnected",) if sys.version_info >= (3, 11) else ()
    assert caught.value.args == expected_args
    assert not input_resource.closed
    assert not output_resource.closed

    release.set()
    await asyncio.wait_for(cleanup_done.wait(), timeout=1)
    assert input_resource.close_calls == 1
    assert output_resource.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_failure_cleans_input_and_consumes_backend_exception():
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    cleanup_done = asyncio.Event()
    release = threading.Event()
    backend_exited = threading.Event()
    input_resource = _Closable()
    cleaned_results: list[object] = []
    loop_errors: list[dict] = []
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _loop, context: loop_errors.append(context),
    )

    def _call():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        assert not input_resource.closed
        backend_exited.set()
        raise RuntimeError("eventual backend failure")

    def _cleanup(result) -> None:
        assert backend_exited.is_set()
        cleaned_results.append(result)
        input_resource.close()
        loop.call_soon_threadsafe(cleanup_done.set)

    try:
        task = asyncio.create_task(offload.run_native_offload(
            _call,
            cleanup_abandoned=_cleanup,
        ))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not input_resource.closed

        release.set()
        await asyncio.wait_for(cleanup_done.wait(), timeout=1)
        for _ in range(20):
            if not offload._ABANDONED_OFFLOADS:
                break
            await asyncio.sleep(0)

        assert input_resource.close_calls == 1
        assert cleaned_results == [None]
        assert not offload._ABANDONED_OFFLOADS
        assert loop_errors == []
    finally:
        loop.set_exception_handler(old_handler)


@pytest.mark.asyncio
async def test_cancelled_while_executor_work_is_unstarted_cleans_inputs(
    monkeypatch,
):
    entered_to_thread = asyncio.Event()
    never_start = asyncio.Event()
    cleanup_done = asyncio.Event()
    input_resource = _Closable()
    cleanup_values: list[object] = []

    async def _queued_to_thread(_call):
        entered_to_thread.set()
        await never_start.wait()
        raise AssertionError("queued native callable unexpectedly started")

    def _cleanup(value) -> None:
        cleanup_values.append(value)
        input_resource.close()
        cleanup_done.set()

    monkeypatch.setattr(offload.asyncio, "to_thread", _queued_to_thread)
    outer = asyncio.create_task(offload.run_native_offload(
        lambda: pytest.fail("native callable unexpectedly ran"),
        cleanup_abandoned=_cleanup,
    ))
    await asyncio.wait_for(entered_to_thread.wait(), timeout=1)

    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert not input_resource.closed

    assert len(offload._ABANDONED_OFFLOADS) == 1
    inner = next(iter(offload._ABANDONED_OFFLOADS))
    inner.cancel()
    await asyncio.wait_for(cleanup_done.wait(), timeout=1)
    await asyncio.sleep(0)

    assert cleanup_values == [None]
    assert input_resource.close_calls == 1
    assert not offload._ABANDONED_OFFLOADS


@pytest.mark.asyncio
async def test_external_base_exception_still_tracks_unstarted_work(monkeypatch):
    class ExternalUnwind(BaseException):
        pass

    entered_to_thread = asyncio.Event()
    never_start = asyncio.Event()
    cleanup_done = asyncio.Event()
    cleanup_values: list[object] = []

    async def _queued_to_thread(_call):
        entered_to_thread.set()
        await never_start.wait()

    async def _raise_external_unwind(_tasks):
        await entered_to_thread.wait()
        raise ExternalUnwind("request coroutine closed")

    def _cleanup(value) -> None:
        cleanup_values.append(value)
        cleanup_done.set()

    monkeypatch.setattr(offload.asyncio, "to_thread", _queued_to_thread)
    monkeypatch.setattr(offload.asyncio, "wait", _raise_external_unwind)

    with pytest.raises(ExternalUnwind, match="request coroutine closed"):
        await offload.run_native_offload(
            lambda: pytest.fail("native callable unexpectedly ran"),
            cleanup_abandoned=_cleanup,
        )

    assert len(offload._ABANDONED_OFFLOADS) == 1
    inner = next(iter(offload._ABANDONED_OFFLOADS))
    inner.cancel()
    await asyncio.wait_for(cleanup_done.wait(), timeout=1)
    await asyncio.sleep(0)

    assert cleanup_values == [None]
    assert not offload._ABANDONED_OFFLOADS


@pytest.mark.asyncio
async def test_backend_cancelled_error_uses_abandoned_cleanup_once():
    input_resource = _Closable()
    cleanup_values: list[object] = []

    def _cancel():
        raise asyncio.CancelledError("backend stopped")

    def _cleanup(value) -> None:
        cleanup_values.append(value)
        input_resource.close()

    with pytest.raises(asyncio.CancelledError, match="backend stopped"):
        await offload.run_native_offload(
            _cancel,
            cleanup_abandoned=_cleanup,
        )
    assert not offload._ABANDONED_OFFLOADS
    await asyncio.sleep(0)

    assert cleanup_values == [None]
    assert input_resource.close_calls == 1


@pytest.mark.parametrize("settle_first", [False, True])
def test_result_ownership_cleans_once_for_both_race_orders(settle_first):
    output_resource = _Closable()
    input_resource = _Closable()
    state = offload._ResultOwnership(
        lambda value: (value.close(), input_resource.close()),
    )

    if settle_first:
        state.settle_success(output_resource)
        assert not output_resource.closed
        state.abandon()
    else:
        state.abandon()
        assert not input_resource.closed
        state.settle_success(output_resource)

    assert output_resource.close_calls == 1
    assert input_resource.close_calls == 1
