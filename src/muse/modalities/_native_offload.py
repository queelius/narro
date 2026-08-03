"""Cancellation-safe ownership for synchronous modality work."""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any, Generic, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")

_ABANDONED_OFFLOADS: set[asyncio.Task[Any]] = set()


def _run_cleanup(
    cleanup: Callable[..., None] | None,
    *args: Any,
) -> None:
    if cleanup is None:
        return
    try:
        cleanup(*args)
    except BaseException:  # noqa: BLE001
        # Cleanup must never replace the backend result, backend exception,
        # or caller cancellation that determined this ownership path.
        logger.exception("native offload cleanup failed")


class _ResultOwnership(Generic[T]):
    """Resolve the cancellation/native-settlement race exactly once."""

    def __init__(
        self,
        cleanup_abandoned: Callable[[T | None], None] | None,
    ) -> None:
        self._cleanup_abandoned = cleanup_abandoned
        self._lock = threading.Lock()
        self._abandoned = False
        self._cleanup_taken = False
        self._settled = False
        self._started = False
        self._start_prevented = False
        self._has_result = False
        self._result: T | None = None
        self._failure: BaseException | None = None

    def try_start(self) -> bool:
        """Claim native execution unless cancellation already owns cleanup."""
        with self._lock:
            if self._start_prevented:
                return False
            self._started = True
            return True

    def settle_success(self, result: T) -> None:
        cleanup_result = False
        with self._lock:
            self._settled = True
            self._result = result
            self._failure = None
            self._has_result = True
            if self._abandoned and not self._cleanup_taken:
                self._cleanup_taken = True
                cleanup_result = True
                self._result = None
                self._has_result = False
        if cleanup_result:
            _run_cleanup(self._cleanup_abandoned, result)

    def settle_failure(self, failure: BaseException) -> None:
        cleanup = False
        with self._lock:
            self._settled = True
            self._has_result = False
            self._result = None
            self._failure = failure
            if self._abandoned and not self._cleanup_taken:
                self._cleanup_taken = True
                cleanup = True
        if cleanup:
            _run_cleanup(self._cleanup_abandoned, None)

    def abandon(self) -> None:
        result: T | None = None
        cleanup = False
        with self._lock:
            self._abandoned = True
            if self._settled and not self._cleanup_taken:
                if self._has_result:
                    result = self._result
                self._result = None
                self._has_result = False
                self._cleanup_taken = True
                cleanup = True
        if cleanup:
            _run_cleanup(self._cleanup_abandoned, result)

    def claim(self) -> None:
        with self._lock:
            self._cleanup_taken = True
            self._has_result = False
            self._result = None
            self._failure = None

    def backend_cancelled_error(self) -> asyncio.CancelledError | None:
        with self._lock:
            if isinstance(self._failure, asyncio.CancelledError):
                return self._failure
            return None

    def cancel_unstarted(self) -> None:
        """Clean an offload whose executor work was cancelled while queued."""
        cleanup = False
        with self._lock:
            if not self._started and not self._settled and not self._cleanup_taken:
                # The native wrapper must consult this flag before touching
                # request resources. Whichever side wins the lock either
                # prevents execution and cleans here, or lets the wrapper run
                # and clean at its eventual settlement.
                self._start_prevented = True
                self._abandoned = True
                self._settled = True
                self._cleanup_taken = True
                self._failure = None
                cleanup = True
        if cleanup:
            _run_cleanup(self._cleanup_abandoned, None)


def _consume_abandoned_task(
    task: asyncio.Task[Any],
    ownership: _ResultOwnership[Any],
) -> None:
    if task.cancelled():
        ownership.cancel_unstarted()
    _ABANDONED_OFFLOADS.discard(task)
    try:
        task.exception()
    except BaseException:
        # The request has already observed cancellation. Retrieving an
        # eventual backend failure only prevents an unhandled-task warning.
        pass


def _track_abandoned_task(
    task: asyncio.Task[Any],
    ownership: _ResultOwnership[Any],
) -> None:
    # asyncio keeps only weak references to tasks. Keep abandoned offloads
    # alive until their executor future settles, and consume any exception.
    _ABANDONED_OFFLOADS.add(task)
    task.add_done_callback(
        lambda done: _consume_abandoned_task(done, ownership)
    )


async def run_native_offload(
    call: Callable[[], T],
    *,
    cleanup_abandoned: Callable[[T | None], None] | None = None,
) -> T:
    """Run synchronous modality work without invalidating live resources.

    On normal settlement the caller retains ownership of inputs and any
    returned value. If the awaiting coroutine is cancelled, cancellation is
    propagated immediately. After ``call`` actually returns or raises,
    ``cleanup_abandoned`` receives its result, or ``None`` on failure, and
    owns every resource captured by the abandoned operation.

    A native callable that raises ``asyncio.CancelledError`` is also treated
    as abandoned: asyncio marks the inner task cancelled, so the helper runs
    cleanup and re-raises the original native exception.

    The ownership handshake lives inside the native wrapper rather than in a
    task done callback. Event-loop shutdown can cancel an asyncio task while
    its executor thread is still running; task completion alone is therefore
    not evidence that native resources are safe to release.

    Cleanup callbacks must be small, synchronous, and thread-safe. Depending
    on which side wins the cancellation/settlement race, cleanup runs either
    in the native worker or in the event-loop thread.
    """
    ownership = _ResultOwnership(cleanup_abandoned)

    def _run() -> T:
        if not ownership.try_start():
            # The asyncio task was cancelled while this work was queued and
            # its done callback already released ownership. The return value
            # cannot be observed because that task is cancelled.
            return None  # type: ignore[return-value]
        try:
            result = call()
        except BaseException as exc:
            ownership.settle_failure(exc)
            raise
        else:
            ownership.settle_success(result)
            return result

    task = asyncio.create_task(asyncio.to_thread(_run))
    try:
        # asyncio.wait never forwards this request task's cancellation into
        # a member task. Unlike asyncio.shield on Python 3.14, it also does
        # not report an eventual abandoned exception independently through
        # the loop exception handler.
        await asyncio.wait((task,))
    except BaseException:
        # Any external unwind abandons the still-independent native task.
        # Cancellation is the common case, but GeneratorExit and other base
        # exceptions must not leave its eventual resources unowned either.
        ownership.abandon()
        _track_abandoned_task(task, ownership)
        raise

    try:
        result = task.result()
    except asyncio.CancelledError:
        # The wait completed, so this cancellation belongs to the native
        # callable rather than the request waiter. Preserve its original
        # exception (including message) while taking abandoned ownership.
        backend_cancelled = ownership.backend_cancelled_error()
        ownership.abandon()
        if backend_cancelled is not None:
            raise backend_cancelled
        raise
    except BaseException:
        ownership.claim()
        raise
    else:
        ownership.claim()
        return result
