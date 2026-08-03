"""LoadDirector: lazy-load coordinator for v0.40.0.

Holds the in-memory map of currently-loaded models, gates concurrent loads
of the same model behind a per-model threading.Event (singleton-load
collapse), and routes decisions to an injected memory probe + enable/
disable callable seam so unit tests don't need a real supervisor.

Three-phase acquire (decide / load / commit):

  Decision (under self.lock)
    - Hot path: increment refcount, update last_touched_at, return port.
    - Cold path: claim ownership of the load by stashing a fresh
      threading.Event in `in_flight_loads`. Other threads racing on the
      same model_id see the Event, drop the lock, and wait. When woken,
      they re-enter the decision phase to read the freshly populated
      LoadEntry.
    - Eviction (Task C): if the requested memory does not fit live free
      (minus headroom), the decision phase returns ("evict_and_retry",
      shortfall_gb, device) WITHOUT claiming an in-flight slot. The
      acquire outer loop runs eviction outside the lock and re-calls
      _decide, which may classify the entry as hot (another thread
      loaded it during our eviction window), or cold-fits, or raise 503
      if eviction can't free enough memory. Re-decide on every retry
      protects against TOCTOU windows during the unlocked eviction.

  Load (lock NOT held)
    - Capture free_before via memory_probe.
    - Call enable_fn(model_id), which is the long-running worker spawn.
    - Capture free_after.
    - Failures here are caught in the commit phase: the in-flight Event
      is popped + set so waiters wake up, no LoadEntry is recorded, and
      the original exception propagates to the caller.

  Commit (under self.lock)
    - Insert LoadEntry, pop the in_flight_loads Event, append a
      DecisionLogEntry, set() the Event so any concurrent waiters wake.
    - Schedule the observed-peak writeback (Task D): a fire-and-forget
      daemon thread compares observed delta vs the recorded
      `measurements.<device>.peak_bytes` and raises it if observation
      exceeds the seed. Errors are logged + swallowed so the request
      hot path is unaffected.

This module does NOT import enable_model / disable_model from
muse.admin.operations at import time (it does lazy-import OperationError
inside the eviction failure path). The callable injection seam
(enable_fn, disable_fn, memory_probe) is what tests use; production
wiring (Task E) will inject the real supervisor's spawn / shutdown
callables.

H4 (event.wait timeout): waiters on an in-flight cold load call
  event.wait(timeout=_INFLIGHT_WAIT_TIMEOUT_SECONDS) rather than an
  unbounded wait(). If the load-winner thread is killed non-cooperatively
  (SIGKILL/OOM during wait_for_ready), _abort/set() never runs and every
  waiter would block forever without a timeout. On timeout, the waiter
  re-enters the decision phase: if the winner succeeded the model will be
  hot; if the winner failed there will be no LoadEntry and the waiter
  becomes the new winner. This avoids both permanent hang and tight-spin.
"""
from __future__ import annotations

import collections
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

# Task D writeback uses the existing catalog read / atomic-write helpers.
# These are imported at module scope so tests can monkeypatch them on the
# load_director module's namespace (see tests/cli_impl/test_load_director.py
# `test_swallows_ioerror_during_write` etc.).
from muse.core.catalog import _CATALOG_WRITE_LOCK, _read_catalog, _write_catalog

# M1: _WRITEBACK_LOCK is an alias to the catalog-level _CATALOG_WRITE_LOCK so
# that the observed-peak writeback in _observed_peak_writeback (which does its
# own _read_catalog -> mutate -> _write_catalog) participates in the same
# shared lock as all other catalog RMW sites (probe.py, set_enabled, etc.).
# Keeping the name _WRITEBACK_LOCK preserves backward compatibility with any
# code or tests that reference it directly (the tests mock it at the
# load_director module level, which still works because this binding points
# at the same Lock object as _CATALOG_WRITE_LOCK in catalog.py).
_WRITEBACK_LOCK = _CATALOG_WRITE_LOCK

# H4: maximum seconds a waiter will block on an in-flight Event before
# re-entering the decision phase. The value must exceed the load timeout
# used by wait_for_ready (120s in production) plus some margin so normal
# loads never time out. 180s is 60s of headroom above the worst-case
# wait_for_ready window. A waiter that times out re-decides: if the
# winner succeeded it takes the hot path; if the winner failed it becomes
# the new winner. This avoids permanent hang when a winner thread is
# killed non-cooperatively (SIGKILL / OOM / hardware reset) without
# introducing a tight spin loop.
_INFLIGHT_WAIT_TIMEOUT_SECONDS = 180.0

from muse.core.memory_probe import (
    available_capacity_gb,
    declared_device,
    resolve_memory_pool,
)

# Task 9: fire-and-forget telemetry. Import the NAME (not the module) so
# `record` is a module global here; tests monkeypatch
# `muse.cli_impl.load_director.record` directly, which is only effective
# because the calls below reference the bare name. The observability
# package-level re-export (`from muse.observability import record`) lands
# in a later task; importing straight from the submodule avoids depending
# on that re-export existing yet. This import is import-light (stdlib +
# two observability siblings, no torch), so it is safe on the CLI import
# path.
from muse.observability.recorder import record

logger = logging.getLogger(__name__)


# (M1: _WRITEBACK_LOCK is now an alias to catalog._CATALOG_WRITE_LOCK,
# defined below near the catalog imports. The old module-level
# threading.Lock() is removed to prevent two locks guarding the same
# resource. See the import block above.)


# Default headroom margins in GB. Subtracted from live free before the
# fit check. Spec defaults: 1 GB GPU, 2 GB CPU. Constructor accepts
# overrides so tests can pin any value they like.
_DEFAULT_GPU_HEADROOM_GB = 1.0
_DEFAULT_CPU_HEADROOM_GB = 2.0


@dataclass
class LoadEntry:
    """One currently-loaded model.

    `memory_gb` is what we accounted for at load time (from the
    manifest's capabilities.memory_gb, falling back to 0.0). Task D
    will add observed-peak writeback so this drifts toward measured
    reality as cold loads happen.

    `last_touched_at` is monotonic seconds, updated on each acquire and
    release. Eviction sorts evictable candidates (refcount == 0) by this
    field ascending (LRU first).
    """
    model_id: str
    worker_port: int
    memory_gb: float
    refcount: int
    last_touched_at: float
    loaded_at: float
    # Concrete accounting pool. The default preserves compatibility for
    # callers that construct LoadEntry directly; director-created entries
    # always stamp the resolved pool explicitly.
    pool: str = "cpu"


@dataclass
class DecisionLogEntry:
    """One load or evict decision, surfaced in /v1/admin/memory.

    `timestamp` is wall-clock seconds (time.time(), not monotonic) so
    the admin endpoint can render an ISO string for humans. The deque
    holding these is capped at maxlen=20 by LoadDirector.

    `evicted` is always a list (possibly empty) so the wire shape is
    uniform across load and evict actions.
    """
    timestamp: float
    model_id: str
    action: Literal["load", "evict"]
    memory_gb: float
    free_before_gb: float
    free_after_gb: float | None
    reason: str
    evicted: list[str] = field(default_factory=list)


@dataclass
class InFlightLoad:
    """A cold load that has been claimed but whose worker has not yet
    allocated its memory.

    Carries `memory_gb` + `pool` so that concurrent `_decide` calls can
    RESERVE the pending load against the live free-memory reading before the
    worker actually consumes VRAM. The live probe does not reflect an
    in-flight load until its worker allocates (seconds later), so without
    this reservation two concurrent cold loads for different models would
    both pass the fit check against the same free reading and over-commit the
    device. The single-event-loop request path masked this by serializing
    every acquire; running acquire off-loop (asyncio.to_thread) exposes it.
    """
    event: threading.Event
    memory_gb: float
    pool: str  # "cuda" or "cpu": the memory pool this load reserves against
    # Lease metadata lets a waiter distinguish a genuinely active owner from
    # an abandoned claim after its bounded wait. Unknown owners (legacy/test
    # records) are reclaimable once the lease expires.
    claimed_at: float = field(default_factory=time.monotonic)
    owner_thread: threading.Thread | None = field(default=None, repr=False)


@dataclass(eq=False)
class ModelEviction:
    """Identity token for one director-owned model teardown.

    The entry is removed from ``loaded`` and, when present, remains accounted
    in the evicting-memory maps until this exact token settles.  Admin disable
    claims additionally keep the model blocked after success; idle claims are
    only transiently blocked by the ordinary ``evicting`` marker.
    """

    model_id: str
    entry: LoadEntry | None
    reason: str
    keep_blocked: bool
    was_blocked: bool
    free_before_gb: float | None
    token: object = field(default_factory=object, repr=False)


class LoadDirector:
    """Coordinates lazy loads + LRU eviction for the supervisor.

    Constructor injection:
      - enable_fn(model_id) -> worker_port: the long-running call to
        spawn (or wake) the per-model worker. In production this is a
        thin wrapper around `muse.admin.operations.enable_model`; in
        tests it's a MagicMock returning a fixed port.
      - disable_fn(model_id) -> None: complement to enable_fn. Called
        by the eviction loop on each LRU victim. The OS reclaims the
        worker's memory after SIGTERM; the director polls free memory
        with a 2s budget per victim before retrying the fit check.
      - memory_probe: any object with .gpu_free_gb() and .cpu_free_gb()
        methods. In production this is a `muse.core.memory_probe.MemoryProbe`;
        in tests it's a MagicMock with .return_value set on each method.

    Concurrency:
      - `lock` is an RLock so the same thread can re-enter (e.g. status
        snapshot from inside an admin route that already holds it).
      - `in_flight_loads` maps model_id to an InFlightLoad record (Event +
        reserved memory_gb + pool). The record is created under the lock at
        decision time; waiters drop the lock, await its Event, then re-enter
        the decision phase. The winner sets the Event in the commit phase (or
        on exception in the cleanup path). The reserved memory is debited
        from available memory by every concurrent `_decide` so that
        off-loop (multi-thread) acquires cannot over-admit the device.
      - `recent_decisions` is a deque(maxlen=20) appended to under the
        lock; admin reads it via list() while holding the lock for a
        consistent snapshot.
    """

    def __init__(
        self,
        *,
        enable_fn: Callable[[str], int],
        disable_fn: Callable[[str], None],
        memory_probe: Any,
        gpu_budget_gb: float | None = None,
        cpu_budget_gb: float | None = None,
        gpu_headroom_gb: float = _DEFAULT_GPU_HEADROOM_GB,
        cpu_headroom_gb: float = _DEFAULT_CPU_HEADROOM_GB,
        cuda_available_fn: Callable[[], bool] | None = None,
    ):
        self.enable_fn = enable_fn
        self.disable_fn = disable_fn
        self.memory_probe = memory_probe

        self.gpu_budget_gb = self._validate_capacity_knob(
            "gpu_budget_gb", gpu_budget_gb, optional=True,
        )
        self.cpu_budget_gb = self._validate_capacity_knob(
            "cpu_budget_gb", cpu_budget_gb, optional=True,
        )
        self.gpu_headroom_gb = self._validate_capacity_knob(
            "gpu_headroom_gb", gpu_headroom_gb,
        )
        self.cpu_headroom_gb = self._validate_capacity_knob(
            "cpu_headroom_gb", cpu_headroom_gb,
        )
        self.cuda_available_fn = cuda_available_fn

        self.loaded: dict[str, LoadEntry] = {}
        self.in_flight_loads: dict[str, InFlightLoad] = {}
        # v0.57.2: model ids whose eviction is mid-flight (victim popped
        # from `loaded`, disable_fn still shutting the worker down, VRAM
        # not yet freed). During that multi-second out-of-lock window the
        # director otherwise looks idle, so concurrent capacity 503s must
        # consult this set to stay retryable. Guarded by self.lock.
        self.evicting: set[str] = set()
        # Concrete pool for director-owned markers. Keep the public set for
        # compatibility/status tests; this side map prevents a CPU eviction
        # from making an unrelated CUDA capacity failure look retryable (and
        # vice versa). Unknown markers inserted by older callers remain
        # conservatively relevant to both pools.
        self._evicting_pools: dict[str, str] = {}
        # Declared memory remains reserved from configured budgets until
        # disable_fn confirms the worker has stopped. Without this, popping
        # a victim from `loaded` makes a static-budget pool look free during
        # the out-of-lock shutdown window and another cold load can overlap
        # the still-resident victim.
        self._evicting_memory_gb: dict[str, float] = {}
        # Explicit admin disable is persistent: once claimed, new acquires
        # remain blocked until the matching enable path calls allow_model().
        # Idle/LRU eviction uses the transient `evicting` marker instead.
        self._blocked_models: set[str] = set()
        # A worker that started but could not be rolled back is conservatively
        # represented in ``loaded`` and kept blocked until explicit teardown
        # succeeds. This marker prevents admin enable from clearing that block
        # and routing to a placeholder/uncertain worker port.
        self._failed_rollbacks: set[str] = set()
        # Exact claim identity prevents a stale finally block from clearing a
        # newer teardown generation's markers or restoring its old LoadEntry.
        self._eviction_claims: dict[str, ModelEviction] = {}
        self.recent_decisions: collections.deque = collections.deque(maxlen=20)
        self.lock = threading.RLock()
        # Monotonic counter bumped under the lock on every memory-mutating
        # event: a cold load claimed (_decide) OR an eviction victim popped
        # (_evict_lru_until_fits). _load_and_commit snapshots it around a
        # load's free_before .. free_after window to detect whether ANOTHER
        # load or an eviction happened during that window (either pollutes the
        # global free-memory delta the observed-peak writeback infers a peak
        # from).
        self._inflight_epoch = 0

        # Bound passive catalog persistence to one drain thread per director.
        # Repeated observations coalesce monotonically by model/device while
        # that worker is busy instead of creating an unbounded thread burst.
        self._writeback_state_lock = threading.Lock()
        self._pending_writebacks: dict[tuple[str, str], int] = {}
        self._writeback_thread: threading.Thread | None = None

        # Pressure decisions serialize only within one accounting pool. CPU
        # and CUDA evictions can still progress independently.
        self._eviction_pool_locks = {
            "cpu": threading.Lock(),
            "cuda": threading.Lock(),
        }

        # Fired (fire-and-forget) whenever capacity MAY have freed: a
        # release() that drops a refcount to 0, or a completed eviction.
        # The supervisor wires this to CapacityNotifier.notify so gateway
        # capacity-waiters wake and re-decide. Never allowed to raise into
        # the caller (wrapped at each call site).
        self.capacity_listener: Callable[[], None] | None = None

    @staticmethod
    def _validate_capacity_knob(
        name: str, value: float | None, *, optional: bool = False,
    ) -> float | None:
        """Return a finite non-negative capacity setting or raise clearly."""
        if value is None:
            if optional:
                return None
            raise ValueError(f"{name} must be a finite non-negative number")
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite non-negative number")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a finite non-negative number",
            ) from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
        return numeric

    @staticmethod
    def _memory_requirement_gb(manifest: dict, model_id: str) -> float:
        """Validate one manifest memory estimate before accounting with it."""
        manifest_valid = isinstance(manifest, dict)
        capabilities = (
            (manifest.get("capabilities") or {})
            if manifest_valid
            else manifest
        )
        capabilities_valid = isinstance(capabilities, dict)
        raw = (
            capabilities.get("memory_gb", 0.0)
            if capabilities_valid
            else capabilities
        )
        if raw is None:
            raw = 0.0
        try:
            if not manifest_valid or not capabilities_valid or isinstance(raw, bool):
                raise ValueError
            memory_gb = float(raw)
        except (TypeError, ValueError):
            memory_gb = math.nan
        if not math.isfinite(memory_gb) or memory_gb < 0.0:
            from muse.admin.operations import OperationError
            raise OperationError(
                "invalid_model_manifest",
                (
                    f"model {model_id!r} has invalid capabilities.memory_gb "
                    f"value {raw!r}; expected a finite non-negative number"
                ),
                status=503,
                retryable=False,
            )
        return memory_gb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, model_id: str, *, manifest: dict) -> int:
        """Increment refcount on a loaded model, or load it cold.

        The three phases run as documented in the module docstring.
        Returns the worker_port hosting `model_id`.

        Raises:
          OperationError("model_too_large_for_device", status=503): when
            memory doesn't fit and on-demand LRU eviction can't free
            enough (no evictable candidates, or sum of evictable memory
            < shortfall).
          Exception: re-raises any exception from enable_fn after
            cleaning up the in-flight Event so concurrent waiters wake
            and re-enter the decision phase.
        """
        return self._acquire_or_warmup(
            model_id, manifest=manifest, bump_refcount=True,
        )

    def warmup(self, model_id: str, *, manifest: dict) -> int:
        """Pre-load a model without serving a request.

        Like `acquire` but does NOT increment refcount on either the hot
        or cold path. The semantic is "load this model now so subsequent
        requests are hot." The loaded entry's initial refcount is 0,
        making it immediately eligible for LRU eviction if pressure
        arises before any request lands.

        Hot path (model already loaded): returns the existing port
        without touching refcount or last_touched_at. The intent is
        idempotency: repeated warmup calls must not skew LRU ordering
        or pin the model artificially.

        Cold path: runs the full decide / load / commit cycle including
        on-demand LRU eviction. The committed LoadEntry has refcount=0,
        which is the only behavioral difference from acquire.

        Returns the worker_port hosting `model_id` after the warmup.

        Raises:
          OperationError("model_too_large_for_device", status=503): when
            memory doesn't fit and on-demand LRU eviction can't free
            enough (same as acquire).
          Exception: re-raises any exception from enable_fn after
            cleaning up the in-flight Event so concurrent waiters wake
            and re-enter the decision phase.
        """
        return self._acquire_or_warmup(
            model_id, manifest=manifest, bump_refcount=False,
        )

    def _acquire_or_warmup(
        self, model_id: str, *, manifest: dict, bump_refcount: bool,
    ) -> int:
        """Shared body for acquire (bump_refcount=True) and warmup (False).

        The two operations differ only in:
          - Hot path: acquire bumps refcount + last_touched; warmup
            doesn't.
          - Cold commit: acquire seeds the LoadEntry refcount at 1;
            warmup seeds at 0.
        """
        # Loop because a thread that lost the singleton race and waited
        # on an in_flight Event may wake to find that the winner failed
        # (no LoadEntry was inserted). On wake, we re-enter the decision
        # phase rather than blindly trusting the winner.
        #
        # The same loop also handles the "evict_and_retry" cycle: when
        # _decide reports the model doesn't fit, we run eviction OUTSIDE
        # the lock, then re-decide. The state may have changed during
        # the unlocked eviction window: another thread may have loaded
        # the model already, or stolen our victim, or pushed yet more
        # memory pressure. Re-decide is the correct recovery.
        while True:
            decision = self._decide(
                model_id, manifest=manifest, bump_refcount=bump_refcount,
            )
            phase = decision[0]

            if phase == "hot":
                # _decide returns the port read under the same lock that
                # classified the entry as loaded. No TOCTOU window: an
                # eviction cannot race between classification and read
                # because both happen inside the lock.
                return decision[1]

            if phase == "wait":
                # We did not own the load; another thread did. Wait for
                # its Event to fire and then re-decide.
                #
                # H4: use a bounded wait instead of unbounded wait().
                # If the winner is killed non-cooperatively (SIGKILL/OOM
                # during the ~120s wait_for_ready), _abort/set() never
                # runs. With no timeout, every waiter blocks forever.
                # With a timeout, the waiter wakes and re-enters _decide:
                # if the winner succeeded the model will be hot; if the
                # winner failed there will be no LoadEntry and this
                # thread becomes the new winner. This breaks the permanent
                # hang without introducing a tight spin loop because each
                # iteration is bounded by _INFLIGHT_WAIT_TIMEOUT_SECONDS.
                claim = decision[1]
                self._wait_for_in_flight(model_id, claim)
                # Loop and re-decide. Either the entry is now hot (winner
                # succeeded) or it's still cold (winner raised or was
                # killed without calling _abort) and this thread will
                # become the new winner.
                continue

            if phase == "evict_and_retry":
                # Run eviction outside the lock so concurrent acquires
                # for already-loaded models can hot-acquire during the
                # disable_fn + memory-release-poll window. Eviction
                # raises OperationError(503) if it can't free enough.
                _, shortfall_gb, device, required_gb = decision
                self._evict_lru_until_fits(
                    model_id=model_id,
                    shortfall_gb=shortfall_gb,
                    device=device,
                    required_gb=required_gb,
                )
                # Re-enter the decision phase. State may have changed:
                # another thread may have loaded our model, or evicted
                # additional models, or grown the memory pressure.
                continue

            # phase == "load": this thread won the singleton race.
            # The Event is already in self.in_flight_loads under our name.
            # Run the (long) load phase outside the lock, then commit.
            initial_refcount = 1 if bump_refcount else 0
            return self._load_and_commit(
                model_id,
                manifest=manifest,
                claim=decision[1],
                initial_refcount=initial_refcount,
            )

    def release(self, model_id: str) -> None:
        """Decrement refcount + bump last_touched_at.

        Release does NOT auto-evict: eviction is on-demand, triggered
        only from a cold acquire that needs room. The release call
        merely flips the refcount toward 0 (making the entry eligible
        for eviction on the next pressure point) and updates the LRU
        timestamp so traffic recency drives the eviction order.

        Releasing an unknown model_id is a no-op (defensive: an evicted
        model whose final request just finished should not crash the
        gateway's `finally:` clause).

        When this release drops the refcount to 0, fires capacity_listener
        (outside the lock) so gateway capacity-waiters get a chance to
        re-decide.
        """
        dropped_to_zero = False
        with self.lock:
            entry = self.loaded.get(model_id)
            if entry is None:
                logger.debug("release(%r): model not in loaded set; ignoring", model_id)
                return
            before = entry.refcount
            entry.refcount = max(0, entry.refcount - 1)
            entry.last_touched_at = time.monotonic()
            dropped_to_zero = before > 0 and entry.refcount == 0
        if dropped_to_zero:
            self._fire_capacity_listener()

    def allow_model(self, model_id: str) -> None:
        """Allow future acquires after an explicit admin enable.

        Clearing is intentionally independent of worker state: the next
        acquire either observes an existing director entry or performs the
        normal singleton load/commit against a worker the admin path warmed.
        """
        from muse.admin.operations import OperationError

        with self.lock:
            if model_id in self._eviction_claims:
                return
            if model_id in self._failed_rollbacks:
                raise OperationError(
                    "model_cleanup_required",
                    f"model {model_id!r} has an uncertain worker from a "
                    "failed load rollback; disable it before enabling",
                    status=503,
                    retryable=True,
                )
            removed = model_id in self._blocked_models
            self._blocked_models.discard(model_id)
        if removed:
            self._fire_capacity_listener()

    def begin_model_disable(self, model_id: str) -> ModelEviction:
        """Atomically block new acquires and claim an idle model for disable.

        The block is installed *before* checking refcounts/in-flight loads, so
        an acquire cannot enter between a busy check and the teardown claim.
        A refused claim restores the prior allow state.  Active requests and
        cold loads are never killed out from under their owners; callers get a
        retryable 409 and may retry once they settle.
        """
        from muse.admin.operations import OperationError

        with self.lock:
            was_blocked = model_id in self._blocked_models
            self._blocked_models.add(model_id)
            entry = self.loaded.get(model_id)
            busy = (
                model_id in self.evicting
                or model_id in self._eviction_claims
                or model_id in self.in_flight_loads
                or (entry is not None and entry.refcount > 0)
            )
            if busy:
                if not was_blocked:
                    self._blocked_models.discard(model_id)
                raise OperationError(
                    "model_in_use",
                    f"model {model_id!r} has active requests or a load in progress",
                    status=409,
                    retryable=True,
                )
            return self._begin_eviction_locked(
                model_id,
                entry=entry,
                reason="admin_disabled",
                keep_blocked=True,
                was_blocked=was_blocked,
            )

    def begin_idle_eviction(
        self,
        model_id: str,
        *,
        expected_entry: LoadEntry,
        idle_before: float,
        reason: str,
    ) -> ModelEviction | None:
        """Claim an exact still-idle entry, or return None after any race.

        Identity, refcount, and freshness are rechecked in the same lock hold
        that publishes the evicting reservation and removes the entry.
        """
        with self.lock:
            if model_id in self._blocked_models or model_id in self.evicting:
                return None
            entry = self.loaded.get(model_id)
            if (
                entry is not expected_entry
                or entry.refcount > 0
                or entry.last_touched_at > idle_before
            ):
                return None
            return self._begin_eviction_locked(
                model_id,
                entry=entry,
                reason=reason,
                keep_blocked=False,
                was_blocked=False,
            )

    def _begin_eviction_locked(
        self,
        model_id: str,
        *,
        entry: LoadEntry | None,
        reason: str,
        keep_blocked: bool,
        was_blocked: bool,
    ) -> ModelEviction:
        """Publish one exact eviction claim. Caller holds ``self.lock``."""
        if model_id in self._eviction_claims:
            raise RuntimeError(f"model {model_id!r} already has an eviction claim")

        free_before_gb: float | None = None
        self.evicting.add(model_id)
        try:
            if entry is not None:
                # Reserve before pop: configured-budget admission must continue
                # to count the worker throughout its out-of-lock shutdown.
                try:
                    free_before_gb = self._free_for_device(entry.pool)
                except Exception:  # noqa: BLE001
                    # Measurement is observability only. A transient probe
                    # failure must not prevent publishing a settleable claim.
                    free_before_gb = None
                    logger.warning(
                        "could not measure memory before eviction of %r",
                        model_id,
                        exc_info=True,
                    )
                self._evicting_pools[model_id] = entry.pool
                self._evicting_memory_gb[model_id] = entry.memory_gb
                self.loaded.pop(model_id, None)
                self._inflight_epoch += 1

            claim = ModelEviction(
                model_id=model_id,
                entry=entry,
                reason=reason,
                keep_blocked=keep_blocked,
                was_blocked=was_blocked,
                free_before_gb=free_before_gb,
            )
            self._eviction_claims[model_id] = claim
            return claim
        except BaseException:
            # No caller received a token it could settle. Roll back every
            # marker under the same lock so an interruption or unexpected
            # publication failure cannot leave the model permanently blocked.
            self._eviction_claims.pop(model_id, None)
            self.evicting.discard(model_id)
            self._evicting_pools.pop(model_id, None)
            self._evicting_memory_gb.pop(model_id, None)
            if entry is not None:
                self.loaded.setdefault(model_id, entry)
            if keep_blocked and not was_blocked:
                self._blocked_models.discard(model_id)
            raise

    def finish_eviction(
        self,
        claim: ModelEviction,
        *,
        success: bool,
        error: BaseException | None = None,
    ) -> bool:
        """Settle one exact claim and wake capacity waiters outside the lock.

        Failed teardown restores the claimed LoadEntry because the worker may
        still own its memory.  Explicit-disable blocking is intentionally not
        cleared here: the admin wrapper decides from the catalog's committed
        state whether failure should remain fail-closed or restore allowance.
        """
        with self.lock:
            if self._eviction_claims.get(claim.model_id) is not claim:
                return False

            entry = claim.entry
            if not success and entry is not None:
                self.loaded.setdefault(claim.model_id, entry)
            elif success:
                self._failed_rollbacks.discard(claim.model_id)

            self.evicting.discard(claim.model_id)
            self._evicting_pools.pop(claim.model_id, None)
            self._evicting_memory_gb.pop(claim.model_id, None)

            if not claim.keep_blocked:
                # Idle eviction is transient and must never leave a persistent
                # disabled state, on either success or rollback.
                self._blocked_models.discard(claim.model_id)

            self._eviction_claims.pop(claim.model_id, None)

            if success and entry is not None:
                try:
                    free_after_gb = self._free_for_device(entry.pool)
                except Exception:  # noqa: BLE001
                    free_after_gb = None
                    logger.warning(
                        "could not measure memory after eviction of %r",
                        claim.model_id,
                        exc_info=True,
                    )
                self.recent_decisions.append(DecisionLogEntry(
                    timestamp=time.time(),
                    model_id=claim.model_id,
                    action="evict",
                    memory_gb=entry.memory_gb,
                    free_before_gb=float(claim.free_before_gb or 0.0),
                    free_after_gb=free_after_gb,
                    reason=claim.reason,
                    evicted=[claim.model_id],
                ))
            elif error is not None:
                logger.warning(
                    "eviction of %r failed; restored director accounting: %s",
                    claim.model_id,
                    error,
                )

        self._fire_capacity_listener()
        return True

    def _fire_capacity_listener(self) -> None:
        """Fire capacity_listener, if set, swallowing any exception.

        Called outside self.lock (never call foreign code under the
        director lock). A listener that raises must never break a
        release or eviction; it is logged and dropped.
        """
        listener = self.capacity_listener
        if listener is None:
            return
        try:
            listener()
        except Exception:  # noqa: BLE001
            logger.warning("capacity_listener raised; ignoring", exc_info=True)

    def status(self) -> dict[str, dict[str, Any]]:
        """Snapshot of currently-loaded models for /v1/models lookup.

        Each value is `{"loaded": True, "worker_port", "last_touched_at",
        "refcount"}`. Models not currently loaded are absent.
        """
        with self.lock:
            return {
                mid: {
                    "loaded": True,
                    "worker_port": e.worker_port,
                    "last_touched_at": e.last_touched_at,
                    "refcount": e.refcount,
                }
                for mid, e in self.loaded.items()
            }

    def observed_peak(
        self,
        model_id: str,
        *,
        observed_peak_bytes: int,
        device: str,
    ) -> threading.Thread | None:
        """Schedule a passive writeback of the observed peak.

        Spec section "Lazy-load passive observation": every cold load
        compares the observed `free_before - free_after` delta against
        the recorded `catalog[model_id].measurements.<device>.peak_bytes`.
        If observed > recorded (or recorded is missing), the catalog gets
        the new larger value via the existing atomic write-then-rename
        in `muse.core.catalog._write_catalog`.

        This is fire-and-forget: a daemon thread does the
        read-modify-write so the request hot path is unaffected.
        Failures are logged at WARNING level and swallowed because a
        corrupted catalog file or transient filesystem error must NOT
        surface as a 500 to the user. Returns the Thread reference (so
        tests can deterministically join it via .join()); production
        callers can ignore the return value.

        `observed_peak_bytes <= 0` is treated as a no-op (negative deltas
        mean another process freed memory during our load window, or the
        measurement was meaningless; we do not corrupt a recorded peak
        with a zero or negative observation).

        `device` is the catalog measurements bucket key. Existing probe
        records use "cuda" / "cpu" / "mps"; the alias "gpu" normalizes
        to "cuda", and the manifest convention "auto" / "" is resolved
        to the device the load actually consumed (cuda if a GPU is
        available, else cpu) so the bucket key matches the probe's.
        """
        if observed_peak_bytes <= 0:
            return None

        # Normalize the bucket key. Three cases:
        #   "gpu" -> "cuda": legacy alias.
        #   "auto" / "": resolved via the SHARED _auto_resolves_to_cuda so
        #     the writeback bucket lands in the same pool the fit/evict
        #     accounting sized against. Without this, manifests declaring
        #     `device: "auto"` would split-brain against probe records that
        #     always write the resolved name.
        #   anything else (cpu / cuda / mps): passed through verbatim, so an
        #     MPS load keeps its own "mps" bucket matching what
        #     `muse models probe` writes -- distinct from the host-RAM POOL
        #     that _resolve_pool_device("mps") sizes against.
        if device == "gpu":
            device_key = "cuda"
        elif device in ("auto", ""):
            device_key = "cuda" if self._auto_resolves_to_cuda() else "cpu"
        else:
            device_key = device

        key = (model_id, device_key)
        with self._writeback_state_lock:
            self._pending_writebacks[key] = max(
                observed_peak_bytes,
                self._pending_writebacks.get(key, 0),
            )
            thread = self._writeback_thread
            if thread is not None and thread.is_alive():
                return thread
            try:
                thread = threading.Thread(
                    target=self._drain_observed_peak_writebacks,
                    daemon=True,
                    name="observed-peak-writeback",
                )
                self._writeback_thread = thread
                thread.start()
                return thread
            except Exception:  # noqa: BLE001
                self._writeback_thread = None
                self._pending_writebacks.clear()
                logger.warning(
                    "could not start observed-peak writeback for %r",
                    model_id,
                    exc_info=True,
                )
                return None

    def _drain_observed_peak_writebacks(self) -> None:
        """Drain coalesced observations until no work remains."""
        while True:
            with self._writeback_state_lock:
                if not self._pending_writebacks:
                    self._writeback_thread = None
                    return
                pending = self._pending_writebacks
                self._pending_writebacks = {}
            for (model_id, device_key), observed_peak_bytes in pending.items():
                self._observed_peak_writeback(
                    model_id,
                    observed_peak_bytes,
                    device_key,
                )

    @staticmethod
    def _observed_peak_writeback(
        model_id: str,
        observed_peak_bytes: int,
        device_key: str,
    ) -> None:
        """Body of the writeback thread; runs OFF the request hot path.

        Held under the module-level _WRITEBACK_LOCK so concurrent cold
        loads of different models do not lose updates via interleaved
        read-modify-write.

        Errors are logged at WARNING and swallowed: the catalog write
        is best-effort, not an invariant.
        """
        try:
            with _WRITEBACK_LOCK:
                catalog = _read_catalog()
                entry = catalog.get(model_id)
                if entry is None:
                    # Model removed (e.g. `muse models remove`) since the
                    # load completed. Nothing to write back; not an error.
                    logger.debug(
                        "observed_peak: %r not in catalog; skipping writeback",
                        model_id,
                    )
                    return

                measurements = entry.setdefault("measurements", {})
                bucket = measurements.get(device_key) or {}
                recorded = int(bucket.get("peak_bytes") or 0)
                if observed_peak_bytes <= recorded:
                    # Estimate is monotonically upward only.
                    return

                bucket["peak_bytes"] = int(observed_peak_bytes)
                bucket["device"] = device_key
                bucket["source"] = "lazy_load_observation"
                bucket["observed_at"] = datetime.now(timezone.utc).isoformat()
                # If `weights_bytes` is absent (no probe ever ran), seed
                # it with the observed peak as a conservative lower bound.
                # This keeps existing /v1/models renderers working even
                # when only lazy-load measurements exist.
                bucket.setdefault("weights_bytes", int(observed_peak_bytes))
                measurements[device_key] = bucket

                _write_catalog(catalog)

                logger.info(
                    "observed_peak writeback: %s (%s): %d bytes (was %d)",
                    model_id,
                    device_key,
                    observed_peak_bytes,
                    recorded,
                )
        except Exception as exc:  # noqa: BLE001
            # Any IO / OS / JSON failure is logged + swallowed. The
            # writeback is best-effort; the next cold load will retry.
            logger.warning(
                "observed_peak writeback failed for %s (%s): %s",
                model_id,
                device_key,
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internals: decision / load / commit
    # ------------------------------------------------------------------

    def _decide(
        self, model_id: str, *, manifest: dict, bump_refcount: bool = True,
    ) -> tuple:
        """First phase, under the lock.

        Returns one of these tuple shapes:
          ("hot", port): entry already loaded; refcount + last_touched
            bumped under the lock (when bump_refcount=True); port is the
            worker_port read under the same lock (no TOCTOU window for
            an evictor to race in).
          ("wait",): another thread owns the load; caller awaits the Event.
          ("load",): we just claimed ownership; run load phase + commit.
          ("evict_and_retry", shortfall_gb, device, required_gb): the
            model does not fit live free minus headroom. Caller runs
            eviction OUTSIDE the lock and re-calls _decide. No in-flight
            slot is claimed on this path so the eviction loop's
            lock-release-and-reacquire pattern is safe across the boundary.
            `required_gb` (the model's memory_gb) lets the eviction loop
            re-check live fit before 503'ing when no candidates remain.

        `bump_refcount` is True for acquire (refcount drives in-flight
        request accounting) and False for warmup (load idempotently
        without skewing LRU ordering or pinning the model).

        On the ("load",) return path, an Event has been stashed in
        self.in_flight_loads[model_id]. The caller is responsible for
        either committing (success) or popping + setting the Event
        (failure). _commit and _abort cover both cases.
        """
        with self.lock:
            if model_id in self._blocked_models:
                from muse.admin.operations import OperationError
                raise OperationError(
                    "model_disabled",
                    f"model {model_id!r} is disabled",
                    status=409,
                    retryable=False,
                )
            if model_id in self.evicting:
                from muse.admin.operations import OperationError
                raise OperationError(
                    "model_eviction_in_progress",
                    f"model {model_id!r} is being unloaded",
                    status=503,
                    retryable=True,
                )
            entry = self.loaded.get(model_id)
            if entry is not None:
                if bump_refcount:
                    entry.refcount += 1
                    entry.last_touched_at = time.monotonic()
                # Read port under the same lock that classified the
                # entry; an evictor cannot race between the classification
                # and this read.
                return ("hot", entry.worker_port)

            existing_claim = self.in_flight_loads.get(model_id)
            if existing_claim is not None:
                return ("wait", existing_claim)

            # We're going to do this load. Decide whether it fits before
            # we claim the in-flight slot, so that if the answer is "no"
            # we don't strand an Event.
            memory_gb = self._memory_requirement_gb(manifest, model_id)
            device = declared_device(manifest.get("capabilities"))
            pool = self._resolve_pool_device(device)

            # Defense in depth for non-gateway callers (notably admin
            # warmup) and models pulled after boot: never evict a working set
            # for a model that cannot fit even an empty physical/configured
            # pool. Older injected probes without total-capacity methods
            # leave this verdict unknown and retain historical behavior.
            hard_capacity_gb = self._hard_capacity_for_device(device)
            if hard_capacity_gb is not None and memory_gb > hard_capacity_gb:
                from muse.admin.operations import OperationError
                raise OperationError(
                    "model_too_large_for_device",
                    (
                        f"cannot fit {model_id!r} on {pool}: "
                        f"{memory_gb:.2f} GB required exceeds "
                        f"{hard_capacity_gb:.2f} GB capacity"
                    ),
                    status=503,
                    retryable=False,
                )

            _, available_gb = self._available_for_device(device)
            # Debit same-pool loads that are claimed but not yet resident.
            # The live probe won't reflect them until their worker allocates
            # VRAM (seconds later), so without this a concurrent _decide for a
            # different model would size against the same free reading and
            # over-admit -> OOM. See InFlightLoad.
            #
            # This is deliberately CONSERVATIVE: while an in-flight load's
            # worker is mid-allocation the live probe has ALREADY dropped by
            # what it consumed so far, yet we still subtract its full declared
            # memory_gb -- a transient double-count that can make a concurrent
            # load that genuinely fits take the evict_and_retry path (a
            # possible spurious evict or 503) until the first load commits and
            # frees its reservation. That is the SAFE direction (never
            # over-admits / OOMs). A precise fix needs per-load allocation
            # accounting against a shared global probe (which is itself
            # ambiguous with multiple concurrent loads); deferred as follow-up.
            effective_available_gb = available_gb - self._reserved_for_pool(pool)
            if memory_gb > effective_available_gb:
                shortfall_gb = memory_gb - effective_available_gb
                # Defer eviction to acquire(), which runs it outside the
                # lock so concurrent hot-acquires keep working during
                # the (potentially multi-second) disable_fn + poll
                # window. We do NOT claim an in_flight_loads slot here:
                # if eviction fails (503), there's nothing to clean up;
                # if eviction succeeds, the retry will re-decide and
                # claim the slot then.
                return ("evict_and_retry", shortfall_gb, device, memory_gb)

            # Claim ownership of the load, reserving its memory + pool so
            # concurrent decisions see it, and bump the epoch so overlapping
            # loads can detect each other for the observed-peak gate.
            claim = InFlightLoad(
                event=threading.Event(),
                memory_gb=memory_gb,
                pool=pool,
                owner_thread=threading.current_thread(),
            )
            self.in_flight_loads[model_id] = claim
            self._inflight_epoch += 1
            return ("load", claim)

    def _wait_for_in_flight(
        self, model_id: str, claim: InFlightLoad,
    ) -> None:
        """Wait once for an exact claim, then recover or fail explicitly.

        A timeout must not simply re-enter ``_decide`` against the unchanged
        record forever. An ownerless/dead-owner claim is removed with an
        identity compare-and-swap; a still-live owner produces a retryable
        timeout instead of spawning a duplicate worker behind its back.
        """
        if claim.event.wait(timeout=_INFLIGHT_WAIT_TIMEOUT_SECONDS):
            return

        try:
            owner_alive = (
                claim.owner_thread is not None
                and claim.owner_thread.is_alive() is True
            )
        except Exception:  # noqa: BLE001
            owner_alive = True

        expired = (
            time.monotonic() - claim.claimed_at
            >= _INFLIGHT_WAIT_TIMEOUT_SECONDS
        )
        reclaimed = False
        with self.lock:
            current = self.in_flight_loads.get(model_id)
            if current is not claim or claim.event.is_set():
                return
            if expired and not owner_alive:
                self.in_flight_loads.pop(model_id, None)
                reclaimed = True

        if reclaimed:
            claim.event.set()
            self._fire_capacity_listener()
            logger.warning("reclaimed abandoned in-flight load for %r", model_id)
            return

        from muse.admin.operations import OperationError
        raise OperationError(
            "model_load_timeout",
            f"timed out waiting for the active load of {model_id!r}",
            status=503,
            retryable=True,
        )

    def _get_in_flight_event(self, model_id: str) -> threading.Event | None:
        """Read the current in-flight Event under the lock.

        The Event may have been removed between _decide returning "wait"
        and this call (the winner committed and pop'd it). In that case
        we return None and the caller falls through to re-decide.
        """
        with self.lock:
            rec = self.in_flight_loads.get(model_id)
            return rec.event if rec is not None else None

    def _reserved_for_pool(self, pool: str, *, exclude: str | None = None) -> float:
        """Sum of memory_gb for claimed-but-not-yet-resident loads drawing on
        `pool`. Callers hold self.lock. `exclude` drops one model_id (its own
        slot) from the sum. Debited from available memory in the fit check so
        concurrent off-loop acquires cannot over-commit the device.
        """
        return sum(
            rec.memory_gb
            for mid, rec in self.in_flight_loads.items()
            if rec.pool == pool and mid != exclude
        )

    def _load_and_commit(
        self,
        model_id: str,
        *,
        manifest: dict,
        claim: InFlightLoad,
        initial_refcount: int = 1,
    ) -> int:
        """Run the load phase outside the lock, then commit.

        On exception, performs cleanup so concurrent waiters wake up,
        no stale LoadEntry exists, and the exception propagates.

        `initial_refcount` is 1 for acquire (a request is being served)
        and 0 for warmup (no request, just pre-load).

        After a successful commit, schedules the observed-peak writeback
        (Task D): a daemon thread compares the observed `free_before -
        free_after` delta against the catalog's recorded peak and raises
        the recorded value if observation exceeds it. The thread starts
        OFF the request hot path so the caller is unaffected by catalog
        IO latency or filesystem failures.
        """
        device = declared_device(manifest.get("capabilities"))

        try:
            with self.lock:
                current_claim = self.in_flight_loads.get(model_id)
                if current_claim is not claim:
                    from muse.admin.operations import OperationError
                    raise OperationError(
                        "model_load_claim_lost",
                        f"load ownership for {model_id!r} expired before startup",
                        status=503,
                        retryable=True,
                    )
                memory_gb = claim.memory_gb
                free_before_gb = self._free_for_device(device)
                # Snapshot in-flight state to detect whether ANOTHER load
                # overlaps our observation window. The epoch also catches a
                # load claimed and released entirely within that window.
                solo_at_start = len(self.in_flight_loads) == 1
                epoch_before = self._inflight_epoch
        except BaseException as exc:
            self._abort(model_id, expected=claim)
            from muse.admin.operations import OperationError
            if isinstance(exc, Exception) and not isinstance(exc, OperationError):
                raise OperationError(
                    "model_load_failed",
                    f"worker for {model_id!r} failed to load: {exc}",
                    status=503,
                ) from exc
            raise
        load_start = time.monotonic()
        enabled = False
        try:
            worker_port = self.enable_fn(model_id)
            enabled = True
            if (
                type(worker_port) is not int
                or worker_port < 1
                or worker_port > 65535
            ):
                raise RuntimeError(
                    f"enable callback returned invalid worker port {worker_port!r}"
                )

            # This probe is passive observation, not a prerequisite for using
            # a worker that enable_fn already proved ready. A transient NVML /
            # psutil failure must not turn success into a leaked failed load.
            try:
                free_after_gb: float | None = self._free_for_device(device)
            except Exception as exc:  # noqa: BLE001
                free_after_gb = None
                logger.warning(
                    "post-load memory observation failed for %r: %s",
                    model_id,
                    exc,
                )

            with self.lock:
                if self.in_flight_loads.get(model_id) is not claim:
                    from muse.admin.operations import OperationError
                    raise OperationError(
                        "model_load_claim_lost",
                        f"load ownership for {model_id!r} expired during startup",
                        status=503,
                        retryable=True,
                    )

                now = time.monotonic()
                entry = LoadEntry(
                    model_id=model_id,
                    worker_port=worker_port,
                    memory_gb=memory_gb,
                    refcount=initial_refcount,
                    last_touched_at=now,
                    loaded_at=now,
                    pool=claim.pool,
                )
                self.loaded[model_id] = entry

                self.recent_decisions.append(DecisionLogEntry(
                    timestamp=time.time(),
                    model_id=model_id,
                    action="load",
                    memory_gb=memory_gb,
                    free_before_gb=free_before_gb,
                    free_after_gb=free_after_gb,
                    reason="fit",
                    evicted=[],
                ))

                try:
                    record(
                        "model_load",
                        model_id=model_id,
                        pool=claim.pool,
                        gb=memory_gb,
                        cold_load_seconds=(time.monotonic() - load_start),
                    )
                except Exception:  # noqa: BLE001
                    pass

                solo_at_end = len(self.in_flight_loads) == 1
                epoch_after = self._inflight_epoch
                rec = self.in_flight_loads.pop(model_id, None)
        except BaseException as e:
            # If startup returned successfully but validation/commit failed,
            # keep the exact claim reserved while disabling the worker. This
            # prevents a waiter from enabling a replacement during rollback.
            rollback_error: BaseException | None = None
            if enabled:
                try:
                    self.disable_fn(model_id)
                except BaseException as cleanup_error:  # noqa: BLE001
                    rollback_error = cleanup_error
                    logger.error(
                        "failed to roll back enabled worker for %r",
                        model_id,
                        exc_info=True,
                    )
            with self.lock:
                loaded = self.loaded.get(model_id)
                if rollback_error is None:
                    if loaded is not None and loaded.worker_port == locals().get(
                        "worker_port"
                    ):
                        self.loaded.pop(model_id, None)
                else:
                    # Teardown did not prove the worker is gone. Keep its
                    # declared memory resident and block routing until an
                    # explicit disable retries cleanup. Port 0 is a deliberate
                    # non-routable placeholder when enable_fn returned an
                    # invalid value before commit.
                    rollback_port = locals().get("worker_port")
                    if (
                        type(rollback_port) is not int
                        or rollback_port < 1
                        or rollback_port > 65535
                    ):
                        rollback_port = 0
                    now = time.monotonic()
                    if loaded is None:
                        loaded = LoadEntry(
                            model_id=model_id,
                            worker_port=rollback_port,
                            memory_gb=memory_gb,
                            refcount=0,
                            last_touched_at=now,
                            loaded_at=now,
                            pool=claim.pool,
                        )
                        self.loaded[model_id] = loaded
                    else:
                        loaded.refcount = 0
                    self._blocked_models.add(model_id)
                    self._failed_rollbacks.add(model_id)
            # Cleanup: pop the Event, set it so waiters wake (they will
            # re-decide against either no entry or the fail-closed block),
            # then surface.
            self._abort(model_id, expected=claim)
            if rollback_error is not None and not isinstance(
                rollback_error, Exception
            ):
                raise rollback_error
            # Translate unexpected load failures (worker spawn OOM, venv
            # missing, health-poll timeout) into OperationError(503) so
            # the gateway and admin callers return an OpenAI-shaped
            # envelope instead of letting a raw exception escape as a
            # bare 500. OperationError passes through unwrapped (it
            # already carries status/code/message); BaseException
            # non-Exceptions (KeyboardInterrupt, SystemExit) propagate
            # untouched.
            from muse.admin.operations import OperationError
            if isinstance(e, Exception) and not isinstance(e, OperationError):
                raise OperationError(
                    "model_load_failed",
                    f"worker for {model_id!r} failed to load: {e}",
                    status=503,
                ) from e
            raise

        # Our load is "solo" (no overlapping load polluting the free-memory
        # delta) iff we were the only in-flight load at BOTH endpoints and no
        # other load was claimed during our window.
        solo_load = solo_at_start and solo_at_end and (epoch_after == epoch_before)

        # rec.event.set() outside the lock is fine; threading.Event is its
        # own synchronization. Doing it inside the lock would block any
        # awakened waiter on its first re-entry into _decide.
        if rec is not None:
            rec.event.set()

        # v0.57.1: a committed load resolves its reservation into real
        # residency (evictable once its refcount drops). Wake parked
        # capacity-waiters to re-decide; a futile wake costs one cheap
        # _decide pass.
        self._fire_capacity_listener()

        # Task D: schedule the observed-peak writeback ONLY for a solo load.
        # With overlapping loads the free_before..free_after delta reflects
        # BOTH models, and the writeback (monotonic-upward only) would
        # permanently inflate this model's recorded peak. Skipping just loses
        # one self-heal opportunity, exactly when it cannot be measured
        # cleanly. observed_peak swallows non-positive values internally.
        if solo_load and free_after_gb is not None:
            observed_bytes = int((free_before_gb - free_after_gb) * 1024**3)
            try:
                self.observed_peak(
                    model_id,
                    observed_peak_bytes=observed_bytes,
                    device=device,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "could not schedule observed-peak writeback for %r",
                    model_id,
                    exc_info=True,
                )

        return worker_port

    def _abort(
        self, model_id: str, *, expected: InFlightLoad | None = None,
    ) -> None:
        """Clean up after a failed load: drop + signal the Event.

        Called from the exception handler in _load_and_commit. After
        this, the model_id is not in self.in_flight_loads and any
        waiting threads have been woken up to re-enter the decision
        phase. No LoadEntry exists, so on re-decide they will become
        the new singleton winner.
        """
        with self.lock:
            current = self.in_flight_loads.get(model_id)
            if expected is not None and current is not expected:
                rec = None
            else:
                rec = self.in_flight_loads.pop(model_id, None)
        if rec is not None:
            rec.event.set()
        # v0.57.1: an aborted load frees its reservation -- capacity may
        # now fit a parked waiter; wake them to re-decide.
        self._fire_capacity_listener()

    # ------------------------------------------------------------------
    # Eviction (Task C)
    # ------------------------------------------------------------------

    def _evict_lru_until_fits(
        self,
        *,
        model_id: str,
        shortfall_gb: float,
        device: str,
        required_gb: float = 0.0,
    ) -> None:
        """Serialize pressure eviction within the target memory pool."""
        target_pool = self._resolve_pool_device(device)
        with self._eviction_pool_locks[target_pool]:
            self._evict_lru_until_fits_serialized(
                model_id=model_id,
                shortfall_gb=shortfall_gb,
                device=device,
                required_gb=required_gb,
            )

    def _evict_lru_until_fits_serialized(
        self,
        *,
        model_id: str,
        shortfall_gb: float,
        device: str,
        required_gb: float = 0.0,
    ) -> None:
        """Evict refcount==0 models in LRU order until shortfall is met.

        `required_gb` is the incoming model's memory_gb. When no evictable
        candidates remain, we re-check live availability against it before
        503'ing: concurrent evictions by OTHER acquires (this path claims
        no in_flight slot, so two acquires for the same evict-needing model
        both land here) may have freed enough since our shortfall was
        computed. If the model now fits, return and let acquire() re-decide
        (-> load) instead of raising a spurious 503 that the gateway would
        surface verbatim with no retry.

        Lock discipline (the critical detail of Task C):
          1. Acquire the lock briefly to snapshot evictable candidates
             and pop the LRU victim from self.loaded. Append a partial
             DecisionLogEntry under the lock so the admin endpoint sees
             the in-progress eviction as soon as it commits to one.
          2. RELEASE the lock for the slow steps (disable_fn invocation
             + memory release polling). Other threads can hot-acquire
             during this window: their _decide call takes the lock,
             observes the (still-loaded) other entries, and proceeds.
             Cold acquires that race here will see a smaller free pool
             until our poll completes, but that's accurate.
          3. REACQUIRE the lock to update the DecisionLogEntry's
             free_after_gb once the poll returns.
          4. If the cumulative freed memory still falls short of the
             original shortfall, loop and pick the next victim. If no
             candidates remain: lazy-import OperationError and raise 503.

        The reacquire step is intentionally minimal (just the entry
        mutation), so other admin endpoints don't stall behind the
        post-poll commit.
        """
        cumulative_freed_gb = 0.0
        # Track victims whose disable_fn raised so we don't re-pick them
        # on the next iteration. A re-inserted victim retains its
        # last_touched_at and would otherwise sort back to LRU position
        # and loop forever. The set is local to this call: a future
        # acquire is welcome to retry the failed victim, since the
        # worker may have died on its own by then.
        failed_victims: set[str] = set()
        target_pool = self._resolve_pool_device(device)

        while cumulative_freed_gb < shortfall_gb:
            # Capacity may have changed since _decide computed the original
            # shortfall (another eviction completed, an external allocation
            # ended, or a previous victim in this loop freed a budget slot).
            # Re-check before selecting another victim so we never evict a
            # healthy idle worker after the incoming model already fits.
            with self.lock:
                _, current_available_gb = self._available_for_device(device)
                current_reserved_gb = self._reserved_for_pool(
                    target_pool, exclude=model_id,
                )
                fits_before_victim = (
                    current_available_gb - current_reserved_gb >= required_gb
                )
            if fits_before_victim:
                self._fire_capacity_listener()
                return

            # fits_now is set inside the lock below only on the "the model
            # fits again without evicting anything further" path; checked
            # right after the lock releases so the capacity_listener fire
            # (which must never run under self.lock) happens outside it.
            fits_now = False
            capacity_failure: BaseException | None = None
            # ---- under-lock: pick + pop a victim ----
            with self.lock:
                # Re-snapshot every iteration: another thread may have
                # released a model since the last pass, making it newly
                # evictable.
                candidates = [
                    e for e in self.loaded.values()
                    if (
                        e.pool == target_pool
                        and e.refcount == 0
                        and e.model_id not in failed_victims
                    )
                ]
                if not candidates:
                    # Re-check live fit before giving up: a concurrent
                    # acquire's eviction (or external memory release) may
                    # have freed enough since our shortfall was computed.
                    # If the model now fits, return so acquire() re-decides
                    # (-> load) instead of 503'ing spuriously. Only raise
                    # when it genuinely still cannot fit an evicted-empty
                    # device (else acquire would spin re-deciding forever).
                    # Net out other same-pool in-flight reservations: without
                    # this, a concurrent pending load (which the live probe
                    # doesn't see yet) would let this path conclude "fits",
                    # acquire re-decides -> evict_and_retry -> no candidates ->
                    # "fits" ... an infinite spin. With it, the model that
                    # can't fit gets the clean 503.
                    _, available_gb = self._available_for_device(device)
                    reserved_gb = self._reserved_for_pool(
                        target_pool, exclude=model_id,
                    )
                    if (available_gb - reserved_gb) >= required_gb:
                        fits_now = True
                    else:
                        # Spec 2026-07-08: if a same-pool loaded model is
                        # currently in-use (refcount > 0), capacity may free
                        # once its request finishes -- worth the gateway
                        # parking and retrying instead of surfacing a hard
                        # 503.
                        # v0.57.1 (stress-test finding): a capacity 503
                        # is ALSO transient when OTHER cold loads are in
                        # flight -- their reservations made the fit fail,
                        # and each will either commit (reservation becomes
                        # evictable residency) or abort (reservation
                        # freed). Without this, a cold-start burst of
                        # distinct models on an empty card mass-503s
                        # instantly instead of queueing behind the first
                        # load.
                        # v0.57.2: an eviction mid-flight (self.evicting)
                        # is the third transient state -- its VRAM frees
                        # when disable_fn completes, so the failure is
                        # worth parking on too.
                        evicting_in_pool = any(
                            self._evicting_pools.get(mid, target_pool)
                            == target_pool
                            for mid in self.evicting
                        )
                        any_inuse = (
                            any(
                                e.pool == target_pool and e.refcount > 0
                                for e in self.loaded.values()
                            )
                            or any(
                                k != model_id and rec.pool == target_pool
                                for k, rec in self.in_flight_loads.items()
                            )
                            or evicting_in_pool
                        )
                        # Lazy import to avoid a cycle: load_director ->
                        # admin.operations -> supervisor -> load_director
                        # (Task E will wire the supervisor to LoadDirector).
                        from muse.admin.operations import OperationError
                        capacity_failure = OperationError(
                            "model_too_large_for_device",
                            (
                                f"cannot fit {model_id!r} on {device}: "
                                f"shortfall {shortfall_gb:.2f} GB; "
                                f"no evictable {target_pool} candidates remain"
                            ),
                            status=503,
                            retryable=any_inuse,
                        )
                else:
                    # LRU first.
                    candidates.sort(key=lambda e: e.last_touched_at)
                    victim = candidates[0]
                    victim_id = victim.model_id
                    victim_memory_gb = victim.memory_gb

                    # Capture before removing the accounting entry. Live
                    # probes are unchanged by this ordering, while a static
                    # budget correctly observes the victim's declared memory
                    # becoming available after the pop.
                    free_before_gb = self._free_for_device(device)

                    # Pop from loaded so other threads see the eviction
                    # commitment immediately. If disable_fn raises, the
                    # state is already consistent (no zombie LoadEntry
                    # claiming a worker port that isn't running).
                    self.loaded.pop(victim_id, None)
                    # Mark the eviction in-flight so concurrent decides that
                    # fail their fit check during the out-of-lock disable_fn
                    # window raise retryable=True (capacity WILL free when
                    # this eviction completes) instead of mass-503ing while
                    # the director looks idle (v0.57.2 stress finding).
                    self.evicting.add(victim_id)
                    self._evicting_pools[victim_id] = target_pool
                    self._evicting_memory_gb[victim_id] = victim_memory_gb
                    # Bump the epoch: an eviction frees VRAM, which pollutes any
                    # concurrent load's free_before..free_after delta just like a
                    # concurrent load does. Counting it here lets _load_and_commit's
                    # solo gate skip that load's observed-peak writeback instead of
                    # recording an under-estimate.
                    self._inflight_epoch += 1

                    decision = DecisionLogEntry(
                        timestamp=time.time(),
                        model_id=victim_id,
                        action="evict",
                        memory_gb=victim_memory_gb,
                        free_before_gb=free_before_gb,
                        free_after_gb=None,
                        reason=f"evicted_for_{model_id}",
                        evicted=[victim_id],
                    )
                    self.recent_decisions.append(decision)

                    # Task 9: fire-and-forget telemetry event for this
                    # eviction. Guarded the same way as the load-site record()
                    # call above: telemetry can never break a real eviction.
                    try:
                        record(
                            "model_evict",
                            model_id=victim_id,
                            pool=self._resolve_pool_device(device),
                            reason=f"evicted_for_{model_id}",
                        )
                    except Exception:  # noqa: BLE001
                        pass

            # ---- lock released ----
            if capacity_failure is not None:
                # Earlier victims in THIS call may have freed memory; notify
                # other parked requests before this request surfaces its own
                # capacity error. Foreign callbacks must never run under the
                # director lock.
                if cumulative_freed_gb > 0.0:
                    self._fire_capacity_listener()
                raise capacity_failure
            if fits_now:
                self._fire_capacity_listener()
                return

            # ---- slow steps ----
            # v0.57.3 (review finding): the evicting marker must clear on
            # EVERY exit of this out-of-lock region -- including a
            # poll/probe exception below, which would otherwise leak the
            # marker and make every future capacity 503 spuriously
            # retryable forever. Hence the try/finally, not inline
            # discards on the known-good paths only.
            disable_failed = False
            try:
                try:
                    self.disable_fn(victim_id)
                except Exception as exc:  # noqa: BLE001
                    # disable_fn raised. The worker is still alive holding
                    # memory: the OS has NOT reclaimed its slot. Polling for
                    # release would observe ~0 freed and the loop would
                    # iterate to the next victim, but the orphan worker
                    # would persist indefinitely with no remediation path.
                    #
                    # Remediation: re-insert the victim's LoadEntry into
                    # self.loaded so accounting matches reality (worker
                    # alive, slot occupied). Append a structured "evict"
                    # decision log entry recording the failure for the
                    # admin endpoint. Skip cumulative_freed_gb (the slot
                    # did not free). Continue to the next candidate.
                    #
                    # KeyboardInterrupt + SystemExit pass through (we catch
                    # Exception, not BaseException) so the user can still
                    # Ctrl-C out of a stuck eviction.
                    logger.error(
                        "disable_fn failed for evicted victim %r; re-inserted into loaded set",
                        victim_id,
                        exc_info=True,
                    )
                    with self.lock:
                        # Re-insert the popped LoadEntry. refcount was 0 at
                        # eviction time (only refcount==0 entries are
                        # candidates) and we preserve that.
                        self.loaded[victim_id] = victim
                        # Update the existing decision entry's reason
                        # in-place. The deque already holds it; mutating
                        # the dataclass is sufficient. evicted=[] because
                        # nothing was actually evicted.
                        decision.reason = f"disable_fn_raised: {exc}"
                        decision.evicted = []
                        # free_after_gb stays None: no poll, no measurement.
                    # Mark this victim as off-limits for the rest of this
                    # eviction call. Re-picking would loop forever (the
                    # re-inserted victim still has the oldest
                    # last_touched_at and would sort to LRU position).
                    failed_victims.add(victim_id)
                    disable_failed = True
                else:
                    # The shutdown owner has settled, so configured-budget
                    # bookkeeping may now release the victim. Keep the
                    # `evicting` marker itself until the whole iteration
                    # settles so concurrent failures remain retryable while
                    # a live driver counter catches up.
                    with self.lock:
                        self._evicting_memory_gb.pop(victim_id, None)
                    # disable_fn returns only after the worker shutdown has
                    # settled. If the model already fits (notably because a
                    # configured working-set budget was the binding limit),
                    # do not spend two seconds waiting for a noisy physical
                    # free-memory counter to show the declared-size delta.
                    with self.lock:
                        _, available_after_shutdown = self._available_for_device(
                            device,
                        )
                        reserved_after_shutdown = self._reserved_for_pool(
                            target_pool, exclude=model_id,
                        )
                        fits_after_shutdown = (
                            available_after_shutdown - reserved_after_shutdown
                            >= required_gb
                        )
                    if fits_after_shutdown:
                        freed_this_round = shortfall_gb - cumulative_freed_gb
                    else:
                        freed_this_round = self._wait_for_memory_release(
                            min_freed_gb=victim_memory_gb,
                            free_at_eviction_start_gb=free_before_gb,
                            device=device,
                        )

                    # ---- reacquire the lock briefly: writeback free_after ----
                    with self.lock:
                        free_after_gb = self._free_for_device(device)
                        decision.free_after_gb = free_after_gb

                    cumulative_freed_gb += freed_this_round
            finally:
                # Eviction settled (worker down + release observed, or
                # disable failed and the victim was re-inserted, or an
                # unexpected exception is propagating): the in-flight
                # marker must not outlive this iteration.
                with self.lock:
                    self.evicting.discard(victim_id)
                    self._evicting_pools.pop(victim_id, None)
                    self._evicting_memory_gb.pop(victim_id, None)
            if disable_failed:
                continue

        # Loop exited normally: cumulative_freed_gb >= shortfall_gb, i.e.
        # eviction freed enough. Outside the lock (the last statement above
        # released it), so this fire is safe.
        self._fire_capacity_listener()
        return

    def _wait_for_memory_release(
        self,
        *,
        min_freed_gb: float,
        free_at_eviction_start_gb: float,
        device: str,
        timeout: float = 2.0,
        poll: float = 0.05,
    ) -> float:
        """Poll memory_probe until enough memory has freed (or timeout).

        Returns the actual freed_gb observed at exit. The caller compares
        this against the expected per-victim freeing to decide whether
        to evict another victim.

        The director lock is NOT held while this runs. Other threads can
        progress through their own decision phases during the poll.

        Negative `freed_gb` is clamped to 0.0: if memory pressure grew
        during the poll (another process allocated VRAM, or our own
        eviction observation lagged), we don't double-count it as a
        freeing.
        """
        deadline = time.monotonic() + timeout
        while True:
            current_free_gb = self._free_for_device(device)
            freed = current_free_gb - free_at_eviction_start_gb
            if freed < 0.0:
                freed = 0.0

            if freed >= min_freed_gb:
                return freed
            if time.monotonic() >= deadline:
                return freed

            time.sleep(poll)

    # ------------------------------------------------------------------
    # Memory accounting helpers
    # ------------------------------------------------------------------

    def _auto_resolves_to_cuda(self) -> bool:
        """Whether a `device: "auto"` model binds to the GPU/VRAM pool here.

        Mirrors the runtime's select_device("auto"): a CUDA-compatible GPU
        is visible when either NVML reports free VRAM or torch reports CUDA
        availability. The latter includes ROCm, whose torch API is CUDA-
        shaped even though NVML is unavailable.
        """
        gpu_free = self.memory_probe.gpu_free_gb()
        if gpu_free is not None:
            return True
        return resolve_memory_pool(
            "auto",
            gpu_free_gb=gpu_free,
            cuda_available=self._cuda_runtime_available(),
        ) == "cuda"

    def _cuda_runtime_available(self) -> bool:
        """Call the injected runtime detector without trusting loose mocks."""
        if self.cuda_available_fn is None:
            return False
        try:
            return bool(self.cuda_available_fn())
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA runtime availability check failed: %s", exc)
            return False

    def _resolve_pool_device(self, device: str) -> str:
        """Map a declared device to the concrete memory POOL it draws from.

        Returns "cuda" or "cpu" -- the two pools muse sizes admission and
        eviction against:

          - "gpu"        -> "cuda" (legacy alias)
          - "auto" / ""  -> "cuda" if a GPU is visible, else "cpu". This is
                            THE fix for the v0.48.0 control-plane bug: an
                            auto-device model on a GPU host loads on the GPU,
                            so it must be sized against the VRAM pool, not
                            host RAM.
          - "cuda"       -> "cuda"
          - "mps" / etc. -> "cpu" (host-RAM accounting; the GPU pool is
                            pynvml/CUDA-only, and MPS is unified memory)

        Note this is the POOL, not the measurements bucket key: "mps" pools
        against host RAM here but keeps its own "mps" bucket in
        _record_observed_peak (matching what `muse models probe` writes).
        """
        normalized = str(device or "auto").lower()
        if normalized not in ("auto", ""):
            return resolve_memory_pool(
                normalized, gpu_free_gb=None, cuda_available=False,
            )
        return "cuda" if self._auto_resolves_to_cuda() else "cpu"

    def _resident_for_pool(self, pool: str) -> float:
        """Accounted resident or shutting-down memory for a budget pool."""
        with self.lock:
            resident_gb = sum(
                entry.memory_gb
                for entry in self.loaded.values()
                if entry.pool == pool
            )
            evicting_gb = sum(
                self._evicting_memory_gb.get(model_id, 0.0)
                for model_id, evicting_pool in self._evicting_pools.items()
                if evicting_pool == pool
            )
            return resident_gb + evicting_gb

    def _optional_total_gb(self, pool: str) -> float | None:
        """Read an optional physical-capacity method from the probe.

        Descriptor inspection avoids treating ``MagicMock``'s fabricated
        child attributes as real probe APIs. Invalid optional values mean
        "unknown" rather than becoming an accidental zero-capacity stamp.
        """
        method_name = "gpu_total_gb" if pool == "cuda" else "cpu_total_gb"
        descriptor = getattr(type(self.memory_probe), method_name, None)
        if not callable(descriptor):
            return None
        method = getattr(self.memory_probe, method_name, None)
        if not callable(method):
            return None
        try:
            value = method()
            if value is None or isinstance(value, bool):
                return None
            result = float(value)
            if not math.isfinite(result) or result < 0.0:
                return None
            return result
        except Exception as exc:  # noqa: BLE001
            logger.debug("optional %s probe failed: %s", method_name, exc)
            return None

    def _hard_capacity_for_device(self, device: str) -> float | None:
        """Empty-working-set capacity after configured headroom.

        Unlike admission, this uses physical total capacity (when the probe
        exposes it) and never current free memory. A configured budget caps
        that ceiling and also serves as the static fallback.
        """
        pool = self._resolve_pool_device(device)
        total_gb = self._optional_total_gb(pool)
        if pool == "cuda":
            budget_gb = self.gpu_budget_gb
            headroom_gb = self.gpu_headroom_gb
        else:
            budget_gb = self.cpu_budget_gb
            headroom_gb = self.cpu_headroom_gb
        if total_gb is None and budget_gb is None:
            return None
        ceiling_gb = (
            float(budget_gb)
            if total_gb is None
            else total_gb
            if budget_gb is None
            else min(total_gb, float(budget_gb))
        )
        return max(0.0, ceiling_gb - float(headroom_gb))

    def _free_for_device(self, device: str) -> float:
        """Live free memory in GB for the relevant device.

        For CPU, use live host RAM. For CUDA, prefer live NVML free VRAM.
        If NVML is unavailable and a GPU budget is configured, treat that
        budget as total capacity and debit director-owned resident models.
        Without either source, return zero so unknown explicit GPU loads do
        not fit accidentally.
        """
        pool = self._resolve_pool_device(device)
        if pool == "cuda":
            free = self.memory_probe.gpu_free_gb()
            if free is not None:
                return float(free)
            if self.gpu_budget_gb is None:
                return 0.0
            return max(
                0.0,
                float(self.gpu_budget_gb) - self._resident_for_pool(pool),
            )
        return float(self.memory_probe.cpu_free_gb())

    def _available_for_device(self, device: str) -> tuple[float, float]:
        """Return (free_before_gb, available_gb).

        available_gb = min(live_free, remaining_budget) - headroom. A
        configured budget is a total Muse working-set cap, so same-pool
        residents are debited before admission; in-flight reservations are
        debited by the decision caller. Negative values are clamped to 0.
        """
        free = self._free_for_device(device)
        pool = self._resolve_pool_device(device)
        if pool == "cuda":
            budget = self.gpu_budget_gb
            headroom = self.gpu_headroom_gb
        else:
            budget = self.cpu_budget_gb
            headroom = self.cpu_headroom_gb

        remaining_budget = None
        if budget is not None:
            remaining_budget = max(
                0.0, float(budget) - self._resident_for_pool(pool),
            )

        available = available_capacity_gb(
            live_free_gb=free,
            budget_gb=remaining_budget,
            headroom_gb=headroom,
        )
        # `_free_for_device` always returns a numeric conservative fallback,
        # so the shared helper cannot return None here.
        return free, float(available or 0.0)
