"""Live memory probing for the lazy-load director (v0.40.0).

Wraps ``pynvml`` (per-device NVIDIA VRAM) and ``psutil`` (system RAM)
behind import-light functions:

- ``gpu_free_gb(device_id)``: VRAM free in gibibytes, or ``None`` when
  pynvml is unavailable, the host has no NVIDIA driver, or the per-device
  query fails (rare; e.g. a TCC device toggled off mid-process).
- ``gpu_total_gb(device_id)``: physical VRAM capacity in gibibytes, with
  the same soft-failure contract.  Supervisor servability checks use total
  capacity; free memory is transient and belongs in admission decisions.
- ``cpu_free_gb()``: RAM available in gibibytes, via
  ``psutil.virtual_memory().available``.
- ``cpu_total_gb()``: physical host RAM in gibibytes.
- ``init_pynvml()``: idempotent. Imports pynvml once, calls ``nvmlInit``
  once, and caches the success / failure verdict for the rest of the
  process.
- ``cuda_runtime_available()``: lazy torch CUDA/ROCm detection.
- ``resolve_memory_pool()`` and ``available_capacity_gb()``: shared
  control-plane policy for device pooling and live-or-budget capacity.

``pynvml`` is a soft dep: muse runs on AMD GPU hosts, Apple Silicon, and
CPU-only CI without it. The director tolerates ``None`` from
``gpu_free_gb`` by either falling back to a static budget (when one is
declared) or refusing the GPU load with a 503.

Deferred-imports pattern: a module-top sentinel ``pynvml: Any = None``
gets populated by ``init_pynvml`` on first use. Tests patch the sentinel
directly (see ``tests/core/test_memory_probe.py``).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Literal

logger = logging.getLogger(__name__)


# Deferred-import sentinels. The first call to ``init_pynvml`` populates
# ``pynvml`` (or leaves it None on failure); ``_init_attempted`` and
# ``_init_ok`` make the call idempotent (sticky verdict, no retry).
pynvml: Any = None
_init_attempted: bool = False
_init_ok: bool = False
_init_lock = threading.Lock()

# 1 GiB in bytes. Both pynvml's ``nvmlDeviceGetMemoryInfo().free`` and
# psutil's ``virtual_memory().available`` are in bytes; the wrapper
# normalizes to gibibytes.
_BYTES_PER_GB: int = 1024 ** 3
_TORCH_UNSET: Any = object()


def init_pynvml() -> bool:
    """Idempotent pynvml init.

    Returns ``True`` once the module has been imported and ``nvmlInit``
    has succeeded; returns ``False`` on any failure (module missing,
    driver missing, AMD card, CPU-only host). The verdict is sticky for
    the remainder of the process: a failure is not retried.

    The pre-commit hook policy bans em-dashes; we use parens above for
    aside clauses.
    """
    global pynvml, _init_attempted, _init_ok
    with _init_lock:
        if _init_attempted:
            return _init_ok
        try:
            import pynvml as _p
            _p.nvmlInit()
        except Exception as e:  # noqa: BLE001
            # ImportError, RuntimeError (NVMLError), OSError on missing libs
            # all collapse to a single "no GPU info" verdict. We log at
            # DEBUG so a CPU-only or AMD host does not get noisy warnings.
            logger.debug("pynvml init failed: %s", e)
            pynvml = None
            _init_ok = False
        else:
            pynvml = _p
            _init_ok = True
        # Publish the sticky attempted flag last. A concurrent caller cannot
        # observe an in-progress initialization as a completed false verdict.
        _init_attempted = True
        return _init_ok


def gpu_free_gb(device_id: int = 0) -> float | None:
    """Live VRAM free on the given NVIDIA GPU, in gibibytes.

    Returns ``None`` when:
    - pynvml is not installed,
    - ``nvmlInit`` failed (no driver, AMD GPU, CPU-only host), or
    - the per-device query raises (rare).

    Triggers a lazy ``init_pynvml`` on first call so callers do not need
    to wire init explicitly.
    """
    if not init_pynvml():
        return None
    # init_pynvml() returning True implies the sentinel is populated. Guard
    # with an explicit None-return (not assert: asserts are stripped under
    # `python -O`, and None is this function's established failure idiom).
    if pynvml is None:  # pragma: no cover - invariant backstop
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    except Exception as e:  # noqa: BLE001
        logger.debug("pynvml query for device %s failed: %s", device_id, e)
        return None
    return float(mem.free) / _BYTES_PER_GB


def gpu_total_gb(device_id: int = 0) -> float | None:
    """Physical VRAM capacity for an NVIDIA GPU, in gibibytes.

    This deliberately does not derive a ceiling from current free VRAM.
    Free memory can increase when Muse evicts one of its workers (or when an
    unrelated process exits), so treating it as a permanent model-fit limit
    creates false ``model_unservable`` stamps.  The soft-failure behavior
    mirrors :func:`gpu_free_gb`.
    """
    if not init_pynvml():
        return None
    if pynvml is None:  # pragma: no cover - invariant backstop
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    except Exception as e:  # noqa: BLE001
        logger.debug("pynvml total-memory query for device %s failed: %s", device_id, e)
        return None
    return float(mem.total) / _BYTES_PER_GB


def cpu_free_gb() -> float:
    """Live RAM available on the host, in gibibytes.

    Backed by ``psutil.virtual_memory().available``: the kernel's view of
    how much memory can be allocated without swapping (Linux: MemAvailable
    from /proc/meminfo; macOS / Windows: the equivalent platform-specific
    counter). psutil is a hard ``museq[server]`` dep so this never fails.
    """
    import psutil
    return float(psutil.virtual_memory().available) / _BYTES_PER_GB


def cpu_total_gb() -> float:
    """Physical host RAM, in gibibytes.

    Servability is a permanent-fit question, so it uses this ceiling while
    the load director continues to use :func:`cpu_free_gb` for live
    admission and eviction.
    """
    import psutil
    return float(psutil.virtual_memory().total) / _BYTES_PER_GB


def cuda_runtime_available(torch_module: Any = _TORCH_UNSET) -> bool:
    """Whether torch can use a CUDA-compatible runtime on this host.

    PyTorch exposes both NVIDIA CUDA and AMD ROCm through ``torch.cuda``.
    NVML alone therefore cannot answer whether an ``auto`` runtime will
    select the CUDA pool.  The optional module argument keeps this helper
    import-light for modality runtimes and fully mockable in tests.  When it
    is omitted, torch is imported lazily; a missing or broken installation
    is treated as unavailable.
    """
    if torch_module is _TORCH_UNSET:
        try:
            import torch as torch_module
        except Exception:  # noqa: BLE001
            return False
    if torch_module is None:
        return False
    try:
        cuda = getattr(torch_module, "cuda", None)
        checker = getattr(cuda, "is_available", None)
        return bool(checker()) if callable(checker) else False
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch CUDA availability check failed: %s", exc)
        return False


def resolve_memory_pool(
    device: str,
    *,
    gpu_free_gb: float | None,
    cuda_available: bool = False,
) -> Literal["cuda", "cpu"]:
    """Resolve a declared device to the memory pool it consumes.

    Explicit CUDA always uses the CUDA pool. ``auto`` uses that pool when
    either a live NVML reading exists or torch reports a CUDA-compatible
    runtime (including ROCm). MPS and unknown devices use host-RAM
    accounting because Muse has no separate unified-memory probe.
    """
    normalized = str(device or "auto").lower()
    if normalized in ("cuda", "gpu") or (
        normalized.startswith("cuda:")
        and normalized.removeprefix("cuda:").isdigit()
    ):
        return "cuda"
    if normalized in ("auto", ""):
        return "cuda" if gpu_free_gb is not None or cuda_available else "cpu"
    return "cpu"


def available_capacity_gb(
    *,
    live_free_gb: float | None,
    budget_gb: float | None,
    headroom_gb: float,
) -> float | None:
    """Usable capacity after budget and headroom policy.

    A live reading is authoritative and is capped by a configured budget.
    When live measurement is unavailable, the configured budget becomes a
    static fallback instead of being combined with a synthetic zero. If
    neither source exists, capacity remains unknown (``None``).
    """
    if live_free_gb is None:
        if budget_gb is None:
            return None
        usable_gb = float(budget_gb)
    else:
        usable_gb = float(live_free_gb)
        if budget_gb is not None:
            usable_gb = min(usable_gb, float(budget_gb))
    return max(0.0, usable_gb - float(headroom_gb))


def declared_device(capabilities: dict | None) -> str:
    """The device a manifest's capabilities declare, defaulting to "auto".

    Shared by every control-plane site that sizes or pools a model
    (LoadDirector admission and commit, IdleSweeper eviction, supervisor
    servability). Absent or empty means the model follows the worker's
    own default (`muse _worker --device auto`), which binds to the GPU
    when one exists, so the control plane must size it against the pool
    "auto" resolves to on this host, NOT against host RAM.

    Regression note (v0.50.1): these sites each defaulted the absent key
    to "cpu". Resolver-pulled manifests never carry a `device` key, so
    the director sized their GPU loads against host RAM, admission always
    "fit", eviction never ran, and workers OOM'd at spawn on a full GPU.
    """
    if not isinstance(capabilities, dict):
        return "auto"
    return str(capabilities.get("device") or "auto").lower()
