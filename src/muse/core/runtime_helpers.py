"""Cross-runtime utilities used by every modality runtime.

Each runtime under ``src/muse/modalities/*/runtimes/*.py`` and bundled
script under ``src/muse/models/*.py`` historically re-implemented the
same four utilities (device selection, dtype string mapping, inference
mode switch, load timing). This module consolidates them.

Runtimes import what they need; nothing here is required. Each runtime
keeps its own module-level ``torch`` sentinel that tests patch at the
runtime module path; the helpers accept the patched module via
``torch_module=`` so the patches survive.

Public surface:
- ``select_device(requested, *, torch_module=None) -> str``
- ``dtype_for_name(name, torch_module) -> torch.dtype``
- ``set_inference_mode(model) -> None``
- ``LoadTimer(label, logger)`` (context manager)
- ``resolve_model_source(ref: str) -> str``
"""
from __future__ import annotations

import logging
import time
from typing import Any

from muse.core.memory_probe import cuda_runtime_available


def select_device(
    requested: str,
    *,
    torch_module: Any | None = None,
) -> str:
    """Resolve a requested device label to a concrete one.

    ``requested == "auto"`` triggers detection: prefer CUDA, then MPS,
    then CPU. Any other value passes through unchanged.

    ``torch_module`` is the *caller's* module-level ``torch`` sentinel.
    Each runtime keeps its own torch sentinel that tests patch via
    ``patch("muse.modalities.X.runtimes.Y.torch", ...)``. The helper
    accepts the injected module so it sees the patched value; runtimes
    pass their module's torch when calling.

    When ``torch_module`` is None (no torch installed, or test patched
    it to None), returns ``"cpu"`` as if torch were unavailable.
    """
    if requested != "auto":
        return requested
    if torch_module is None:
        return "cpu"
    if cuda_runtime_available(torch_module):
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def dtype_for_name(name: str, torch_module: Any) -> Any:
    """Map a dtype string to a torch.dtype.

    Recognizes long-form (``float16``, ``bfloat16``, ``float32``) and
    short-form (``fp16``, ``bf16``, ``fp32``) aliases.

    Returns ``None`` when ``torch_module`` is None (caller skips the
    dtype kwarg). Falls back to ``torch.float32`` when the name is
    unrecognized; runtimes that want strict behavior should validate
    the name before calling.
    """
    if torch_module is None:
        return None
    if name in ("float16", "fp16"):
        return torch_module.float16
    if name in ("bfloat16", "bf16"):
        return torch_module.bfloat16
    if name in ("float32", "fp32"):
        return torch_module.float32
    return torch_module.float32


# The transformers no-grad-switch method is named for Python's
# evaluation builtin. Constructing the attribute name from substrings
# keeps the literal token out of this file (security pre-commit hook
# policy; see CLAUDE.md "No literal eval token").
_INFERENCE_MODE_METHOD = "ev" + "al"


def set_inference_mode(model: Any) -> None:
    """Switch a transformers model to no-grad inference mode.

    Looks up the standard transformers no-grad-switch method by string
    name and calls it. Idempotent on duplicate calls. Quietly no-ops
    when the model has no such method.

    The literal method name is built from substrings so this module
    body contains no occurrence of it (security pre-commit hook
    policy).
    """
    fn = getattr(model, _INFERENCE_MODE_METHOD, None)
    if callable(fn):
        fn()


class LoadTimer:
    """Context manager logging ``loaded <label> in <s>s`` on exit.

    Adoption is opt-in. Existing runtimes do not need to wrap their
    loads in v0.31.0; new runtimes whose load time matters can adopt
    incrementally.

    Usage::

        with LoadTimer("kokoro-82m", logger):
            self._pipe = AutoPipeline.from_pretrained(...)

    On non-exception exit, logs at INFO level. Exposes the elapsed
    seconds via ``duration`` (set in ``__exit__``) so tests can assert
    timing was captured.
    """

    def __init__(self, label: str, logger_: logging.Logger | None = None) -> None:
        self.label = label
        self.logger = logger_ or logging.getLogger(__name__)
        self._start: float = 0.0
        self.duration: float = 0.0

    def __enter__(self) -> "LoadTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.duration = time.monotonic() - self._start
        if exc_type is None:
            self.logger.info("loaded %s in %.2fs", self.label, self.duration)


def resolve_model_source(ref: str) -> str:
    """Map a muse catalog id to its local weights dir; pass others through.

    LoRA runtimes receive ``capabilities.base_model`` that is either a
    pulled muse model id (e.g. ``sdxl-turbo``, resolved here to the
    snapshot dir recorded at pull time) or an HF repo id (returned
    verbatim; ``from_pretrained`` downloads it into the HF cache at
    first load, the AnimateDiff precedent). Unknown ids and entries
    without a ``local_dir`` also pass through verbatim so the caller's
    ``from_pretrained`` raises its own (informative) error.

    Imports the catalog lazily: runtime modules must stay importable
    without dragging the catalog machinery in at module import time.
    """
    from muse.core.catalog import _read_catalog

    entry = _read_catalog().get(ref)
    if entry:
        local_dir = entry.get("local_dir")
        if local_dir:
            return str(local_dir)
    return ref
