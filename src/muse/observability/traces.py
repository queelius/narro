"""Request-local trace context shared by the gateway and load director.

``asyncio.to_thread`` copies context variables into its worker thread. That
makes a ContextVar a useful, low-coupling seam here: the gateway owns the
request trace while LoadDirector can annotate the exact cold load and LRU
evictions performed on behalf of that request without changing acquire's
long-standing integer return type.
"""
from __future__ import annotations

import contextvars
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestTrace:
    request_id: str
    model_id: str
    modality: str
    cold: bool = False
    load_ms: float = 0.0
    queued_ms: float = 0.0
    forward_ms: float = 0.0
    forward_started_at: float | None = None
    evicted_models: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def note_load(self, model_id: str, seconds: float) -> None:
        if model_id != self.model_id:
            return
        with self._lock:
            self.cold = True
            self.load_ms += max(0.0, float(seconds)) * 1000.0

    def note_eviction(self, model_id: str) -> None:
        with self._lock:
            if model_id not in self.evicted_models:
                self.evicted_models.append(model_id)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "request_id": self.request_id,
                "model_id": self.model_id,
                "modality": self.modality,
                "cold": self.cold,
                "load_ms": self.load_ms,
                "queued_ms": self.queued_ms,
                "forward_ms": self.forward_ms,
                "evicted_models": list(self.evicted_models),
            }


_CURRENT_TRACE: contextvars.ContextVar[RequestTrace | None] = contextvars.ContextVar(
    "muse_request_trace", default=None,
)


def begin_request_trace(
    model_id: str, modality: str,
) -> tuple[RequestTrace, contextvars.Token[RequestTrace | None]]:
    trace = RequestTrace(
        request_id=uuid.uuid4().hex,
        model_id=model_id,
        modality=modality,
    )
    return trace, _CURRENT_TRACE.set(trace)


def reset_request_trace(token: contextvars.Token[RequestTrace | None]) -> None:
    _CURRENT_TRACE.reset(token)


def current_request_id() -> str | None:
    trace = _CURRENT_TRACE.get()
    return trace.request_id if trace is not None else None


def note_model_load(model_id: str, seconds: float) -> None:
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        trace.note_load(model_id, seconds)


def note_model_eviction(model_id: str) -> None:
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        trace.note_eviction(model_id)
