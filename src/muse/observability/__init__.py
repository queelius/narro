"""Observability package: telemetry recording, storage, and the /dashboard UI.

Import-hygiene contract: this module MUST stay import-light. Submodules
like `muse.observability.recorder` are imported from the hot request path
(director/gateway/supervisor), and importing any submodule runs this
`__init__.py` first. So only stdlib-only names are re-exported eagerly
here; the fastapi/sse_starlette-touching names (`dashboard`,
`dashboard_auth`) are re-exported lazily via PEP 562 module `__getattr__`
so `import muse.observability.recorder` never drags fastapi in.
"""
from __future__ import annotations

from muse.observability.events import EVENT_COLUMNS, event_to_row
from muse.observability.store import TelemetryStore
from muse.observability.recorder import (
    TelemetryRecorder,
    record,
    get_recorder,
    init_recorder,
    reset_recorder,
)
from muse.observability.logs import LogHub
from muse.observability.sampler import Sampler, VramTracker

__all__ = [
    "EVENT_COLUMNS",
    "event_to_row",
    "TelemetryStore",
    "TelemetryRecorder",
    "record",
    "get_recorder",
    "init_recorder",
    "reset_recorder",
    "LogHub",
    "Sampler",
    "VramTracker",
    "build_dashboard_router",
    "DASHBOARD_HTML",
    "require_dashboard_auth",
    "check_dashboard_token",
]


def __getattr__(name: str):
    if name in ("build_dashboard_router", "DASHBOARD_HTML"):
        from muse.observability import dashboard
        return getattr(dashboard, name)
    if name in ("require_dashboard_auth", "check_dashboard_token"):
        from muse.observability import dashboard_auth
        return getattr(dashboard_auth, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
