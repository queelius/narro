"""Node-state model + pure refresh reducer for the muse federation coordinator.

`build_node_state` folds three polled payloads (`/v1/models`, `/health`,
`/v1/telemetry/summary`) for one node into a `NodeState` snapshot. It is
pure: no network calls, no clock reads. The caller fetches the payloads
and passes the current time in via `now`, so this module stays stdlib
only (dataclasses) and is trivially testable without mocking a clock or
an HTTP client.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from muse.federation.nodes import NodeSpec

_MAX_NODE_MODELS = 4096


@dataclass
class ModelAvail:
    loaded: bool
    enabled: bool


@dataclass
class NodeState:
    spec: NodeSpec
    reachable: bool
    models: dict[str, ModelAvail] = field(default_factory=dict)
    in_flight: int | None = None
    last_poll_ts: float = 0.0


def build_node_state(
    spec: NodeSpec,
    *,
    models_payload: dict | None,
    health_payload: dict | None,
    summary_payload: dict | None,
    now: float,
) -> NodeState:
    """Fold polled payloads for one node into a NodeState snapshot.

    - reachable = models_payload is a mapping with a list-shaped ``data``
      field (a node whose /v1/models we cannot safely interpret is
      unroutable).
    - models: one ModelAvail per entry in models_payload["data"], keyed by
      entry["id"]. Missing/absent "data" (or models_payload is None)
      yields an empty dict. Entries lacking a usable string "id" (falsy
      or non-string) are skipped rather than raising, so one malformed
      entry does not nuke the rest of the node's model list.
    - in_flight accepts only a non-negative JSON integer; malformed remote
      telemetry degrades to None so it can never break numeric routing.
    - last_poll_ts = now (passed in, never read from a clock here).
    """
    reachable = isinstance(models_payload, dict)

    models: dict[str, ModelAvail] = {}
    if reachable:
        entries = models_payload.get("data", [])
        if not isinstance(entries, list):
            reachable = False
            entries = []
        elif len(entries) > _MAX_NODE_MODELS:
            # Do not silently cache a truncated routing view: treating an
            # over-limit node as unreachable is deterministic and bounded.
            reachable = False
            entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if not model_id or not isinstance(model_id, str):
                continue
            loaded = entry.get("loaded", False)
            models[model_id] = ModelAvail(
                loaded=loaded if isinstance(loaded, bool) else False,
                enabled=True,
            )

    raw_in_flight = (
        summary_payload.get("in_flight")
        if isinstance(summary_payload, dict)
        else None
    )
    in_flight = (
        raw_in_flight
        if type(raw_in_flight) is int and raw_in_flight >= 0
        else None
    )

    return NodeState(
        spec=spec,
        reachable=reachable,
        models=models,
        in_flight=in_flight,
        last_poll_ts=now,
    )
