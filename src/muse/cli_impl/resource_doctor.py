"""CLI presentation for Muse-owned runtime resource diagnostics."""
from __future__ import annotations

from muse.core.resource_registry import (
    ResourceRegistryError,
    inspect_resources,
    repair_stale_resources,
)


def run_resource_doctor(*, repair: bool = False, grace: float = 5.0) -> int:
    """Print owned-resource state and optionally repair verified orphans."""
    try:
        statuses = inspect_resources()
    except ResourceRegistryError as exc:
        print(f"resource registry error: {exc}")
        return 2

    if not statuses:
        print("no Muse-owned runtime resources are registered")
    else:
        print("KIND       PID      STATE        PORT   MODELS")
        for status in statuses:
            record = status.record
            models = ",".join(record.models) if record.models else "-"
            port = str(record.port) if record.port is not None else "-"
            print(
                f"{record.kind:<10} {record.pid:<8} {status.state:<12} "
                f"{port:<6} {models}"
            )
            print(f"  {record.resource_id}: {status.detail}")

    if repair:
        try:
            results = repair_stale_resources(grace=grace)
        except (ResourceRegistryError, ValueError) as exc:
            print(f"repair failed: {exc}")
            return 2
        for result in results:
            print(f"repair {result.resource_id}: {result.action} ({result.detail})")
        # A refusal is actionable and must not look like a successful repair.
        return 2 if any(result.action == "refused" for result in results) else 0

    return 1 if any(
        status.state in {"dead", "pid_reused", "orphaned", "unverifiable"}
        for status in statuses
    ) else 0
