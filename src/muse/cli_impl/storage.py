"""Human and JSON presentation for Muse storage diagnostics and pruning."""
from __future__ import annotations

import json
import sys

from muse.core import config
from muse.core.catalog import CatalogError
from muse.core.storage import (
    PruneResult,
    StorageInspectionError,
    StorageItem,
    StorageReport,
    inspect_storage,
    plan_prune,
    prune_storage,
)


def _human_size(value: int) -> str:
    size = float(max(0, value))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit in {"B", "KiB"} else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _item_payload(item: StorageItem) -> dict:
    return {
        "kind": item.kind,
        "status": item.status,
        "path": str(item.path),
        "physical_bytes": item.physical_bytes,
        "latest_mtime": item.latest_mtime,
        "model_id": item.model_id,
        "reason": item.reason,
    }


def _report_payload(report: StorageReport) -> dict:
    safe_plan = plan_prune(report)
    unreferenced_plan = plan_prune(report, include_unreferenced=True)
    return {
        "catalog_dir": str(report.catalog_dir),
        "scanned_at": report.scanned_at,
        "filesystem": {
            "total_bytes": report.filesystem_total_bytes,
            "free_bytes": report.filesystem_free_bytes,
        },
        "muse_owned": {
            "total_bytes": report.muse_bytes,
            "venv_bytes": report.venv_bytes,
            "weights_bytes": report.weights_bytes,
            "owned_hf_recognized_bytes": report.owned_hf_recognized_bytes,
        },
        "referenced": {
            "venvs": [_item_payload(item) for item in report.referenced_venvs],
            "weights": [_item_payload(item) for item in report.referenced_weights],
        },
        "unreferenced": {
            "venvs": [_item_payload(item) for item in report.unreferenced_venvs],
            "weights": [_item_payload(item) for item in report.unreferenced_weights],
        },
        "safe_garbage": {
            "incomplete_downloads": [
                _item_payload(item) for item in report.incomplete_downloads
            ],
            "abandoned_staging": [
                _item_payload(item) for item in report.abandoned_staging
            ],
            "recovery_workspaces": [
                _item_payload(item) for item in report.recovery_workspaces
            ],
            "eligible_bytes": safe_plan.estimated_bytes,
        },
        "unreferenced_eligible_bytes": max(
            0, unreferenced_plan.estimated_bytes - safe_plan.estimated_bytes,
        ),
        "shared": {
            "huggingface_cache": (
                str(report.shared_hf_cache) if report.shared_hf_cache else None
            ),
            "huggingface_bytes": report.shared_hf_bytes,
            "huggingface_catalog_referenced_bytes": (
                report.shared_hf_referenced_bytes
            ),
            "pip_cache": str(report.pip_cache) if report.pip_cache else None,
            "pip_bytes": report.pip_cache_bytes,
        },
        "automatic_policy": _automatic_policy_payload(report),
        "issues": [
            {
                "kind": issue.kind,
                "model_id": issue.model_id,
                "path": str(issue.path) if issue.path else None,
                "detail": issue.detail,
            }
            for issue in report.issues
        ],
        "owned_hf_warnings": list(report.owned_hf_warnings),
    }


def _sum_items(items: tuple[StorageItem, ...]) -> int:
    return sum(item.physical_bytes for item in items)


def _automatic_policy_payload(report: StorageReport) -> dict:
    enabled = bool(config.get("storage.auto_prune_before_pull"))
    grace_hours = float(config.get("storage.auto_prune_grace_hours"))
    min_free_gb = float(config.get("storage.auto_prune_min_free_gb"))
    min_free_percent = float(
        config.get("storage.auto_prune_min_free_percent")
    )
    threshold_bytes = max(
        int(min_free_gb * 1024**3),
        int(report.filesystem_total_bytes * min_free_percent / 100.0),
    )
    return {
        "enabled_before_pull": enabled,
        "grace_hours": grace_hours,
        "min_free_gb": min_free_gb,
        "min_free_percent": min_free_percent,
        "trigger_below_bytes": threshold_bytes,
        "below_trigger": report.filesystem_free_bytes < threshold_bytes,
        "safe_categories_only": True,
    }


def _print_report(report: StorageReport) -> None:
    safe_plan = plan_prune(report)
    all_plan = plan_prune(report, include_unreferenced=True)
    unreferenced_bytes = max(0, all_plan.estimated_bytes - safe_plan.estimated_bytes)
    free_percent = (
        100.0 * report.filesystem_free_bytes / report.filesystem_total_bytes
        if report.filesystem_total_bytes else 0.0
    )

    print(f"Muse storage: {report.catalog_dir}")
    print(
        f"  Muse-owned total:  {_human_size(report.muse_bytes)} "
        f"(venvs {_human_size(report.venv_bytes)}, "
        f"weights {_human_size(report.weights_bytes)})"
    )
    print(
        f"  Filesystem free:   {_human_size(report.filesystem_free_bytes)} "
        f"({free_percent:.1f}%)"
    )
    print(
        f"  Catalog references: {len(report.referenced_venvs)} venv(s), "
        f"{len(report.referenced_weights)} owned weight store(s)"
    )
    print(
        f"  Unreferenced:      {len(report.unreferenced_venvs)} venv(s) / "
        f"{_human_size(_sum_items(report.unreferenced_venvs))}, "
        f"{len(report.unreferenced_weights)} weight store(s) / "
        f"{_human_size(_sum_items(report.unreferenced_weights))}"
    )
    print(
        f"  Old safe garbage:  {len(safe_plan.candidates)} item(s) / "
        f"{_human_size(safe_plan.estimated_bytes)} eligible now"
    )
    policy = _automatic_policy_payload(report)
    if policy["enabled_before_pull"]:
        print(
            "  Automatic cleanup: enabled before pulls below "
            f"{policy['min_free_gb']:g} GiB or "
            f"{policy['min_free_percent']:g}% free "
            f"({policy['grace_hours']:g}h grace; safe transient data only)"
        )
    else:
        print("  Automatic cleanup: disabled")
    if report.recovery_workspaces:
        print(
            f"  Recovery data:     {len(report.recovery_workspaces)} workspace(s) "
            "preserved for manual inspection"
        )

    if report.shared_hf_cache is not None:
        print("Shared caches (report only; never deleted by Muse):")
        print(
            f"  Hugging Face:      {_human_size(report.shared_hf_bytes)} at "
            f"{report.shared_hf_cache}"
        )
        if report.shared_hf_referenced_bytes:
            print(
                f"    catalog-referenced repositories: "
                f"{_human_size(report.shared_hf_referenced_bytes)}"
            )
    if report.pip_cache is not None:
        print(
            f"  pip:               {_human_size(report.pip_cache_bytes)} at "
            f"{report.pip_cache}"
        )

    if report.issues or report.owned_hf_warnings:
        print("Issues:")
        for issue in report.issues:
            owner = f" [{issue.model_id}]" if issue.model_id else ""
            location = f" ({issue.path})" if issue.path else ""
            print(f"  - {issue.kind}{owner}: {issue.detail}{location}")
        for warning in report.owned_hf_warnings:
            print(f"  - owned-hf-cache: {warning}")

    if safe_plan.estimated_bytes:
        print(
            f"Run `muse storage prune` to reclaim about "
            f"{_human_size(safe_plan.estimated_bytes)} of old partial/staging data."
        )
    if unreferenced_bytes:
        print(
            f"Run `muse storage prune --include-unreferenced --dry-run` to review "
            f"another {_human_size(unreferenced_bytes)} that may be retained data."
        )


def run_storage_doctor(*, json_output: bool = False) -> int:
    """Inspect storage; return 1 for actionable findings and 2 on failure."""
    try:
        report = inspect_storage(include_shared=True)
    except (OSError, StorageInspectionError, ValueError) as exc:
        print(f"storage inspection failed: {exc}", file=sys.stderr)
        return 2
    if json_output:
        print(json.dumps(_report_payload(report), indent=2, sort_keys=True))
    else:
        _print_report(report)

    low_space = bool(_automatic_policy_payload(report)["below_trigger"])
    actionable = bool(
        report.issues
        or report.owned_hf_warnings
        or report.incomplete_downloads
        or report.abandoned_staging
        or report.recovery_workspaces
        or report.unreferenced_venvs
        or report.unreferenced_weights
        or low_space
    )
    return 1 if actionable else 0


def _result_payload(result: PruneResult) -> dict:
    return {
        "dry_run": result.dry_run,
        "reclaimed_bytes": result.reclaimed_bytes,
        "outcomes": [
            {
                "action": outcome.action,
                "kind": outcome.item.kind,
                "path": str(outcome.item.path),
                "estimated_bytes": outcome.item.physical_bytes,
                "reclaimed_bytes": outcome.reclaimed_bytes,
                "detail": outcome.detail,
            }
            for outcome in result.outcomes
        ],
    }


def run_storage_prune(
    *,
    dry_run: bool = False,
    include_unreferenced: bool = False,
    older_than_hours: float = 24.0,
    json_output: bool = False,
) -> int:
    """Plan and apply conservative Muse-owned storage cleanup."""
    try:
        plan, result = prune_storage(
            dry_run=dry_run,
            include_unreferenced=include_unreferenced,
            older_than_seconds=float(older_than_hours) * 3600.0,
        )
    except (CatalogError, OSError, StorageInspectionError, ValueError) as exc:
        print(f"storage prune failed: {exc}", file=sys.stderr)
        return 2

    if json_output:
        payload = _result_payload(result)
        payload["estimated_bytes"] = plan.estimated_bytes
        payload["notices"] = list(plan.notices)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for notice in plan.notices:
            print(f"notice: {notice}")
        if not result.outcomes:
            print("nothing eligible to prune")
        for outcome in result.outcomes:
            size = _human_size(outcome.item.physical_bytes)
            print(
                f"{outcome.action:<12} {size:>10}  {outcome.item.path}"
                + (f" ({outcome.detail})" if outcome.detail else "")
            )
        verb = "would reclaim" if dry_run else "reclaimed"
        amount = plan.estimated_bytes if dry_run else result.reclaimed_bytes
        print(f"{verb} {_human_size(amount)}")

    return 2 if result.failures else 0
