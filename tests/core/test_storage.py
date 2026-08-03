"""Storage inventory and conservative pruning tests."""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from muse.core import config
from muse.core.catalog import (
    CatalogError,
    _reset_known_models_cache,
    _reset_read_catalog_cache,
    _write_catalog,
)
from muse.core.storage import (
    StorageInspectionError,
    automatic_prune_before_pull,
    execute_prune,
    inspect_storage,
    plan_prune,
    prune_storage,
)


@pytest.fixture
def storage_root(tmp_path, monkeypatch):
    root = tmp_path / "muse"
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MUSE_STORAGE_AUTO_PRUNE_BEFORE_PULL", "1")
    config.reset_config()
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    yield root
    config.reset_config()
    _reset_known_models_cache()
    _reset_read_catalog_cache()


def _complete_venv(path: Path) -> None:
    python = path / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o700)


def _age_tree(path: Path, *, seconds: float = 48 * 3600) -> None:
    timestamp = time.time() - seconds
    for child in sorted(path.rglob("*"), reverse=True):
        os.utime(child, (timestamp, timestamp), follow_symlinks=False)
    os.utime(path, (timestamp, timestamp), follow_symlinks=False)


def test_empty_inspection_is_read_only(storage_root):
    assert not storage_root.exists()

    report = inspect_storage(include_shared=False)

    assert report.muse_bytes == 0
    assert report.referenced_venvs == ()
    assert report.unreferenced_weights == ()
    assert not storage_root.exists()


def test_referenced_and_unreferenced_resources_are_separate(storage_root):
    active_venv = storage_root / "venvs" / "active"
    retained_venv = storage_root / "venvs" / "retained"
    active_weights = storage_root / "weights" / "active-weights"
    retained_weights = storage_root / "weights" / "retained-weights"
    _complete_venv(active_venv)
    _complete_venv(retained_venv)
    active_weights.mkdir(parents=True)
    retained_weights.mkdir()
    (active_weights / "model.bin").write_bytes(b"active")
    (retained_weights / "model.bin").write_bytes(b"retained")
    _write_catalog({
        "active": {
            "venv_path": str(active_venv),
            "local_dir": str(active_weights),
            "enabled": False,
        },
    })

    report = inspect_storage(include_shared=False)

    assert [item.path for item in report.referenced_venvs] == [active_venv]
    assert [item.path for item in report.unreferenced_venvs] == [retained_venv]
    assert [item.path for item in report.referenced_weights] == [active_weights]
    assert [item.path for item in report.unreferenced_weights] == [retained_weights]


def test_default_prune_deletes_only_old_incomplete_download(storage_root):
    partial = storage_root / "weights" / "models--org--demo" / "blobs" / "x.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    _age_tree(partial.parent.parent)

    report = inspect_storage(include_shared=False)
    plan = plan_prune(report)

    assert [item.path for item in plan.candidates] == [partial]
    result = execute_prune(plan)
    assert not result.failures
    assert not partial.exists()


def test_dry_run_never_deletes(storage_root):
    partial = (
        storage_root / "weights" / "models--org--demo" / "blobs"
        / "old.incomplete"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    _age_tree(partial.parent)
    plan = plan_prune(inspect_storage(include_shared=False))

    result = execute_prune(plan, dry_run=True)

    assert partial.exists()
    assert [outcome.action for outcome in result.outcomes] == ["would-delete"]


def test_young_partial_is_preserved(storage_root):
    partial = (
        storage_root / "weights" / "models--org--demo" / "blobs"
        / "new.incomplete"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")

    plan = plan_prune(inspect_storage(include_shared=False))

    assert plan.candidates == ()
    assert partial.exists()


def test_flat_model_payload_named_incomplete_is_never_transient_garbage(
    storage_root,
):
    bundle = storage_root / "weights" / "artifact-bundle"
    payload = bundle / "model.incomplete"
    bundle.mkdir(parents=True)
    payload.write_bytes(b"valid model payload")
    _complete_venv(storage_root / "venvs" / "bundle")
    _write_catalog({
        "bundle": {
            "local_dir": str(bundle),
            "venv_path": str(storage_root / "venvs" / "bundle"),
        },
    })
    _age_tree(bundle)

    report = inspect_storage(include_shared=False)

    assert report.incomplete_downloads == ()
    assert plan_prune(report).candidates == ()
    assert payload.read_bytes() == b"valid model payload"


def test_unreferenced_requires_explicit_opt_in(storage_root):
    venv = storage_root / "venvs" / "retained"
    weights = storage_root / "weights" / "retained"
    _complete_venv(venv)
    weights.mkdir(parents=True)
    (weights / "model.bin").write_bytes(b"weights")
    _age_tree(venv)
    _age_tree(weights)

    report = inspect_storage(include_shared=False)
    assert plan_prune(report).candidates == ()

    plan = plan_prune(report, include_unreferenced=True)
    assert {item.path for item in plan.candidates} == {venv, weights}
    result = execute_prune(plan)
    assert not result.failures
    assert not venv.exists()
    assert not weights.exists()


def test_new_catalog_reference_wins_over_old_plan(storage_root):
    weights = storage_root / "weights" / "retained"
    weights.mkdir(parents=True)
    (weights / "model.bin").write_bytes(b"weights")
    _age_tree(weights)
    plan = plan_prune(
        inspect_storage(include_shared=False), include_unreferenced=True,
    )
    assert [item.path for item in plan.candidates] == [weights]

    _write_catalog({
        "new-owner": {
            "local_dir": str(weights),
            "venv_path": str(storage_root / "venvs" / "new-owner"),
        },
    })
    result = execute_prune(plan)

    assert weights.exists()
    assert [outcome.action for outcome in result.outcomes] == ["skipped"]
    assert "reference" in result.outcomes[0].detail


def test_recent_descendant_protects_old_staging_directory(storage_root):
    staging = storage_root / "weights" / ".bundle.staging-abc"
    staging.mkdir(parents=True)
    child = staging / "active.part"
    child.write_bytes(b"still changing")
    old = time.time() - 48 * 3600
    os.utime(staging, (old, old))

    report = inspect_storage(include_shared=False)
    plan = plan_prune(report)

    assert report.abandoned_staging
    assert plan.candidates == ()
    assert staging.exists()


def test_transaction_without_complete_canonical_venv_is_recovery_data(
    storage_root,
):
    transaction = storage_root / "venvs" / ".demo.transaction-abc"
    _complete_venv(transaction / "prior")
    _age_tree(transaction)

    report = inspect_storage(include_shared=False)

    assert [item.path for item in report.recovery_workspaces] == [transaction]
    assert plan_prune(report).candidates == ()


def test_staging_for_leading_underscore_model_id_is_classified(storage_root):
    staging = storage_root / "venvs" / "._demo.staging-abc"
    _complete_venv(staging)
    _age_tree(staging)

    report = inspect_storage(include_shared=False)

    assert [item.path for item in report.abandoned_staging] == [staging]
    assert [item.model_id for item in report.abandoned_staging] == ["_demo"]


def test_referenced_dot_prefixed_venv_is_not_hidden_from_inventory(storage_root):
    venv = storage_root / "venvs" / ".demo"
    weights = storage_root / "weights" / "demo"
    _complete_venv(venv)
    weights.mkdir(parents=True)
    _write_catalog({
        ".demo": {"venv_path": str(venv), "local_dir": str(weights)},
    })

    report = inspect_storage(include_shared=False)

    assert [item.path for item in report.referenced_venvs] == [venv]


def test_hf_cache_tag_is_known_metadata_not_an_unsafe_path(storage_root):
    weights_root = storage_root / "weights"
    weights_root.mkdir(parents=True)
    (weights_root / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172789f06886806bc55\n")

    report = inspect_storage(include_shared=False)

    assert not any(
        issue.path == weights_root / "CACHEDIR.TAG" for issue in report.issues
    )


def test_symlink_in_owned_root_is_never_followed_or_pruned(storage_root, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "keep"
    marker.write_bytes(b"important")
    venv_root = storage_root / "venvs"
    venv_root.mkdir(parents=True)
    link = venv_root / "escape"
    link.symlink_to(external, target_is_directory=True)

    report = inspect_storage(include_shared=False)
    plan = plan_prune(report, include_unreferenced=True, older_than_seconds=0)

    assert plan.candidates == ()
    assert marker.read_bytes() == b"important"
    assert any(issue.kind == "unsafe-owned-path" for issue in report.issues)


@pytest.mark.parametrize("value", (-1, float("nan"), float("inf")))
def test_invalid_age_is_rejected(storage_root, value):
    report = inspect_storage(include_shared=False)
    with pytest.raises(ValueError, match="finite non-negative"):
        plan_prune(report, older_than_seconds=value)


def test_prune_rejects_invalid_age_before_scanning(storage_root):
    with patch(
        "muse.core.storage.inspect_storage",
        side_effect=AssertionError("inspection must not run"),
    ):
        with pytest.raises(ValueError, match="finite non-negative"):
            prune_storage(older_than_seconds=float("nan"))


def test_incomplete_directory_inspection_is_never_planned(storage_root, monkeypatch):
    from muse.core import storage as storage_module

    staging = storage_root / "weights" / ".bundle.staging-abc"
    staging.mkdir(parents=True)
    (staging / "unknown").write_bytes(b"data")
    _age_tree(staging)
    real_scandir = storage_module.os.scandir

    def refuse_staging(path):
        if Path(path) == staging:
            raise PermissionError("cannot inspect staging descendants")
        return real_scandir(path)

    monkeypatch.setattr(storage_module.os, "scandir", refuse_staging)

    report = inspect_storage(include_shared=False)

    assert report.abandoned_staging == ()
    assert plan_prune(report).candidates == ()
    assert staging.exists()
    assert any("cannot inspect staging" in issue.detail for issue in report.issues)


def test_apply_fails_closed_when_candidate_traversal_becomes_incomplete(
    storage_root,
    monkeypatch,
):
    from muse.core import storage as storage_module

    staging = storage_root / "weights" / ".bundle.staging-abc"
    staging.mkdir(parents=True)
    (staging / "unknown").write_bytes(b"data")
    _age_tree(staging)
    plan = plan_prune(inspect_storage(include_shared=False))
    assert [item.path for item in plan.candidates] == [staging]
    real_scandir = storage_module.os.scandir

    def refuse_staging(path):
        if Path(path) == staging:
            raise PermissionError("cannot revalidate staging descendants")
        return real_scandir(path)

    monkeypatch.setattr(storage_module.os, "scandir", refuse_staging)

    result = execute_prune(plan)

    assert staging.exists()
    assert [outcome.action for outcome in result.outcomes] == ["failed"]
    assert "cannot revalidate" in result.outcomes[0].detail


def test_catalog_error_during_venv_deletion_is_failure_not_busy(
    storage_root,
):
    staging = storage_root / "venvs" / ".demo.staging-abc"
    _complete_venv(staging)
    _age_tree(staging)
    plan = plan_prune(inspect_storage(include_shared=False))

    with patch(
        "muse.core.storage._delete_item",
        side_effect=CatalogError("unsafe deletion refused"),
    ):
        result = execute_prune(plan)

    assert staging.exists()
    assert [outcome.action for outcome in result.outcomes] == ["failed"]


def test_automatic_prune_under_low_space_only_removes_transient_data(
    storage_root,
):
    partial = (
        storage_root / "weights" / "models--org--demo" / "blobs"
        / "old.incomplete"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    retained = storage_root / "venvs" / "retained"
    _complete_venv(retained)
    _age_tree(partial.parent)
    _age_tree(retained)
    gib = 1024**3

    with patch(
        "muse.core.storage._filesystem_usage",
        return_value=(100 * gib, 5 * gib),
    ):
        maintenance = automatic_prune_before_pull()

    assert maintenance is not None
    assert not partial.exists()
    assert retained.exists(), "automatic cleanup must not include unreferenced venvs"
    assert maintenance.result.reclaimed_bytes > 0


def test_automatic_prune_skips_when_space_is_above_combined_threshold(
    storage_root,
):
    partial = (
        storage_root / "weights" / "models--org--demo" / "blobs"
        / "old.incomplete"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    _age_tree(partial.parent)
    gib = 1024**3

    with patch(
        "muse.core.storage._filesystem_usage",
        return_value=(100 * gib, 60 * gib),
    ):
        maintenance = automatic_prune_before_pull()

    assert maintenance is None
    assert partial.exists()


def test_automatic_prune_absolute_floor_triggers_on_small_filesystem(
    storage_root,
):
    partial = (
        storage_root / "weights" / "models--org--demo" / "blobs"
        / "old.incomplete"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    _age_tree(partial.parent)
    gib = 1024**3

    with patch(
        "muse.core.storage._filesystem_usage",
        return_value=(100 * gib, 15 * gib),
    ):
        maintenance = automatic_prune_before_pull()

    assert maintenance is not None
    assert not partial.exists()


def test_automatic_prune_percentage_floor_triggers_on_large_filesystem(
    storage_root,
):
    partial = (
        storage_root / "weights" / "models--org--demo" / "blobs"
        / "old.incomplete"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    _age_tree(partial.parent)
    gib = 1024**3

    with patch(
        "muse.core.storage._filesystem_usage",
        return_value=(2 * 1024 * gib, 80 * gib),
    ):
        maintenance = automatic_prune_before_pull()

    assert maintenance is not None
    assert not partial.exists()


def test_automatic_prune_can_be_disabled(storage_root, monkeypatch):
    partial = (
        storage_root / "weights" / "models--org--demo" / "blobs"
        / "old.incomplete"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    _age_tree(partial.parent)
    monkeypatch.setenv("MUSE_STORAGE_AUTO_PRUNE_BEFORE_PULL", "0")
    config.reset_config()

    with patch(
        "muse.core.storage._filesystem_usage",
        side_effect=AssertionError("disabled maintenance must not inspect space"),
    ):
        maintenance = automatic_prune_before_pull()

    assert maintenance is None
    assert partial.exists()
