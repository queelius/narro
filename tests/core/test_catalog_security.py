"""Filesystem and serialization trust-boundary regressions for catalog I/O."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from muse.core import catalog


@pytest.fixture(autouse=True)
def _isolated_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    catalog._reset_known_models_cache()
    catalog._reset_read_catalog_cache()
    yield
    catalog._reset_known_models_cache()
    catalog._reset_read_catalog_cache()


def test_catalog_read_refuses_symlink(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external.json"
    external.write_text(json.dumps({"outside": {"enabled": True}}))
    (tmp_path / "catalog.json").symlink_to(external)

    with pytest.raises(catalog.CatalogError, match="non-symlink"):
        catalog._read_catalog()

    assert json.loads(external.read_text()) == {"outside": {"enabled": True}}


def test_catalog_transaction_refuses_symlink_lock(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-lock-target"
    external.write_text("do not touch")
    (tmp_path / ".catalog.lock").symlink_to(external)

    with pytest.raises(catalog.CatalogError, match="safely open"):
        catalog._write_catalog({"model": {"enabled": True}})

    assert external.read_text() == "do not touch"


def test_pull_lock_refuses_symlink_directory(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external-locks"
    external.mkdir()
    (tmp_path / "locks").symlink_to(external, target_is_directory=True)

    with pytest.raises(catalog.CatalogError, match="safe directory"):
        with catalog._model_pull_lock("model"):
            pytest.fail("unsafe lock unexpectedly acquired")

    assert list(external.iterdir()) == []


def test_catalog_write_rejects_nonfinite_json() -> None:
    with pytest.raises(catalog.CatalogError, match="finite numbers"):
        catalog._write_catalog({"model": {"memory_gb": float("nan")}})


@pytest.mark.parametrize("model_id", ["../escape", "/absolute", ".", ".."])
def test_catalog_rejects_filesystem_unsafe_model_ids(model_id: str) -> None:
    with pytest.raises(catalog.CatalogError, match="invalid model id"):
        catalog._write_catalog({model_id: {"enabled": True}})


def test_catalog_read_rejects_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catalog, "_MAX_CATALOG_BYTES", 32)
    (tmp_path / "catalog.json").write_text(json.dumps({"model": {"x": "y" * 64}}))

    with pytest.raises(catalog.CatalogError, match="exceeds 32 bytes"):
        catalog._read_catalog()


@pytest.mark.parametrize("other", ["weights/root/child", "weights"])
def test_recursive_purge_reference_detects_parent_child_overlap(
    tmp_path: Path, other: str,
) -> None:
    target = (tmp_path / "weights" / "root").resolve()
    referenced = (tmp_path / other).resolve()
    data = {"other": {"local_dir": str(referenced)}}

    assert catalog._other_catalog_path_reference(
        data,
        model_id="owner",
        field_name="local_dir",
        target=target,
    ) == "other"
