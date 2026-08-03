from __future__ import annotations

import multiprocessing
import os
import threading
from unittest.mock import patch

import pytest


def _hold_model_pull_lock(
    catalog_dir: str, identity: str, holding, release,
) -> None:
    os.environ["MUSE_CATALOG_DIR"] = catalog_dir
    from muse.core import config
    from muse.core.catalog import _model_pull_lock

    config.reset_config()
    with _model_pull_lock(identity):
        holding.set()
        release.wait(10)


def _acquire_model_pull_lock(
    catalog_dir: str, identity: str, attempting, acquired,
) -> None:
    os.environ["MUSE_CATALOG_DIR"] = catalog_dir
    from muse.core import config
    from muse.core.catalog import _model_pull_lock

    config.reset_config()
    attempting.set()
    with _model_pull_lock(identity):
        acquired.set()


def _hold_storage_cache_lock(catalog_dir: str, holding, release) -> None:
    os.environ["MUSE_CATALOG_DIR"] = catalog_dir
    from muse.core import config
    from muse.core.catalog import _storage_cache_lock

    config.reset_config()
    with _storage_cache_lock():
        holding.set()
        release.wait(10)


def _try_storage_cache_lock(catalog_dir: str, attempted, refused) -> None:
    os.environ["MUSE_CATALOG_DIR"] = catalog_dir
    from muse.core import config
    from muse.core.catalog import StorageBusyError, _storage_cache_lock

    config.reset_config()
    attempted.set()
    try:
        with _storage_cache_lock(wait=False):
            pass
    except StorageBusyError:
        refused.set()


def test_same_model_pull_lock_serializes_independent_processes(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork-capable platform")
    context = multiprocessing.get_context("fork")
    holding = context.Event()
    release = context.Event()
    attempting = context.Event()
    acquired = context.Event()
    first = context.Process(
        target=_hold_model_pull_lock,
        args=(str(tmp_path), "same-model", holding, release),
    )
    second = context.Process(
        target=_acquire_model_pull_lock,
        args=(str(tmp_path), "same-model", attempting, acquired),
    )
    first.start()
    try:
        assert holding.wait(5)
        second.start()
        assert attempting.wait(5)
        assert not acquired.wait(0.2)
        release.set()
        assert acquired.wait(5)
        first.join(5)
        second.join(5)
        assert first.exitcode == 0
        assert second.exitcode == 0
    finally:
        release.set()
        first.join(1)
        if second.pid is not None:
            second.join(1)


def test_different_model_pull_locks_do_not_serialize(tmp_path, monkeypatch):
    from muse.core import config
    from muse.core.catalog import _model_pull_lock

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    config.reset_config()
    first_holding = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()

    def first() -> None:
        with _model_pull_lock("model-a"):
            first_holding.set()
            release_first.wait(5)

    def second() -> None:
        with _model_pull_lock("model-b"):
            second_acquired.set()

    thread_a = threading.Thread(target=first)
    thread_b = threading.Thread(target=second)
    thread_a.start()
    assert first_holding.wait(5)
    thread_b.start()
    try:
        assert second_acquired.wait(1)
    finally:
        release_first.set()
        thread_a.join(5)
        thread_b.join(5)
        config.reset_config()


def test_pull_lock_paths_are_hashed_and_private(tmp_path, monkeypatch):
    from muse.core import config
    from muse.core.catalog import _model_pull_lock

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    config.reset_config()
    identity = "hf://organization/model@revision/../../not-a-path"
    with _model_pull_lock(identity):
        lock_files = list((tmp_path / "locks").glob("pull-*.lock"))
        assert len(lock_files) == 1
        assert "organization" not in lock_files[0].name
        assert lock_files[0].stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "locks").stat().st_mode & 0o777 == 0o700
    config.reset_config()


def test_storage_cache_lock_is_nonblocking_for_cleanup(tmp_path, monkeypatch):
    from muse.core import config
    from muse.core.catalog import StorageBusyError, _storage_cache_lock

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    config.reset_config()
    attempted = threading.Event()
    refused = threading.Event()

    def contender() -> None:
        attempted.set()
        try:
            with _storage_cache_lock(wait=False):
                pass
        except StorageBusyError:
            refused.set()

    with _storage_cache_lock():
        thread = threading.Thread(target=contender)
        thread.start()
        assert attempted.wait(1)
        assert refused.wait(1)
        thread.join(2)
        lock_path = tmp_path / "locks" / "storage-cache.lock"
        assert lock_path.stat().st_mode & 0o777 == 0o600
    config.reset_config()


def test_storage_cache_lock_serializes_independent_processes(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork-capable platform")
    context = multiprocessing.get_context("fork")
    holding = context.Event()
    release = context.Event()
    attempted = context.Event()
    refused = context.Event()
    holder = context.Process(
        target=_hold_storage_cache_lock,
        args=(str(tmp_path), holding, release),
    )
    contender = context.Process(
        target=_try_storage_cache_lock,
        args=(str(tmp_path), attempted, refused),
    )
    holder.start()
    try:
        assert holding.wait(5)
        contender.start()
        assert attempted.wait(5)
        assert refused.wait(5)
        contender.join(5)
        assert contender.exitcode == 0
    finally:
        release.set()
        holder.join(5)
        if contender.pid is not None:
            contender.join(1)
    assert holder.exitcode == 0


def test_remove_waits_for_same_model_pull_lock(tmp_path, monkeypatch):
    from muse.core import config
    from muse.core.catalog import _model_pull_lock, remove

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    config.reset_config()
    holding = threading.Event()
    release = threading.Event()
    removed = threading.Event()

    def hold_pull() -> None:
        with _model_pull_lock("same-model"):
            holding.set()
            release.wait(5)

    def perform_remove() -> None:
        remove("same-model", purge=True)
        removed.set()

    holder = threading.Thread(target=hold_pull)
    remover = threading.Thread(target=perform_remove)
    with patch("muse.core.catalog._remove_locked") as remove_locked:
        holder.start()
        assert holding.wait(2)
        remover.start()
        try:
            assert not removed.wait(0.1)
            remove_locked.assert_not_called()
            release.set()
            assert removed.wait(2)
        finally:
            release.set()
            holder.join(2)
            remover.join(2)
    remove_locked.assert_called_once_with("same-model", purge=True)
    config.reset_config()
