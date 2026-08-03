from __future__ import annotations

import json
import math
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from muse.core import resource_registry as registry


def _identity_map(monkeypatch, values: dict[int, float | None]) -> None:
    monkeypatch.setattr(
        registry, "_process_create_time", lambda pid: values.get(pid),
    )


def _private_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    runtime.chmod(0o700)
    return runtime


def _private_write(path, content: str | bytes) -> None:
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def test_empty_read_does_not_create_runtime_directory(tmp_path):
    assert registry.list_resources(catalog_dir=tmp_path) == []
    assert not (tmp_path / "runtime").exists()


def test_register_list_unregister_round_trip(tmp_path, monkeypatch):
    _identity_map(monkeypatch, {200: 10.0, 100: 5.0})
    monkeypatch.setattr(registry, "_isolated_process_group", lambda pid: pid)

    resource_id = registry.register_process(
        kind="worker",
        pid=200,
        owner_pid=100,
        port=9001,
        models=["demo"],
        catalog_dir=tmp_path,
    )

    records = registry.list_resources(catalog_dir=tmp_path)
    assert len(records) == 1
    assert records[0].resource_id == resource_id
    assert records[0].models == ("demo",)
    assert records[0].process_group == 200
    assert registry.registry_path(tmp_path).stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "runtime").stat().st_mode & 0o777 == 0o700

    assert registry.unregister_process(resource_id, catalog_dir=tmp_path) is True
    assert registry.list_resources(catalog_dir=tmp_path) == []


def test_schema_validation_rejects_valid_but_malformed_json(tmp_path):
    runtime = _private_runtime(tmp_path)
    _private_write(runtime / "resources.lock", b"\0")
    _private_write(
        runtime / "resources.json", json.dumps({"version": 1, "resources": []}),
    )

    with pytest.raises(registry.ResourceRegistryError, match="resources must be"):
        registry.list_resources(catalog_dir=tmp_path)


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_schema_validation_requires_exact_integer_version(tmp_path, version):
    runtime = _private_runtime(tmp_path)
    _private_write(runtime / "resources.lock", b"\0")
    _private_write(
        runtime / "resources.json",
        json.dumps({"version": version, "resources": {}}),
    )

    with pytest.raises(registry.ResourceRegistryError, match="supported"):
        registry.list_resources(catalog_dir=tmp_path)


def test_read_refuses_group_writable_runtime_directory(tmp_path):
    runtime = _private_runtime(tmp_path)
    runtime.chmod(0o770)
    _private_write(runtime / "resources.lock", b"\0")
    _private_write(
        runtime / "resources.json", json.dumps({"version": 1, "resources": {}}),
    )

    with pytest.raises(registry.ResourceRegistryError, match="group/other writable"):
        registry.list_resources(catalog_dir=tmp_path)


def test_read_refuses_group_writable_registry_file(tmp_path):
    runtime = _private_runtime(tmp_path)
    _private_write(runtime / "resources.lock", b"\0")
    data = runtime / "resources.json"
    _private_write(data, json.dumps({"version": 1, "resources": {}}))
    data.chmod(0o660)

    with pytest.raises(registry.ResourceRegistryError, match="group/other writable"):
        registry.list_resources(catalog_dir=tmp_path)


def test_read_refuses_oversized_registry_before_json_decode(tmp_path):
    runtime = _private_runtime(tmp_path)
    _private_write(runtime / "resources.lock", b"\0")
    data = runtime / "resources.json"
    with data.open("wb") as handle:
        handle.truncate(registry._MAX_REGISTRY_BYTES + 1)
    data.chmod(0o600)

    with pytest.raises(registry.ResourceRegistryError, match="exceeds"):
        registry.list_resources(catalog_dir=tmp_path)


def test_read_refuses_excessive_resource_count_before_entry_parsing(tmp_path):
    runtime = _private_runtime(tmp_path)
    _private_write(runtime / "resources.lock", b"\0")
    resources = {str(index): {} for index in range(registry._MAX_RESOURCE_RECORDS + 1)}
    _private_write(
        runtime / "resources.json",
        json.dumps({"version": 1, "resources": resources}),
    )

    with pytest.raises(registry.ResourceRegistryError, match="more than"):
        registry.list_resources(catalog_dir=tmp_path)


@pytest.mark.parametrize("field", ["create_time", "owner_create_time", "created_at"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_schema_validation_rejects_nonfinite_identity_times(tmp_path, field, value):
    runtime = _private_runtime(tmp_path)
    _private_write(runtime / "resources.lock", b"\0")
    record = {
        "resource_id": "r",
        "kind": "worker",
        "pid": 200,
        "create_time": 10.0,
        "owner_pid": 100,
        "owner_create_time": 5.0,
        "process_group": 200,
        "port": None,
        "models": [],
        "created_at": 1.0,
    }
    record[field] = value
    _private_write(
        runtime / "resources.json",
        json.dumps({"version": 1, "resources": {"r": record}}),
    )

    with pytest.raises(registry.ResourceRegistryError, match="finite"):
        registry.list_resources(catalog_dir=tmp_path)


def test_read_refuses_registry_symlink_without_following_it(tmp_path):
    runtime = _private_runtime(tmp_path)
    _private_write(runtime / "resources.lock", b"\0")
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"version": 1, "resources": {}}), encoding="utf-8")
    (runtime / "resources.json").symlink_to(outside)

    with pytest.raises(registry.ResourceRegistryError, match="regular file|safely open"):
        registry.list_resources(catalog_dir=tmp_path)


def test_read_refuses_lock_symlink_without_touching_target(tmp_path):
    runtime = _private_runtime(tmp_path)
    outside = tmp_path / "outside.lock"
    outside.write_bytes(b"sentinel")
    (runtime / "resources.lock").symlink_to(outside)
    _private_write(
        runtime / "resources.json", json.dumps({"version": 1, "resources": {}}),
    )

    with pytest.raises(registry.ResourceRegistryError, match="safely open"):
        registry.list_resources(catalog_dir=tmp_path)
    assert outside.read_bytes() == b"sentinel"


def test_read_refuses_existing_registry_without_lock_and_creates_nothing(tmp_path):
    runtime = _private_runtime(tmp_path)
    data = runtime / "resources.json"
    _private_write(data, json.dumps({"version": 1, "resources": {}}))

    with pytest.raises(registry.ResourceRegistryError, match="lock is missing"):
        registry.list_resources(catalog_dir=tmp_path)
    assert sorted(path.name for path in runtime.iterdir()) == ["resources.json"]


def test_read_wraps_advisory_lock_failure(tmp_path, monkeypatch):
    runtime = _private_runtime(tmp_path)
    _private_write(runtime / "resources.lock", b"\0")
    _private_write(
        runtime / "resources.json", json.dumps({"version": 1, "resources": {}}),
    )
    monkeypatch.setattr(
        registry,
        "_acquire_file_lock",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("lock unavailable")),
    )

    with pytest.raises(registry.ResourceRegistryError, match="cannot lock"):
        registry.list_resources(catalog_dir=tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason="symlink loops are POSIX-specific")
def test_read_wraps_catalog_symlink_resolution_loop(tmp_path):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)

    with pytest.raises(registry.ResourceRegistryError, match="cannot resolve"):
        registry.list_resources(catalog_dir=loop)


def test_register_refuses_runtime_directory_symlink(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "runtime").symlink_to(outside, target_is_directory=True)
    _identity_map(monkeypatch, {200: 10.0})

    with pytest.raises(registry.ResourceRegistryError, match="safe directory"):
        registry.register_process(kind="worker", pid=200, catalog_dir=tmp_path)
    assert list(outside.iterdir()) == []


def test_read_refuses_runtime_directory_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "resources.lock").write_bytes(b"\0")
    (outside / "resources.json").write_text(
        json.dumps({"version": 1, "resources": {}}), encoding="utf-8",
    )
    (tmp_path / "runtime").symlink_to(outside, target_is_directory=True)

    with pytest.raises(registry.ResourceRegistryError, match="safe directory"):
        registry.list_resources(catalog_dir=tmp_path)


def test_registration_validates_metadata_before_identity_lookup(tmp_path, monkeypatch):
    monkeypatch.setattr(
        registry,
        "_process_create_time",
        lambda pid: pytest.fail("invalid metadata must fail before identity lookup"),
    )

    with pytest.raises(registry.ResourceRegistryError, match="owner pid"):
        registry.register_process(
            kind="worker", pid=200, owner_pid="100", catalog_dir=tmp_path,
        )
    with pytest.raises(registry.ResourceRegistryError, match="models"):
        registry.register_process(
            kind="worker", pid=200, models="demo", catalog_dir=tmp_path,
        )


def test_concurrent_thread_registration_preserves_every_record(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_process_create_time", lambda pid: float(pid))
    monkeypatch.setattr(registry, "_isolated_process_group", lambda pid: pid)

    def register(pid: int) -> str:
        return registry.register_process(
            kind="worker", pid=pid, models=[f"model-{pid}"], catalog_dir=tmp_path,
        )

    pids = list(range(200, 220))
    with ThreadPoolExecutor(max_workers=8) as executor:
        resource_ids = list(executor.map(register, pids))

    records = registry.list_resources(catalog_dir=tmp_path)
    assert {record.resource_id for record in records} == set(resource_ids)
    assert {record.pid for record in records} == set(pids)


def test_oversized_registration_preserves_existing_registry(tmp_path, monkeypatch):
    _identity_map(monkeypatch, {200: 10.0, 201: 11.0})
    first_id = registry.register_process(
        kind="worker", pid=200, models=["small"], catalog_dir=tmp_path,
    )

    with pytest.raises(registry.ResourceRegistryError, match="cannot exceed"):
        registry.register_process(
            kind="worker",
            pid=201,
            models=["x" * registry._MAX_REGISTRY_BYTES],
            catalog_dir=tmp_path,
        )

    records = registry.list_resources(catalog_dir=tmp_path)
    assert [record.resource_id for record in records] == [first_id]


def test_unregister_compare_and_remove_preserves_replacement(tmp_path, monkeypatch):
    _identity_map(monkeypatch, {200: 10.0})
    resource_id = registry.register_process(
        kind="worker", pid=200, catalog_dir=tmp_path,
    )
    original = registry.list_resources(catalog_dir=tmp_path)[0]
    replacement = replace(original, created_at=original.created_at + 1.0)
    with registry._registry_lock(tmp_path):
        records = registry._read_unlocked(tmp_path)
        records[resource_id] = replacement
        registry._write_unlocked(records, tmp_path)

    assert registry.unregister_process(
        resource_id, catalog_dir=tmp_path, expected=original,
    ) is False
    assert registry.list_resources(catalog_dir=tmp_path) == [replacement]


def test_inspection_distinguishes_dead_reused_orphan_and_running(monkeypatch):
    base = registry.ResourceRecord(
        resource_id="r",
        kind="worker",
        pid=200,
        create_time=10.0,
        owner_pid=100,
        owner_create_time=5.0,
        process_group=200,
        port=9001,
        models=("demo",),
        created_at=1.0,
    )

    _identity_map(monkeypatch, {})
    assert registry.inspect_resource(base).state == "dead"
    _identity_map(monkeypatch, {200: 11.0})
    assert registry.inspect_resource(base).state == "pid_reused"
    _identity_map(monkeypatch, {200: 10.0})
    assert registry.inspect_resource(base).state == "orphaned"
    _identity_map(monkeypatch, {200: 10.0, 100: 5.0})
    assert registry.inspect_resource(base).state == "running"


def test_process_create_time_treats_zombie_as_gone(monkeypatch):
    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    class ZombieProcess(Exception):
        pass

    class FakeProcess:
        def __init__(self, pid):
            assert pid == 200

        def create_time(self):
            return 10.0

        def status(self):
            return "zombie"

    fake_psutil = SimpleNamespace(
        Process=FakeProcess,
        NoSuchProcess=NoSuchProcess,
        AccessDenied=AccessDenied,
        ZombieProcess=ZombieProcess,
        STATUS_ZOMBIE="zombie",
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert registry._process_create_time(200) is None


@pytest.mark.parametrize("grace", [math.nan, math.inf, -math.inf, True, "5"])
def test_repair_rejects_nonfinite_or_nonnumeric_grace_without_inspection(
    grace, monkeypatch,
):
    monkeypatch.setattr(
        registry,
        "list_resources",
        lambda **kwargs: pytest.fail("invalid grace must fail before inspection"),
    )

    with pytest.raises(ValueError, match="finite non-negative"):
        registry.repair_stale_resources(grace=grace)


def test_repair_removes_dead_record_without_signalling(tmp_path, monkeypatch):
    values = {200: 10.0, 100: 5.0}
    _identity_map(monkeypatch, values)
    monkeypatch.setattr(registry, "_isolated_process_group", lambda pid: pid)
    registry.register_process(
        kind="worker", pid=200, owner_pid=100, catalog_dir=tmp_path,
    )
    values[200] = None
    monkeypatch.setattr(
        registry,
        "_terminate_verified",
        lambda *args, **kwargs: pytest.fail("dead record must not be signalled"),
    )

    results = registry.repair_stale_resources(catalog_dir=tmp_path)
    assert [result.action for result in results] == ["removed_record"]
    assert registry.list_resources(catalog_dir=tmp_path) == []


def test_repair_terminates_only_verified_orphan_child(tmp_path, monkeypatch):
    values = {200: 10.0, 100: 5.0}
    _identity_map(monkeypatch, values)
    monkeypatch.setattr(registry, "_isolated_process_group", lambda pid: pid)
    resource_id = registry.register_process(
        kind="admin_job", pid=200, owner_pid=100, catalog_dir=tmp_path,
    )
    values[100] = None
    terminated: list[str] = []
    monkeypatch.setattr(
        registry,
        "_terminate_verified",
        lambda record, grace: terminated.append(record.resource_id) or "terminated",
    )

    results = registry.repair_stale_resources(catalog_dir=tmp_path)
    assert terminated == [resource_id]
    assert [result.action for result in results] == ["terminated"]
    assert registry.list_resources(catalog_dir=tmp_path) == []


def test_repair_terminates_verified_orphan_supervisor(tmp_path, monkeypatch):
    values = {200: 10.0, 100: 5.0}
    _identity_map(monkeypatch, values)
    resource_id = registry.register_process(
        kind="supervisor", pid=200, owner_pid=100, catalog_dir=tmp_path,
    )
    values[100] = None
    terminated: list[str] = []
    monkeypatch.setattr(
        registry,
        "_terminate_verified",
        lambda record, grace: terminated.append(record.resource_id) or "terminated",
    )

    results = registry.repair_stale_resources(catalog_dir=tmp_path)
    assert terminated == [resource_id]
    assert [result.action for result in results] == ["terminated"]
    assert registry.list_resources(catalog_dir=tmp_path) == []


def test_repair_reclassifies_worker_after_terminating_its_supervisor(
    tmp_path, monkeypatch,
):
    values = {300: 30.0, 200: 20.0, 100: 10.0}
    _identity_map(monkeypatch, values)
    created_at = iter((1.0, 2.0))
    monkeypatch.setattr(registry.time, "time", lambda: next(created_at))
    supervisor_id = registry.register_process(
        kind="supervisor", pid=200, owner_pid=100, catalog_dir=tmp_path,
    )
    worker_id = registry.register_process(
        kind="worker", pid=300, owner_pid=200, catalog_dir=tmp_path,
    )
    values[100] = None
    terminated: list[int] = []

    def terminate(record, grace):
        terminated.append(record.pid)
        values[record.pid] = None
        return "terminated"

    monkeypatch.setattr(registry, "_terminate_verified", terminate)

    results = registry.repair_stale_resources(catalog_dir=tmp_path)

    assert terminated == [200, 300]
    assert [(result.resource_id, result.action) for result in results] == [
        (supervisor_id, "terminated"),
        (worker_id, "terminated"),
    ]
    assert registry.list_resources(catalog_dir=tmp_path) == []


def test_repair_refuses_unknown_kind_even_if_owner_is_gone(tmp_path, monkeypatch):
    values = {200: 10.0, 100: 5.0}
    _identity_map(monkeypatch, values)
    registry.register_process(
        kind="external", pid=200, owner_pid=100, catalog_dir=tmp_path,
    )
    values[100] = None

    results = registry.repair_stale_resources(catalog_dir=tmp_path)
    assert [result.action for result in results] == ["refused"]
    assert len(registry.list_resources(catalog_dir=tmp_path)) == 1


def test_unverifiable_identity_is_never_treated_as_dead(monkeypatch):
    record = registry.ResourceRecord(
        resource_id="r",
        kind="worker",
        pid=200,
        create_time=10.0,
        owner_pid=100,
        owner_create_time=5.0,
        process_group=200,
        port=None,
        models=(),
        created_at=1.0,
    )
    monkeypatch.setattr(
        registry,
        "_process_create_time",
        lambda pid: (_ for _ in ()).throw(
            registry.ResourceIdentityUnavailable("inspection denied")
        ),
    )
    assert registry.inspect_resource(record).state == "unverifiable"


def test_repair_reports_unverifiable_identity_as_refused(tmp_path, monkeypatch):
    values = {200: 10.0, 100: 5.0}
    _identity_map(monkeypatch, values)
    resource_id = registry.register_process(
        kind="worker", pid=200, owner_pid=100, catalog_dir=tmp_path,
    )
    monkeypatch.setattr(
        registry,
        "inspect_resource",
        lambda record: registry.ResourceStatus(
            record, "unverifiable", "inspection denied",
        ),
    )

    results = registry.repair_stale_resources(catalog_dir=tmp_path)
    assert results == [registry.RepairResult(resource_id, "refused", "inspection denied")]
    assert len(registry.list_resources(catalog_dir=tmp_path)) == 1


def test_signal_revalidates_state_immediately_before_targeting(monkeypatch):
    record = registry.ResourceRecord(
        resource_id="r",
        kind="worker",
        pid=200,
        create_time=10.0,
        owner_pid=100,
        owner_create_time=5.0,
        process_group=200,
        port=None,
        models=(),
        created_at=1.0,
    )
    monkeypatch.setattr(
        registry,
        "inspect_resource",
        lambda value: registry.ResourceStatus(value, "running", "owner recovered"),
    )
    signal_calls: list[tuple[int, signal.Signals]] = []
    closed: list[int] = []
    monkeypatch.setattr(registry, "_pidfd_open", lambda pid: 77)
    monkeypatch.setattr(
        registry, "_pidfd_signal", lambda fd, sig: signal_calls.append((fd, sig)),
    )
    monkeypatch.setattr(registry, "_pidfd_close", closed.append)
    monkeypatch.setattr(
        registry.os, "kill", lambda *args: pytest.fail("numeric kill is forbidden"),
    )
    monkeypatch.setattr(
        registry.os, "killpg", lambda *args: pytest.fail("group kill is forbidden"),
    )

    with pytest.raises(registry.ResourceRegistryError, match="changed state"):
        registry._signal_verified(record, signal.SIGTERM)
    assert signal_calls == []
    assert closed == [77]


def test_signal_uses_identity_bound_handle_and_never_numeric_group(monkeypatch):
    record = registry.ResourceRecord(
        resource_id="r",
        kind="worker",
        pid=200,
        create_time=10.0,
        owner_pid=100,
        owner_create_time=5.0,
        process_group=200,
        port=None,
        models=(),
        created_at=1.0,
    )
    monkeypatch.setattr(
        registry,
        "inspect_resource",
        lambda value: registry.ResourceStatus(value, "orphaned", "owner gone"),
    )
    events: list[tuple] = []
    monkeypatch.setattr(
        registry, "_pidfd_open", lambda pid: events.append(("open", pid)) or 88,
    )
    monkeypatch.setattr(
        registry,
        "_pidfd_signal",
        lambda fd, sig: events.append(("signal", fd, sig)),
    )
    monkeypatch.setattr(
        registry, "_pidfd_close", lambda fd: events.append(("close", fd)),
    )
    monkeypatch.setattr(
        registry.os, "kill", lambda *args: pytest.fail("numeric kill is forbidden"),
    )
    monkeypatch.setattr(
        registry.os, "killpg", lambda *args: pytest.fail("group kill is forbidden"),
    )

    registry._signal_verified(record, signal.SIGTERM)

    assert events == [
        ("open", 200),
        ("signal", 88, signal.SIGTERM),
        ("close", 88),
    ]


def test_signal_refuses_when_identity_bound_handles_are_unavailable(monkeypatch):
    record = registry.ResourceRecord(
        resource_id="r",
        kind="worker",
        pid=200,
        create_time=10.0,
        owner_pid=100,
        owner_create_time=5.0,
        process_group=200,
        port=None,
        models=(),
        created_at=1.0,
    )
    monkeypatch.setattr(
        registry,
        "_pidfd_open",
        lambda pid: (_ for _ in ()).throw(
            registry.ResourceIdentityUnavailable("pidfd unavailable")
        ),
    )
    monkeypatch.setattr(
        registry.os, "kill", lambda *args: pytest.fail("numeric kill is forbidden"),
    )
    monkeypatch.setattr(
        registry.os, "killpg", lambda *args: pytest.fail("group kill is forbidden"),
    )

    with pytest.raises(registry.ResourceIdentityUnavailable, match="pidfd unavailable"):
        registry._signal_verified(record, signal.SIGTERM)


def test_pidfd_wrappers_use_feature_detected_identity_bound_calls(monkeypatch):
    opened: list[tuple[int, int]] = []
    sent: list[tuple] = []
    monkeypatch.setattr(
        registry.os,
        "pidfd_open",
        lambda pid, flags: opened.append((pid, flags)) or 77,
        raising=False,
    )
    monkeypatch.setattr(
        registry.signal,
        "pidfd_send_signal",
        lambda *args: sent.append(args),
        raising=False,
    )

    descriptor = registry._pidfd_open(200)
    registry._pidfd_signal(descriptor, signal.SIGTERM)

    assert descriptor == 77
    assert opened == [(200, 0)]
    assert sent == [(77, signal.SIGTERM, None, 0)]


def test_pidfd_open_refuses_when_kernel_wrapper_is_unavailable(monkeypatch):
    monkeypatch.setattr(registry.os, "pidfd_open", None, raising=False)
    monkeypatch.setattr(registry.signal, "pidfd_send_signal", None, raising=False)

    with pytest.raises(registry.ResourceIdentityUnavailable, match="pidfd support"):
        registry._pidfd_open(200)


def test_repair_compare_and_remove_refuses_concurrent_record_change(monkeypatch):
    record = registry.ResourceRecord(
        resource_id="r",
        kind="worker",
        pid=200,
        create_time=10.0,
        owner_pid=100,
        owner_create_time=5.0,
        process_group=200,
        port=None,
        models=(),
        created_at=1.0,
    )
    monkeypatch.setattr(
        registry,
        "list_resources",
        lambda **kwargs: [record],
    )
    monkeypatch.setattr(
        registry,
        "inspect_resource",
        lambda value: registry.ResourceStatus(value, "dead", "process gone"),
    )
    seen: list[registry.ResourceRecord | None] = []
    monkeypatch.setattr(
        registry,
        "unregister_process",
        lambda resource_id, **kwargs: seen.append(kwargs.get("expected")) or False,
    )

    assert registry.repair_stale_resources() == [
        registry.RepairResult("r", "refused", "resource record changed during repair")
    ]
    assert seen == [record]


def test_repair_reports_post_termination_record_change_as_refused(monkeypatch):
    record = registry.ResourceRecord(
        resource_id="r",
        kind="worker",
        pid=200,
        create_time=10.0,
        owner_pid=100,
        owner_create_time=5.0,
        process_group=200,
        port=None,
        models=(),
        created_at=1.0,
    )
    monkeypatch.setattr(
        registry,
        "list_resources",
        lambda **kwargs: [record],
    )
    monkeypatch.setattr(
        registry,
        "inspect_resource",
        lambda value: registry.ResourceStatus(value, "orphaned", "owner gone"),
    )
    monkeypatch.setattr(registry, "_terminate_verified", lambda *args: "terminated")
    monkeypatch.setattr(registry, "unregister_process", lambda *args, **kwargs: False)

    assert registry.repair_stale_resources() == [
        registry.RepairResult(
            "r",
            "refused",
            "process was terminated but resource record changed during repair",
        )
    ]
