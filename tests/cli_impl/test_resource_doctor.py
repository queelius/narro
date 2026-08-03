from __future__ import annotations

from muse.cli_impl import resource_doctor
from muse.core.resource_registry import RepairResult, ResourceRecord, ResourceStatus


def _status(state: str) -> ResourceStatus:
    record = ResourceRecord(
        resource_id="abc",
        kind="worker",
        pid=123,
        create_time=10.0,
        owner_pid=100,
        owner_create_time=5.0,
        process_group=123,
        port=9001,
        models=("demo",),
        created_at=1.0,
    )
    return ResourceStatus(record, state, "detail")


def test_doctor_empty_is_success(monkeypatch, capsys):
    monkeypatch.setattr(resource_doctor, "inspect_resources", lambda: [])
    assert resource_doctor.run_resource_doctor() == 0
    assert "no Muse-owned" in capsys.readouterr().out


def test_doctor_reports_stale_nonzero_without_repair(monkeypatch, capsys):
    monkeypatch.setattr(resource_doctor, "inspect_resources", lambda: [_status("dead")])
    assert resource_doctor.run_resource_doctor() == 1
    output = capsys.readouterr().out
    assert "worker" in output
    assert "dead" in output
    assert "demo" in output


def test_doctor_repair_surfaces_refusal(monkeypatch, capsys):
    monkeypatch.setattr(resource_doctor, "inspect_resources", lambda: [_status("orphaned")])
    monkeypatch.setattr(
        resource_doctor,
        "repair_stale_resources",
        lambda grace: [RepairResult("abc", "refused", "identity changed")],
    )
    assert resource_doctor.run_resource_doctor(repair=True) == 2
    assert "refused" in capsys.readouterr().out
