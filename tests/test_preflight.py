"""Tests for scripts/preflight.py (loaded as a module; stdlib-only)."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"


def _load():
    spec = importlib.util.spec_from_file_location("preflight", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_only_exits_zero_and_does_not_run_pytest(monkeypatch):
    pf = _load()
    monkeypatch.setattr(pf, "missing_deps", lambda: [])
    called = {}
    monkeypatch.setattr(pf.subprocess, "run",
                        lambda *a, **k: called.setdefault("ran", True))
    assert pf.main(["--check-only"]) == 0
    assert "ran" not in called


def test_missing_dep_exits_nonzero_and_prints_install_cmd(monkeypatch, capsys):
    pf = _load()
    monkeypatch.setattr(pf, "missing_deps",
                        lambda: [("torch", "audio", "torch")])
    rc = pf.main(["--check-only"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "torch" in err
    assert "download.pytorch.org/whl/cpu" in err


def test_required_covers_mcp_transport_and_sse_versions():
    pf = _load()
    requirements = {import_name: requirement for import_name, _, requirement in pf.REQUIRED}
    assert requirements["mcp.server.streamable_http_manager"] == "mcp>=1.8.0,<2"
    assert requirements["sse_starlette"] == "sse-starlette>=3.0.0,<3.1.0"


def test_importable_but_incompatible_version_is_reported(monkeypatch):
    pf = _load()
    monkeypatch.setattr(pf.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(
        pf,
        "_requirement_satisfied",
        lambda requirement: requirement != "mcp>=1.8.0,<2",
    )

    assert (
        "mcp.server.streamable_http_manager",
        "server",
        "mcp>=1.8.0,<2",
    ) in pf.missing_deps()


@pytest.mark.parametrize(
    ("installed", "satisfied"),
    [("1.7.9", False), ("1.8.0", True), ("1.27.2", True), ("2.0.0", False)],
)
def test_requirement_satisfied_enforces_mcp_interval(
    monkeypatch, installed, satisfied,
):
    pf = _load()
    monkeypatch.setattr(pf.importlib.metadata, "version", lambda _name: installed)
    assert pf._requirement_satisfied("mcp>=1.8.0,<2") is satisfied


def test_missing_packaging_is_reported_without_version_checks(monkeypatch):
    pf = _load()
    monkeypatch.setattr(pf, "Requirement", None)
    assert pf.missing_deps() == [
        ("packaging", "dev", "packaging>=23.0"),
    ]


def test_runs_fast_lane_when_all_present(monkeypatch):
    pf = _load()
    monkeypatch.setattr(pf, "missing_deps", lambda: [])
    captured = {}

    class FakeProc:
        returncode = 0

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(pf.subprocess, "run", fake_run)
    assert pf.main([]) == 0
    assert "-m" in captured["cmd"]
    assert "not slow" in captured["cmd"]


def test_forwards_trailing_args_to_pytest(monkeypatch):
    pf = _load()
    monkeypatch.setattr(pf, "missing_deps", lambda: [])
    captured = {}

    class FakeProc:
        returncode = 0

    monkeypatch.setattr(pf.subprocess, "run",
                        lambda cmd, *a, **k: captured.__setitem__("cmd", cmd) or FakeProc())
    assert pf.main(["--", "-k", "resolver"]) == 0
    assert captured["cmd"][-2:] == ["-k", "resolver"]
    assert "--" not in captured["cmd"]


def test_propagates_pytest_returncode(monkeypatch):
    pf = _load()
    monkeypatch.setattr(pf, "missing_deps", lambda: [])

    class FakeProc:
        returncode = 5

    monkeypatch.setattr(pf.subprocess, "run", lambda *a, **k: FakeProc())
    assert pf.main([]) == 5
