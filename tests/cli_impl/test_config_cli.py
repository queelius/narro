"""Tests for the `muse config` CLI group (generate/show/path/get/set).

Mirrors the existing `muse models` CLI test conventions: invoked
in-process via typer's CliRunner against the real `app`, with
MUSE_CATALOG_DIR pointed at a tmp_path so nothing touches ~/.muse.
"""
import json
import stat

import yaml
from typer.testing import CliRunner

from muse.core import config as cfg
from muse.cli import app

runner = CliRunner()


def test_config_path(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["config", "path"])
    assert r.exit_code == 0
    assert str(tmp_path / "config.yaml") in r.stdout


def test_config_generate_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["config", "generate"])
    assert r.exit_code == 0
    body = (tmp_path / "config.yaml").read_text()
    assert "server:" in body
    # refuses overwrite without --force
    r2 = runner.invoke(app, ["config", "generate"])
    assert r2.exit_code != 0
    r3 = runner.invoke(app, ["config", "generate", "--force"])
    assert r3.exit_code == 0


def test_config_generate_hardens_catalog_and_file_permissions(
    monkeypatch, tmp_path,
):
    tmp_path.chmod(0o755)
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()

    r = runner.invoke(app, ["config", "generate"])

    assert r.exit_code == 0
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "config.yaml").stat().st_mode) == 0o600


def test_config_get(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "5")
    cfg.reset_config()
    r = runner.invoke(app, ["config", "get", "limits.rerank_max_documents"])
    assert r.exit_code == 0 and "5" in r.stdout


def test_config_get_unknown_key_exits_2(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["config", "get", "nope.not_a_key"])
    assert r.exit_code == 2


def test_config_set_then_show_json(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["config", "set", "server.gpu_headroom_gb", "2.5"])
    assert r.exit_code == 0
    data = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert data["server"]["gpu_headroom_gb"] == 2.5
    cfg.reset_config()
    r2 = runner.invoke(app, ["config", "show", "--json"])
    assert r2.exit_code == 0
    rows = json.loads(r2.stdout)
    row = next(x for x in rows if x["key"] == "server.gpu_headroom_gb")
    assert row["value"] == 2.5 and row["source"] == "file"


def test_config_set_bad_value_nonzero(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["config", "set", "limits.rerank_max_documents", "abc"])
    assert r.exit_code != 0


def test_config_set_out_of_domain_value_nonzero(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(
        app, ["config", "set", "federation.refresh_interval_seconds", "0"]
    )
    assert r.exit_code == 2
    assert "must be > 0" in r.stderr


def test_config_set_unknown_key_exits_2(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["config", "set", "nope.not_a_key", "5"])
    assert r.exit_code == 2


def test_config_show_redacts_admin_token(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "super-secret-value")
    cfg.reset_config()
    r = runner.invoke(app, ["config", "show", "--json"])
    assert "super-secret-value" not in r.stdout


def test_config_show_redacts_admin_token_table(monkeypatch, tmp_path):
    """Redaction must also apply to the non-JSON (plain/table) render path."""
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "super-secret-value")
    cfg.reset_config()
    r = runner.invoke(app, ["config", "show"])
    assert r.exit_code == 0
    assert "super-secret-value" not in r.stdout


def test_config_show_admin_token_set_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    cfg.reset_config()
    r = runner.invoke(app, ["config", "show", "--json"])
    assert r.exit_code == 0
    rows = json.loads(r.stdout)
    row = next(x for x in rows if x["key"] == "admin.token")
    assert row["value"] == "unset"

    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "abc123")
    cfg.reset_config()
    r2 = runner.invoke(app, ["config", "show", "--json"])
    rows2 = json.loads(r2.stdout)
    row2 = next(x for x in rows2 if x["key"] == "admin.token")
    assert row2["value"] == "set"


def test_config_get_admin_token_redacted(monkeypatch, tmp_path):
    """`config get admin.token` must never echo the real token value."""
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "super-secret-abc")
    cfg.reset_config()
    r = runner.invoke(app, ["config", "get", "admin.token"])
    assert r.exit_code == 0
    assert "super-secret-abc" not in r.stdout
    assert r.stdout.strip() == "set"

    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    cfg.reset_config()
    r2 = runner.invoke(app, ["config", "get", "admin.token"])
    assert r2.exit_code == 0
    assert r2.stdout.strip() == "unset"


def test_config_show_and_get_agree_on_token_redaction(monkeypatch, tmp_path):
    """`config show --json` and `config get` must render identical
    redaction for admin.token, proving the two surfaces share one
    redaction source instead of two copies that could drift."""
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.setenv("MUSE_ADMIN_TOKEN", "another-secret-xyz")
    cfg.reset_config()

    r_show = runner.invoke(app, ["config", "show", "--json"])
    assert r_show.exit_code == 0
    rows = json.loads(r_show.stdout)
    row = next(x for x in rows if x["key"] == "admin.token")
    assert row["value"] == "set"
    assert "another-secret-xyz" not in r_show.stdout

    cfg.reset_config()
    r_get = runner.invoke(app, ["config", "get", "admin.token"])
    assert r_get.exit_code == 0
    assert r_get.stdout.strip() == "set"
    assert "another-secret-xyz" not in r_get.stdout


def test_config_unset(monkeypatch, tmp_path):
    import yaml
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    runner.invoke(app, ["config", "set", "server.gpu_headroom_gb", "2.5"])
    r = runner.invoke(app, ["config", "unset", "server.gpu_headroom_gb"])
    assert r.exit_code == 0
    cfg.reset_config()
    data = yaml.safe_load((tmp_path / "config.yaml").read_text()) or {}
    assert "gpu_headroom_gb" not in data.get("server", {})


def test_config_unset_unknown_key_nonzero(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    cfg.reset_config()
    r = runner.invoke(app, ["config", "unset", "no.such.key"])
    assert r.exit_code != 0
