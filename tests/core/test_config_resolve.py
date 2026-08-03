import pytest
from muse.core import config as cfg


@pytest.fixture(autouse=True)
def _reset():
    cfg.reset_config()
    yield
    cfg.reset_config()


def _cfg(tmp_path, text=None):
    p = tmp_path / "config.yaml"
    if text is not None:
        p.write_text(text)
    return cfg.Config(path=p)


def test_default_when_nothing_set(tmp_path):
    c = _cfg(tmp_path)
    assert c.get("limits.rerank_max_documents") == 1000
    assert c.source("limits.rerank_max_documents") == "default"


def test_env_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "5")
    c = _cfg(tmp_path)
    assert c.get("limits.rerank_max_documents") == 5
    assert c.source("limits.rerank_max_documents") == "env"


def test_file_overrides_default(tmp_path):
    c = _cfg(tmp_path, "limits:\n  rerank_max_documents: 7\n")
    assert c.get("limits.rerank_max_documents") == 7
    assert c.source("limits.rerank_max_documents") == "file"


def test_env_beats_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "9")
    c = _cfg(tmp_path, "limits:\n  rerank_max_documents: 7\n")
    assert c.get("limits.rerank_max_documents") == 9


def test_override_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "9")
    c = _cfg(tmp_path)
    assert c.get("limits.rerank_max_documents", override=3) == 3
    assert c.source("limits.rerank_max_documents") == "env"  # source ignores per-call override


def test_env_live_reread(tmp_path, monkeypatch):
    c = _cfg(tmp_path)
    assert c.get("limits.rerank_max_documents") == 1000
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "11")
    assert c.get("limits.rerank_max_documents") == 11  # not cached


def test_bad_env_warns_and_defaults(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("MUSE_RERANK_MAX_DOCUMENTS", "abc")
    c = _cfg(tmp_path)
    with caplog.at_level("WARNING"):
        assert c.get("limits.rerank_max_documents") == 1000  # lenient
    assert any("MUSE_RERANK_MAX_DOCUMENTS" in r.message for r in caplog.records)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
def test_non_finite_env_warns_and_defaults(
    tmp_path, monkeypatch, caplog, raw,
):
    monkeypatch.setenv("MUSE_FEDERATION_REFRESH_INTERVAL_SECONDS", raw)
    c = _cfg(tmp_path)
    with caplog.at_level("WARNING"):
        assert c.get("federation.refresh_interval_seconds") == 3.0
    assert any("must be finite" in r.message for r in caplog.records)


def test_out_of_domain_env_warns_and_defaults(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("MUSE_FEDERATION_REFRESH_INTERVAL_SECONDS", "0")
    c = _cfg(tmp_path)
    with caplog.at_level("WARNING"):
        assert c.get("federation.refresh_interval_seconds") == 3.0
    assert any("must be > 0" in r.message for r in caplog.records)


def test_max_request_body_env_is_validated(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("MUSE_MAX_REQUEST_BODY_MB", "32")
    c = _cfg(tmp_path)
    assert c.get("server.max_request_body_mb") == 32

    monkeypatch.setenv("MUSE_MAX_REQUEST_BODY_MB", "0")
    with caplog.at_level("WARNING"):
        assert c.get("server.max_request_body_mb") == 64
    assert any("MUSE_MAX_REQUEST_BODY_MB" in r.message for r in caplog.records)


def test_out_of_domain_file_warns_and_defaults(tmp_path, caplog):
    c = _cfg(tmp_path, "telemetry:\n  sample_interval_seconds: -1\n")
    with caplog.at_level("WARNING"):
        assert c.get("telemetry.sample_interval_seconds") == 10.0
    assert any("must be > 0" in r.message for r in caplog.records)


def test_invalid_programmatic_override_warns_and_defaults(tmp_path, caplog):
    c = _cfg(tmp_path)
    with caplog.at_level("WARNING"):
        assert c.get("server.max_queue_depth", override=-1) == 256
    assert any("must be >= 0" in r.message for r in caplog.records)


def test_enum_env_is_canonicalized_and_invalid_value_defaults(
    tmp_path, monkeypatch, caplog,
):
    monkeypatch.setenv("MUSE_DEVICE", " CUDA ")
    c = _cfg(tmp_path)
    assert c.get("server.device") == "cuda"

    monkeypatch.setenv("MUSE_DEVICE", "tpu")
    with caplog.at_level("WARNING"):
        assert c.get("server.device") == "auto"
    assert any("must be one of" in r.message for r in caplog.records)


def test_legacy_video_offload_false_alias_still_forces_off(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("MUSE_VIDEO_CPU_OFFLOAD", "0")
    c = _cfg(tmp_path)
    assert c.get("server.video_cpu_offload") == "0"

    cfg.reset_config()
    from muse.modalities.video_generation.runtimes._offload import (
        resolve_offload_mode,
    )

    assert resolve_offload_mode("sequential") is None


def test_documented_zero_and_negative_disable_values_are_preserved(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("MUSE_QUEUE_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS", "-1")
    c = _cfg(tmp_path)
    assert c.get("server.queue_timeout_seconds") == 0.0
    assert c.get("server.idle_timeout_seconds") == -1.0


def test_opt_float_empty_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_SHUTDOWN_GRACE_SECONDS", "")
    c = _cfg(tmp_path)
    assert c.get("server.shutdown_grace_seconds") is None


def test_idle_timeout_default_600(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS", raising=False)
    c = _cfg(tmp_path)
    assert c.get("server.idle_timeout_seconds") == 600.0


def test_unknown_file_key_ignored_with_warning(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        c = _cfg(tmp_path, "limits:\n  bogus_key: 1\nnope:\n  x: 2\n")
        c.file_values()
    msgs = " ".join(r.message for r in caplog.records)
    assert "bogus_key" in msgs and "nope" in msgs


def test_unknown_key_raises_keyerror(tmp_path):
    c = _cfg(tmp_path)
    with pytest.raises(KeyError):
        c.get("no.such.key")


def test_singleton_and_reset(monkeypatch):
    a = cfg.get_config()
    b = cfg.get_config()
    assert a is b
    cfg.reset_config()
    assert cfg.get_config() is not a


def test_yaml_null_opt_setting_is_none(tmp_path):
    c = _cfg(tmp_path, "server:\n  idle_timeout_seconds: null\n")
    assert c.get("server.idle_timeout_seconds") is None
    assert c.source("server.idle_timeout_seconds") == "file"


def test_yaml_null_non_opt_setting_warns_and_defaults(tmp_path, caplog):
    c = _cfg(tmp_path, "server:\n  gpu_headroom_gb: null\n")
    with caplog.at_level("WARNING"):
        assert c.get("server.gpu_headroom_gb") == 1.0  # default
    assert any("gpu_headroom_gb" in r.message for r in caplog.records)


def test_strict_config_rejects_malformed_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("server: [\n")

    with pytest.raises(cfg.ConfigError, match="cannot parse"):
        cfg.Config(path=path, strict=True).validate()


def test_strict_config_rejects_unknown_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("server:\n  typo_headroom_gb: 1\n")

    with pytest.raises(cfg.ConfigError, match="unknown config key"):
        cfg.Config(path=path, strict=True).validate()


def test_strict_config_validates_every_file_value(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("telemetry:\n  sample_interval_seconds: -1\n")

    with pytest.raises(cfg.ConfigError, match="must be > 0"):
        cfg.Config(path=path, strict=True).validate()


def test_strict_config_rejects_invalid_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_MAX_REQUEST_BODY_MB", "not-an-int")

    with pytest.raises(cfg.ConfigError, match="MUSE_MAX_REQUEST_BODY_MB"):
        cfg.Config(
            path=tmp_path / "missing.yaml", strict=True,
        ).validate()


def test_config_path_is_directory_degrades_to_empty(tmp_path):
    c = cfg.Config(path=tmp_path)
    assert c.file_values() == {}
    assert c.get("limits.rerank_max_documents") == 1000


def test_owned_config_read_hardens_legacy_permissions(tmp_path):
    import stat

    path = tmp_path / "config.yaml"
    path.write_text("limits:\n  rerank_max_documents: 7\n")
    path.chmod(0o644)

    c = cfg.Config(path=path)

    assert c.get("limits.rerank_max_documents") == 7
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_owned_group_writable_config_is_hardened_before_use(tmp_path):
    import stat

    path = tmp_path / "config.yaml"
    path.write_text("limits:\n  rerank_max_documents: 7\n")
    path.chmod(0o660)

    c = cfg.Config(path=path)

    assert c.get("limits.rerank_max_documents") == 7
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_symlink_config_read_degrades_to_defaults_without_following(tmp_path):
    external = tmp_path / "external.yaml"
    external.write_text("limits:\n  rerank_max_documents: 7\n")
    path = tmp_path / "config.yaml"
    path.symlink_to(external)

    c = cfg.Config(path=path)

    assert c.get("limits.rerank_max_documents") == 1000
    assert path.is_symlink()


# --- bootstrap-key invariant: paths.catalog_dir / paths.config_file are
# documented as env+default ONLY (the file cannot redirect the path that
# locates itself). Config.get and .source must never let a config.yaml
# value win for these two keys.


def test_bootstrap_catalog_dir_ignores_file_value(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSE_CATALOG_DIR", raising=False)
    c = _cfg(tmp_path, "paths:\n  catalog_dir: /tmp/somewhere-else\n")
    assert c.get("paths.catalog_dir") == "~/.muse"
    assert c.source("paths.catalog_dir") == "default"


def test_bootstrap_config_file_ignores_file_value(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSE_CONFIG", raising=False)
    c = _cfg(tmp_path, "paths:\n  config_file: /tmp/other-config.yaml\n")
    assert c.get("paths.config_file") is None
    assert c.source("paths.config_file") == "default"


def test_bootstrap_catalog_dir_env_still_wins_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", "/env/dir")
    c = _cfg(tmp_path, "paths:\n  catalog_dir: /tmp/somewhere-else\n")
    assert c.get("paths.catalog_dir") == "/env/dir"
    assert c.source("paths.catalog_dir") == "env"


def test_non_bootstrap_key_still_reads_file(tmp_path):
    """Sanity: the bootstrap-key skip must not disable file reads for
    ordinary settings."""
    c = _cfg(tmp_path, "limits:\n  rerank_max_documents: 7\n")
    assert c.get("limits.rerank_max_documents") == 7
    assert c.source("limits.rerank_max_documents") == "file"
