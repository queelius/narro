import os
import pytest
from muse.core import config as cfg


def test_registry_keys_unique_and_dotted():
    keys = [s.key for s in cfg.SETTINGS]
    assert len(keys) == len(set(keys)), "duplicate keys"
    assert all("." in k for k in keys), "keys must be group.leaf"


def test_registry_envs_unique_and_prefixed():
    envs = [s.env for s in cfg.SETTINGS]
    assert len(envs) == len(set(envs)), "duplicate env vars"
    assert all(e.startswith("MUSE_") for e in envs)


def test_key_group_matches_prefix():
    for s in cfg.SETTINGS:
        assert s.key.split(".")[0] == s.group


def test_lookup_maps_cover_all():
    assert set(cfg.SETTINGS_BY_KEY) == {s.key for s in cfg.SETTINGS}
    assert set(cfg.SETTINGS_BY_ENV) == {s.env for s in cfg.SETTINGS}


def test_expected_settings_present():
    # spot-check representative rows across groups
    for key in (
        "server.idle_timeout_seconds",
        "server.gpu_headroom_gb",
        "server.video_cpu_offload",
        "admin.token",
        "client.server_url",
        "paths.catalog_dir",
        "fetch.allow_private",
        "limits.upscale_max_input_side",
        "limits.vectorization_max_input_side",
        "limits.audio_quality_max_bytes",
        "limits.audio_quality_max_duration_seconds",
        "limits.audio_alignment_max_bytes",
        "limits.audio_alignment_max_duration_seconds",
        "limits.audio_alignment_max_text_chars",
        "limits.rerank_max_documents",
    ):
        assert key in cfg.SETTINGS_BY_KEY, key


def test_video_cpu_offload_setting_shape():
    s = cfg.SETTINGS_BY_KEY["server.video_cpu_offload"]
    assert s.env == "MUSE_VIDEO_CPU_OFFLOAD"
    assert s.type == "opt_str"
    assert s.default is None
    assert s.group == "server"


def test_idle_timeout_default_is_600():
    assert cfg.SETTINGS_BY_KEY["server.idle_timeout_seconds"].default == 600.0


@pytest.mark.parametrize("t,raw,expected", [
    ("int", "42", 42),
    ("float", "1.5", 1.5),
    ("str", "hi", "hi"),
    ("bool", "1", True),
    ("bool", "true", True),
    ("bool", "0", False),
    ("bool", "false", False),
    ("opt_int", "", None),
    ("opt_int", "7", 7),
    ("opt_float", "", None),
    ("opt_str", "", None),
    ("opt_str", "x", "x"),
])
def test_coerce_types(t, raw, expected):
    s = cfg.Setting(key="g.k", env="MUSE_K", type=t, default=None, group="g", help="h")
    assert cfg.coerce(s, raw) == expected


def test_coerce_bad_int_raises():
    s = cfg.Setting(key="g.k", env="MUSE_K", type="int", default=0, group="g", help="h")
    with pytest.raises(cfg.ConfigError):
        cfg.coerce(s, "abc")


def test_catalog_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    assert cfg._catalog_dir() == tmp_path


def test_config_path_defaults_under_catalog_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.delenv("MUSE_CONFIG", raising=False)
    assert cfg.config_path() == tmp_path / "config.yaml"


def test_config_path_explicit_override(monkeypatch, tmp_path):
    p = tmp_path / "custom.yaml"
    monkeypatch.setenv("MUSE_CONFIG", str(p))
    assert cfg.config_path() == p


def test_telemetry_settings_present():
    from muse.core import config as cfg
    for key, default in [
        ("telemetry.enabled", True),
        ("telemetry.retention_days", 7),
        ("telemetry.log_buffer_kb", 64),
        ("telemetry.sample_interval_seconds", 10.0),
    ]:
        assert key in cfg.SETTINGS_BY_KEY, key
        assert cfg.SETTINGS_BY_KEY[key].default == default
