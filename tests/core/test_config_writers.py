import os
import stat
import threading

import pytest
import yaml
from muse.core import config as cfg


def test_template_has_every_setting():
    body = cfg.render_template()
    for s in cfg.SETTINGS:
        assert s.env in body           # env name in a comment
    # parseable once comment-only lines / commented bootstrap are stripped by yaml
    assert "server:" in body and "limits:" in body


def test_set_value_creates_and_coerces(tmp_path):
    p = tmp_path / "config.yaml"
    out = cfg.set_value("limits.rerank_max_documents", "42", path=p)
    assert out == 42
    data = yaml.safe_load(p.read_text())
    assert data["limits"]["rerank_max_documents"] == 42


def test_set_value_preserves_other_keys(tmp_path):
    p = tmp_path / "config.yaml"
    cfg.set_value("limits.rerank_max_documents", "42", path=p)
    cfg.set_value("server.gpu_headroom_gb", "2.5", path=p)
    data = yaml.safe_load(p.read_text())
    assert data["limits"]["rerank_max_documents"] == 42
    assert data["server"]["gpu_headroom_gb"] == 2.5


def test_set_value_bad_value_raises_and_no_write(tmp_path):
    p = tmp_path / "config.yaml"
    with pytest.raises(cfg.ConfigError):
        cfg.set_value("limits.rerank_max_documents", "abc", path=p)
    assert not p.exists()


@pytest.mark.parametrize(
    "key,raw",
    [
        ("limits.rerank_max_documents", "-1"),
        ("server.aggregation_timeout_seconds", "0"),
        ("server.max_request_body_mb", "0"),
        ("server.gpu_headroom_gb", "nan"),
        ("server.device", "tpu"),
    ],
)
def test_set_value_rejects_out_of_domain_value_without_writing(
    tmp_path, key, raw,
):
    p = tmp_path / "config.yaml"
    with pytest.raises(cfg.ConfigError):
        cfg.set_value(key, raw, path=p)
    assert not p.exists()


def test_set_value_canonicalizes_enum(tmp_path):
    p = tmp_path / "config.yaml"
    assert cfg.set_value("server.device", " CUDA ", path=p) == "cuda"
    assert yaml.safe_load(p.read_text())["server"]["device"] == "cuda"


def test_set_value_creates_private_directory_and_file_despite_umask(tmp_path):
    p = tmp_path / "private" / "config.yaml"
    previous_umask = os.umask(0)
    try:
        cfg.set_value("admin.token", "secret", path=p)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(p.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_set_and_unset_restore_private_file_mode(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("server:\n  gpu_headroom_gb: 2.5\n")
    p.chmod(0o644)

    cfg.set_value("server.cpu_headroom_gb", "3.0", path=p)
    assert stat.S_IMODE(p.stat().st_mode) == 0o600

    p.chmod(0o644)
    assert cfg.unset_value("server.cpu_headroom_gb", path=p) is True
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_set_value_refuses_symlink_config_without_touching_target(tmp_path):
    external = tmp_path / "external.yaml"
    external.write_text("admin:\n  token: original\n")
    p = tmp_path / "config.yaml"
    p.symlink_to(external)

    with pytest.raises(cfg.ConfigError, match="safely read|symlink"):
        cfg.set_value("admin.token", "replacement", path=p)

    assert external.read_text() == "admin:\n  token: original\n"
    assert p.is_symlink()


def test_set_value_refuses_symlink_transaction_lock(tmp_path):
    external = tmp_path / "external-lock"
    external.write_text("do not touch")
    (tmp_path / ".config.yaml.lock").symlink_to(external)

    with pytest.raises(cfg.ConfigError, match="config lock"):
        cfg.set_value("admin.token", "secret", path=tmp_path / "config.yaml")

    assert external.read_text() == "do not touch"


def test_set_value_refuses_symlink_parent_without_touching_target(tmp_path):
    external = tmp_path / "external-dir"
    external.mkdir()
    linked_parent = tmp_path / "linked-config"
    linked_parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(cfg.ConfigError, match="config parent"):
        cfg.set_value(
            "admin.token", "secret", path=linked_parent / "config.yaml",
        )

    assert list(external.iterdir()) == []
    assert linked_parent.is_symlink()


def test_set_value_refuses_group_writable_existing_parent(tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o770)

    with pytest.raises(cfg.ConfigError, match="group/other writable"):
        cfg.set_value("admin.token", "secret", path=parent / "config.yaml")

    assert list(parent.iterdir()) == []


def test_set_value_refuses_hardlinked_config(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text("admin:\n  token: original\n")
    target = tmp_path / "config.yaml"
    os.link(source, target)

    with pytest.raises(cfg.ConfigError, match="multiple links"):
        cfg.set_value("admin.token", "replacement", path=target)

    assert source.read_text() == "admin:\n  token: original\n"


def test_concurrent_config_updates_preserve_distinct_keys(tmp_path):
    p = tmp_path / "config.yaml"
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def update(key: str, value: str) -> None:
        try:
            barrier.wait()
            cfg.set_value(key, value, path=p)
        except BaseException as exc:  # pragma: no cover - regression detail
            errors.append(exc)

    threads = [
        threading.Thread(
            target=update,
            args=("server.gpu_headroom_gb", "2.5"),
        ),
        threading.Thread(
            target=update,
            args=("server.cpu_headroom_gb", "3.5"),
        ),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert not any(thread.is_alive() for thread in threads)
    data = yaml.safe_load(p.read_text())
    assert data["server"] == {
        "cpu_headroom_gb": 3.5,
        "gpu_headroom_gb": 2.5,
    }


def test_concurrent_create_config_never_clobbers_winner(tmp_path):
    target = tmp_path / "config.yaml"
    barrier = threading.Barrier(3)
    outcomes: list[tuple[str, bool]] = []

    def create(text: str) -> None:
        barrier.wait()
        outcomes.append((text, cfg.create_config_text(text, path=target)))

    threads = [
        threading.Thread(target=create, args=("first\n",)),
        threading.Thread(target=create, args=("second\n",)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2.0)

    assert not any(thread.is_alive() for thread in threads)
    assert sorted(created for _, created in outcomes) == [False, True]
    winner = next(text for text, created in outcomes if created)
    assert target.read_text() == winner


def test_create_config_refuses_existing_symlink_without_touching_target(tmp_path):
    external = tmp_path / "external.yaml"
    external.write_text("original\n")
    target = tmp_path / "config.yaml"
    target.symlink_to(external)

    assert cfg.create_config_text("replacement\n", path=target) is False
    assert target.is_symlink()
    assert external.read_text() == "original\n"


def test_set_value_unknown_key_raises(tmp_path):
    with pytest.raises(KeyError):
        cfg.set_value("no.such.key", "1", path=tmp_path / "config.yaml")


def test_template_is_valid_yaml_and_roundtrips():
    import yaml
    from muse.core import config as cfg
    body = cfg.render_template()
    # no bare document-end markers
    assert not any(line.strip() == "..." for line in body.splitlines())
    data = yaml.safe_load(body)          # must parse without raising
    assert isinstance(data, dict)
    # active (non-bootstrap) settings round-trip to their declared default
    for key in ("server.idle_timeout_seconds", "limits.rerank_max_documents",
                "client.server_url", "fetch.allow_private"):
        group, leaf = key.split(".", 1)
        assert data[group][leaf] == cfg.SETTINGS_BY_KEY[key].default
    # bootstrap keys are commented out -> NOT present as active keys
    assert "catalog_dir" not in data.get("paths", {})
    assert "config_file" not in data.get("paths", {})


# --- unset_value: remove a key so it falls back to env/default ---

def test_unset_value_removes_key_preserves_others(tmp_path):
    import yaml
    from muse.core import config as cfg
    p = tmp_path / "config.yaml"
    cfg.set_value("limits.rerank_max_documents", "42", path=p)
    cfg.set_value("server.gpu_headroom_gb", "2.5", path=p)
    assert cfg.unset_value("limits.rerank_max_documents", path=p) is True
    data = yaml.safe_load(p.read_text())
    assert "rerank_max_documents" not in data.get("limits", {})
    assert data["server"]["gpu_headroom_gb"] == 2.5


def test_unset_value_prunes_empty_group(tmp_path):
    import yaml
    from muse.core import config as cfg
    p = tmp_path / "config.yaml"
    cfg.set_value("limits.rerank_max_documents", "42", path=p)
    cfg.unset_value("limits.rerank_max_documents", path=p)
    data = yaml.safe_load(p.read_text()) or {}
    assert "limits" not in data


def test_unset_value_absent_key_is_noop(tmp_path):
    import yaml
    from muse.core import config as cfg
    p = tmp_path / "config.yaml"
    cfg.set_value("server.gpu_headroom_gb", "2.5", path=p)
    assert cfg.unset_value("limits.rerank_max_documents", path=p) is False
    assert yaml.safe_load(p.read_text())["server"]["gpu_headroom_gb"] == 2.5


def test_unset_value_no_file_is_noop(tmp_path):
    from muse.core import config as cfg
    p = tmp_path / "config.yaml"
    assert cfg.unset_value("server.gpu_headroom_gb", path=p) is False
    assert not p.exists()


def test_unset_value_unknown_key_raises(tmp_path):
    import pytest
    from muse.core import config as cfg
    with pytest.raises(KeyError):
        cfg.unset_value("no.such.key", path=tmp_path / "config.yaml")


# --- singleton reset: writes to the ACTIVE config path must invalidate
# the process-wide Config singleton so a later config.get() in the SAME
# process sees the new value instead of a stale cached parse. Writes to
# an explicit non-active test path (all the tests above) must NOT reset
# the singleton.


@pytest.fixture
def _reset_singleton():
    from muse.core import config as cfg
    cfg.reset_config()
    yield
    cfg.reset_config()


def test_set_value_default_path_resets_singleton_for_get(
    tmp_path, monkeypatch, _reset_singleton,
):
    from muse.core import config as cfg
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.delenv("MUSE_CONFIG", raising=False)
    monkeypatch.delenv("MUSE_GPU_HEADROOM_GB", raising=False)

    # Prime the singleton with the default value before any write.
    assert cfg.get("server.gpu_headroom_gb") == 1.0

    cfg.set_value("server.gpu_headroom_gb", "3.5")  # default (active) path

    assert cfg.get("server.gpu_headroom_gb") == 3.5, (
        "set_value on the active config path must reset the module "
        "singleton so a same-process get() sees the new value"
    )


def test_unset_value_default_path_resets_singleton_for_get(
    tmp_path, monkeypatch, _reset_singleton,
):
    from muse.core import config as cfg
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.delenv("MUSE_CONFIG", raising=False)
    monkeypatch.delenv("MUSE_GPU_HEADROOM_GB", raising=False)

    cfg.set_value("server.gpu_headroom_gb", "3.5")  # active path, resets singleton
    assert cfg.get("server.gpu_headroom_gb") == 3.5

    cfg.unset_value("server.gpu_headroom_gb")  # active path

    assert cfg.get("server.gpu_headroom_gb") == 1.0, (
        "unset_value on the active config path must reset the module "
        "singleton so a same-process get() reverts to the default"
    )


def test_set_value_refuses_bootstrap_catalog_dir(tmp_path):
    p = tmp_path / "config.yaml"
    with pytest.raises(cfg.ConfigError):
        cfg.set_value("paths.catalog_dir", "/tmp/x", path=p)
    assert not p.exists()


def test_set_value_refuses_bootstrap_config_file(tmp_path):
    p = tmp_path / "config.yaml"
    with pytest.raises(cfg.ConfigError):
        cfg.set_value("paths.config_file", "/tmp/other.yaml", path=p)
    assert not p.exists()


def test_unset_value_still_allows_bootstrap_key_as_noop(tmp_path):
    """unset stays allowed: it's harmless cleanup of a stale value that
    was never able to take effect anyway (no override value exists for
    "use the lower-precedence default"; unset is the counterpart)."""
    p = tmp_path / "config.yaml"
    assert cfg.unset_value("paths.catalog_dir", path=p) is False


def test_set_value_explicit_test_path_does_not_reset_singleton(
    tmp_path, monkeypatch, _reset_singleton,
):
    """Writing to an explicit, non-active path (the common test pattern
    used throughout this file) must NOT clobber the process singleton --
    only writes to the resolved active config_path() do."""
    from muse.core import config as cfg
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path / "active"))
    monkeypatch.delenv("MUSE_CONFIG", raising=False)
    monkeypatch.delenv("MUSE_GPU_HEADROOM_GB", raising=False)

    # Prime the singleton before touching an unrelated explicit path.
    assert cfg.get("server.gpu_headroom_gb") == 1.0

    other_path = tmp_path / "unrelated.yaml"
    cfg.set_value("server.gpu_headroom_gb", "9.0", path=other_path)

    assert cfg.get("server.gpu_headroom_gb") == 1.0, (
        "a write to an explicit non-active path must not reset the "
        "process singleton"
    )
