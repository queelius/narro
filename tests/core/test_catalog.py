"""Tests for the discovery-driven catalog and pull()."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from muse.core.catalog import (
    CatalogEntry,
    pull,
    is_pulled,
    known_models,
    list_known,
    load_backend,
    remove,
    _read_catalog,
    _reset_known_models_cache,
    _reset_read_catalog_cache,
)
from muse.core.discovery import modality_tags


@pytest.fixture(autouse=True)
def _isolate_catalog_cache():
    """Reset the known-models AND read-catalog mtime caches around every test.

    Both caches persist in process memory and would otherwise bleed state
    across tests. Bundled-only discovery is cheap and the catalog file is
    tiny, so re-running each per test is fine.
    """
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    yield
    _reset_known_models_cache()
    _reset_read_catalog_cache()


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    """Point catalog state at a temp file."""
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    yield tmp_path


def test_known_models_entries_have_valid_modality():
    """Every catalog entry's modality must match a discovered modality.

    The valid set comes from `modality_tags()` (single source of truth).
    Adding a new modality package is enough to make this test accept it;
    no test-side hardcoded list to forget to update.
    """
    valid = set(modality_tags())
    for model_id, entry in known_models().items():
        assert entry.modality in valid, \
            f"model {model_id} has invalid modality {entry.modality!r}"


def test_known_models_seeded_with_required_entries():
    """Bundled src/muse/models/ discovery picks up every built-in model."""
    catalog = known_models()
    assert "soprano-80m" in catalog
    assert "kokoro-82m" in catalog
    assert "bark-small" in catalog
    assert "sd-turbo" in catalog
    assert "nv-embed-v2" in catalog


def test_catalog_entry_reflects_manifest_capabilities():
    """MANIFEST['capabilities'] flows into CatalogEntry.extra."""
    kokoro = known_models()["kokoro-82m"]
    assert kokoro.modality == "audio/speech"
    assert kokoro.hf_repo == "hexgrad/Kokoro-82M"
    assert "sample_rate" in kokoro.extra
    assert kokoro.extra["sample_rate"] == 24000


def test_catalog_backend_path_points_at_discovered_model_class():
    """backend_path is synthesized from the Model class, not the MANIFEST."""
    kokoro = known_models()["kokoro-82m"]
    assert kokoro.backend_path == "muse.models.kokoro_82m:Model"


def test_known_models_cache_is_reusable():
    """Second call within the same cache window returns the same dict."""
    first = known_models()
    second = known_models()
    assert first is second


def _write_user_model(user_dir, filename, model_id, modality="audio/speech", hf_repo="fake/repo"):
    """Helper: write a minimal valid model script into user_dir."""
    import textwrap
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / filename).write_text(textwrap.dedent(f"""
        MANIFEST = {{
            "model_id": {model_id!r},
            "modality": {modality!r},
            "hf_repo": {hf_repo!r},
        }}
        class Model:
            model_id = {model_id!r}
    """).lstrip())


def test_known_models_picks_up_user_models_dir(tmp_path, monkeypatch):
    """Scripts in ~/.muse/models/ show up in known_models()."""
    monkeypatch.setenv("HOME", str(tmp_path))
    user_dir = tmp_path / ".muse" / "models"
    _write_user_model(user_dir, "my_custom.py", "my-custom-tts")

    _reset_known_models_cache()
    catalog = known_models()
    assert "my-custom-tts" in catalog
    assert catalog["my-custom-tts"].modality == "audio/speech"


def test_known_models_picks_up_env_override_dir(tmp_path, monkeypatch):
    """$MUSE_MODELS_DIR is scanned after the user dir."""
    env_dir = tmp_path / "env-muse-models"
    _write_user_model(env_dir, "experimental.py", "experimental-tts")
    monkeypatch.setenv("MUSE_MODELS_DIR", str(env_dir))

    _reset_known_models_cache()
    catalog = known_models()
    assert "experimental-tts" in catalog


def test_bundled_models_shadow_user_models_on_collision(tmp_path, monkeypatch, caplog):
    """First-found-wins: bundled entries beat user entries with the same id.

    Users cannot silently replace a bundled model. A warning is logged
    when a user script collides with a bundled one.
    """
    import logging
    monkeypatch.setenv("HOME", str(tmp_path))
    user_dir = tmp_path / ".muse" / "models"
    # User "kokoro-82m" points at a bogus repo; bundled one points at
    # hexgrad/Kokoro-82M. We expect the bundled manifest to win.
    _write_user_model(
        user_dir, "kokoro_82m.py", "kokoro-82m",
        hf_repo="user/override-repo",
    )

    caplog.set_level(logging.WARNING)
    _reset_known_models_cache()
    catalog = known_models()
    assert catalog["kokoro-82m"].hf_repo == "hexgrad/Kokoro-82M"
    # Collision should be noted
    assert "kokoro-82m" in caplog.text


def test_nonexistent_user_dir_is_silently_skipped(tmp_path, monkeypatch):
    """No ~/.muse/models/ dir = discovery carries on without warnings."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Deliberately do NOT create the user dir
    _reset_known_models_cache()
    catalog = known_models()
    # Bundled set still intact
    assert "kokoro-82m" in catalog
    assert "sd-turbo" in catalog


def test_list_known_filters_by_modality():
    audio = list_known("audio/speech")
    assert all(e.modality == "audio/speech" for e in audio)
    assert len(audio) >= 1
    images = list_known("image/generation")
    assert all(e.modality == "image/generation" for e in images)
    assert len(images) >= 1


def test_list_known_all():
    all_entries = list_known()
    modalities = {e.modality for e in all_entries}
    assert "audio/speech" in modalities
    assert "image/generation" in modalities


def test_is_pulled_false_when_not_in_catalog(tmp_catalog):
    assert not is_pulled("soprano-80m")


def test_pull_installs_pip_downloads_and_writes_catalog(tmp_catalog):
    with patch("muse.core.catalog.create_venv") as mock_create, \
         patch("muse.core.catalog.install_into_venv") as mock_install, \
         patch("muse.core.catalog.snapshot_download") as mock_download, \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        mock_download.return_value = "/fake/cache/soprano"
        pull("soprano-80m")
        mock_create.assert_called_once()
        # install_into_venv called twice: once for museq[server], once for
        # the model's pip_extras.
        assert mock_install.call_count == 2
        mock_download.assert_called_once()
        assert is_pulled("soprano-80m")


def test_pull_unknown_raises():
    with pytest.raises(KeyError, match="unknown model"):
        pull("does-not-exist-xyz")


def test_pull_warns_on_missing_system_packages(tmp_catalog, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=["espeak-ng"]):
        pull("kokoro-82m")
        assert "espeak-ng" in caplog.text


def test_remove_clears_from_catalog(tmp_catalog):
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
        assert is_pulled("soprano-80m")
        remove("soprano-80m")
        assert not is_pulled("soprano-80m")


def test_remove_default_leaves_venv_on_disk(tmp_catalog):
    """Default remove() unregisters from catalog only; venv persists."""
    venv_path = tmp_catalog / "venvs" / "soprano-80m"
    venv_path.mkdir(parents=True)
    (venv_path / "marker").write_text("present")
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    remove("soprano-80m")
    assert venv_path.exists(), "default remove must not delete the venv"
    assert (venv_path / "marker").exists()


def test_remove_with_purge_deletes_venv(tmp_catalog):
    """remove(purge=True) also wipes the per-model venv directory."""
    venv_path = tmp_catalog / "venvs" / "soprano-80m"
    venv_path.mkdir(parents=True)
    (venv_path / "marker").write_text("present")
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    remove("soprano-80m", purge=True)
    assert not venv_path.exists(), "purge must delete the venv directory"


def test_remove_with_purge_tolerates_missing_venv(tmp_catalog):
    """purge=True must not raise if the venv directory is already gone."""
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    venv_path = tmp_catalog / "venvs" / "soprano-80m"
    if venv_path.exists():
        import shutil
        shutil.rmtree(venv_path)
    # Should not raise
    remove("soprano-80m", purge=True)
    assert not is_pulled("soprano-80m")


# ---- v0.34.0 finding #15: purge cleans resolver weights cache ----


def test_remove_with_purge_cleans_resolver_weights(tmp_catalog):
    """v0.34.0 finding #15: remove(purge=True) MUST also rmtree the
    resolver weights cache when local_dir lives under the muse-owned
    ~/.muse/weights/ tree. Otherwise a user pulling and removing 10
    large models accumulates 100GB+ of orphans the muse CLI cannot
    clean up."""
    from muse.core.catalog import _catalog_path, _read_catalog, _write_catalog

    weights_root = tmp_catalog / "weights"
    weights_dir = weights_root / "resolver-pulled-x"
    weights_dir.mkdir(parents=True)
    (weights_dir / "model.safetensors").write_bytes(b"x" * 100)

    venv_path = tmp_catalog / "venvs" / "x"
    venv_path.mkdir(parents=True)
    (venv_path / "marker").write_text("v")

    # Hand-build a catalog entry shaped like a resolver pull.
    _write_catalog({
        "x": {
            "pulled_at": "2026-05-02T00:00:00Z",
            "hf_repo": "fake/x",
            "local_dir": str(weights_dir),
            "venv_path": str(venv_path),
            "python_path": str(venv_path / "bin" / "python"),
            "enabled": False,
        },
    })

    remove("x", purge=True)

    assert not weights_dir.exists(), "purge must rmtree resolver weights"
    assert not venv_path.exists(), "purge must rmtree venv"
    assert "x" not in _read_catalog()


def test_remove_with_purge_leaves_external_weights(tmp_catalog):
    """A local_dir outside ~/.muse/weights/ (typically the HF shared
    cache at ~/.cache/huggingface) must NOT be rmtree'd, even under
    purge=True. muse does not own that cache."""
    from muse.core.catalog import _read_catalog, _write_catalog

    # Outside the weights tree.
    external_weights = tmp_catalog / "outside" / "hf_cache" / "models--foo"
    external_weights.mkdir(parents=True)
    (external_weights / "model.safetensors").write_bytes(b"y" * 50)

    venv_path = tmp_catalog / "venvs" / "y"
    venv_path.mkdir(parents=True)

    _write_catalog({
        "y": {
            "pulled_at": "2026-05-02T00:00:00Z",
            "hf_repo": "fake/y",
            "local_dir": str(external_weights),
            "venv_path": str(venv_path),
            "enabled": False,
        },
    })

    remove("y", purge=True)

    assert external_weights.exists(), \
        "external cache must be left alone (not under ~/.muse/weights/)"
    assert (external_weights / "model.safetensors").exists()
    assert not venv_path.exists(), "venv still gets purged"


def test_remove_with_purge_tolerates_missing_weights_dir(tmp_catalog):
    """purge=True must not raise if the resolver weights dir is
    already gone (caller may have rm-rf'd it manually)."""
    from muse.core.catalog import _write_catalog

    weights_root = tmp_catalog / "weights"
    weights_dir = weights_root / "z"
    # Don't create it; it's missing.

    _write_catalog({
        "z": {
            "pulled_at": "2026-05-02T00:00:00Z",
            "hf_repo": "fake/z",
            "local_dir": str(weights_dir),
            "venv_path": str(tmp_catalog / "venvs" / "z"),
            "enabled": False,
        },
    })

    # Should not raise even though the dir doesn't exist.
    remove("z", purge=True)


def test_remove_default_leaves_resolver_weights(tmp_catalog):
    """Without purge=True, weights persist (mirrors apt-remove semantics)."""
    from muse.core.catalog import _read_catalog, _write_catalog

    weights_root = tmp_catalog / "weights"
    weights_dir = weights_root / "w"
    weights_dir.mkdir(parents=True)
    (weights_dir / "marker").write_bytes(b"keep me")

    _write_catalog({
        "w": {
            "pulled_at": "2026-05-02T00:00:00Z",
            "hf_repo": "fake/w",
            "local_dir": str(weights_dir),
            "venv_path": str(tmp_catalog / "venvs" / "w"),
            "enabled": False,
        },
    })

    remove("w")  # default: no purge

    assert weights_dir.exists(), "default remove must leave weights on disk"
    assert (weights_dir / "marker").exists()


def test_load_backend_raises_when_not_pulled(tmp_catalog):
    with pytest.raises(RuntimeError, match="not pulled"):
        load_backend("soprano-80m")


def test_load_backend_imports_and_constructs(tmp_catalog):
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake/local"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.Model = fake_class
    with patch("muse.core.catalog._import_backend_module", return_value=fake_module):
        load_backend("soprano-80m", device="cpu")
    fake_class.assert_called_once()
    # Verify the constructor got hf_repo, local_dir, and device kwargs
    kwargs = fake_class.call_args.kwargs
    assert kwargs["local_dir"] == "/fake/local"
    assert kwargs["hf_repo"] == "ekwek/Soprano-1.1-80M"
    assert kwargs["device"] == "cpu"


def test_load_backend_raises_keyerror_on_unknown_model(tmp_catalog):
    with pytest.raises(KeyError, match="unknown model"):
        load_backend("bogus-model-xyz")


def test_load_backend_handles_backend_path_with_extra_colon(tmp_catalog):
    """load_backend must split backend_path the same way get_manifest does
    (split(":", 1)): an unbounded split(":") raises "too many values to
    unpack" for any backend_path containing a second colon."""
    _write_persisted_resolver_entry(
        tmp_catalog,
        model_id="weird-model",
        backend_path="muse.fake.module:Weird:Extra",
    )

    fake_class = MagicMock()
    fake_module = MagicMock()
    setattr(fake_module, "Weird:Extra", fake_class)

    with patch("muse.core.catalog._import_backend_module", return_value=fake_module):
        load_backend("weird-model", device="cpu")

    fake_class.assert_called_once()


def test_write_catalog_is_atomic_no_tmp_leftover(tmp_catalog):
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    # After successful write, the .tmp file must not exist
    tmp_files = list(tmp_catalog.glob("*.tmp"))
    assert tmp_files == [], f"leftover tmp files: {tmp_files}"
    # And catalog.json must have the entry
    catalog_file = tmp_catalog / "catalog.json"
    assert catalog_file.exists()


def test_pull_creates_venv_under_muse_catalog_dir(tmp_catalog):
    """pull() must create a venv at <MUSE_CATALOG_DIR>/venvs/<model-id>/."""
    with patch("muse.core.catalog.create_venv") as mock_create, \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
        mock_create.assert_called_once()
        venv_target = mock_create.call_args[0][0]
        expected = tmp_catalog / "venvs" / "soprano-80m"
        assert venv_target == expected


def test_pull_installs_pip_extras_into_venv_not_system(tmp_catalog):
    """pip_extras go into the venv, never the supervisor's env."""
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv") as mock_install, \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    # Find the call that installed soprano's actual pip_extras. There are
    # two calls in total: museq[server] first, then the model's pip_extras.
    model_call = next(
        c for c in mock_install.call_args_list
        if any("transformers" in p for p in c.args[1])
    )
    venv_arg, packages_arg = model_call.args
    assert venv_arg == tmp_catalog / "venvs" / "soprano-80m"
    assert any("transformers" in p for p in packages_arg)


def test_pull_installs_muse_editable_with_server_extras(tmp_catalog):
    """Worker venvs must have muse installed so python -m muse.cli works.

    Without this, `<venv>/bin/python -m muse.cli _worker` crashes with
    ModuleNotFoundError: No module named 'muse'. The supervisor can't
    spawn workers.
    """
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv") as mock_install, \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")

    muse_call = next(
        (c for c in mock_install.call_args_list if "-e" in c.args[1]),
        None,
    )
    assert muse_call is not None, "muse was not installed into the venv"
    venv_arg, packages_arg = muse_call.args
    assert venv_arg == tmp_catalog / "venvs" / "soprano-80m"
    # Format: ["-e", "<repo-root>[server]"]
    assert packages_arg[0] == "-e"
    assert "[server]" in packages_arg[1]


def test_pull_records_venv_path_and_python_in_catalog(tmp_catalog):
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    catalog = _read_catalog()
    entry = catalog["soprano-80m"]
    assert "venv_path" in entry
    assert entry["venv_path"] == str(tmp_catalog / "venvs" / "soprano-80m")
    assert "python_path" in entry
    assert entry["python_path"] == str(tmp_catalog / "venvs" / "soprano-80m" / "bin" / "python")


def test_catalog_has_no_system_install_pip_extras(tmp_catalog):
    """The old system-wide install_pip_extras (dead footgun: it would have
    installed into sys.executable's env, the supervisor env, defeating
    per-model venv isolation) has been removed entirely -- catalog.py no
    longer imports or exposes it. pip installs are venv-scoped via
    `install_into_venv` only."""
    import muse.core.catalog as catalog_module
    assert not hasattr(catalog_module, "install_pip_extras")
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")


def test_pull_records_enabled_true_by_default(tmp_catalog):
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    catalog = _read_catalog()
    assert catalog["soprano-80m"]["enabled"] is True


def test_read_catalog_backfills_enabled_for_legacy_entries(tmp_catalog):
    """Old catalog.json entries without `enabled` are treated as enabled.

    This is the migration path: no destructive writes, just a default
    when reading. Existing entries stay valid after the schema change.
    """
    import json
    from muse.core.catalog import _catalog_path
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write a legacy entry (no `enabled` field)
    p.write_text(json.dumps({
        "legacy-model": {
            "pulled_at": "...",
            "hf_repo": "x",
            "local_dir": "/x",
            "venv_path": "/v",
            "python_path": "/v/bin/python",
        },
    }))

    catalog = _read_catalog()
    assert catalog["legacy-model"]["enabled"] is True


def test_is_enabled_helper_returns_true_for_entry_with_flag(tmp_catalog):
    from muse.core.catalog import is_enabled
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    assert is_enabled("soprano-80m") is True


def test_is_enabled_helper_returns_false_after_set_enabled_false(tmp_catalog):
    from muse.core.catalog import is_enabled, set_enabled
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    set_enabled("soprano-80m", False)
    assert is_enabled("soprano-80m") is False


def test_set_enabled_raises_on_unknown_model(tmp_catalog):
    from muse.core.catalog import set_enabled
    with pytest.raises(KeyError, match="not pulled"):
        set_enabled("not-pulled-model", True)


def test_set_enabled_preserves_other_fields(tmp_catalog):
    from muse.core.catalog import set_enabled
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    before = _read_catalog()["soprano-80m"]
    set_enabled("soprano-80m", False)
    after = _read_catalog()["soprano-80m"]
    # Everything except `enabled` is preserved
    for key in ("pulled_at", "hf_repo", "local_dir", "venv_path", "python_path"):
        assert before[key] == after[key]
    assert after["enabled"] is False


# --- F1: catalog merges resolver-persisted manifests ------------------------


def _write_persisted_resolver_entry(
    tmp_catalog,
    *,
    model_id,
    modality="chat/completion",
    hf_repo="fake/repo",
    backend_path="muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel",
    capabilities=None,
):
    """Write a catalog.json entry mimicking what _pull_via_resolver would persist."""
    import json
    from muse.core.catalog import _catalog_path
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(p.read_text()) if p.exists() else {}
    existing[model_id] = {
        "pulled_at": "2026-04-15T00:00:00Z",
        "hf_repo": hf_repo,
        "local_dir": str(tmp_catalog / "weights" / model_id),
        "venv_path": str(tmp_catalog / "venvs" / model_id),
        "python_path": str(tmp_catalog / "venvs" / model_id / "bin" / "python"),
        "enabled": True,
        "source": f"hf://{hf_repo}@variant",
        "manifest": {
            "model_id": model_id,
            "modality": modality,
            "hf_repo": hf_repo,
            "backend_path": backend_path,
            "description": f"resolver-pulled {model_id}",
            "pip_extras": [],
            "system_packages": [],
            "capabilities": capabilities or {},
        },
    }
    p.write_text(json.dumps(existing))


def test_known_models_merges_resolver_persisted_entries(tmp_catalog):
    """Catalog entries with a `manifest` field show up in known_models()."""
    _write_persisted_resolver_entry(
        tmp_catalog,
        model_id="qwen3-8b-gguf-q4-k-m",
        capabilities={"gguf_file": "qwen3-8b-q4_k_m.gguf", "supports_tools": True},
    )
    _reset_known_models_cache()
    entries = known_models()
    assert "qwen3-8b-gguf-q4-k-m" in entries
    e = entries["qwen3-8b-gguf-q4-k-m"]
    assert e.modality == "chat/completion"
    assert e.backend_path.endswith(":LlamaCppModel")
    assert e.extra["gguf_file"] == "qwen3-8b-q4_k_m.gguf"
    assert e.extra["supports_tools"] is True


def _bump_catalog_mtime():
    """Nudge catalog.json's mtime_ns forward deterministically.

    Two writes within one test can land inside the same mtime_ns tick on
    coarse-grained filesystems; the explicit bump removes that flake."""
    import os
    from muse.core.catalog import _catalog_path
    p = _catalog_path()
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def test_known_models_self_invalidates_on_catalog_change(tmp_catalog):
    """A catalog.json write from ANOTHER process must become visible without
    a manual cache reset.

    Regression: the supervisor's known_models() cache froze at first call.
    The admin pull endpoint runs `muse pull` as a subprocess (whose own
    cache resets are invisible here), so `enable` and the gateway route
    404'd "unknown model" for anything pulled after the cache was built
    (the all-in-one-pixel-model incident), even though catalog.json and
    /v1/models both showed the entry.
    """
    baseline = known_models()
    assert "pulled-after-cache" not in baseline

    # Simulate the subprocess: write the file directly, no muse helpers,
    # no _reset_known_models_cache().
    _write_persisted_resolver_entry(tmp_catalog, model_id="pulled-after-cache")
    _bump_catalog_mtime()

    entries = known_models()
    assert "pulled-after-cache" in entries
    assert entries["pulled-after-cache"].modality == "chat/completion"


def test_known_models_drops_entries_removed_from_catalog(tmp_catalog):
    """The inverse staleness: an entry deleted from catalog.json (e.g.
    `muse models remove` in another shell) must disappear from
    known_models() without a process restart."""
    from muse.core.catalog import _catalog_path

    _write_persisted_resolver_entry(tmp_catalog, model_id="soon-removed")
    _reset_known_models_cache()
    assert "soon-removed" in known_models()

    _catalog_path().write_text("{}")
    _bump_catalog_mtime()

    assert "soon-removed" not in known_models()


def test_known_models_identity_stable_while_catalog_unchanged(tmp_catalog):
    """Hot path stays memoized: same catalog mtime -> same dict object."""
    _write_persisted_resolver_entry(tmp_catalog, model_id="stable-entry")
    _reset_known_models_cache()
    first = known_models()
    second = known_models()
    assert first is second


def test_known_models_does_not_rerun_discovery_on_catalog_change(
    tmp_catalog, monkeypatch,
):
    """Catalog-change invalidation must NOT re-run discover_models: script
    discovery does importlib imports (module bodies execute), so it stays
    cached for the process lifetime; only the cheap merge re-runs."""
    import muse.core.catalog as cat

    calls = {"n": 0}
    real_discover = cat.discover_models

    def counting_discover(dirs):
        calls["n"] += 1
        return real_discover(dirs)

    monkeypatch.setattr(cat, "discover_models", counting_discover)
    _reset_known_models_cache()

    known_models()
    _write_persisted_resolver_entry(tmp_catalog, model_id="new-after-build")
    _bump_catalog_mtime()

    assert "new-after-build" in known_models()
    assert calls["n"] == 1


def test_bundled_scripts_win_on_collision_with_resolver_manifest(tmp_catalog):
    """A persisted manifest with the same id as a bundled script is shadowed."""
    _write_persisted_resolver_entry(
        tmp_catalog,
        model_id="kokoro-82m",
        modality="audio/speech",
        hf_repo="malicious/fake",
        backend_path="muse.models.kokoro_82m:Model",
    )
    _reset_known_models_cache()
    entries = known_models()
    # Bundled wins: hf_repo from the script, not from the malicious manifest
    assert entries["kokoro-82m"].hf_repo == "hexgrad/Kokoro-82M"


def test_legacy_catalog_entries_without_manifest_are_skipped_in_merge(tmp_catalog):
    """Old catalog entries (pre-resolver) lack `manifest`; they don't break merge."""
    import json
    from muse.core.catalog import _catalog_path
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "kokoro-82m": {
            "pulled_at": "2026-04-15T00:00:00Z",
            "hf_repo": "hexgrad/Kokoro-82M",
            "local_dir": "/some/path",
            "venv_path": "/v",
            "python_path": "/v/bin/python",
            "enabled": True,
            # no `manifest` key
        },
        "alien-legacy-model": {
            "pulled_at": "2026-04-15T00:00:00Z",
            "hf_repo": "alien/repo",
            "local_dir": "/a",
            "venv_path": "/av",
            "python_path": "/av/bin/python",
            "enabled": True,
            # no `manifest` key
        },
    }))
    _reset_known_models_cache()
    entries = known_models()
    assert "kokoro-82m" in entries  # bundled script discovery
    assert "alien-legacy-model" not in entries  # no manifest, no script -> skip


def test_get_manifest_returns_persisted_manifest_for_resolver_entry(tmp_catalog):
    """get_manifest() returns the catalog-persisted manifest for resolver-pulled models."""
    from muse.core.catalog import get_manifest
    _write_persisted_resolver_entry(
        tmp_catalog,
        model_id="q3-gguf-q4",
        capabilities={"gguf_file": "q4.gguf"},
    )
    _reset_known_models_cache()
    m = get_manifest("q3-gguf-q4")
    assert m["model_id"] == "q3-gguf-q4"
    assert m["capabilities"]["gguf_file"] == "q4.gguf"


def test_get_manifest_falls_back_to_script_module_for_bundled(tmp_catalog):
    """Bundled-script models have no persisted manifest; get_manifest reads the module."""
    from muse.core.catalog import get_manifest
    m = get_manifest("kokoro-82m")
    assert m["model_id"] == "kokoro-82m"
    assert m["modality"] == "audio/speech"


# --- _read_catalog mtime cache ---------------------------------------------
#
# `get_manifest` runs on the gateway hot path (every request). Without
# caching, each request re-reads catalog.json + re-parses JSON, which adds
# unnecessary file I/O per request. The cache stores (mtime, parsed_dict)
# keyed by catalog path; invalidates when the file's mtime changes (writes
# go through `_write_catalog`'s atomic rename, which updates mtime).


def test_read_catalog_caches_consecutive_reads(tmp_catalog):
    """Two consecutive _read_catalog() calls hit disk only once.

    The cache uses mtime invalidation, so back-to-back reads against an
    unchanged file return the cached result without re-parsing.
    """
    from unittest.mock import patch
    from muse.core.catalog import (
        _read_catalog,
        _reset_read_catalog_cache,
        _write_catalog,
    )

    _write_catalog({"alpha": {"hf_repo": "x/y", "enabled": True}})
    _reset_read_catalog_cache()

    real_path_read_text = type(tmp_catalog).read_text
    call_count = {"n": 0}

    def counting_read_text(self, *a, **kw):
        call_count["n"] += 1
        return real_path_read_text(self, *a, **kw)

    with patch.object(type(tmp_catalog), "read_text", counting_read_text):
        c1 = _read_catalog()
        c2 = _read_catalog()
        c3 = _read_catalog()

    assert c1 == c2 == c3
    # Three calls, exactly ONE disk read. Cache hits skip Path.read_text.
    assert call_count["n"] == 1, (
        f"_read_catalog hit disk {call_count['n']} times across 3 calls; "
        "expected 1 (mtime cache)"
    )


def test_read_catalog_cache_invalidates_on_mtime_change(tmp_catalog):
    """Writing through `_write_catalog` updates mtime; next read sees new data."""
    import os
    import time
    from muse.core.catalog import (
        _read_catalog,
        _reset_read_catalog_cache,
        _catalog_path,
        _write_catalog,
    )

    _write_catalog({"alpha": {"hf_repo": "x/y", "enabled": True}})
    _reset_read_catalog_cache()

    first = _read_catalog()
    assert "alpha" in first
    assert "beta" not in first

    # mtime resolution can be coarse (1s on some filesystems); bump
    # explicitly so the cache invalidates even on fast back-to-back writes.
    time.sleep(0.01)
    _write_catalog({"alpha": {"hf_repo": "x/y", "enabled": True},
                    "beta": {"hf_repo": "p/q", "enabled": True}})
    p = _catalog_path()
    now = time.time()
    os.utime(p, (now, now + 1.0))

    second = _read_catalog()
    assert "beta" in second, "_read_catalog cache failed to invalidate after _write_catalog"


# --- _read_catalog corrupt-file guard ---------------------------------------
#
# A corrupt/truncated catalog.json must NOT silently look like an empty-but-
# valid catalog: every consumer (get_manifest, known_models, /v1/models,
# is_pulled) would then behave as "no models," 404-ing or emptying results
# without any signal that the catalog itself is broken.


def test_read_catalog_corrupt_with_no_prior_cache_raises(tmp_catalog):
    """No last-known-good cache exists yet (first read ever): corruption
    must raise a distinct error, not return {}."""
    from muse.core.catalog import CatalogError, _catalog_path, _read_catalog

    _catalog_path().parent.mkdir(parents=True, exist_ok=True)
    _catalog_path().write_text("{not valid json")

    with pytest.raises(CatalogError):
        _read_catalog()


def test_read_catalog_corrupt_falls_back_to_last_known_good(tmp_catalog):
    """A cache from a prior good read exists: corruption on a later read
    must serve the cached data, not an empty dict."""
    from muse.core.catalog import (
        _catalog_path,
        _read_catalog,
        _reset_read_catalog_cache,
        _write_catalog,
    )

    _write_catalog({"alpha": {"hf_repo": "x/y", "enabled": True}})
    _reset_read_catalog_cache()
    good = _read_catalog()
    assert "alpha" in good

    # Corrupt the file in place (same path, different mtime/content).
    _catalog_path().write_text("{not valid json")

    result = _read_catalog()
    assert "alpha" in result, (
        "corrupt catalog must fall back to the last-known-good cached "
        "parse, not silently degrade to an empty model set"
    )


def test_get_manifest_does_not_re_read_catalog_each_call(tmp_catalog):
    """get_manifest is on the gateway hot path; back-to-back calls must
    hit the catalog file only once thanks to the mtime cache.
    """
    from unittest.mock import patch
    from muse.core.catalog import (
        get_manifest,
        _reset_known_models_cache,
        _reset_read_catalog_cache,
    )

    _write_persisted_resolver_entry(
        tmp_catalog,
        model_id="q3-gguf-q4",
        capabilities={"gguf_file": "q4.gguf"},
    )
    _reset_known_models_cache()
    _reset_read_catalog_cache()

    real_path_read_text = type(tmp_catalog).read_text
    call_count = {"n": 0}

    def counting_read_text(self, *a, **kw):
        call_count["n"] += 1
        return real_path_read_text(self, *a, **kw)

    with patch.object(type(tmp_catalog), "read_text", counting_read_text):
        get_manifest("q3-gguf-q4")
        baseline = call_count["n"]
        # Second call MUST be a cache hit. Without the cache, this would
        # re-read the catalog file from disk (the bug Issue 2 flagged).
        get_manifest("q3-gguf-q4")
        after_second = call_count["n"]

    assert after_second == baseline, (
        f"second get_manifest call hit disk {after_second - baseline} extra times; "
        "expected 0 (cached)"
    )


# --- F2: pull() dispatch on URI vs bare id ----------------------------------


def test_pull_dispatches_to_resolver_for_uri(tmp_catalog):
    """`muse pull hf://...` routes through the resolver and persists the manifest."""
    from muse.core.catalog import pull
    from muse.core.resolvers import (
        Resolver, ResolvedModel, register_resolver, _reset_registry_for_tests,
    )

    class _FakeResolver(Resolver):
        scheme = "fake"

        def resolve(self, uri):
            return ResolvedModel(
                manifest={
                    "model_id": "pulled-from-resolver",
                    "modality": "chat/completion",
                    "hf_repo": "fake/repo",
                    "backend_path": "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel",
                    "pip_extras": ["llama-cpp-python"],
                    "capabilities": {"gguf_file": "x.gguf"},
                },
                backend_path="muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel",
                download=lambda cache: cache / "weights",
            )

        def search(self, q, **k):
            return []

    _reset_registry_for_tests()
    register_resolver(_FakeResolver())

    try:
        with patch("muse.core.catalog.create_venv"), \
             patch("muse.core.catalog.install_into_venv"), \
             patch("muse.core.catalog.check_system_packages", return_value=[]):
            pull("fake://some/repo@variant")
    finally:
        _reset_registry_for_tests()

    catalog = _read_catalog()
    assert "pulled-from-resolver" in catalog
    entry = catalog["pulled-from-resolver"]
    assert entry["source"] == "fake://some/repo@variant"
    assert entry["manifest"]["modality"] == "chat/completion"
    assert entry["manifest"]["capabilities"]["gguf_file"] == "x.gguf"
    # Cache invalidation: the new model must show up in known_models()
    assert "pulled-from-resolver" in known_models()


def test_pull_bare_id_still_uses_bundled_path(tmp_catalog):
    """Regression: non-URI pull goes through known_models() / scripts."""
    from muse.core.catalog import pull, is_pulled
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("kokoro-82m")
    assert is_pulled("kokoro-82m")
    # Bundled-path entries do NOT carry a `manifest` field (legacy shape preserved)
    assert "manifest" not in _read_catalog()["kokoro-82m"]


class TestMuseServerInstallArgs:
    """M2 companion: `muse pull` must install the published `museq` from PyPI
    when muse is not running from a source checkout, mirroring the refresh fix.
    _muse_repo_root() would otherwise return site-packages and
    `pip install -e <site-packages>[server]` fails, so a PyPI-installed muse
    could never pull any model."""

    def test_editable_target_when_source_tree(self):
        from muse.core.catalog import _muse_server_install_args
        with patch("muse.core.catalog._muse_repo_root", return_value=Path("/src/muse")):
            args = _muse_server_install_args()
        assert args == ["-e", "/src/muse[server]"]

    def test_pypi_target_when_no_source_tree(self):
        from muse.core.catalog import _muse_server_install_args
        with patch("muse.core.catalog._muse_repo_root", return_value=None):
            args = _muse_server_install_args()
        assert args == ["museq[server]"]
        assert "-e" not in args

    def test_pull_bundled_installs_museq_from_pypi_when_not_source_tree(
        self, tmp_catalog,
    ):
        from muse.core.catalog import pull
        with patch("muse.core.catalog._muse_repo_root", return_value=None), \
             patch("muse.core.catalog.create_venv"), \
             patch("muse.core.catalog.install_into_venv") as mock_install, \
             patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
             patch("muse.core.catalog.check_system_packages", return_value=[]):
            pull("kokoro-82m")
        # The muse-self install (first call) must be the published dist, no -e.
        first_args = mock_install.call_args_list[0].args[1]
        assert first_args == ["museq[server]"]


def test_repull_resolver_model_by_bare_id_routes_through_resolver(tmp_catalog):
    """M3: re-pulling a resolver-pulled model by its friendly bare id must
    go back through _pull_via_resolver, not _pull_bundled. The bare-id
    branch used to call _pull_bundled, which overwrote the entry with a
    dict lacking `manifest`/`source`; the next known_models() rebuild then
    dropped the (no-manifest, no-script) entry and the model vanished with
    a spurious 'unknown model' error until re-pulled by full URI."""
    from muse.core.catalog import pull

    _write_persisted_resolver_entry(
        tmp_catalog,
        model_id="qwen3-8b-gguf-q4-k-m",
        hf_repo="unsloth/Qwen3-8B-GGUF",
    )
    _reset_known_models_cache()

    with patch("muse.core.catalog._pull_via_resolver") as mock_resolver, \
         patch("muse.core.catalog._pull_bundled") as mock_bundled:
        pull("qwen3-8b-gguf-q4-k-m")

    mock_bundled.assert_not_called()
    mock_resolver.assert_called_once()
    # Routed back through the stored source URI, keyed to the same id.
    assert mock_resolver.call_args.args[0] == "hf://unsloth/Qwen3-8B-GGUF@variant"
    assert mock_resolver.call_args.kwargs.get("model_id_override") == (
        "qwen3-8b-gguf-q4-k-m"
    )


def test_pull_invalidates_known_models_cache_on_resolver_pull(tmp_catalog):
    """After resolver pull, known_models() must reflect the new entry without
    needing a manual cache reset."""
    from muse.core.catalog import pull
    from muse.core.resolvers import (
        Resolver, ResolvedModel, register_resolver, _reset_registry_for_tests,
    )

    baseline = set(known_models())
    assert "freshly-resolved" not in baseline

    class _FakeResolver(Resolver):
        scheme = "fake"

        def resolve(self, uri):
            return ResolvedModel(
                manifest={
                    "model_id": "freshly-resolved",
                    "modality": "chat/completion",
                    "hf_repo": "x/y",
                    "backend_path": "x.y:Z",
                },
                backend_path="x.y:Z",
                download=lambda cache: cache / "w",
            )

        def search(self, q, **k):
            return []

    _reset_registry_for_tests()
    register_resolver(_FakeResolver())

    try:
        with patch("muse.core.catalog.create_venv"), \
             patch("muse.core.catalog.install_into_venv"), \
             patch("muse.core.catalog.check_system_packages", return_value=[]):
            pull("fake://anything")
    finally:
        _reset_registry_for_tests()

    assert "freshly-resolved" in known_models()


def test_load_backend_merges_persisted_capabilities_into_kwargs(tmp_catalog):
    """LlamaCppModel-style runtimes need gguf_file from the manifest;
    load_backend must merge capabilities + inject model_id."""
    _write_persisted_resolver_entry(
        tmp_catalog,
        model_id="llama-mock",
        capabilities={
            "gguf_file": "model.gguf",
            "context_length": 4096,
            "chat_template": "chatml",
        },
    )
    _reset_known_models_cache()

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.LlamaCppModel = fake_class
    with patch("muse.core.catalog._import_backend_module", return_value=fake_module):
        load_backend("llama-mock", device="cpu")

    fake_class.assert_called_once()
    kwargs = fake_class.call_args.kwargs
    assert kwargs["model_id"] == "llama-mock"
    assert kwargs["hf_repo"] == "fake/repo"
    assert kwargs["gguf_file"] == "model.gguf"
    assert kwargs["context_length"] == 4096
    assert kwargs["chat_template"] == "chatml"
    assert kwargs["device"] == "cpu"


def test_load_backend_caller_kwargs_override_manifest_capabilities(tmp_catalog):
    """If the caller passes a kwarg that's also in capabilities, caller wins."""
    _write_persisted_resolver_entry(
        tmp_catalog,
        model_id="llama-mock-2",
        capabilities={"chat_template": "chatml"},
    )
    _reset_known_models_cache()

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.LlamaCppModel = fake_class
    with patch("muse.core.catalog._import_backend_module", return_value=fake_module):
        load_backend("llama-mock-2", chat_template="qwen", device="cpu")

    kwargs = fake_class.call_args.kwargs
    assert kwargs["chat_template"] == "qwen"


def test_load_backend_capability_device_overrides_caller_kwargs(tmp_catalog):
    """Capability device declaration wins over supervisor --device flag."""
    from unittest.mock import patch, MagicMock
    from muse.core.catalog import load_backend, _write_catalog

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.FakeRuntime = fake_class

    catalog_state = {
        "test-cpu-model": {
            "pulled_at": "2026-01-01T00:00:00+00:00",
            "hf_repo": "org/repo",
            "local_dir": "/fake/weights",
            "venv_path": "/fake/venv",
            "python_path": "/fake/py",
            "enabled": True,
            "manifest": {
                "model_id": "test-cpu-model",
                "modality": "embedding/text",
                "hf_repo": "org/repo",
                "pip_extras": [],
                "system_packages": [],
                "capabilities": {"device": "cpu"},
                "backend_path": "fake.module:FakeRuntime",
            },
        },
    }
    _write_catalog(catalog_state)

    with patch("muse.core.catalog._import_backend_module", return_value=fake_module), \
         patch("muse.core.catalog._reset_known_models_cache"):
        from muse.core.catalog import _reset_known_models_cache
        _reset_known_models_cache()
        load_backend("test-cpu-model", device="cuda")

    # Verify the runtime got device="cpu" (capability wins over kwargs)
    assert fake_class.call_args.kwargs["device"] == "cpu"


def test_load_backend_kwargs_win_when_capability_device_is_auto(tmp_catalog):
    """Capability device='auto' defers to caller's device kwarg."""
    from unittest.mock import patch, MagicMock
    from muse.core.catalog import load_backend, _write_catalog, _reset_known_models_cache

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.FakeRuntime = fake_class

    catalog_state = {
        "test-auto-model": {
            "pulled_at": "2026-01-01T00:00:00+00:00",
            "hf_repo": "org/repo",
            "local_dir": "/fake/weights",
            "venv_path": "/fake/venv",
            "python_path": "/fake/py",
            "enabled": True,
            "manifest": {
                "model_id": "test-auto-model",
                "modality": "embedding/text",
                "hf_repo": "org/repo",
                "pip_extras": [],
                "system_packages": [],
                "capabilities": {"device": "auto"},
                "backend_path": "fake.module:FakeRuntime",
            },
        },
    }
    _write_catalog(catalog_state)

    with patch("muse.core.catalog._import_backend_module", return_value=fake_module):
        _reset_known_models_cache()
        load_backend("test-auto-model", device="cuda")

    assert fake_class.call_args.kwargs["device"] == "cuda"


def test_load_backend_kwargs_win_when_no_capability_device(tmp_catalog):
    """When capabilities omits device entirely, caller kwarg wins (existing behavior)."""
    from unittest.mock import patch, MagicMock
    from muse.core.catalog import load_backend, _write_catalog, _reset_known_models_cache

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.FakeRuntime = fake_class

    catalog_state = {
        "test-no-pref": {
            "pulled_at": "2026-01-01T00:00:00+00:00",
            "hf_repo": "org/repo",
            "local_dir": "/fake/weights",
            "venv_path": "/fake/venv",
            "python_path": "/fake/py",
            "enabled": True,
            "manifest": {
                "model_id": "test-no-pref",
                "modality": "embedding/text",
                "hf_repo": "org/repo",
                "pip_extras": [],
                "system_packages": [],
                "capabilities": {},
                "backend_path": "fake.module:FakeRuntime",
            },
        },
    }
    _write_catalog(catalog_state)

    with patch("muse.core.catalog._import_backend_module", return_value=fake_module):
        _reset_known_models_cache()
        load_backend("test-no-pref", device="mps")

    assert fake_class.call_args.kwargs["device"] == "mps"


def _override_catalog_state(device_override=None, capability_device="cpu"):
    """Build a one-model catalog dict for device_override precedence tests."""
    caps = {}
    if capability_device is not None:
        caps["device"] = capability_device
    entry = {
        "pulled_at": "2026-01-01T00:00:00+00:00",
        "hf_repo": "org/repo",
        "local_dir": "/fake/weights",
        "venv_path": "/fake/venv",
        "python_path": "/fake/py",
        "enabled": True,
        "manifest": {
            "model_id": "ov-model",
            "modality": "embedding/text",
            "hf_repo": "org/repo",
            "pip_extras": [],
            "system_packages": [],
            "capabilities": caps,
            "backend_path": "fake.module:FakeRuntime",
        },
    }
    if device_override is not None:
        entry["device_override"] = device_override
    return {"ov-model": entry}


def test_load_backend_device_override_beats_capability_pin(tmp_catalog):
    """A catalog device_override outranks the manifest capabilities.device pin.

    kokoro-style models pin device='cpu'; an operator `set-device` override
    must force the requested device anyway.
    """
    from unittest.mock import patch, MagicMock
    from muse.core.catalog import load_backend, _write_catalog, _reset_known_models_cache

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.FakeRuntime = fake_class

    _write_catalog(_override_catalog_state(device_override="cuda", capability_device="cpu"))

    with patch("muse.core.catalog._import_backend_module", return_value=fake_module):
        _reset_known_models_cache()
        load_backend("ov-model", device="cpu")

    assert fake_class.call_args.kwargs["device"] == "cuda"


def test_load_backend_device_override_beats_caller_kwarg(tmp_catalog):
    """device_override wins even when there is no manifest pin."""
    from unittest.mock import patch, MagicMock
    from muse.core.catalog import load_backend, _write_catalog, _reset_known_models_cache

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.FakeRuntime = fake_class

    _write_catalog(_override_catalog_state(device_override="cuda", capability_device=None))

    with patch("muse.core.catalog._import_backend_module", return_value=fake_module):
        _reset_known_models_cache()
        load_backend("ov-model", device="cpu")

    assert fake_class.call_args.kwargs["device"] == "cuda"


def test_load_backend_device_override_auto_unpins_cpu(tmp_catalog):
    """override='auto' un-pins a cpu-pinned model: device flows as 'auto'
    so the runtime's select_device picks cuda when a GPU is present."""
    from unittest.mock import patch, MagicMock
    from muse.core.catalog import load_backend, _write_catalog, _reset_known_models_cache

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.FakeRuntime = fake_class

    _write_catalog(_override_catalog_state(device_override="auto", capability_device="cpu"))

    with patch("muse.core.catalog._import_backend_module", return_value=fake_module):
        _reset_known_models_cache()
        load_backend("ov-model", device="cpu")

    assert fake_class.call_args.kwargs["device"] == "auto"


def test_set_device_override_writes_field(tmp_catalog):
    from muse.core.catalog import (
        pull, set_device_override, _read_catalog,
    )
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake/local"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")

    set_device_override("soprano-80m", "cuda")
    assert _read_catalog()["soprano-80m"]["device_override"] == "cuda"


def test_set_device_override_clears_with_none(tmp_catalog):
    from muse.core.catalog import (
        pull, set_device_override, _read_catalog,
    )
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake/local"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")

    set_device_override("soprano-80m", "cuda")
    set_device_override("soprano-80m", None)
    assert "device_override" not in _read_catalog()["soprano-80m"]


def test_set_device_override_validates_device(tmp_catalog):
    from muse.core.catalog import (
        pull, set_device_override,
    )
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake/local"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")

    with pytest.raises(ValueError, match="device"):
        set_device_override("soprano-80m", "gpu")  # not a valid device label


def test_set_device_override_raises_for_unpulled(tmp_catalog):
    from muse.core.catalog import set_device_override
    with pytest.raises(KeyError, match="not pulled"):
        set_device_override("never-pulled-xyz", "cuda")


def test_load_backend_bundled_path_unchanged(tmp_catalog):
    """Regression: bundled-path load_backend still works (no manifest in catalog).

    The new merging logic gates on `persisted_manifest` being non-empty;
    bundled entries have no manifest field, so the merge is a no-op
    apart from injecting `model_id` (which bundled scripts already
    accept via **_).
    """
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake/local"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.Model = fake_class
    with patch("muse.core.catalog._import_backend_module", return_value=fake_module):
        load_backend("soprano-80m", device="cpu")
    kwargs = fake_class.call_args.kwargs
    # model_id injected
    assert kwargs["model_id"] == "soprano-80m"
    # bundled path semantics preserved
    assert kwargs["hf_repo"] == "ekwek/Soprano-1.1-80M"
    assert kwargs["local_dir"] == "/fake/local"
    assert kwargs["device"] == "cpu"


# --- v0.11.0: curated alias dispatch in pull() -----------------------------


def test_pull_curated_resolver_id_uses_curated_id_in_catalog(tmp_catalog):
    """`muse pull qwen3-8b-q4` (curated alias) persists under `qwen3-8b-q4`,
    not under the resolver's synthesized id like `qwen3-8b-instruct-gguf-q4-k-m`."""
    from muse.core.catalog import pull
    from muse.core.curated import CuratedEntry, _reset_curated_cache_for_tests
    from muse.core.resolvers import (
        Resolver, ResolvedModel, register_resolver, _reset_registry_for_tests,
    )

    class _FakeResolver(Resolver):
        scheme = "fake"
        def resolve(self, uri):
            return ResolvedModel(
                manifest={
                    "model_id": "long-ugly-synthesized-id",
                    "modality": "chat/completion",
                    "hf_repo": "fake/repo",
                    "backend_path": "x.y:Z",
                },
                backend_path="x.y:Z",
                download=lambda cache: cache / "w",
            )
        def search(self, q, **k):
            return []

    _reset_registry_for_tests()
    _reset_curated_cache_for_tests()
    register_resolver(_FakeResolver())

    # Patch find_curated to return a curated alias for "friendly-id"
    fake_curated = CuratedEntry(
        id="friendly-id",
        bundled=False,
        uri="fake://anything",
        modality="chat/completion",
        size_gb=1.0,
        description="aliased",
        tags=(),
    )
    try:
        with patch("muse.core.catalog.find_curated", return_value=fake_curated), \
             patch("muse.core.catalog.create_venv"), \
             patch("muse.core.catalog.install_into_venv"), \
             patch("muse.core.catalog.check_system_packages", return_value=[]):
            pull("friendly-id")
    finally:
        _reset_registry_for_tests()
        _reset_curated_cache_for_tests()

    catalog = _read_catalog()
    # Curated id wins over the resolver's synthesized id
    assert "friendly-id" in catalog
    assert "long-ugly-synthesized-id" not in catalog
    assert catalog["friendly-id"]["manifest"]["model_id"] == "friendly-id"
    assert catalog["friendly-id"]["source"] == "fake://anything"


def test_pull_curated_bundled_alias_uses_bundled_path(tmp_catalog):
    """A curated entry with `bundled: true` should route through the
    bundled-script path, not the resolver path."""
    from muse.core.catalog import pull, is_pulled
    from muse.core.curated import CuratedEntry

    fake_curated = CuratedEntry(
        id="kokoro-82m",
        bundled=True,
        uri=None,
        modality="audio/speech",
        size_gb=None,
        description=None,
        tags=(),
    )
    with patch("muse.core.catalog.find_curated", return_value=fake_curated), \
         patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("kokoro-82m")
    assert is_pulled("kokoro-82m")
    # Bundled path: no `manifest` field persisted (legacy shape)
    assert "manifest" not in _read_catalog()["kokoro-82m"]


def test_pull_bare_id_unaffected_by_curated_cache(tmp_catalog):
    """Regression: pulling a bare bundled id that's NOT in the curated list
    works exactly as before (no spurious dispatch)."""
    from muse.core.catalog import pull, is_pulled
    from muse.core.curated import _reset_curated_cache_for_tests

    _reset_curated_cache_for_tests()
    # find_curated returns None for the real "soprano-80m" if the YAML's
    # bundled entry is `kokoro-82m` etc. — make it explicit:
    with patch("muse.core.catalog.find_curated", return_value=None), \
         patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    assert is_pulled("soprano-80m")


# --- v0.11.3: unknown-id error includes curated ids + did-you-mean ----------


def test_pull_unknown_id_suggests_close_curated_match(tmp_catalog):
    """Typing a stale curated id (e.g. `qwen3-8b-q4` after rename to
    `qwen3.5-9b-q4`) should surface close matches from the curated list
    in the error, not just bundled ids."""
    from muse.core.catalog import pull
    from muse.core.curated import CuratedEntry

    fake_curated = [
        CuratedEntry(
            id="qwen3.5-9b-q4", bundled=False,
            uri="hf://x/y", modality="chat/completion",
            size_gb=5.0, description="close match", tags=(),
        ),
    ]
    with patch("muse.core.catalog.find_curated", return_value=None), \
         patch("muse.core.curated.load_curated", return_value=fake_curated):
        with pytest.raises(KeyError) as exc_info:
            pull("qwen3-8b-q4")
    msg = str(exc_info.value)
    assert "qwen3-8b-q4" in msg
    assert "did you mean" in msg.lower()
    assert "qwen3.5-9b-q4" in msg


def test_pull_unknown_id_with_no_close_matches_suggests_models_list(tmp_catalog):
    """If no close match, the error should tell the user how to find ids."""
    from muse.core.catalog import pull

    # patch curated to empty so only bundled ids are candidates
    with patch("muse.core.catalog.find_curated", return_value=None), \
         patch("muse.core.curated.load_curated", return_value=[]):
        with pytest.raises(KeyError) as exc_info:
            pull("total-gibberish-no-match-zzzzz")
    msg = str(exc_info.value).lower()
    assert "total-gibberish-no-match-zzzzz" in msg
    assert "muse models list" in msg
    # Should NOT show the "did you mean" since nothing is close
    assert "did you mean" not in msg


def test_pull_unknown_id_error_includes_curated_ids_in_did_you_mean(tmp_catalog):
    """The did-you-mean list draws from both bundled AND curated."""
    from muse.core.catalog import pull
    from muse.core.curated import CuratedEntry

    fake_curated = [
        CuratedEntry(
            id="llama-3.2-3b-q4", bundled=False, uri="hf://b/a",
            modality="chat/completion", size_gb=2.0,
            description="", tags=(),
        ),
        CuratedEntry(
            id="qwen3.5-9b-q4", bundled=False, uri="hf://b/c",
            modality="chat/completion", size_gb=5.0,
            description="", tags=(),
        ),
    ]
    with patch("muse.core.catalog.find_curated", return_value=None), \
         patch("muse.core.curated.load_curated", return_value=fake_curated):
        with pytest.raises(KeyError) as exc_info:
            pull("llama-3.2-3b")  # close to llama-3.2-3b-q4
    msg = str(exc_info.value)
    assert "did you mean" in msg.lower()
    assert "llama-3.2-3b-q4" in msg


# --- Task 2: curated capabilities overlay merges into persisted manifest -----


def test_pull_via_resolver_merges_curated_capabilities_overlay(tmp_catalog):
    """Curated entries may carry a capabilities dict; it merges into the
    persisted manifest's capabilities so the runtime gets the overlay."""
    from unittest.mock import patch
    from muse.core.catalog import pull, _read_catalog
    from muse.core.curated import CuratedEntry
    from muse.core.resolvers import ResolvedModel

    fake_curated = CuratedEntry(
        id="my-model",
        bundled=False,
        uri="hf://org/repo",
        modality="embedding/text",
        size_gb=0.5,
        description="custom",
        tags=(),
        capabilities={"trust_remote_code": True, "custom_flag": 42},
    )

    fake_resolved = ResolvedModel(
        manifest={
            "model_id": "repo",
            "modality": "embedding/text",
            "hf_repo": "org/repo",
            "pip_extras": [],
            "system_packages": [],
            "capabilities": {"base_caps_key": "base_val"},
        },
        backend_path="fake.mod:Cls",
        download=lambda cache_root: cache_root / "weights" / "my-model",
    )

    # resolve() is imported locally inside _pull_via_resolver
    # (`from muse.core.resolvers import resolve`), so the patch must
    # target the source module, not muse.core.catalog.
    with patch("muse.core.catalog.find_curated", return_value=fake_curated), \
         patch("muse.core.resolvers.resolve", return_value=fake_resolved), \
         patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]), \
         patch("muse.core.catalog.venv_python", return_value="/fake/py"):
        pull("my-model")

    catalog = _read_catalog()
    assert "my-model" in catalog
    persisted = catalog["my-model"]["manifest"]
    assert persisted["capabilities"] == {
        "base_caps_key": "base_val",
        "trust_remote_code": True,
        "custom_flag": 42,
    }


def test_pull_via_resolver_overlay_wins_on_collision(tmp_catalog):
    """On key collision, curated capabilities win (curated is hand-edited
    source of truth; resolver output is heuristic)."""
    from unittest.mock import patch
    from muse.core.catalog import pull, _read_catalog
    from muse.core.curated import CuratedEntry
    from muse.core.resolvers import ResolvedModel

    fake_curated = CuratedEntry(
        id="collide",
        bundled=False,
        uri="hf://org/repo",
        modality="embedding/text",
        size_gb=None,
        description=None,
        tags=(),
        capabilities={"shared_key": "curated_wins"},
    )
    fake_resolved = ResolvedModel(
        manifest={
            "model_id": "repo",
            "modality": "embedding/text",
            "hf_repo": "org/repo",
            "pip_extras": [],
            "system_packages": [],
            "capabilities": {"shared_key": "resolver_loses"},
        },
        backend_path="fake.mod:Cls",
        download=lambda cache_root: cache_root / "weights" / "collide",
    )
    with patch("muse.core.catalog.find_curated", return_value=fake_curated), \
         patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]), \
         patch("muse.core.catalog.venv_python", return_value="/fake/py"), \
         patch("muse.core.resolvers.resolve", return_value=fake_resolved):
        pull("collide")

    persisted = _read_catalog()["collide"]["manifest"]
    assert persisted["capabilities"]["shared_key"] == "curated_wins"


def test_pull_uri_direct_inherits_curated_capabilities(tmp_catalog):
    """`muse pull hf://...` should pick up the curated overlay when the URI
    matches a curated entry, even if the user didn't type the curated id.

    Regression for the C2 bug: copying a URI from `muse search` and
    pasting it into `muse pull` previously produced a broken model
    (no `safe_labels` for KoalaAI), because find_curated() only matched
    by id and the URI path bypassed curated entirely.
    """
    from unittest.mock import patch
    from muse.core.catalog import pull, _read_catalog
    from muse.core.curated import CuratedEntry
    from muse.core.resolvers import ResolvedModel

    fake_curated = CuratedEntry(
        id="text-moderation",
        bundled=False,
        uri="hf://KoalaAI/Text-Moderation",
        modality="text/classification",
        size_gb=0.14,
        description="9-cat",
        tags=(),
        capabilities={"safe_labels": ["OK"]},
    )
    fake_resolved = ResolvedModel(
        manifest={
            "model_id": "text-moderation",
            "modality": "text/classification",
            "hf_repo": "KoalaAI/Text-Moderation",
            "pip_extras": [],
            "system_packages": [],
            "capabilities": {},
        },
        backend_path="fake.mod:Cls",
        download=lambda cache_root: cache_root / "weights" / "tm",
    )
    # find_curated by id returns None (user didn't type curated id);
    # find_curated_by_uri returns the entry (URI matches curated.yaml).
    with patch("muse.core.catalog.find_curated", return_value=None), \
         patch("muse.core.catalog.find_curated_by_uri", return_value=fake_curated), \
         patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]), \
         patch("muse.core.catalog.venv_python", return_value="/fake/py"), \
         patch("muse.core.resolvers.resolve", return_value=fake_resolved):
        pull("hf://KoalaAI/Text-Moderation")

    persisted = _read_catalog()["text-moderation"]["manifest"]
    assert persisted["capabilities"]["safe_labels"] == ["OK"]


# --- v0.17.3: re-apply curated capabilities overlay at known_models() time ---


def test_known_models_reapplies_curated_capabilities_overlay_at_runtime(
    tmp_catalog,
):
    """Edits to curated.yaml take effect on next known_models() call without
    requiring a re-pull. Critical: a v0.16.2-style curated.yaml edit that
    adds `device: cpu` to an already-pulled model must surface in
    catalog entries on next process restart.
    """
    from unittest.mock import patch
    from muse.core.catalog import known_models, _write_catalog, _reset_known_models_cache
    from muse.core.curated import CuratedEntry

    # Persisted manifest has device:cuda (stale; from before v0.16.2 edit)
    catalog_state = {
        "stale-model": {
            "pulled_at": "2026-01-01T00:00:00+00:00",
            "hf_repo": "org/repo",
            "local_dir": "/fake",
            "venv_path": "/fake/venv",
            "python_path": "/fake/py",
            "enabled": True,
            "source": "hf://org/repo",
            "manifest": {
                "model_id": "stale-model",
                "modality": "embedding/text",
                "hf_repo": "org/repo",
                "backend_path": "fake.module:Cls",
                "pip_extras": [],
                "system_packages": [],
                "capabilities": {"device": "cuda", "old_key": "old_val"},
            },
        },
    }
    _write_catalog(catalog_state)

    # Curated.yaml (post-edit) declares device: cpu for this id
    fake_curated = CuratedEntry(
        id="stale-model",
        bundled=False,
        uri="hf://org/repo",
        modality="embedding/text",
        size_gb=None, description=None, tags=(),
        capabilities={"device": "cpu", "new_key": "new_val"},
    )

    with patch("muse.core.catalog.find_curated", return_value=fake_curated):
        _reset_known_models_cache()
        models = known_models()

    entry = models["stale-model"]
    # Curated overlay applied: device:cpu wins over device:cuda
    assert entry.extra["device"] == "cpu"
    # New curated key surfaces
    assert entry.extra["new_key"] == "new_val"
    # Persisted-manifest-only key still present (overlay shallow-merges, doesn't replace)
    assert entry.extra["old_key"] == "old_val"


def test_get_manifest_reapplies_curated_capabilities_overlay_at_runtime(
    tmp_catalog,
):
    """get_manifest() must apply the same curated-capabilities overlay as
    known_models(), so a curated.yaml edit takes effect for BOTH the
    entry known_models() surfaces (which load_backend/entry.extra reads
    at construction time) and the manifest get_manifest() returns (which
    worker.py uses to register/gate the model). Before the fix these two
    read paths could diverge after a curated.yaml edit + restart.
    """
    from unittest.mock import patch
    from muse.core.catalog import get_manifest, _write_catalog, _reset_known_models_cache
    from muse.core.curated import CuratedEntry

    catalog_state = {
        "stale-model": {
            "pulled_at": "2026-01-01T00:00:00+00:00",
            "hf_repo": "org/repo",
            "local_dir": "/fake",
            "venv_path": "/fake/venv",
            "python_path": "/fake/py",
            "enabled": True,
            "source": "hf://org/repo",
            "manifest": {
                "model_id": "stale-model",
                "modality": "embedding/text",
                "hf_repo": "org/repo",
                "backend_path": "fake.module:Cls",
                "pip_extras": [],
                "system_packages": [],
                "capabilities": {"device": "cuda", "old_key": "old_val"},
            },
        },
    }
    _write_catalog(catalog_state)

    fake_curated = CuratedEntry(
        id="stale-model",
        bundled=False,
        uri="hf://org/repo",
        modality="embedding/text",
        size_gb=None, description=None, tags=(),
        capabilities={"device": "cpu", "new_key": "new_val"},
    )

    with patch("muse.core.catalog.find_curated", return_value=fake_curated):
        _reset_known_models_cache()
        manifest = get_manifest("stale-model")

    assert manifest["capabilities"]["device"] == "cpu"
    assert manifest["capabilities"]["new_key"] == "new_val"
    assert manifest["capabilities"]["old_key"] == "old_val"


def test_known_models_uri_based_curated_lookup_when_id_misses(tmp_catalog):
    """If find_curated(id) misses, fall back to find_curated_by_uri(source)."""
    from unittest.mock import patch
    from muse.core.catalog import known_models, _write_catalog, _reset_known_models_cache
    from muse.core.curated import CuratedEntry

    catalog_state = {
        "weird-id": {
            "pulled_at": "2026-01-01T00:00:00+00:00",
            "hf_repo": "org/repo",
            "local_dir": "/fake",
            "venv_path": "/fake/venv",
            "python_path": "/fake/py",
            "enabled": True,
            "source": "hf://org/repo",
            "manifest": {
                "model_id": "weird-id",
                "modality": "embedding/text",
                "hf_repo": "org/repo",
                "backend_path": "fake.module:Cls",
                "pip_extras": [],
                "system_packages": [],
                "capabilities": {},
            },
        },
    }
    _write_catalog(catalog_state)

    fake_uri_curated = CuratedEntry(
        id="canonical-id",  # different from "weird-id"
        bundled=False,
        uri="hf://org/repo",
        modality="embedding/text",
        size_gb=None, description=None, tags=(),
        capabilities={"device": "cpu"},
    )

    with patch("muse.core.catalog.find_curated", return_value=None), \
         patch("muse.core.catalog.find_curated_by_uri", return_value=fake_uri_curated):
        _reset_known_models_cache()
        models = known_models()

    assert models["weird-id"].extra["device"] == "cpu"


def test_load_backend_bundled_script_device_capability_honored(tmp_catalog):
    """Bundled scripts (no persisted manifest) must have their MANIFEST
    capabilities.device honored, not just resolver-pulled models.

    Regression: pre-v0.18.1 read capabilities only from persisted_manifest,
    leaving bundled scripts' capabilities silently ignored at load time.
    A bundled-pull catalog entry has no `manifest` field (only resolver
    pulls persist one); the fix routes through entry.extra which
    known_models() populates from BOTH sources.
    """
    from muse.core.catalog import load_backend, _write_catalog

    # Simulate a bundled-pull catalog entry: NO persisted manifest field.
    # Just the bare-id pull metadata.
    catalog_state = {
        "kokoro-82m": {
            "pulled_at": "2026-01-01T00:00:00+00:00",
            "hf_repo": "hexgrad/Kokoro-82M",
            "local_dir": "/fake/kokoro",
            "venv_path": "/fake/venv",
            "python_path": "/fake/py",
            "enabled": True,
            # no "manifest" key (bundled-pull contract)
        },
    }
    _write_catalog(catalog_state)

    fake_class = MagicMock()
    fake_module = MagicMock()
    fake_module.Model = fake_class

    # Build a fake known_models result where kokoro-82m's extra holds
    # device: cpu (the bundled MANIFEST's capabilities)
    fake_entry = CatalogEntry(
        model_id="kokoro-82m",
        modality="audio/speech",
        backend_path="muse.models.kokoro_82m:Model",
        hf_repo="hexgrad/Kokoro-82M",
        description="...",
        pip_extras=(),
        system_packages=(),
        extra={"device": "cpu"},  # bundled MANIFEST capability
    )

    with patch(
        "muse.core.catalog.known_models",
        return_value={"kokoro-82m": fake_entry},
    ), patch(
        "muse.core.catalog.is_pulled",
        return_value=True,
    ), patch(
        "muse.core.catalog._import_backend_module",
        return_value=fake_module,
    ):
        load_backend("kokoro-82m", device="cuda")

    # Capability cpu MUST win over kwarg cuda. Pre-v0.18.1 this assertion failed.
    assert fake_class.call_args.kwargs["device"] == "cpu"


def test_dir_size_bytes_returns_zero_for_missing_path(tmp_path):
    """_dir_size_bytes returns 0 for a path that doesn't exist."""
    from muse.core.catalog import _dir_size_bytes
    assert _dir_size_bytes(str(tmp_path / "does-not-exist")) == 0


def test_dir_size_bytes_returns_zero_for_empty_dir(tmp_path):
    """_dir_size_bytes returns 0 for an empty directory."""
    from muse.core.catalog import _dir_size_bytes
    assert _dir_size_bytes(str(tmp_path)) == 0


def test_dir_size_bytes_sums_file_sizes_recursively(tmp_path):
    """_dir_size_bytes walks subdirectories and totals all file sizes."""
    from muse.core.catalog import _dir_size_bytes
    (tmp_path / "a.bin").write_bytes(b"x" * 1024)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * 2048)
    deeper = sub / "deeper"
    deeper.mkdir()
    (deeper / "c.bin").write_bytes(b"z" * 4096)
    assert _dir_size_bytes(str(tmp_path)) == 1024 + 2048 + 4096


def test_dir_size_bytes_does_not_follow_symlinks(tmp_path):
    """Symlinks must not double-count contents from elsewhere."""
    import os
    from muse.core.catalog import _dir_size_bytes
    target = tmp_path / "target"
    target.mkdir()
    (target / "big.bin").write_bytes(b"x" * 1024 * 1024)  # 1 MB
    pulled = tmp_path / "pulled"
    pulled.mkdir()
    os.symlink(str(target), str(pulled / "link"))
    # _dir_size_bytes(pulled) walks pulled/ but does not descend into
    # the symlinked target directory.
    assert _dir_size_bytes(str(pulled)) == 0


@pytest.mark.parametrize("nbytes,expected", [
    (0, "-"),
    (1024, "1 KB"),
    (1024 * 500, "500 KB"),
    (1024 * 1024, "1 MB"),
    (1024 * 1024 * 250, "250 MB"),
    (1024**3, "1.0 GB"),
    (int(2.5 * 1024**3), "2.5 GB"),
    (7 * 1024**3, "7.0 GB"),
])
def test_human_size_formats_bytes(nbytes, expected):
    """_human_size renders bytes as '- / N KB / N MB / N.N GB'."""
    from muse.core.catalog import _human_size
    assert _human_size(nbytes) == expected


def test_pull_bundled_honors_allow_patterns_capability(tmp_catalog):
    """MANIFEST.capabilities.allow_patterns flows into snapshot_download."""
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake") as mock_download, \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        # sd-turbo MANIFEST declares allow_patterns in v0.18.2
        pull("sd-turbo")
    # Verify snapshot_download was called with allow_patterns
    mock_download.assert_called_once()
    kwargs = mock_download.call_args.kwargs
    assert "allow_patterns" in kwargs
    patterns = kwargs["allow_patterns"]
    # spot-check a few patterns we know should be there
    assert any("fp16" in p for p in patterns)
    assert any("unet" in p for p in patterns)


def test_pull_bundled_no_allow_patterns_calls_default(tmp_catalog):
    """Models without allow_patterns capability call snapshot_download without it."""
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake") as mock_download, \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("kokoro-82m")  # no allow_patterns in MANIFEST
    mock_download.assert_called_once()
    kwargs = mock_download.call_args.kwargs
    assert "allow_patterns" not in kwargs
    assert "revision" not in kwargs


def test_pull_bundled_honors_pinned_revision(tmp_catalog):
    """Exact custom architectures must download the reviewed snapshot."""
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch(
             "muse.core.catalog.snapshot_download",
             return_value="/fake/starvector",
         ) as mock_download, \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("starvector-1b-im2svg")

    kwargs = mock_download.call_args.kwargs
    revision = (
        "380ab95d25a8e9ab1dc825debe238b4953ae13b9"
    )
    assert kwargs["revision"] == revision
    assert "*.safetensors" in kwargs["allow_patterns"]
    assert "*.py" not in kwargs["allow_patterns"]
    assert _read_catalog()["starvector-1b-im2svg"]["revision"] == revision


def test_known_models_surfaces_memory_annotation():
    """Bundled MANIFESTs that declare capabilities.memory_gb expose it on the entry."""
    catalog = known_models()
    # kokoro-82m got memory_gb in v0.18.2
    assert "memory_gb" in catalog["kokoro-82m"].extra
    assert catalog["kokoro-82m"].extra["memory_gb"] == 0.5
    # sd-turbo also got it
    assert catalog["sd-turbo"].extra["memory_gb"] == 4.0


def test_known_models_no_curated_match_leaves_manifest_unchanged(tmp_catalog):
    """If no curated entry matches, the persisted manifest is used as-is."""
    from unittest.mock import patch
    from muse.core.catalog import known_models, _write_catalog, _reset_known_models_cache

    catalog_state = {
        "orphan": {
            "pulled_at": "2026-01-01T00:00:00+00:00",
            "hf_repo": "org/repo",
            "local_dir": "/fake",
            "venv_path": "/fake/venv",
            "python_path": "/fake/py",
            "enabled": True,
            "manifest": {
                "model_id": "orphan",
                "modality": "embedding/text",
                "hf_repo": "org/repo",
                "backend_path": "fake.module:Cls",
                "pip_extras": [],
                "system_packages": [],
                "capabilities": {"device": "cuda"},
            },
        },
    }
    _write_catalog(catalog_state)

    with patch("muse.core.catalog.find_curated", return_value=None), \
         patch("muse.core.catalog.find_curated_by_uri", return_value=None):
        _reset_known_models_cache()
        models = known_models()

    # Unchanged
    assert models["orphan"].extra["device"] == "cuda"


class TestResetKnownModelsCacheLocking:
    """L9: _reset_known_models_cache must take _KNOWN_MODELS_LOCK.

    Without the lock, the invalidator's `cache = None` races the
    lock-guarded rebuild in known_models(); a slow rebuild that read a
    pre-mutation catalog can write its stale snapshot back AFTER the
    invalidator ran, resurrecting a cache that hides a just-pulled model.
    """

    def test_invalidator_blocks_while_lock_held(self):
        import threading

        from muse.core import catalog as catalog_mod

        # Seed a sentinel cache so we can observe when it gets cleared.
        catalog_mod._known_models_cache = {"sentinel": object()}  # type: ignore[assignment]
        started = threading.Event()
        finished = threading.Event()

        def invalidate():
            started.set()
            catalog_mod._reset_known_models_cache()
            finished.set()

        with catalog_mod._KNOWN_MODELS_LOCK:
            t = threading.Thread(target=invalidate)
            t.start()
            # The thread starts but must block on the lock we hold: the
            # cache stays the sentinel and finished never fires.
            assert started.wait(1.0)
            assert not finished.wait(0.2), "invalidator ran without the lock"
            assert catalog_mod._known_models_cache is not None
        # Lock released: the invalidator now completes and clears the cache.
        assert finished.wait(1.0)
        t.join(1.0)
        assert catalog_mod._known_models_cache is None


class TestLoraPullValidation:
    def _lora_manifest(self, caps):
        return {
            "model_id": "some-lora",
            "modality": "image/generation",
            "hf_repo": "org/some-lora",
            "backend_path": "muse.modalities.image_generation.runtimes.diffusers:DiffusersText2ImageModel",
            "capabilities": caps,
        }

    def test_lora_without_base_model_raises_actionable(self, tmp_catalog):
        import pytest
        from muse.core.catalog import _validate_lora_capabilities
        from muse.core.resolvers import ResolverError

        with pytest.raises(ResolverError, match="--base"):
            _validate_lora_capabilities(
                self._lora_manifest({"lora_adapter": True})
            )

    def test_lora_with_unpulled_muse_base_raises_actionable(self, tmp_catalog):
        import pytest
        from muse.core.catalog import _validate_lora_capabilities
        from muse.core.resolvers import ResolverError

        with pytest.raises(ResolverError, match="muse pull sdxl-turbo"):
            _validate_lora_capabilities(self._lora_manifest(
                {"lora_adapter": True, "base_model": "sdxl-turbo"}
            ))

    def test_lora_with_pulled_muse_base_passes(self, tmp_catalog):
        import json
        from muse.core.catalog import (
            _catalog_path, _reset_read_catalog_cache,
            _validate_lora_capabilities,
        )

        p = _catalog_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "sdxl-turbo": {"local_dir": "/w/sdxl-turbo", "enabled": True},
        }))
        _reset_read_catalog_cache()
        _validate_lora_capabilities(self._lora_manifest(
            {"lora_adapter": True, "base_model": "sdxl-turbo"}
        ))  # no raise

    def test_lora_with_hf_repo_base_passes_without_catalog(self, tmp_catalog):
        from muse.core.catalog import _validate_lora_capabilities

        _validate_lora_capabilities(self._lora_manifest({
            "lora_adapter": True,
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        }))  # no raise: HF-repo bases download at load time

    def test_non_lora_manifest_is_ignored(self, tmp_catalog):
        from muse.core.catalog import _validate_lora_capabilities

        _validate_lora_capabilities(self._lora_manifest({}))  # no raise


class TestPullBaseOverrideDispatch:
    """End-to-end dispatch coverage for `pull(identifier, base_override=...)`.

    Mocks `_pull_via_resolver` itself (rather than the deeper venv/download
    machinery) so each test can assert exactly what `pull()` decided to
    forward, across all three `://`-URI sub-branches plus the named-curated
    alias branch.
    """

    def _capture(self, monkeypatch):
        import muse.core.catalog as cat
        calls = {}

        def fake_pvr(uri, **kwargs):
            calls["uri"] = uri
            calls.update(kwargs)

        monkeypatch.setattr(cat, "_pull_via_resolver", fake_pvr)
        return calls

    def test_uri_branch_threads_base_override(self, tmp_catalog, monkeypatch):
        """base_override is a dedicated kwarg to _pull_via_resolver (C1),
        NOT merged into capabilities_overlay: the durable operator pin
        lives as a top-level catalog field, not a manifest capability
        that a later curated overlay could clobber."""
        from muse.core.catalog import pull

        calls = self._capture(monkeypatch)
        pull("hf://org/some-lora", base_override="sdxl-turbo")
        assert calls["base_override"] == "sdxl-turbo"
        assert calls.get("capabilities_overlay") is None

    def test_uri_branch_without_override_passes_none(self, tmp_catalog, monkeypatch):
        from muse.core.catalog import pull

        calls = self._capture(monkeypatch)
        pull("hf://org/some-lora")
        assert calls.get("base_override") is None
        assert calls.get("capabilities_overlay") is None

    def test_curated_alias_branch_threads_base_override_kwarg(
        self, tmp_catalog, monkeypatch,
    ):
        import muse.core.catalog as cat
        from muse.core.catalog import pull
        from muse.core.curated import CuratedEntry

        fake_curated = CuratedEntry(
            id="some-lora-alias",
            bundled=False,
            uri="hf://org/some-lora",
            modality="image/generation",
            size_gb=0.2,
            description="a lora",
            tags=(),
            capabilities={"trust_remote_code": True},
        )
        monkeypatch.setattr(cat, "find_curated", lambda ident: fake_curated)

        calls = self._capture(monkeypatch)
        pull("some-lora-alias", base_override="sdxl-turbo")

        assert calls["base_override"] == "sdxl-turbo"
        # Pre-existing curated capability keys still flow through the
        # (unrelated) capabilities_overlay.
        assert calls["capabilities_overlay"]["trust_remote_code"] is True
        assert "base_model" not in calls["capabilities_overlay"]
        assert calls["model_id_override"] == "some-lora-alias"

    def test_uri_matching_curated_by_uri_threads_base_override(
        self, tmp_catalog, monkeypatch,
    ):
        """Finding 1 regression: `--base` must not be dropped when the raw
        URI happens to match a curated entry by URI (as opposed to the user
        typing the curated alias id).
        """
        import muse.core.catalog as cat
        from muse.core.catalog import pull
        from muse.core.curated import CuratedEntry

        fake_uri_curated = CuratedEntry(
            id="some-lora-alias",
            bundled=False,
            uri="hf://org/some-lora",
            modality="image/generation",
            size_gb=0.2,
            description="a lora",
            tags=(),
            capabilities={"trust_remote_code": True},
        )
        monkeypatch.setattr(cat, "find_curated", lambda ident: None)
        monkeypatch.setattr(cat, "find_curated_by_uri", lambda uri: fake_uri_curated)

        calls = self._capture(monkeypatch)
        pull("hf://org/some-lora", base_override="sdxl-turbo")

        assert calls["base_override"] == "sdxl-turbo"
        assert calls["capabilities_overlay"]["trust_remote_code"] is True

    def test_bare_id_resolver_sourced_threads_base_override(
        self, tmp_catalog, monkeypatch,
    ):
        """A resolver-sourced bare id (re-pull-by-friendly-id path) must
        thread --base through to _pull_via_resolver, not warn-and-ignore.
        The warning is reserved for true bundled-script pulls, which have
        no LoRA base to set."""
        import muse.core.catalog as cat
        from muse.core.catalog import pull

        _write_persisted_resolver_entry(
            tmp_catalog,
            model_id="pixel-art-xl",
            modality="image/generation",
            hf_repo="nerijs/pixel-art-xl",
            capabilities={"lora_adapter": True},
        )
        _reset_known_models_cache()
        monkeypatch.setattr(cat, "find_curated_by_uri", lambda uri: None)

        calls = self._capture(monkeypatch)
        pull("pixel-art-xl", base_override="sdxl-turbo")

        assert calls["base_override"] == "sdxl-turbo"
        assert calls["model_id_override"] == "pixel-art-xl"

    def test_bundled_bare_id_with_base_override_warns_and_ignores(
        self, tmp_catalog, monkeypatch, caplog,
    ):
        """A true bundled-script pull (no curated alias, no resolver
        source) has no LoRA base to set; --base is warned-and-ignored,
        unchanged from before."""
        import logging
        from muse.core.catalog import pull, known_models
        from muse.core.curated import find_curated

        caplog.set_level(logging.WARNING)
        # Pick a bundled id with no curated alias, so dispatch falls all
        # the way through to the bare-id / no-source-uri warning branch
        # rather than short-circuiting via the curated-bundled path.
        bundled_id = next(
            mid for mid in known_models() if find_curated(mid) is None
        )
        with patch("muse.core.catalog._pull_bundled") as mock_bundled:
            pull(bundled_id, base_override="sdxl-turbo")
        mock_bundled.assert_called_once_with(bundled_id)
        assert "--base only applies" in caplog.text


class TestBaseOverrideDurability:
    """C1/I4: an operator --base pin is a top-level catalog field (mirrors
    device_override), so it survives curated-overlay re-application in
    known_models() and get_manifest(), and survives re-pulls that omit
    --base."""

    def test_known_models_applies_base_override_over_curated_base_model(
        self, tmp_catalog, monkeypatch,
    ):
        import muse.core.catalog as cat
        from muse.core.catalog import known_models, _catalog_path
        from muse.core.curated import CuratedEntry
        import json

        _write_persisted_resolver_entry(
            tmp_catalog,
            model_id="pixel-art-xl",
            modality="image/generation",
            hf_repo="nerijs/pixel-art-xl",
            capabilities={"lora_adapter": True, "base_model": "sdxl-turbo"},
        )
        # Mark the entry with a durable operator override that conflicts
        # with what curated.yaml declares below.
        p = _catalog_path()
        data = json.loads(p.read_text())
        data["pixel-art-xl"]["base_override"] = "flux-schnell"
        p.write_text(json.dumps(data))
        _reset_known_models_cache()

        fake_curated = CuratedEntry(
            id="pixel-art-xl",
            bundled=False,
            uri="hf://nerijs/pixel-art-xl",
            modality="image/generation",
            size_gb=0.05,
            description="pixel art lora",
            tags=(),
            capabilities={"base_model": "sdxl-turbo"},
        )
        monkeypatch.setattr(cat, "find_curated", lambda ident: fake_curated)

        entries = known_models()
        assert entries["pixel-art-xl"].extra["base_model"] == "flux-schnell"

    def test_get_manifest_applies_base_override(self, tmp_catalog):
        from muse.core.catalog import get_manifest, _catalog_path
        import json

        _write_persisted_resolver_entry(
            tmp_catalog,
            model_id="pixel-art-xl",
            modality="image/generation",
            hf_repo="nerijs/pixel-art-xl",
            capabilities={"lora_adapter": True, "base_model": "sdxl-turbo"},
        )
        p = _catalog_path()
        data = json.loads(p.read_text())
        data["pixel-art-xl"]["base_override"] = "flux-schnell"
        p.write_text(json.dumps(data))
        _reset_read_catalog_cache()
        _reset_known_models_cache()

        manifest = get_manifest("pixel-art-xl")
        assert manifest["capabilities"]["base_model"] == "flux-schnell"

    def test_repull_without_base_preserves_prior_base_override(
        self, tmp_catalog, monkeypatch,
    ):
        """Re-pulling (no --base) must NOT revert a previously-set operator
        override back to the tag-declared / curated base."""
        import json
        from muse.core.catalog import _pull_via_resolver, _catalog_path
        from muse.core.resolvers import ResolvedModel

        _write_persisted_resolver_entry(
            tmp_catalog,
            model_id="pixel-art-xl",
            modality="image/generation",
            hf_repo="nerijs/pixel-art-xl",
            capabilities={"lora_adapter": True, "base_model": "sdxl-turbo"},
        )
        p = _catalog_path()
        data = json.loads(p.read_text())
        data["pixel-art-xl"]["base_override"] = "sdxl-turbo-pinned"
        # The override target must itself be "pulled" for
        # _validate_lora_capabilities to accept the preserved override.
        data["sdxl-turbo-pinned"] = {"local_dir": "/w/sdxl-turbo-pinned", "enabled": True}
        p.write_text(json.dumps(data))
        _reset_read_catalog_cache()
        _reset_known_models_cache()

        def fake_resolve(uri, *, modality=None, base_override=None):
            return ResolvedModel(
                manifest={
                    "model_id": "pixel-art-xl",
                    "modality": "image/generation",
                    "hf_repo": "nerijs/pixel-art-xl",
                    "backend_path": (
                        "muse.modalities.image_generation.runtimes.diffusers"
                        ":DiffusersText2ImageModel"
                    ),
                    "capabilities": {
                        "lora_adapter": True,
                        "base_model": "sdxl-turbo",
                    },
                },
                backend_path=(
                    "muse.modalities.image_generation.runtimes.diffusers"
                    ":DiffusersText2ImageModel"
                ),
                download=lambda cache: cache / "w",
            )

        monkeypatch.setattr("muse.core.resolvers.resolve", fake_resolve)
        with patch("muse.core.catalog.create_venv"), \
             patch("muse.core.catalog.install_into_venv"), \
             patch("muse.core.catalog.check_system_packages", return_value=[]):
            _pull_via_resolver(
                "hf://nerijs/pixel-art-xl",
                model_id_override="pixel-art-xl",
            )

        catalog = _read_catalog()
        entry = catalog["pixel-art-xl"]
        assert entry.get("base_override") == "sdxl-turbo-pinned"
        assert entry["manifest"]["capabilities"]["base_model"] == "sdxl-turbo-pinned"


class TestGpuLayersOverride:
    """Spec 2026-07-08: operator pin for llama.cpp n_gpu_layers."""

    def _seed(self, tmp_path, monkeypatch, capabilities=None):
        """Seed a resolver-pulled-style catalog entry with a persisted
        manifest so known_models() picks up capabilities."""
        import json
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        from muse.core.catalog import _reset_known_models_cache
        entry = {
            "pulled_at": "...", "hf_repo": "org/repo", "local_dir": "/w",
            "venv_path": "/v", "python_path": "/v/bin/python",
            "enabled": True, "source": "hf://org/repo",
            "manifest": {
                "model_id": "test-gguf", "modality": "chat/completion",
                "hf_repo": "org/repo",
                "backend_path": "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel",
                "capabilities": capabilities or {"gguf_file": "m.gguf"},
            },
        }
        (tmp_path / "catalog.json").write_text(json.dumps({"test-gguf": entry}))
        _reset_known_models_cache()

    def test_set_and_clear_round_trip(self, tmp_path, monkeypatch):
        from muse.core.catalog import _read_catalog, set_gpu_layers_override
        self._seed(tmp_path, monkeypatch)
        set_gpu_layers_override("test-gguf", 30)
        assert _read_catalog()["test-gguf"]["gpu_layers_override"] == 30
        set_gpu_layers_override("test-gguf", None)
        assert "gpu_layers_override" not in _read_catalog()["test-gguf"]

    def test_minus_one_and_zero_are_valid(self, tmp_path, monkeypatch):
        from muse.core.catalog import _read_catalog, set_gpu_layers_override
        self._seed(tmp_path, monkeypatch)
        set_gpu_layers_override("test-gguf", -1)
        assert _read_catalog()["test-gguf"]["gpu_layers_override"] == -1
        set_gpu_layers_override("test-gguf", 0)
        assert _read_catalog()["test-gguf"]["gpu_layers_override"] == 0

    def test_invalid_values_raise(self, tmp_path, monkeypatch):
        from muse.core.catalog import set_gpu_layers_override
        self._seed(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            set_gpu_layers_override("test-gguf", -2)
        with pytest.raises(ValueError):
            set_gpu_layers_override("test-gguf", "thirty")

    def test_unknown_model_raises_keyerror(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        from muse.core.catalog import set_gpu_layers_override
        with pytest.raises(KeyError):
            set_gpu_layers_override("never-pulled", 10)

    def test_load_backend_pin_beats_capability(self, tmp_path, monkeypatch):
        """Precedence: catalog pin > capabilities.n_gpu_layers > default."""
        from unittest.mock import MagicMock, patch as mpatch
        from muse.core.catalog import load_backend, set_gpu_layers_override
        self._seed(tmp_path, monkeypatch,
                   capabilities={"gguf_file": "m.gguf", "n_gpu_layers": 10})
        set_gpu_layers_override("test-gguf", 30)
        fake_cls = MagicMock()
        fake_module = MagicMock()
        fake_module.LlamaCppModel = fake_cls
        with mpatch("muse.core.catalog._import_backend_module",
                    return_value=fake_module), \
             mpatch("muse.core.catalog.is_pulled", return_value=True):
            load_backend("test-gguf")
        assert fake_cls.call_args.kwargs["n_gpu_layers"] == 30

    def test_load_backend_capability_used_without_pin(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch as mpatch
        from muse.core.catalog import load_backend
        self._seed(tmp_path, monkeypatch,
                   capabilities={"gguf_file": "m.gguf", "n_gpu_layers": 10})
        fake_cls = MagicMock()
        fake_module = MagicMock()
        fake_module.LlamaCppModel = fake_cls
        with mpatch("muse.core.catalog._import_backend_module",
                    return_value=fake_module), \
             mpatch("muse.core.catalog.is_pulled", return_value=True):
            load_backend("test-gguf")
        assert fake_cls.call_args.kwargs["n_gpu_layers"] == 10

    def test_load_backend_pin_beats_caller_kwarg(self, tmp_path, monkeypatch):
        """The pin wins over BOTH the manifest capability and an explicit
        caller kwarg (mirrors test_load_backend_device_override_beats_caller_kwarg)."""
        from unittest.mock import MagicMock, patch as mpatch
        from muse.core.catalog import load_backend, set_gpu_layers_override
        self._seed(tmp_path, monkeypatch,
                   capabilities={"gguf_file": "m.gguf", "n_gpu_layers": 10})
        set_gpu_layers_override("test-gguf", 30)
        fake_cls = MagicMock()
        fake_module = MagicMock()
        fake_module.LlamaCppModel = fake_cls
        with mpatch("muse.core.catalog._import_backend_module",
                    return_value=fake_module), \
             mpatch("muse.core.catalog.is_pulled", return_value=True):
            load_backend("test-gguf", n_gpu_layers=5)
        assert fake_cls.call_args.kwargs["n_gpu_layers"] == 30

    def test_load_backend_absent_everywhere_passes_nothing(self, tmp_path, monkeypatch):
        """No pin + no capability: n_gpu_layers not in kwargs; the runtime
        default (-1) governs."""
        from unittest.mock import MagicMock, patch as mpatch
        from muse.core.catalog import load_backend
        self._seed(tmp_path, monkeypatch)  # gguf_file only
        fake_cls = MagicMock()
        fake_module = MagicMock()
        fake_module.LlamaCppModel = fake_cls
        with mpatch("muse.core.catalog._import_backend_module",
                    return_value=fake_module), \
             mpatch("muse.core.catalog.is_pulled", return_value=True):
            load_backend("test-gguf")
        assert "n_gpu_layers" not in fake_cls.call_args.kwargs

    def test_probe_flow_gets_pin_via_load_backend(self, tmp_path, monkeypatch):
        """probe_worker constructs via load_backend(model_id, device=...),
        so the pin flows into the probed construction with zero probe code.
        This test binds that seam: a caller-passed device kwarg must NOT
        displace the injected n_gpu_layers."""
        from unittest.mock import MagicMock, patch as mpatch
        from muse.core.catalog import load_backend, set_gpu_layers_override
        self._seed(tmp_path, monkeypatch)
        set_gpu_layers_override("test-gguf", 25)
        fake_cls = MagicMock()
        fake_module = MagicMock()
        fake_module.LlamaCppModel = fake_cls
        with mpatch("muse.core.catalog._import_backend_module",
                    return_value=fake_module), \
             mpatch("muse.core.catalog.is_pulled", return_value=True):
            load_backend("test-gguf", device="cuda")  # probe-style call
        assert fake_cls.call_args.kwargs["n_gpu_layers"] == 25
        assert fake_cls.call_args.kwargs["device"] == "cuda"


class TestModelIdFilesystemSafety:
    """model_id becomes `<catalog_dir>/venvs/<model_id>` (and, for
    resolver pulls, feeds the weights cache path too). A hostile or
    malformed identifier must be refused before any venv/weights path
    is constructed, for every identifier shape: bare id (this class),
    the resolver-synthesized id, and a curated alias resolution result.
    """

    def test_pull_bare_id_rejects_path_traversal(self, tmp_catalog):
        with pytest.raises(ValueError, match="invalid model id"):
            pull("../evil")
        assert not (tmp_catalog / "venvs").exists()

    def test_pull_bare_id_rejects_path_separator(self, tmp_catalog):
        with pytest.raises(ValueError, match="invalid model id"):
            pull("a/b")
        assert not (tmp_catalog / "venvs").exists()

    def test_pull_bare_id_rejects_dot_alone(self, tmp_catalog):
        with pytest.raises(ValueError, match="invalid model id"):
            pull(".")
        assert not (tmp_catalog / "venvs").exists()

    def test_pull_via_resolver_rejects_bad_model_id_override(self, tmp_catalog):
        """A curated alias / --base override that resolves to a malformed
        model_id must be refused before venv_path is constructed, not just
        the raw bare-id shape."""
        from muse.core.catalog import _pull_via_resolver
        from muse.core.resolvers import ResolvedModel

        fake_resolved = ResolvedModel(
            manifest={
                "model_id": "safe-id",
                "modality": "embedding/text",
                "hf_repo": "org/repo",
                "backend_path": "fake.module:Cls",
            },
            backend_path="fake.module:Cls",
            download=lambda cache_dir: str(cache_dir / "weights"),
        )
        with patch("muse.core.resolvers.resolve", return_value=fake_resolved):
            with pytest.raises(ValueError, match="invalid model id"):
                _pull_via_resolver(
                    "hf://org/repo", model_id_override="../evil",
                )
        assert not (tmp_catalog / "venvs").exists()

    def test_pull_bundled_rejects_bad_model_id(self, tmp_catalog):
        """_pull_bundled validates its model_id parameter directly, so a
        curated-bundled-alias id with unsafe characters is refused too."""
        from muse.core.catalog import _pull_bundled

        with pytest.raises(ValueError, match="invalid model id"):
            _pull_bundled("a/b")
        assert not (tmp_catalog / "venvs").exists()
