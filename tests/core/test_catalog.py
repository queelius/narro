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
)


@pytest.fixture(autouse=True)
def _isolate_catalog_cache():
    """Reset the known-models cache around every test.

    The cache persists in process memory and would otherwise bleed
    state between tests. Bundled-only discovery is cheap, so re-running
    it per test is fine.
    """
    _reset_known_models_cache()
    yield
    _reset_known_models_cache()


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    """Point catalog state at a temp file."""
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    yield tmp_path


def test_known_models_entries_have_valid_modality():
    valid = {"audio/speech", "image/generation", "embedding/text"}
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
    assert "all-minilm-l6-v2" in catalog
    assert "qwen3-embedding-0.6b" in catalog
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
        # install_into_venv called twice: once for muse[server], once for
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
    with patch("muse.core.catalog.importlib.import_module", return_value=fake_module):
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
    # two calls in total: muse[server] first, then the model's pip_extras.
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


def test_pull_does_not_call_system_install_pip_extras(tmp_catalog):
    """The old system-wide install_pip_extras must NOT be called; it's venv-scoped now."""
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.install_pip_extras") as mock_system_install, \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    mock_system_install.assert_not_called()


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
