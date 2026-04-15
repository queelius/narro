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
    with patch("muse.core.catalog.importlib.import_module", return_value=fake_module):
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
    with patch("muse.core.catalog.importlib.import_module", return_value=fake_module):
        load_backend("llama-mock-2", chat_template="qwen", device="cpu")

    kwargs = fake_class.call_args.kwargs
    assert kwargs["chat_template"] == "qwen"


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
    with patch("muse.core.catalog.importlib.import_module", return_value=fake_module):
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
