"""Tests for muse.core.discovery.

Discovery scans directories of .py files (models) or subpackages
(modalities) and extracts MANIFEST + Model class (models) or
MODALITY tag + build_router (modalities). Errors during discovery
are logged and skipped; discovery never raises.
"""
import textwrap
from pathlib import Path

import pytest

from muse.core.discovery import (
    DiscoveredModel,
    discover_models,
    discover_modalities,
)


def _write_model_script(tmp_path: Path, filename: str, content: str) -> Path:
    """Helper: write a .py file with given content to tmp_path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content).lstrip())
    return p


def _write_modality_package(tmp_path: Path, name: str, content: str) -> Path:
    """Helper: write a subpackage (__init__.py only) under tmp_path/name/."""
    pkg = tmp_path / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(textwrap.dedent(content).lstrip())
    return pkg


# ---------- Model discovery ----------

class TestDiscoverModels:
    def test_empty_directory_yields_no_models(self, tmp_path):
        result = discover_models([tmp_path])
        assert result == {}

    def test_script_with_manifest_and_model_class_is_discovered(self, tmp_path):
        _write_model_script(tmp_path, "fake_model.py", """
            MANIFEST = {
                "model_id": "fake-model",
                "modality": "audio/speech",
                "hf_repo": "fake/repo",
            }
            class Model:
                model_id = "fake-model"
        """)
        result = discover_models([tmp_path])
        assert "fake-model" in result
        entry = result["fake-model"]
        assert isinstance(entry, DiscoveredModel)
        assert entry.manifest["model_id"] == "fake-model"
        assert entry.manifest["modality"] == "audio/speech"
        assert entry.model_class.__name__ == "Model"
        assert entry.source_path == tmp_path / "fake_model.py"

    def test_script_without_manifest_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "noisy.py", """
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "MANIFEST" in caplog.text or "noisy" in caplog.text

    def test_script_without_model_class_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "manifest_only.py", """
            MANIFEST = {
                "model_id": "half-model",
                "modality": "audio/speech",
                "hf_repo": "x/y",
            }
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "Model" in caplog.text or "half-model" in caplog.text or "manifest_only" in caplog.text

    def test_script_with_import_error_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "broken.py", """
            import definitely_not_a_real_module_xyz
            MANIFEST = {"model_id": "x", "modality": "y", "hf_repo": "z"}
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        # Discovery must not raise; just log
        assert "broken" in caplog.text or "ImportError" in caplog.text or "definitely_not" in caplog.text

    def test_files_starting_with_underscore_are_ignored(self, tmp_path):
        _write_model_script(tmp_path, "_private.py", """
            MANIFEST = {"model_id": "p", "modality": "m", "hf_repo": "r"}
            class Model: ...
        """)
        _write_model_script(tmp_path, "__init__.py", "")
        result = discover_models([tmp_path])
        assert result == {}

    def test_manifest_missing_required_fields_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "bad_manifest.py", """
            MANIFEST = {"model_id": "x", "modality": "audio/speech"}
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "hf_repo" in caplog.text or "required" in caplog.text.lower()

    def test_multiple_directories_scanned_in_order(self, tmp_path):
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_model_script(d1, "model_a.py", """
            MANIFEST = {"model_id": "a", "modality": "m", "hf_repo": "r"}
            class Model: ...
        """)
        _write_model_script(d2, "model_b.py", """
            MANIFEST = {"model_id": "b", "modality": "m", "hf_repo": "r"}
            class Model: ...
        """)
        result = discover_models([d1, d2])
        assert {"a", "b"} == set(result.keys())

    def test_first_found_wins_on_model_id_collision(self, tmp_path, caplog):
        d1 = tmp_path / "bundled"
        d2 = tmp_path / "user"
        d1.mkdir()
        d2.mkdir()
        _write_model_script(d1, "m.py", """
            MANIFEST = {"model_id": "collide", "modality": "m", "hf_repo": "bundled-repo"}
            class Model: ...
        """)
        _write_model_script(d2, "m.py", """
            MANIFEST = {"model_id": "collide", "modality": "m", "hf_repo": "user-repo"}
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([d1, d2])
        assert len(result) == 1
        assert result["collide"].manifest["hf_repo"] == "bundled-repo"
        assert "collide" in caplog.text

    def test_nonexistent_directory_is_silently_skipped(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        result = discover_models([missing])
        assert result == {}


# ---------- Modality discovery ----------

class TestDiscoverModalities:
    def test_empty_directory_yields_no_modalities(self, tmp_path):
        result = discover_modalities([tmp_path])
        assert result == {}

    def test_subpackage_with_MODALITY_and_build_router_is_discovered(self, tmp_path):
        _write_modality_package(tmp_path, "fake_modality", """
            MODALITY = "fake/type"
            def build_router(registry):
                from fastapi import APIRouter
                return APIRouter()
        """)
        result = discover_modalities([tmp_path])
        assert "fake/type" in result
        build_fn = result["fake/type"]
        assert callable(build_fn)

    def test_subpackage_without_MODALITY_is_skipped(self, tmp_path, caplog):
        _write_modality_package(tmp_path, "no_tag", """
            def build_router(registry):
                return None
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([tmp_path])
        assert result == {}
        assert "MODALITY" in caplog.text or "no_tag" in caplog.text

    def test_subpackage_without_build_router_is_skipped(self, tmp_path, caplog):
        _write_modality_package(tmp_path, "no_router", """
            MODALITY = "x/y"
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([tmp_path])
        assert result == {}
        assert "build_router" in caplog.text or "no_router" in caplog.text

    def test_plain_py_files_are_not_treated_as_modalities(self, tmp_path):
        (tmp_path / "not_a_package.py").write_text(
            'MODALITY = "wrong/form"\ndef build_router(r): pass\n'
        )
        result = discover_modalities([tmp_path])
        assert result == {}

    def test_first_found_wins_on_modality_tag_collision(self, tmp_path, caplog):
        d1 = tmp_path / "bundled"
        d2 = tmp_path / "escape"
        d1.mkdir()
        d2.mkdir()
        _write_modality_package(d1, "my_mod", """
            MODALITY = "collide/tag"
            def build_router(r): return ("bundled",)
        """)
        _write_modality_package(d2, "my_mod", """
            MODALITY = "collide/tag"
            def build_router(r): return ("escape",)
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([d1, d2])
        assert len(result) == 1
        assert result["collide/tag"](None) == ("bundled",)
        assert "collide/tag" in caplog.text


# ---------- Edge-case regression tests (B2) ----------
#
# These guard the defensive checks in the B1 implementation:
#   - isinstance(manifest, dict)
#   - isinstance(model_class, type)
#   - isinstance(tag, str) and non-empty
#   - callable(build_router)
#   - repeat-scan produces consistent results (no leaked state)

class TestDiscoveryEdgeCases:
    def test_non_dict_MANIFEST_is_skipped(self, tmp_path, caplog):
        """MANIFEST defined as something other than a dict must not crash."""
        _write_model_script(tmp_path, "string_manifest.py", """
            MANIFEST = "this should be a dict"
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "MANIFEST" in caplog.text or "string_manifest" in caplog.text

    def test_Model_as_instance_not_class_is_skipped(self, tmp_path, caplog):
        """`Model = SomeClass()` (instance, not class) must be rejected."""
        _write_model_script(tmp_path, "instance_model.py", """
            MANIFEST = {
                "model_id": "inst",
                "modality": "audio/speech",
                "hf_repo": "x/y",
            }
            class _RealModel: ...
            Model = _RealModel()
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "Model" in caplog.text or "instance_model" in caplog.text

    def test_non_string_MODALITY_is_skipped(self, tmp_path, caplog):
        """MODALITY = 42 (or any non-str) must be rejected."""
        _write_modality_package(tmp_path, "numeric_tag", """
            MODALITY = 42
            def build_router(r): return None
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([tmp_path])
        assert result == {}
        assert "MODALITY" in caplog.text or "numeric_tag" in caplog.text

    def test_non_callable_build_router_is_skipped(self, tmp_path, caplog):
        """build_router = "not a function" must be rejected."""
        _write_modality_package(tmp_path, "stringy_router", """
            MODALITY = "x/y"
            build_router = "not a callable"
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([tmp_path])
        assert result == {}
        assert "build_router" in caplog.text or "stringy_router" in caplog.text

    def test_bundled_model_scripts_import_without_any_ml_deps(self):
        """Regression: bundled scripts must discover cleanly even if torch/
        diffusers/transformers/sentence-transformers are broken on the
        supervisor env.

        Discovery runs outside any per-model venv. A script that imports
        a heavy dep at module top leaks the dep failure into discovery
        and hides every sibling model from `muse models list`. The fix
        is deferred imports via an `_ensure_deps()` helper called from
        `Model.__init__`. This test installs a meta-path finder that
        raises on those deps and verifies all bundled models still show
        up.
        """
        import sys
        from muse.core.catalog import _bundled_models_dir

        banned = {
            "torch", "diffusers", "transformers",
            "sentence_transformers",
        }

        class _BlockingFinder:
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split(".")[0]
                if root in banned:
                    raise RuntimeError(f"simulated broken dep: {fullname}")
                return None

        # Evict already-imported copies so the finder gets hit next time.
        saved = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k.split(".")[0] in banned
        }
        finder = _BlockingFinder()
        sys.meta_path.insert(0, finder)
        try:
            for key in list(sys.modules):
                if key.startswith("muse.models."):
                    sys.modules.pop(key)
            found = discover_models([_bundled_models_dir()])
        finally:
            sys.meta_path.remove(finder)
            sys.modules.update(saved)

        expected = {
            "soprano-80m", "kokoro-82m", "bark-small",
            "sd-turbo",
            "all-minilm-l6-v2", "qwen3-embedding-0.6b", "nv-embed-v2",
        }
        assert expected.issubset(found.keys()), (
            f"missing after dep block: {expected - found.keys()}"
        )

    def test_discovery_is_isolated_per_scan(self, tmp_path):
        """Repeat scans are idempotent and don't pollute top-level sys.modules.

        Guards the mangled-module-name pattern in `_load_script`. Two
        concerns: (1) scanning the same directory twice returns the
        same model_ids; (2) model filenames like `soprano.py` must NOT
        end up in sys.modules under their bare name, because that
        would collide with user code or other discovery roots.
        """
        import sys
        _write_model_script(tmp_path, "soprano.py", """
            MANIFEST = {
                "model_id": "soprano-test",
                "modality": "audio/speech",
                "hf_repo": "fake/soprano",
            }
            class Model: ...
        """)

        before_keys = set(sys.modules.keys())

        first = discover_models([tmp_path])
        second = discover_models([tmp_path])

        # (1) Idempotent: same keys, same manifest content
        assert set(first.keys()) == set(second.keys()) == {"soprano-test"}
        assert first["soprano-test"].manifest == second["soprano-test"].manifest

        # (2) sys.modules hygiene: bare "soprano" must not be registered.
        # Only mangled `_muse_discover_*` names are allowed to appear.
        new_keys = set(sys.modules.keys()) - before_keys
        assert "soprano" not in sys.modules
        for k in new_keys:
            assert k.startswith("_muse_discover_"), (
                f"discovery leaked non-mangled module name into sys.modules: {k}"
            )
