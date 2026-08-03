"""Unit tests for scripts/smoke_fresh_venv.py.

Heavy operations (venv creation, subprocess runs, HF download) are
mocked so the unit tests do not actually spawn pip or download weights.
The CI workflow exercises the full flow end-to-end.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_smoke_module():
    """Import scripts/smoke_fresh_venv.py as a module by file path.

    The script lives outside any importable package; tests load it
    directly via importlib.util.spec_from_file_location so the test
    file does not need a sys.path hack. Module is registered in
    sys.modules before exec_module so dataclasses.dataclass can resolve
    the module's __module__ attribute back to its globals dict.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "smoke_fresh_venv.py"
    spec = importlib.util.spec_from_file_location(
        "muse_smoke_fresh_venv", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


smoke = _load_smoke_module()


@pytest.fixture
def fake_known_models():
    """Provide a known_models()-shaped dict with one fake bundled entry."""
    entry = MagicMock()
    entry.model_id = "fake-model"
    entry.modality = "embedding/text"
    entry.pip_extras = ("torch>=2.1.0", "transformers>=4.36.0")
    return {"fake-model": entry}


def test_smoke_one_success_path(tmp_path, fake_known_models):
    """A bundled model routes through the isolated real-pull pipeline."""
    expected = smoke.SmokeResult(
        model_id="fake-model",
        ok=True,
        error=None,
        duration_s=1.0,
        label="fake-model: OK (1.0s)",
    )
    with patch.object(
             smoke, "_smoke_pulled_model", return_value=expected,
         ) as pulled, \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        result = smoke.smoke_one("fake-model", tmp_path)

    assert result is expected
    pulled.assert_called_once_with("fake-model", tmp_path)


def test_smoke_one_unknown_model(tmp_path, fake_known_models):
    """Model not in known_models() returns ok=False with 'unknown model'."""
    with patch("muse.core.catalog.known_models", return_value=fake_known_models):
        result = smoke.smoke_one("missing-model", tmp_path)

    assert result.ok is False
    assert "unknown" in result.error
    assert "unknown model" in result.label


def test_extract_failure_reason_modulenotfound():
    """_extract_failure_reason picks the missing module name."""
    captured = "ModuleNotFoundError: No module named 'safetensors'\n"
    assert smoke._extract_failure_reason(captured) == "missing dep: safetensors"


def test_extract_failure_reason_importerror():
    """_extract_failure_reason picks ImportError lines."""
    captured = (
        "Traceback ...\n"
        "ImportError: cannot import name 'foo' from 'bar'\n"
    )
    out = smoke._extract_failure_reason(captured)
    assert "ImportError" in out


def test_extract_failure_reason_load_failed():
    """_extract_failure_reason picks 'load failed:' lines from the probe worker."""
    captured = "baseline RAM=1.2 GB\nload failed: model file not found\n"
    out = smoke._extract_failure_reason(captured)
    assert "load failed" in out


def test_extract_failure_reason_skips_blank_load_prefix():
    captured = "baseline RAM=1.2 GB\nload failed:\nAssertionError\n"
    assert smoke._extract_failure_reason(captured) == "AssertionError"


def test_extract_failure_reason_empty():
    """_extract_failure_reason handles empty input gracefully."""
    assert smoke._extract_failure_reason("") == "no output"


def test_run_inference_probe_requires_successful_inference(tmp_path):
    proc = MagicMock(
        returncode=0,
        stdout=json.dumps({
            "ran_inference": False,
            "inference_error": "TorchCodec is required",
        }) + "\n",
        stderr="",
    )
    with patch.object(smoke.subprocess, "run", return_value=proc):
        rc, captured = smoke._run_inference_probe(
            tmp_path / "python", "utmos",
        )
    assert rc == 1
    assert "TorchCodec is required" in captured


def test_run_inference_probe_accepts_completed_inference(tmp_path):
    proc = MagicMock(
        returncode=0,
        stdout=json.dumps({"ran_inference": True}) + "\n",
        stderr="",
    )
    with patch.object(smoke.subprocess, "run", return_value=proc):
        rc, _ = smoke._run_inference_probe(tmp_path / "python", "utmos")
    assert rc == 0


def test_main_human_output(tmp_path, fake_known_models, capsys):
    """main() without --json prints the SmokeResult.label to stdout."""
    result = smoke.SmokeResult(
        "fake-model", True, None, 1.0, "fake-model: OK (1.0s)",
    )
    with patch.object(smoke, "_smoke_pulled_model", return_value=result), \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        rc = smoke.main([
            "--model_id", "fake-model",
            "--venv_root", str(tmp_path),
        ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "fake-model" in out
    assert "OK" in out


def test_main_json_output(tmp_path, fake_known_models, capsys):
    """main() with --json prints a parseable JSON record."""
    result = smoke.SmokeResult(
        "fake-model", True, None, 1.0, "fake-model: OK (1.0s)",
    )
    with patch.object(smoke, "_smoke_pulled_model", return_value=result), \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        rc = smoke.main([
            "--model_id", "fake-model",
            "--venv_root", str(tmp_path),
            "--json",
        ])

    assert rc == 0
    out = capsys.readouterr().out
    record = json.loads(out)
    assert record["model_id"] == "fake-model"
    assert record["ok"] is True
    assert record["error"] is None
    assert "duration_s" in record
    assert "label" in record


def test_main_failure_returns_non_zero(tmp_path, fake_known_models, capsys):
    """main() returns 1 when the smoke test fails."""
    result = smoke.SmokeResult(
        "fake-model",
        False,
        "load failed: missing dep: librosa",
        1.0,
        "fake-model: FAIL (missing dep: librosa)",
    )
    with patch.object(smoke, "_smoke_pulled_model", return_value=result), \
         patch("muse.core.catalog.known_models", return_value=fake_known_models):
        rc = smoke.main([
            "--model_id", "fake-model",
            "--venv_root", str(tmp_path),
            "--json",
        ])

    assert rc == 1
    out = capsys.readouterr().out
    record = json.loads(out)
    assert record["ok"] is False
    assert "librosa" in record["label"]


class _FakeCuratedEntry:
    """Minimal stand-in for muse.core.curated.CuratedEntry (uri-only)."""

    def __init__(self, id: str, uri: str | None):
        self.id = id
        self.uri = uri


def test_smoke_one_dispatches_curated_resolver_entry(tmp_path, fake_known_models):
    """A model_id absent from known_models() but present in curated.yaml
    with a `uri` (resolver-based, e.g. opus-mt-en-es) routes to
    _smoke_curated_resolver instead of failing 'unknown model'."""
    fake_result = smoke.SmokeResult(
        model_id="opus-mt-en-es", ok=True, error=None,
        duration_s=1.0, label="opus-mt-en-es: OK (1.0s)",
    )
    with patch("muse.core.catalog.known_models", return_value=fake_known_models), \
         patch(
             "muse.core.curated.find_curated",
             return_value=_FakeCuratedEntry("opus-mt-en-es", "hf://Helsinki-NLP/opus-mt-en-es"),
         ), \
         patch.object(smoke, "_smoke_curated_resolver", return_value=fake_result) as mock_dispatch:
        result = smoke.smoke_one("opus-mt-en-es", tmp_path)

    mock_dispatch.assert_called_once_with(
        "opus-mt-en-es", "hf://Helsinki-NLP/opus-mt-en-es", tmp_path,
    )
    assert result is fake_result


def test_smoke_one_unknown_model_not_in_curated_either(tmp_path, fake_known_models):
    """A model_id absent from both known_models() and curated.yaml still
    fails 'unknown model' (real find_curated lookup, no mock)."""
    with patch("muse.core.catalog.known_models", return_value=fake_known_models):
        result = smoke.smoke_one("definitely-not-a-real-model-id", tmp_path)

    assert result.ok is False
    assert "unknown model" in result.label


def test_smoke_curated_resolver_success(tmp_path):
    """muse pull succeeds, catalog.json carries python_path, load-only
    probe succeeds -> ok=True."""
    def _fake_pull(cmd, capture_output, text, env, cwd):
        catalog_dir = Path(env["MUSE_CATALOG_DIR"])
        catalog_dir.mkdir(parents=True, exist_ok=True)
        (catalog_dir / "catalog.json").write_text(json.dumps({
            "opus-mt-en-es": {
                "python_path": str(catalog_dir / "venvs" / "opus-mt-en-es" / "bin" / "python"),
            },
        }))
        return MagicMock(returncode=0, stdout="pulled opus-mt-en-es\n", stderr="")

    with patch.object(
        smoke.subprocess, "run", side_effect=_fake_pull,
    ) as pull_run, patch.object(
        smoke, "_run_load_only", return_value=(0, '{"ok": 1}'),
    ) as probe:
        result = smoke._smoke_curated_resolver(
            "opus-mt-en-es", "hf://Helsinki-NLP/opus-mt-en-es", tmp_path,
        )

    assert result.ok is True
    assert result.model_id == "opus-mt-en-es"
    assert "OK" in result.label
    pull_cmd = pull_run.call_args.args[0]
    assert pull_cmd[-3:] == ["pull", "opus-mt-en-es", "--no-probe"]
    probe_env = probe.call_args.kwargs["env"]
    catalog_dir = Path(probe_env["MUSE_CATALOG_DIR"])
    assert catalog_dir.parent == tmp_path
    assert catalog_dir.name.startswith("muse-catalog-")


def test_smoke_curated_resolver_pull_fails(tmp_path, caplog):
    """muse pull exits non-zero -> ok=False, error mentions 'pull'."""
    fail = MagicMock(
        returncode=1,
        stdout="",
        stderr="ModuleNotFoundError: No module named 'sentencepiece'\n",
    )
    with patch.object(smoke.subprocess, "run", return_value=fail):
        result = smoke._smoke_curated_resolver(
            "opus-mt-en-es", "hf://Helsinki-NLP/opus-mt-en-es", tmp_path,
        )

    assert result.ok is False
    assert "pull" in result.error
    assert "FAIL" in result.label
    assert "sentencepiece" in caplog.text


def test_smoke_curated_resolver_missing_python_path(tmp_path):
    """Pull 'succeeds' but the persisted entry has no python_path -> FAIL."""
    def _fake_pull(cmd, capture_output, text, env, cwd):
        catalog_dir = Path(env["MUSE_CATALOG_DIR"])
        catalog_dir.mkdir(parents=True, exist_ok=True)
        (catalog_dir / "catalog.json").write_text(json.dumps({
            "opus-mt-en-es": {},
        }))
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(smoke.subprocess, "run", side_effect=_fake_pull):
        result = smoke._smoke_curated_resolver(
            "opus-mt-en-es", "hf://Helsinki-NLP/opus-mt-en-es", tmp_path,
        )

    assert result.ok is False
    assert "python_path" in result.error


def test_smoke_curated_resolver_load_fails(tmp_path):
    """Pull succeeds but the load-only probe fails -> FAIL with reason."""
    def _fake_pull(cmd, capture_output, text, env, cwd):
        catalog_dir = Path(env["MUSE_CATALOG_DIR"])
        catalog_dir.mkdir(parents=True, exist_ok=True)
        (catalog_dir / "catalog.json").write_text(json.dumps({
            "opus-mt-en-es": {
                "python_path": str(catalog_dir / "venvs" / "opus-mt-en-es" / "bin" / "python"),
            },
        }))
        return MagicMock(returncode=0, stdout="", stderr="")

    captured = "ModuleNotFoundError: No module named 'sentencepiece'\n"
    with patch.object(smoke.subprocess, "run", side_effect=_fake_pull), \
         patch.object(smoke, "_run_load_only", return_value=(1, captured)):
        result = smoke._smoke_curated_resolver(
            "opus-mt-en-es", "hf://Helsinki-NLP/opus-mt-en-es", tmp_path,
        )

    assert result.ok is False
    assert "missing dep: sentencepiece" in result.label


@pytest.mark.parametrize("model_id,modality,uri", [
    ("utmos", "audio/quality", "hf://Blinorot/UTMOS-PyTorch"),
    (
        "qwen3-forced-aligner-0.6b",
        "audio/alignment",
        "hf://Qwen/Qwen3-ForcedAligner-0.6B-hf",
    ),
])
def test_smoke_curated_audio_analysis_runs_inference_probe(
    tmp_path, model_id, modality, uri,
):
    def _fake_pull(cmd, capture_output, text, env, cwd):
        catalog_dir = Path(env["MUSE_CATALOG_DIR"])
        catalog_dir.mkdir(parents=True, exist_ok=True)
        (catalog_dir / "catalog.json").write_text(json.dumps({
            model_id: {
                "python_path": str(
                    catalog_dir / "venvs" / model_id / "bin" / "python"
                ),
                "manifest": {"modality": modality},
            },
        }))
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(smoke.subprocess, "run", side_effect=_fake_pull), \
         patch.object(
             smoke,
             "_run_inference_probe",
             return_value=(0, '{"ran_inference": true}'),
        ) as inference, \
         patch.object(smoke, "_run_load_only") as load_only:
        result = smoke._smoke_curated_resolver(model_id, uri, tmp_path)

    assert result.ok is True
    inference.assert_called_once()
    load_only.assert_not_called()


def test_repo_root_finds_pyproject():
    """_repo_root() locates the muse repo via pyproject.toml ancestor walk."""
    root = smoke._repo_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "muse").exists()
