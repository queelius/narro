"""Smoke tests for the top-level `muse` CLI.

The CLI surface is deliberately modality-agnostic:
    muse serve / pull / models {list,info,remove}

No per-modality subcommands — those would be hardcoded modality→verb
mappings (the anti-pattern this CLI design rejects).
"""
import subprocess
import sys


def _run(*args, timeout=30):
    return subprocess.run(
        [sys.executable, "-m", "muse.cli", *args],
        capture_output=True, text=True, timeout=timeout,
    )


def test_no_args_prints_help():
    r = _run()
    assert r.returncode in (0, 2)
    combined = r.stdout + r.stderr
    assert "muse" in combined.lower()


def test_top_level_help_lists_only_admin_subcommands():
    """serve, pull, models — and nothing modality-specific."""
    r = _run("--help")
    combined = r.stdout + r.stderr
    for cmd in ("serve", "pull", "models"):
        assert cmd in combined, f"{cmd!r} missing from top-level help"
    # The per-modality and shortcut subcommands must NOT appear
    for removed in ("speak", "imagine", "audio ", "images "):
        assert removed not in combined, f"removed {removed!r} still in top-level help"


def test_models_help_lists_subcommands():
    r = _run("models", "--help")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    for cmd in ("list", "info", "remove"):
        assert cmd in combined, f"models {cmd!r} missing from help"


def test_models_list_shows_entries_across_all_modalities():
    """Without filter, list shows audio.speech AND images.generations models."""
    r = _run("models", "list")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    # Expect at least one audio.speech and one images.generations model
    assert any(m in combined for m in ("soprano", "kokoro", "bark"))
    assert "sd-turbo" in combined


def test_models_list_shows_modality_column():
    """Each listed model must include its modality so the output is self-describing."""
    r = _run("models", "list")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "audio.speech" in combined
    assert "images.generations" in combined


def test_models_list_modality_filter():
    r = _run("models", "list", "--modality", "images.generations")
    assert r.returncode == 0
    assert "sd-turbo" in r.stdout
    # audio.speech models must NOT appear under this filter
    for m in ("soprano", "kokoro", "bark"):
        assert m not in r.stdout


def test_models_list_empty_filter_reports_empty():
    r = _run("models", "list", "--modality", "video.generations")
    assert r.returncode == 0
    combined = (r.stdout + r.stderr).lower()
    assert "no known models" in combined


def test_models_info_on_known_model():
    r = _run("models", "info", "soprano-80m")
    assert r.returncode == 0
    assert "soprano" in r.stdout.lower()
    assert "audio.speech" in r.stdout


def test_models_info_unknown_nonzero():
    r = _run("models", "info", "no-such-model")
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "unknown" in combined


def test_pull_unknown_model_nonzero_exit():
    r = _run("pull", "no-such-model-12345")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "unknown" in combined.lower() or "not found" in combined.lower()


def test_help_is_fast(tmp_path):
    """muse --help must not load heavy libs (torch, diffusers, transformers)."""
    import time
    start = time.time()
    r = _run("--help")
    elapsed = time.time() - start
    assert r.returncode in (0, 2)
    assert elapsed < 5.0, f"muse --help took {elapsed:.1f}s; heavy imports leaked into CLI"
