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
    """Without filter, list shows audio/speech AND images.generations models."""
    r = _run("models", "list")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    # Expect at least one audio/speech and one images.generations model
    assert any(m in combined for m in ("soprano", "kokoro", "bark"))
    assert "sd-turbo" in combined


def test_models_list_shows_modality_column():
    """Each listed model must include its modality so the output is self-describing."""
    r = _run("models", "list")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "audio/speech" in combined
    assert "image/generation" in combined


def test_models_list_modality_filter():
    r = _run("models", "list", "--modality", "image/generation")
    assert r.returncode == 0
    assert "sd-turbo" in r.stdout
    # audio/speech models must NOT appear under this filter
    for m in ("soprano", "kokoro", "bark"):
        assert m not in r.stdout


def test_models_list_empty_filter_reports_empty():
    r = _run("models", "list", "--modality", "video.generations")
    assert r.returncode == 0
    combined = (r.stdout + r.stderr).lower()
    assert "no models" in combined


def test_models_info_on_known_model():
    r = _run("models", "info", "soprano-80m")
    assert r.returncode == 0
    assert "soprano" in r.stdout.lower()
    assert "audio/speech" in r.stdout


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


def test_worker_subcommand_accepts_port_and_model():
    """`muse _worker --port N --model X` must parse without error."""
    r = _run("_worker", "--port", "9999", "--model", "soprano-80m", "--help")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "--port" in combined
    assert "--model" in combined


def test_models_enable_subcommand_parses():
    r = _run("models", "enable", "soprano-80m", "--help")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "enable" in combined.lower()


def test_models_disable_subcommand_parses():
    r = _run("models", "disable", "soprano-80m", "--help")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "disable" in combined.lower()


def test_models_enable_unknown_model_nonzero_exit():
    """enable on a non-pulled model should nonzero with a clear message."""
    r = _run("models", "enable", "bogus-model-xyz")
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "not pulled" in combined or "error" in combined


def test_models_disable_unknown_model_nonzero_exit():
    r = _run("models", "disable", "bogus-model-xyz")
    assert r.returncode != 0


def test_models_list_shows_known_model_regardless_of_pull_status():
    """List includes soprano-80m with whatever status it has."""
    r = _run("models", "list")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "soprano-80m" in combined


# --- v0.11.0: curated recommendations + filters in `muse models list` ------


def test_models_list_shows_recommended_status_for_curated_unpulled():
    """A curated model that hasn't been pulled shows up as [recommended]."""
    r = _run("models", "list")
    assert r.returncode == 0
    # The bundled curated.yaml includes resolver entries (e.g. qwen3-8b-q4)
    # that are not pulled in this fresh test env -> they show as recommended.
    combined = r.stdout + r.stderr
    assert "recommended" in combined.lower()


def test_models_list_filter_modality_chat_completion():
    r = _run("models", "list", "--modality", "chat/completion")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    # Curated chat models should appear; non-chat ones should NOT
    assert "chat/completion" in combined
    # kokoro is audio/speech; should be filtered out
    assert "kokoro" not in combined.lower() or "no models" in combined.lower()


def test_models_list_filter_available_excludes_disabled_and_enabled():
    """--available shows only models you could install."""
    r = _run("models", "list", "--available")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    # No enabled or disabled status should appear
    assert "[enabled" not in combined
    assert "[disabled" not in combined


def test_models_list_filter_installed_only():
    """--installed shows only catalog entries (none in fresh test env -> empty)."""
    r = _run("models", "list", "--installed")
    assert r.returncode == 0
    # In CI / fresh runs there's nothing pulled, so this is empty or close to it.
    # We assert that no [recommended] or [available] rows appear.
    combined = r.stdout + r.stderr
    assert "[recommended" not in combined
    assert "[available" not in combined
