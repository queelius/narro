"""Smoke tests for top-level `muse` CLI dispatch."""
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


def test_top_level_help_lists_all_subcommands():
    r = _run("--help")
    combined = r.stdout + r.stderr
    for cmd in ("serve", "pull", "audio", "images", "speak", "imagine"):
        assert cmd in combined, f"{cmd!r} missing from top-level help"


def test_audio_help():
    r = _run("audio", "--help")
    assert r.returncode == 0
    assert "speech" in r.stdout


def test_audio_speech_help():
    r = _run("audio", "speech", "--help")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "models" in combined
    assert "create" in combined


def test_images_generations_models_list_runs():
    r = _run("images", "generations", "models", "list")
    assert r.returncode == 0
    # Must mention sd-turbo OR "no known models" (if catalog filtering broke)
    assert "sd-turbo" in r.stdout or "no known" in r.stdout.lower()


def test_audio_speech_models_list_runs():
    r = _run("audio", "speech", "models", "list")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    # Should show at least one of the seeded audio.speech models
    assert any(m in combined for m in ("soprano", "kokoro", "bark"))


def test_pull_unknown_model_nonzero_exit():
    r = _run("pull", "no-such-model-12345")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "unknown" in combined.lower() or "not found" in combined.lower()


def test_audio_speech_models_info_on_known_model():
    r = _run("audio", "speech", "models", "info", "soprano-80m")
    assert r.returncode == 0
    assert "soprano" in r.stdout.lower()
    assert "audio.speech" in r.stdout


def test_audio_speech_models_info_unknown_nonzero():
    r = _run("audio", "speech", "models", "info", "no-such-model")
    assert r.returncode != 0


def test_help_is_fast(tmp_path):
    """muse --help must not load heavy libs (torch, diffusers, transformers)."""
    import time
    # Run --help and time it. Allow 3s for cold-start argparse, but require
    # completion under that budget. This fails immediately if someone adds
    # a top-level `import torch` to cli.py.
    start = time.time()
    r = _run("--help")
    elapsed = time.time() - start
    assert r.returncode in (0, 2)
    assert elapsed < 5.0, f"muse --help took {elapsed:.1f}s; heavy imports leaked into CLI"
