"""Smoke tests for the top-level `muse` CLI.

The CLI surface is deliberately modality-agnostic:
    muse serve / pull / models {list,info,remove}

No per-modality subcommands — those would be hardcoded modality→verb
mappings (the anti-pattern this CLI design rejects).
"""
import os
import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def _isolated_muse_state(tmp_path, monkeypatch):
    """Keep CLI subprocesses away from the developer's Muse state."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path / "catalog"))
    monkeypatch.setenv("MUSE_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.delenv("MUSE_MODELS_DIR", raising=False)
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)


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


def test_no_args_does_not_print_traceback():
    """`muse` with no args should print help and exit cleanly.

    Regression watchdog for the typer + standalone_mode=False bug
    where click's NoArgsIsHelpError leaked through main()'s try/except,
    leaving a Python traceback chasing the rendered help.
    """
    r = _run()
    combined = r.stdout + r.stderr
    assert "Traceback" not in combined, (
        f"Python traceback should not be printed for `muse` no-args:\n{combined}"
    )
    assert "NoArgsIsHelpError" not in combined, (
        f"click internal exception leaked through main():\n{combined}"
    )


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


def test_doctor_resources_command_parses_without_inspecting_processes():
    import typer

    from muse.cli import app

    doctor = typer.main.get_command(app).commands["doctor"]
    resources = doctor.commands["resources"]
    opts = [opt for param in resources.params for opt in param.opts]
    assert "--repair" in opts
    assert "--grace" in opts


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


def test_models_info_curated_only_renders_card(tmp_path):
    """Regression for v0.40.2: a curated-only id (in curated.yaml but
    not bundled and not pulled) must render an info card with the
    description / uri / install hint, not 'unknown model'.

    Scope to a fresh empty catalog via MUSE_CATALOG_DIR so the test
    is hermetic regardless of what the dev box has pulled.
    """
    import os
    import subprocess
    import sys
    env = os.environ.copy()
    env["MUSE_CATALOG_DIR"] = str(tmp_path)
    r = subprocess.run(
        [sys.executable, "-m", "muse.cli", "models", "info", "flux-schnell"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "recommended, not pulled" in out
    assert "image/generation" in out
    assert "muse pull flux-schnell" in out


def test_pull_unknown_model_nonzero_exit():
    r = _run("pull", "no-such-model-12345")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "unknown" in combined.lower() or "not found" in combined.lower()


def test_pull_help_documents_no_probe_flag():
    """The `pull` command must expose the --no-probe opt-out flag.

    `--help` must run cleanly; the flag's presence is verified by typer
    introspection rather than by grepping the rendered help. The rendered
    help's line-wrapping is environment-dependent: some CI runners pipe
    stdout, so os.get_terminal_size() reports 0 columns and rich wraps
    option names character-by-character, dropping "--no-probe" as a literal
    substring. Introspecting the command's options is render-independent.
    """
    import typer
    from muse.cli import app

    r = _run("pull", "--help")
    assert r.returncode == 0
    pull = typer.main.get_command(app).commands["pull"]
    opts = [opt for param in pull.params for opt in param.opts]
    assert "--no-probe" in opts


def test_pull_curated_alias_registers_hf_resolver():
    """Regression (v0.11.1): pulling a curated id that expands to an hf://
    URI must register the HF resolver before pull() dispatches. v0.11.0
    only registered on URIs typed directly, so curated aliases crashed
    with 'no resolver for scheme "hf"'.

    We don't actually complete the pull (that would download weights);
    we just check that the failure mode is NOT the resolver-not-registered
    error. The pull will likely fail later (network, missing HF repo,
    etc.) but that's a different, acceptable failure.
    """
    # Pull a real curated id whose URI is hf://... (qwen3-8b-q4 -> a GGUF
    # repo). HF_HUB_OFFLINE forces the post-registration HF call to fail
    # fast instead of downloading ~5GB of weights: registration happens
    # before resolve() touches the network, so the resolver-not-registered
    # error must NOT appear, while the offline error is an acceptable
    # post-registration failure.
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env["MUSE_CATALOG_DIR"] = tmp
        env["HF_HUB_OFFLINE"] = "1"
        import subprocess
        r = subprocess.run(
            ["muse", "pull", "qwen3-8b-q4"],
            capture_output=True, text=True, timeout=60, env=env,
        )
        # Must NOT fail with "no resolver for scheme 'hf'"
        combined = r.stdout + r.stderr
        assert "no resolver for scheme 'hf'" not in combined, (
            f"HF resolver not registered when pulling curated alias:\n{combined}"
        )


def test_help_is_fast(tmp_path):
    """muse --help must not load heavy libs (torch, diffusers, transformers)."""
    import time
    start = time.time()
    r = _run("--help")
    elapsed = time.time() - start
    assert r.returncode in (0, 2)
    assert elapsed < 5.0, f"muse --help took {elapsed:.1f}s; heavy imports leaked into CLI"


def test_worker_subcommand_accepts_port_and_model():
    """`muse _worker` must accept --port and --model.

    `--help` must run cleanly; option presence is verified by typer
    introspection (see test_pull_help_documents_no_probe_flag for why
    grepping rendered help is environment-fragile).
    """
    import typer
    from muse.cli import app

    r = _run("_worker", "--port", "9999", "--model", "soprano-80m", "--help")
    assert r.returncode == 0
    worker = typer.main.get_command(app).commands["_worker"]
    opts = [opt for param in worker.params for opt in param.opts]
    assert "--port" in opts
    assert "--model" in opts


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


def test_models_warmup_subcommand_parses():
    """`muse models warmup --help` must parse and document its argument."""
    r = _run("models", "warmup", "soprano-80m", "--help")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "warmup" in combined.lower() or "pre-load" in combined.lower()


def test_models_help_lists_warmup_subcommand():
    """models --help must include the new warmup verb."""
    r = _run("models", "--help")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "warmup" in combined.lower()


def test_models_warmup_without_admin_token_emits_clear_error(tmp_path):
    """Warmup is a runtime operation and has no offline equivalent.

    When `MUSE_ADMIN_TOKEN` is unset, the CLI must emit a clear
    "warmup requires a running `muse serve`" error rather than silently
    falling through to a catalog mutation.
    """
    import os
    env = os.environ.copy()
    # Strip the admin token to exercise the error path.
    env.pop("MUSE_ADMIN_TOKEN", None)
    env["MUSE_CATALOG_DIR"] = str(tmp_path)
    r = subprocess.run(
        [sys.executable, "-m", "muse.cli", "models", "warmup", "kokoro-82m"],
        capture_output=True, text=True, timeout=15, env=env,
    )
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "muse serve" in combined or "admin_token" in combined or "admin token" in combined


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
