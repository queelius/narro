"""`muse models refresh`: re-install museq[server,<extras>] into per-model venvs.

Use case: museq[server] gets a new dep (e.g. python-multipart in v0.13.1)
and existing per-model venvs created by older `muse pull` calls are
stale. Without this command, hitting the new code path inside an old
venv crashes the worker; the user has fixed this by hand multiple
times. `muse models refresh --all` upgrades every venv in one pass.

Behavior:
  - Inspect catalog.json for the target model_id(s).
  - For each: invoke <venv>/bin/pip install --upgrade <muse-target>[server,<modality-extras>],
    where <muse-target> is `-e <source-tree>` from a checkout or the
    published `museq` distribution from a wheel/PyPI install.
  - Then (unless --no-extras): refresh the model's pip extras and immutable
    non-packaged Python sources.
  - Continue past failures; aggregate at the end.

The supervisor is NOT restarted. To pick up a refreshed venv, the
operator runs `Ctrl+C; muse serve` themselves. This is documented.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from muse.core.catalog import (
    ModelInUseError,
    _catalog_dir,
    _ensure_owned_directory,
    _installed_muse_requirement,
    _is_muse_pyproject,
    _model_pull_lock,
    _model_resource_lease,
    _read_catalog,
    _validate_model_id_for_fs,
    get_manifest,
    is_enabled,
)
from muse.core.venv import (
    install_python_sources,
    run_owned_command,
    venv_python,
    venv_transaction,
)

logger = logging.getLogger(__name__)


# 30 minutes: long enough for a slow PyPI mirror to complete a full
# torch+diffusers install, short enough that a hung mirror is detected
# before the operator gives up. Probe and admin/operations.refresh use
# the same value.
_PIP_TIMEOUT = 1800


# Map a modality tag to the muse-side optional-deps extras names.
# `server` is added unconditionally on top of these. The map is
# hand-maintained: each entry corresponds to a pyproject [project.
# optional-dependencies] block (see pyproject.toml). New modalities
# either bring a new extra (then add a row here) or reuse `server`
# (then map to []).
MODALITY_EXTRAS: dict[str, list[str]] = {
    "audio/speech": ["audio"],
    "audio/transcription": [],
    "audio/embedding": [],
    "audio/quality": [],
    "audio/alignment": [],
    "audio/generation": [],
    "image/generation": ["images"],
    "image/animation": ["images"],
    "image/upscale": ["images"],
    "image/embedding": [],
    "image/segmentation": [],
    "image/vectorization": [],
    "embedding/text": ["embeddings"],
    "text/classification": [],
    "text/rerank": [],
    "text/summarization": [],
    "video/generation": ["images"],
    "chat/completion": [],
}


@dataclass
class RefreshResult:
    """Outcome record for one model's venv refresh."""

    model_id: str
    state: str  # "ok" | "failed" | "skipped"
    message: str = ""
    pip_output: str = ""
    extras: list[str] = field(default_factory=list)


def _infer_extras(modality: str) -> list[str]:
    """Look up museq[server,<extras>] for a modality tag. Unknown -> []."""
    return list(MODALITY_EXTRAS.get(modality, []))


# The source-tree sniff and exact installed-distribution requirement are shared
# with core.catalog, which applies the same fork to `muse pull` venv creation.


def _muse_repo_root() -> Path | None:
    """Locate the muse source tree for an editable refresh, or None.

    Walks parents of this file for a pyproject.toml that actually
    declares the museq project. Returns None when running from a
    wheel/PyPI install (no such pyproject in any parent), so the caller
    installs the published `museq` distribution from PyPI instead of
    editable-installing whatever unrelated project happens to sit in the
    current working directory.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.exists() and _is_muse_pyproject(pyproject):
            return parent
    return None


def _pip_target(extras: list[str]) -> str:
    """Build the museq install spec: <target>[server,extra1,extra2,...].

    `<target>` is the local source tree (editable refresh) when muse runs
    from a checkout, else the exact invoking `museq` distribution version.
    `server` is always present; modality extras append. Bracket-comma syntax
    matches PEP 508 extras.
    """
    root = _muse_repo_root()
    if root is not None:
        spec = ",".join(["server", *extras])
        return f"{root}[{spec}]"
    return _installed_muse_requirement(["server", *extras])


def _pip_target_args(extras: list[str]) -> list[str]:
    """pip target args for refreshing muse inside a per-model venv.

    Source-tree detection is performed once so a concurrent filesystem change
    cannot pair an editable flag with a distribution target (or vice versa).
    """
    root = _muse_repo_root()
    if root is not None:
        spec = ",".join(["server", *extras])
        return ["-e", f"{root}[{spec}]"]
    return [_installed_muse_requirement(["server", *extras])]


def _validated_refresh_venv(model_id: str, entry: dict) -> tuple[Path, Path]:
    """Require the catalog entry to name this model's canonical owned venv."""
    _validate_model_id_for_fs(model_id)
    venvs_root = _catalog_dir() / "venvs"
    _ensure_owned_directory(venvs_root, private=True)
    expected_venv = Path(os.path.abspath(venvs_root / model_id))
    expected_python = Path(os.path.abspath(venv_python(expected_venv)))

    raw_venv = entry.get("venv_path")
    if raw_venv is not None:
        catalog_venv = Path(os.path.abspath(os.fspath(raw_venv)))
        if catalog_venv != expected_venv:
            raise RuntimeError(
                f"catalog venv_path for {model_id!r} is outside its owned path: "
                f"expected {expected_venv}, got {raw_venv}"
            )

    raw_python = entry.get("python_path")
    if not raw_python:
        raise RuntimeError(f"catalog entry for {model_id!r} is missing python_path")
    catalog_python = Path(os.path.abspath(os.fspath(raw_python)))
    if catalog_python != expected_python:
        raise RuntimeError(
            f"catalog python_path for {model_id!r} is not its owned interpreter: "
            f"expected {expected_python}, got {raw_python}"
        )
    return expected_venv, expected_python


def refresh_one(
    model_id: str,
    *,
    no_extras: bool = False,
) -> RefreshResult:
    """Refresh one model while excluding pull/remove/parallel refresh."""
    with _model_pull_lock(model_id):
        try:
            with _model_resource_lease(model_id):
                return _refresh_one_locked(model_id, no_extras=no_extras)
        except ModelInUseError as exc:
            return RefreshResult(model_id, "failed", str(exc))


def _refresh_one_locked(
    model_id: str,
    *,
    no_extras: bool = False,
) -> RefreshResult:
    """Refresh a single model's venv.

    The caller holds the model's cross-thread/process pull lock for this
    complete catalog-read and venv-mutation transaction.

    Two pip invocations plus an optional reviewed-source refresh:
      1. install --upgrade <muse-target>[server,<modality-extras>]
         (editable `-e <tree>` from a checkout, else `museq` from PyPI)
      2. install --upgrade <model's pip_extras...>  (skipped if --no-extras
         or pip_extras is empty)
      3. synchronize manifest `python_sources` into the same venv, including
         revoking removed hooks when empty (skipped only with --no-extras)

    On any non-zero pip exit, returns RefreshResult(state='failed') with
    captured stdout+stderr and skips step 2. Catalog entries that point
    at a missing python_path return state='skipped'.
    """
    catalog = _read_catalog()
    entry = catalog.get(model_id)
    if not entry:
        return RefreshResult(model_id, "skipped", "not in catalog")
    try:
        venv_path, validated_python = _validated_refresh_venv(model_id, entry)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return RefreshResult(model_id, "failed", f"unsafe catalog venv: {exc}")
    python_path = str(validated_python)
    if not validated_python.exists():
        return RefreshResult(
            model_id,
            "skipped",
            f"python_path missing or not found: {python_path}",
        )

    try:
        manifest = get_manifest(model_id)
    except KeyError:
        manifest = {}
    modality = manifest.get("modality") or entry.get("modality") or ""
    pip_extras_list = list(manifest.get("pip_extras") or ())
    python_sources = list(manifest.get("python_sources") or ())
    muse_extras = _infer_extras(modality)
    try:
        target_args = _pip_target_args(muse_extras)
    except RuntimeError as exc:
        return RefreshResult(model_id, "failed", str(exc), extras=muse_extras)
    try:
        with venv_transaction(venv_path) as tx:
            result = _refresh_staged_venv(
                model_id,
                python_path=str(venv_python(tx.path)),
                venv_path=tx.path,
                target_args=target_args,
                pip_extras_list=pip_extras_list,
                python_sources=python_sources,
                muse_extras=muse_extras,
                no_extras=no_extras,
            )
            if result.state == "ok":
                tx.commit()
            return result
    except (OSError, RuntimeError, ValueError) as exc:
        return RefreshResult(
            model_id,
            "failed",
            f"transactional venv refresh failed: {exc}",
            extras=muse_extras,
        )


def _refresh_staged_venv(
    model_id: str,
    *,
    python_path: str,
    venv_path: Path,
    target_args: list[str],
    pip_extras_list: list[str],
    python_sources: list[object],
    muse_extras: list[str],
    no_extras: bool,
) -> RefreshResult:
    """Apply one refresh to the transaction's private working copy."""

    cmd = [python_path, "-m", "pip", "install", "--upgrade", *target_args]
    logger.info("refresh %s: %s", model_id, " ".join(cmd))
    try:
        proc = run_owned_command(
            cmd, capture_output=True, timeout=_PIP_TIMEOUT, check=False,
        )
    except subprocess.TimeoutExpired:
        return RefreshResult(
            model_id,
            "failed",
            f"museq[server] install timed out after {_PIP_TIMEOUT}s",
            extras=muse_extras,
        )
    except (OSError, ValueError) as exc:
        return RefreshResult(
            model_id,
            "failed",
            f"could not start museq[server] install: {exc}",
            extras=muse_extras,
        )
    if proc.returncode != 0:
        return RefreshResult(
            model_id,
            "failed",
            "museq[server] install failed",
            (proc.stdout or "") + (proc.stderr or ""),
            extras=muse_extras,
        )

    if not no_extras and pip_extras_list:
        cmd2 = [python_path, "-m", "pip", "install", "--upgrade", *pip_extras_list]
        logger.info("refresh %s extras: %s", model_id, " ".join(cmd2))
        try:
            proc2 = run_owned_command(
                cmd2, capture_output=True, timeout=_PIP_TIMEOUT, check=False,
            )
        except subprocess.TimeoutExpired:
            return RefreshResult(
                model_id,
                "failed",
                f"pip_extras install timed out after {_PIP_TIMEOUT}s",
                extras=muse_extras,
            )
        except (OSError, ValueError) as exc:
            return RefreshResult(
                model_id,
                "failed",
                f"could not start pip_extras install: {exc}",
                extras=muse_extras,
            )
        if proc2.returncode != 0:
            return RefreshResult(
                model_id,
                "failed",
                "pip_extras install failed",
                (proc2.stdout or "") + (proc2.stderr or ""),
                extras=muse_extras,
            )

    if not no_extras:
        try:
            install_python_sources(venv_path, python_sources)
        except subprocess.TimeoutExpired as exc:
            output = "".join(
                str(part or "") for part in (exc.stdout, exc.stderr)
            )
            return RefreshResult(
                model_id,
                "failed",
                f"python_sources install timed out after {_PIP_TIMEOUT}s",
                output,
                extras=muse_extras,
            )
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            output = "".join(
                str(part or "")
                for part in (
                    getattr(exc, "stdout", None),
                    getattr(exc, "stderr", None),
                )
            )
            return RefreshResult(
                model_id,
                "failed",
                f"python_sources install failed: {exc}",
                output,
                extras=muse_extras,
            )

    return RefreshResult(model_id, "ok", extras=muse_extras)


def _select_targets(
    *,
    model_id: str | None,
    all_: bool,
    enabled_only: bool,
) -> list[str] | None:
    """Resolve the --all/--enabled/<id> flags to a sorted list of targets.

    Returns None on usage error (caller prints a message + exits 2).
    Alphabetical order keeps `--all` deterministic across runs (same
    output ordering, same JSON shape).
    """
    catalog = _read_catalog()
    if all_:
        return sorted(catalog.keys())
    if enabled_only:
        return sorted(mid for mid in catalog if is_enabled(mid))
    if model_id is not None:
        return [model_id]
    return None


def run_refresh(
    *,
    model_id: str | None = None,
    all_: bool = False,
    enabled_only: bool = False,
    no_extras: bool = False,
    as_json: bool = False,
) -> int:
    """Entry point for `muse models refresh`.

    Returns 0 if every selected target succeeded or was skipped, 1 if
    any failed, 2 on usage error (no targets selected).
    """
    targets = _select_targets(
        model_id=model_id, all_=all_, enabled_only=enabled_only,
    )
    if targets is None:
        print(
            "error: pass <model_id>, --all, or --enabled",
            file=sys.stderr,
        )
        return 2

    if not targets:
        print("no targets selected")
        return 0

    results = [refresh_one(mid, no_extras=no_extras) for mid in targets]

    if as_json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        _render_refresh_summary(results)

    n_failed = sum(1 for r in results if r.state == "failed")
    return 0 if n_failed == 0 else 1


# Status glyphs for the refresh summary. Mirrors the encoding used in
# `muse models list` (●/○/★/·) but maps to refresh outcomes
# specifically. Single-cell narrow chars only.
_REFRESH_GLYPHS = {
    "ok": ("✓", "bold green"),
    "failed": ("✗", "bold red"),
    "skipped": ("·", "dim"),
}


def _render_refresh_summary(results: list) -> None:
    """Render results: rich.Table on TTY, plain aligned text otherwise."""
    if sys.stdout.isatty():
        _render_rich_refresh(results)
    else:
        _render_plain_refresh(results)

    n_ok = sum(1 for r in results if r.state == "ok")
    n_failed = sum(1 for r in results if r.state == "failed")
    n_skipped = sum(1 for r in results if r.state == "skipped")
    print()
    print(f"{n_ok} ok, {n_failed} failed, {n_skipped} skipped")


def _render_rich_refresh(results: list) -> None:
    from rich import box
    from rich.table import Table
    from rich.text import Text

    from muse.cli_impl.console import get_console

    console = get_console()
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold",
                  pad_edge=False, expand=True)
    table.add_column("", width=1, no_wrap=True)
    table.add_column("model_id", no_wrap=True, style="cyan")
    table.add_column("state", no_wrap=True)
    table.add_column("message", overflow="fold", ratio=1)
    for r in results:
        glyph, style = _REFRESH_GLYPHS.get(r.state, ("?", "dim"))
        table.add_row(
            Text(glyph, style=style),
            r.model_id,
            r.state,
            r.message,
        )
    console.print(table)
    # Pip-output tails for failures (rich Panel-free since the content
    # is shell output, not pretty-printable).
    for r in results:
        if r.state == "failed" and r.pip_output:
            tail = "\n".join(r.pip_output.strip().splitlines()[-5:])
            console.print(
                Text(f"\n  {r.model_id} pip output (last 5 lines):", style="dim red")
            )
            console.print(Text(f"    {tail}", style="dim"))


def _render_plain_refresh(results: list) -> None:
    id_w = max((len(r.model_id) for r in results), default=0)
    state_w = max((len(r.state) for r in results), default=0)
    for r in results:
        glyph, _ = _REFRESH_GLYPHS.get(r.state, ("?", ""))
        print(
            f"  {glyph} "
            f"{r.model_id:<{id_w}s}  "
            f"{r.state:<{state_w}s}  "
            f"{r.message}"
        )
        if r.state == "failed" and r.pip_output:
            tail = "\n".join(r.pip_output.strip().splitlines()[-5:])
            print(f"    pip output (last 5 lines):\n    {tail}")
