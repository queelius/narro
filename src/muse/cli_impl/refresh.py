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
  - Then (unless --no-extras): pip install --upgrade <model's pip_extras...>.
  - Continue past failures; aggregate at the end.

The supervisor is NOT restarted. To pick up a refreshed venv, the
operator runs `Ctrl+C; muse serve` themselves. This is documented.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from muse.core.catalog import (
    _PYPI_DIST,
    _is_muse_pyproject,
    _read_catalog,
    get_manifest,
    is_enabled,
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


# `_PYPI_DIST` (published wheel name) and `_is_muse_pyproject` (source-tree
# sniff) are shared with core.catalog, which applies the same wheel-vs-source
# fork to `muse pull`'s venv creation. Imported rather than duplicated.


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
    from a checkout, else the published `museq` distribution so pip
    upgrades from PyPI. `server` is always present; modality extras
    append. Bracket-comma syntax matches PEP 508 extras.
    """
    spec = ",".join(["server", *extras])
    root = _muse_repo_root()
    base = str(root) if root is not None else _PYPI_DIST
    return f"{base}[{spec}]"


def _pip_target_args(extras: list[str]) -> list[str]:
    """pip target args for refreshing muse inside a per-model venv.

    From a source checkout: ``-e <root>[extras]`` (editable, tracks the
    working tree). From a wheel/PyPI install: ``museq[extras]`` (no -e;
    pip resolves museq from PyPI). Returning args rather than a bare
    string lets the caller splat them without re-deciding editability.
    """
    target = _pip_target(extras)
    if _muse_repo_root() is not None:
        return ["-e", target]
    return [target]


def refresh_one(
    model_id: str,
    *,
    no_extras: bool = False,
) -> RefreshResult:
    """Refresh a single model's venv.

    Two pip invocations:
      1. install --upgrade <muse-target>[server,<modality-extras>]
         (editable `-e <tree>` from a checkout, else `museq` from PyPI)
      2. install --upgrade <model's pip_extras...>  (skipped if --no-extras
         or pip_extras is empty)

    On any non-zero pip exit, returns RefreshResult(state='failed') with
    captured stdout+stderr and skips step 2. Catalog entries that point
    at a missing python_path return state='skipped'.
    """
    catalog = _read_catalog()
    entry = catalog.get(model_id)
    if not entry:
        return RefreshResult(model_id, "skipped", "not in catalog")
    python_path = entry.get("python_path")
    if not python_path or not Path(python_path).exists():
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
    muse_extras = _infer_extras(modality)
    target_args = _pip_target_args(muse_extras)

    cmd = [python_path, "-m", "pip", "install", "--upgrade", *target_args]
    logger.info("refresh %s: %s", model_id, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_PIP_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return RefreshResult(
            model_id,
            "failed",
            f"museq[server] install timed out after {_PIP_TIMEOUT}s",
            extras=muse_extras,
        )
    if proc.returncode != 0:
        return RefreshResult(
            model_id,
            "failed",
            "museq[server] install failed",
            proc.stdout + proc.stderr,
            extras=muse_extras,
        )

    if not no_extras and pip_extras_list:
        cmd2 = [python_path, "-m", "pip", "install", "--upgrade", *pip_extras_list]
        logger.info("refresh %s extras: %s", model_id, " ".join(cmd2))
        try:
            proc2 = subprocess.run(
                cmd2, capture_output=True, text=True, timeout=_PIP_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return RefreshResult(
                model_id,
                "failed",
                f"pip_extras install timed out after {_PIP_TIMEOUT}s",
                extras=muse_extras,
            )
        if proc2.returncode != 0:
            return RefreshResult(
                model_id,
                "failed",
                "pip_extras install failed",
                proc2.stdout + proc2.stderr,
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
