"""`muse config` command implementations (generate/show/path/get/set).

Thin logic layer behind the `config_app` typer sub-app registered in
`muse.cli`. Mirrors the `muse models` convention: command functions
here return a process rc (0 success, nonzero on error); `cli.py` wraps
each call in `raise typer.Exit(rc or 0)`.

Import-light by design (typer/rich/muse.core.config only) so `muse
config --help` stays fast, matching the rest of the CLI.

Every command calls `reset_config()` first so a config.yaml written by
a prior command in the same process (or an env var change) is picked
up fresh rather than served from `get_config()`'s memoized `Config`
singleton.
"""
from __future__ import annotations

import json
import sys

import typer

from muse.core import config as cfg

# admin.token must never leak its raw value to `muse config show`
# output (JSON or table). This is the one setting-level exception to
# "show renders config.get(key) verbatim".
_REDACTED_KEY = "admin.token"


def run_path() -> int:
    typer.echo(str(cfg.config_path()))
    return 0


def run_generate(force: bool) -> int:
    cfg.reset_config()
    target = cfg.config_path()
    template = cfg.render_template()
    if force:
        cfg.write_config_text(template, path=target)
    elif not cfg.create_config_text(template, path=target):
        typer.echo(
            f"error: {target} already exists; pass --force to overwrite",
            err=True,
        )
        return 1
    typer.echo(f"wrote {target}")
    return 0


def run_get(key: str) -> int:
    cfg.reset_config()
    if key not in cfg.SETTINGS_BY_KEY:
        typer.echo(f"error: unknown config key {key!r}", err=True)
        return 2
    value = cfg.get(key)
    typer.echo(_display_value(key, value))
    return 0


def run_set(key: str, value: str) -> int:
    cfg.reset_config()
    try:
        coerced = cfg.set_value(key, value)
    except KeyError:
        typer.echo(f"error: unknown config key {key!r}", err=True)
        return 2
    except cfg.ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        return 2
    typer.echo(f"set {key} = {_display_value(key, coerced)}")
    return 0


def run_unset(key: str) -> int:
    cfg.reset_config()
    try:
        removed = cfg.unset_value(key)
    except KeyError:
        typer.echo(f"error: unknown config key {key!r}", err=True)
        return 2
    if removed:
        typer.echo(f"unset {key} (now uses env/default)")
    else:
        typer.echo(f"{key} was not set (no change)")
    return 0


def run_show(as_json: bool) -> int:
    cfg.reset_config()
    rows = _build_rows()
    if as_json:
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    _render(rows)
    return 0


def _redacted_value(key: str, value):
    """The ONE place that decides admin.token redaction.

    admin.token must never leak its raw value through any `muse config`
    surface (`show` table, `show --json`, or `get`). Both `_display_value`
    (get/set) and `_build_rows` (show) funnel through this helper so
    there is exactly one redaction site to audit; a second, independent
    copy is how a drift/regression could leak the real token.
    """
    if key == _REDACTED_KEY:
        return "set" if value else "unset"
    return value


def _display_value(key: str, value) -> str:
    """String form of a setting's value, redacting admin.token."""
    return str(_redacted_value(key, value))


def _build_rows() -> list[dict]:
    rows: list[dict] = []
    for setting in cfg.SETTINGS:
        value = cfg.get(setting.key)
        rows.append({
            "key": setting.key,
            "value": _redacted_value(setting.key, value),
            "source": cfg.source(setting.key),
            "env": setting.env,
        })
    return rows


def _render(rows: list[dict]) -> None:
    """TTY -> rich.Table; non-TTY (pipe/redirect/subprocess) -> plain text."""
    if sys.stdout.isatty():
        _render_rich_table(rows)
    else:
        _render_plain_table(rows)


def _render_rich_table(rows: list[dict]) -> None:
    from rich import box
    from rich.table import Table

    from muse.cli_impl.console import get_console

    console = get_console()
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
        pad_edge=False,
        expand=True,
    )
    table.add_column("key", no_wrap=True, style="cyan")
    table.add_column("value", overflow="ellipsis", no_wrap=True, ratio=1)
    table.add_column("source", no_wrap=True)
    table.add_column("env", no_wrap=True, style="dim")

    for r in rows:
        table.add_row(r["key"], str(r["value"]), r["source"], r["env"])
    console.print(table)


def _render_plain_table(rows: list[dict]) -> None:
    key_w = max((len(r["key"]) for r in rows), default=0)
    value_w = max((len(str(r["value"])) for r in rows), default=0)
    source_w = max((len(r["source"]) for r in rows), default=0)

    for r in rows:
        line = (
            f"{r['key']:<{key_w}s}  "
            f"{str(r['value']):<{value_w}s}  "
            f"{r['source']:<{source_w}s}  "
            f"{r['env']}"
        )
        print(line)
