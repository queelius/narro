"""`muse` CLI -- admin commands only.

The CLI surface is deliberately minimal and modality-agnostic:

    muse serve                    start the HTTP server
    muse pull <model-id>          download weights + install deps
    muse models list              list known/pulled models (all modalities)
    muse models info <model-id>   show catalog entry
    muse models remove <model-id> unregister from catalog

Generation endpoints are reached via HTTP (the canonical interface):
    - Python: muse.modalities.audio_speech.SpeechClient,
              muse.modalities.image_generation.GenerationsClient
    - Shell:  curl -X POST http://host:8000/v1/audio/speech ...
    - LLMs:   muse mcp (MCP server bridging muse to LLM clients)

Built on typer (a thin click wrapper); argparse-era v0.x.x shipped its
own parser, the typer port landed in v0.39.0. Existing tests that
invoke `python -m muse.cli ...` keep working unchanged because typer
preserves argv handling.

Heavy imports (torch, diffusers) are kept out of this module so
`muse --help` stays instant. Command implementations live in
`muse.cli_impl.*` and import what they need when invoked.
"""
from __future__ import annotations

import logging
import sys
from enum import Enum
from typing import Optional

import typer
from typing_extensions import Annotated


log = logging.getLogger("muse")


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class AdminActionOutcome(str, Enum):
    """Three-way result of `_try_admin_action`.

    - SUCCESS: the admin API handled the action; caller returns (exit 0).
    - FAILED: the admin API was consulted and the action definitively
      failed (job finished non-done, or a non-503 AdminClientError);
      caller raises `typer.Exit(2)`.
    - TIMEOUT: a distinguished FAILED sub-case -- the async job's status
      poll timed out before reaching a terminal state. This is genuinely
      ambiguous (the job may still complete server-side), so it is NOT
      folded into FAILED; callers raise `typer.Exit(3)` to give operators
      a distinct signal from a hard failure.
    - NOT_HANDLED: no admin token configured, the admin API is
      unreachable, or it reports 503 admin_disabled -- caller falls
      through to the catalog-only path exactly as if no admin API
      existed.
    """

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NOT_HANDLED = "not_handled"


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class SearchSort(str, Enum):
    downloads = "downloads"
    lastModified = "lastModified"
    likes = "likes"


class McpFilter(str, Enum):
    all = "all"
    admin = "admin"
    inference = "inference"


# Top-level Typer app. rich_markup_mode lets help strings use [bold]
# and similar inline rich tags; we don't lean on it heavily but it
# costs nothing.
app = typer.Typer(
    name="muse",
    help="Multi-modality generation server + admin CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
models_app = typer.Typer(
    name="models",
    help="manage the model catalog",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(models_app, name="models")
config_app = typer.Typer(
    name="config",
    help="inspect and edit muse's config.yaml",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(config_app, name="config")
doctor_app = typer.Typer(
    name="doctor",
    help="diagnose local muse state and owned runtime resources",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(doctor_app, name="doctor")
storage_app = typer.Typer(
    name="storage",
    help="inspect and reclaim Muse-owned disk space",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(storage_app, name="storage")


def _version_callback(value: bool) -> bool:
    """Print the installed distribution version for eager version flags."""
    if value:
        from muse import __version__

        typer.echo(__version__)
        raise typer.Exit()
    return value


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="show the installed muse version and exit",
        ),
    ] = False,
    log_level: Annotated[
        LogLevel,
        typer.Option("--log-level", help="root logger verbosity"),
    ] = LogLevel.INFO,
) -> None:
    """Root callback: configure logging before any subcommand runs."""
    logging.basicConfig(
        level=getattr(logging, log_level.value), format="%(message)s",
    )


# Top-level subcommands ------------------------------------------------------


@doctor_app.command("resources")
def doctor_resources(
    repair: Annotated[
        bool,
        typer.Option(
            "--repair",
            help=(
                "remove stale records and terminate only identity-verified "
                "orphan Muse process leaders (supervisors, workers, and "
                "admin jobs)"
            ),
        ),
    ] = False,
    grace: Annotated[
        float,
        typer.Option(
            "--grace",
            min=0.0,
            help="seconds to wait before force-killing a verified orphan",
        ),
    ] = 5.0,
) -> None:
    """Inspect Muse's registry without scanning unrelated host processes."""
    from muse.cli_impl.resource_doctor import run_resource_doctor

    raise typer.Exit(run_resource_doctor(repair=repair, grace=grace))


@doctor_app.command("storage")
def doctor_storage(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="emit a machine-readable storage report"),
    ] = False,
) -> None:
    """Inspect owned and adjacent shared storage without changing it."""
    from muse.cli_impl.storage import run_storage_doctor

    raise typer.Exit(run_storage_doctor(json_output=json_output))


@storage_app.command("prune")
def storage_prune(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="show the cleanup plan without deleting"),
    ] = False,
    include_unreferenced: Annotated[
        bool,
        typer.Option(
            "--include-unreferenced",
            help=(
                "also delete old Muse-owned venvs/weights with no catalog "
                "reference; these may have been intentionally retained"
            ),
        ),
    ] = False,
    older_than_hours: Annotated[
        float,
        typer.Option(
            "--older-than-hours",
            min=0.0,
            help="only prune items untouched for at least this many hours",
        ),
    ] = 24.0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="emit machine-readable results"),
    ] = False,
) -> None:
    """Delete old partial/staging data; unreferenced models require opt-in."""
    from muse.cli_impl.storage import run_storage_prune

    raise typer.Exit(run_storage_prune(
        dry_run=dry_run,
        include_unreferenced=include_unreferenced,
        older_than_hours=older_than_hours,
        json_output=json_output,
    ))


def _resolve_serve_device(cli_device: "Device | None") -> str:
    """Resolve the effective `muse serve` device.

    Precedence: explicit `--device` flag > `server.device` config
    (MUSE_DEVICE env / config.yaml) > registry default ("auto").
    `cli_device` is None when the flag was not passed, which is what
    lets the config/env value take over; an explicit flag always wins
    since it is passed as `override`.
    """
    from muse.core import config
    override = cli_device.value if cli_device is not None else None
    return config.get("server.device", override=override)


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option(help="bind address")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="bind port")] = 8000,
    device: Annotated[
        Optional[Device],
        typer.Option(
            help="default device for models (auto|cpu|cuda|mps); "
            "default: $MUSE_DEVICE or config server.device, else auto",
        ),
    ] = None,
) -> None:
    """Start the HTTP gateway (spawns one worker per venv)."""
    from muse.cli_impl.serve import run_serve
    raise typer.Exit(
        run_serve(host=host, port=port, device=_resolve_serve_device(device)) or 0
    )


@app.command("pull")
def pull(
    identifier: Annotated[
        str,
        typer.Argument(
            help=(
                "bundled model_id (e.g. `kokoro-82m`) OR resolver URI "
                "(e.g. `hf://Qwen/Qwen3-8B-GGUF@q4_k_m`)"
            ),
        ),
    ],
    no_probe: Annotated[
        bool,
        typer.Option(
            "--no-probe",
            help=(
                "skip the post-pull memory probe (use for cross-device pulls, "
                "e.g. pulling on a CPU box for later GPU deployment)"
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v",
            help=(
                "stream pip's full dep-resolution output and HF tqdm "
                "progress bars (default: quiet, only stage markers + "
                "errors)"
            ),
        ),
    ] = False,
    base: Annotated[
        str | None,
        typer.Option(
            "--base",
            help=(
                "(LoRA pulls) pair the adapter with this base model: a "
                "pulled muse id (e.g. sdxl-turbo) or an HF repo "
                "(org/name). Overrides the base the repo declares."
            ),
        ),
    ] = None,
) -> None:
    """Download weights + install deps for a model.

    By default the pull is quiet: pip runs with `-q`, HF download
    progress bars are suppressed, and the user sees only the stage
    markers (creating venv, installing museq[server], installing
    pip_extras, downloaded weights, probed). On non-zero pip exit
    the captured output is printed to stderr so dep failures are
    diagnosable. Use `-v` / `--verbose` to stream the full pip +
    HF firehose (the v0.40.2 default).

    The post-pull probe runs by default to populate the new entry's
    `measurements.<device>.peak_bytes`; without that data the v0.40.0
    supervisor flags the model as unservable. `--no-probe` opts out
    for cross-device scenarios.
    """
    from muse.core.catalog import ModelInUseError, _read_catalog, pull as _pull
    from muse.core.venv import install_output_mode
    # Always register the HF resolver before dispatching. The arg may
    # be a URI directly, OR a curated alias that expands to a URI
    # inside pull(); the old conditional "only import when :// is in
    # the arg" missed the second case and crashed with "no resolver
    # for scheme 'hf'". Importing is near-free (heavy huggingface_hub
    # imports happen on actual resolve(), not on module import).
    import muse.core.resolvers_hf  # noqa: F401  (registers HFResolver on import)

    # Capture the catalog state pre-pull so the post-pull probe can
    # identify which entry is new. URIs and curated aliases that synth
    # an override don't reveal the resolved model_id from the pull call
    # itself; the diff is the cheapest, most reliable way to find it.
    try:
        before_keys = set(_read_catalog().keys())
    except Exception:  # noqa: BLE001
        before_keys = set()

    with install_output_mode(verbose=verbose):
        try:
            _pull(identifier, base_override=base)
        except KeyError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(2)
        except ModelInUseError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(2)
        except Exception as e:  # noqa: BLE001
            # A gated/private HF repo pulled without auth raises a deep
            # huggingface_hub error whose uncaught traceback is ~200 lines
            # of noise. Translate the recognized access failures into a
            # one-line actionable message; re-raise anything else so real
            # bugs still surface with a full traceback.
            from muse.cli_impl.pull_errors import friendly_pull_error
            msg = friendly_pull_error(identifier, e)
            if msg is None:
                raise
            typer.echo(msg, err=True)
            raise typer.Exit(1)
        typer.echo(f"pulled {identifier}")

        if no_probe:
            return

        # Probe-on-pull. Failures are logged to stderr by run_probe_for_pull
        # but do not raise; we keep the exit code as the pull's success.
        try:
            from muse.cli_impl.probe import run_probe_for_pull
            run_probe_for_pull(identifier, before_keys=before_keys)
        except Exception as e:  # noqa: BLE001
            typer.echo(
                f"warning: probe-on-pull skipped due to internal error: {e}",
                err=True,
            )


@app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="search query")],
    modality: Annotated[
        Optional[str],
        typer.Option(help="filter by modality (e.g., audio/speech)"),
    ] = None,
    limit: Annotated[int, typer.Option(help="max rows to return")] = 20,
    sort: Annotated[SearchSort, typer.Option()] = SearchSort.downloads,
    max_size_gb: Annotated[
        Optional[float],
        typer.Option(
            "--max-size-gb",
            help=(
                "filter out rows whose size exceeds this "
                "(rows with unknown size pass through)"
            ),
        ),
    ] = None,
    backend: Annotated[
        Optional[str],
        typer.Option(
            help="resolver backend (default: only-registered, or pick when ambiguous)",
        ),
    ] = None,
) -> None:
    """Search resolvers (e.g. HuggingFace) for pullable models."""
    from muse.cli_impl.search import run_search
    from muse.core.discovery import modality_tags
    if modality is not None and modality not in modality_tags():
        valid = ", ".join(sorted(modality_tags()))
        raise typer.BadParameter(
            f"unknown modality {modality!r}; valid: {valid}",
            param_hint="--modality",
        )
    # Register resolver backends. Today only HF; future backends slot
    # in by importing their resolvers_<scheme> module here.
    import muse.core.resolvers_hf  # noqa: F401
    raise typer.Exit(run_search(
        query=query, modality=modality, limit=limit, sort=sort.value,
        max_size_gb=max_size_gb, backend=backend,
    ) or 0)


@app.command("mcp")
def mcp(
    http: Annotated[
        bool,
        typer.Option(
            "--http/--stdio",
            help="HTTP+SSE mode instead of stdio (default: stdio for desktop apps)",
        ),
    ] = False,
    port: Annotated[
        int, typer.Option(help="port for HTTP+SSE mode"),
    ] = 8088,
    server: Annotated[
        Optional[str],
        typer.Option(help="muse server URL (default: $MUSE_SERVER or http://localhost:8000)"),
    ] = None,
    admin_token: Annotated[
        Optional[str],
        typer.Option(
            "--admin-token",
            help="admin bearer token for /v1/admin/* tools (default: $MUSE_ADMIN_TOKEN)",
        ),
    ] = None,
    filter_kind: Annotated[
        McpFilter,
        typer.Option(
            "--filter",
            help="restrict tool surface (default: all 31 tools)",
        ),
    ] = McpFilter.all,
) -> None:
    """Run an MCP server bridging muse to LLM clients (Claude Desktop, Cursor)."""
    from muse.cli_impl.mcp_server import run_mcp_server
    from muse.core import config
    server_url = server or config.get("client.server_url")
    admin_token = admin_token or config.get("admin.token")
    raise typer.Exit(run_mcp_server(
        http=http, port=port, server_url=server_url,
        admin_token=admin_token, filter_kind=filter_kind.value,
    ) or 0)


@app.command("federate")
def federate(
    host: Annotated[str, typer.Option(help="bind address")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="bind port")] = 8100,
    node: Annotated[
        list[str],
        typer.Option(
            "--node",
            help=(
                "remote muse node URL (repeatable), e.g. "
                "http://192.168.0.204:8000 or name=http://host:8000"
            ),
        ),
    ] = [],
    config: Annotated[
        Optional[str],
        typer.Option(
            "--config",
            help=(
                "path to a federation node-list yaml (default: "
                "$MUSE_FEDERATION_CONFIG, else <catalog_dir>/federation.yaml "
                "if it exists)"
            ),
        ),
    ] = None,
) -> None:
    """Start the federation coordinator (fronts a fixed set of muse nodes).

    Merges `--node` entries with the resolved yaml file, routes each
    request by model-locality across the node cluster, and exits 2
    without starting a server if no nodes resolve from either source.
    """
    from muse.cli_impl.federation import run_coordinator
    raise typer.Exit(run_coordinator(
        host=host, port=port, cli_nodes=node or None, config_path=config,
    ) or 0)


# Hidden internal commands invoked by the supervisor as subprocesses.
# They never show up in `muse --help` and aren't documented for users.

@app.command("_worker", hidden=True)
def _worker(
    port: Annotated[int, typer.Option()],
    model: Annotated[
        list[str],
        typer.Option(help="model to load (repeatable)"),
    ],
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    device: Annotated[Device, typer.Option()] = Device.auto,
) -> None:
    """internal: run a single worker (invoked by `muse serve`)."""
    from muse.cli_impl.worker import run_worker
    raise typer.Exit(run_worker(
        host=host, port=port, models=model, device=device.value,
    ) or 0)


@app.command("_probe_worker", hidden=True)
def _probe_worker(
    model: Annotated[str, typer.Option()],
    device: Annotated[str, typer.Option()] = "auto",
    no_inference: Annotated[
        bool,
        typer.Option("--no-inference", help="load only; skip inference"),
    ] = False,
) -> None:
    """internal: run a probe in this venv (invoked by `muse models probe`)."""
    from muse.cli_impl.probe_worker import run_probe_worker
    raise typer.Exit(run_probe_worker(
        model_id=model, device=device, run_inference=not no_inference,
    ) or 0)


# `muse models <verb>` ------------------------------------------------------


@models_app.command("list")
def models_list(
    modality: Annotated[
        Optional[str],
        typer.Option(help="filter by modality (e.g., audio/speech)"),
    ] = None,
    installed: Annotated[
        bool,
        typer.Option(
            "--installed",
            help="only models with a catalog.json entry (enabled or disabled)",
        ),
    ] = False,
    available: Annotated[
        bool,
        typer.Option(
            "--available",
            help="only models you could install (recommended or available bundled)",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="machine-readable JSON instead of the human table",
        ),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="disable color and table styling"),
    ] = False,
) -> None:
    """List known models (bundled scripts + curated recommendations + pulled)."""
    from muse.cli_impl.models_list import run_models_list
    raise typer.Exit(run_models_list(
        modality=modality, installed=installed, available=available,
        as_json=as_json, no_color=no_color,
    ) or 0)


@models_app.command("info")
def models_info(
    model_id: Annotated[str, typer.Argument()],
) -> None:
    """Show catalog entry for a model.

    Recognizes three sources, in priority order:
      1. Bundled scripts and resolver-pulled entries (`known_models()`)
      2. Curated recommendations not yet pulled (`curated.yaml`)
      3. Unknown -> 2 with a clear error.

    For curated-only rows the output shows description / hf_repo /
    license / suggested memory plus a `muse pull <id>` hint, so
    `muse models info <id>` matches the rows surfaced by
    `muse models list` symmetrically.
    """
    from muse.core.catalog import _read_catalog, known_models
    from muse.cli_impl.models_info_display import format_info
    from muse.core.curated import load_curated

    catalog_known = known_models()
    if model_id in catalog_known:
        catalog_data = _read_catalog().get(model_id, {}) or {}
        online_status = _probe_online_worker_status(model_id)
        typer.echo(format_info(
            model_id,
            catalog_known=catalog_known,
            catalog_data=catalog_data,
            online_status=online_status,
        ))
        return

    # Fall through: maybe it's a curated-only recommendation that
    # hasn't been pulled. Render a minimal info card from the YAML.
    curated_match = next(
        (c for c in load_curated() if c.id == model_id), None,
    )
    if curated_match is not None:
        typer.echo(_format_curated_info(curated_match))
        return

    typer.echo(f"error: unknown model {model_id!r}", err=True)
    raise typer.Exit(2)


def _format_curated_info(c) -> str:
    """Render a `CuratedEntry` as the same shape as `format_info` for
    curated-only (not-yet-pulled) entries.
    """
    lines: list[str] = []
    lines.append(f"{c.id}".ljust(36) + "  [recommended, not pulled]")
    lines.append("")
    lines.append("Basics:")
    if c.modality:
        lines.append(f"  modality:        {c.modality}")
    if c.uri:
        lines.append(f"  uri:             {c.uri}")
    if c.description:
        lines.append(f"  description:     {c.description}")
    if c.size_gb is not None:
        lines.append(f"  size on HF:      ~{c.size_gb:.1f} GB")
    if c.tags:
        lines.append(f"  tags:            {', '.join(c.tags)}")
    if c.capabilities:
        lines.append("")
        lines.append("Capabilities (would be applied at pull):")
        for k, v in c.capabilities.items():
            lines.append(f"  {k + ':':17s}{v}")
    lines.append("")
    lines.append(f"To install: muse pull {c.id}")
    return "\n".join(lines)


@models_app.command("remove")
def models_remove(
    model_id: Annotated[str, typer.Argument()],
    purge: Annotated[
        bool,
        typer.Option(
            "--purge",
            help=(
                "also delete the per-model venv and unshared Muse-owned "
                "weights; legacy weights in a shared HF cache are preserved"
            ),
        ),
    ] = False,
) -> None:
    """Unregister a model from the catalog."""
    from muse.core.catalog import CatalogError, remove
    try:
        remove(model_id, purge=purge)
    except CatalogError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    suffix = " (purged unshared Muse-owned files)" if purge else ""
    typer.echo(f"removed {model_id} from catalog{suffix}")


@models_app.command("enable")
def models_enable(
    model_id: Annotated[str, typer.Argument()],
) -> None:
    """Enable a pulled model for serving.

    When MUSE_ADMIN_TOKEN is set AND the supervisor is running, this
    routes through the admin API so the running gateway picks up the
    change live (worker spawn or restart-in-place). Otherwise it falls
    back to mutating catalog.json directly with a warning.
    """
    outcome = _try_admin_action("enable", model_id)
    if outcome is AdminActionOutcome.SUCCESS:
        return
    if outcome is AdminActionOutcome.FAILED:
        raise typer.Exit(2)
    if outcome is AdminActionOutcome.TIMEOUT:
        raise typer.Exit(3)
    # NOT_HANDLED: no admin token / admin unreachable / admin disabled
    # (503) -- fall through to the catalog-only path exactly as today.

    from muse.core.catalog import set_enabled
    try:
        set_enabled(model_id, True)
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    typer.echo(
        f"enabled {model_id} (catalog only; supervisor will pick this up on restart)"
    )


@models_app.command("disable")
def models_disable(
    model_id: Annotated[str, typer.Argument()],
) -> None:
    """Disable a pulled model.

    Stays in catalog, not loaded by `muse serve`. Same admin-API
    routing as `enable`.
    """
    outcome = _try_admin_action("disable", model_id)
    if outcome is AdminActionOutcome.SUCCESS:
        return
    if outcome is AdminActionOutcome.FAILED:
        raise typer.Exit(2)
    if outcome is AdminActionOutcome.TIMEOUT:
        raise typer.Exit(3)
    # NOT_HANDLED: fall through to the catalog-only path exactly as today.

    from muse.core.catalog import set_enabled
    try:
        set_enabled(model_id, False)
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    typer.echo(
        f"disabled {model_id} (catalog only; supervisor will pick this up on restart)"
    )


@models_app.command("set-device")
def models_set_device(
    model_id: Annotated[str, typer.Argument()],
    device: Annotated[
        Optional[Device],
        typer.Argument(
            help="device to pin the model to (auto|cpu|cuda|mps); omit with --clear",
        ),
    ] = None,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="remove the override (revert to manifest / --device)"),
    ] = False,
) -> None:
    """Pin a model's load device, overriding its manifest device + --device flag.

    Writes a per-model `device_override` to the catalog. Precedence at
    load time: override > manifest capabilities.device pin > --device flag
    > auto. Use `cuda` to force a cpu-pinned model (e.g. kokoro-82m) onto
    the GPU, `cpu` to keep a model off scarce VRAM, or `auto` to un-pin and
    let muse pick cuda-if-available. `--clear` removes the override.

    Catalog-only state: takes effect on the model's NEXT cold load. To
    apply it to an already-resident worker, evict it (idle/memory pressure)
    or restart the supervisor.
    """
    from muse.core.catalog import set_device_override

    if clear:
        target = None
    elif device is None:
        typer.echo(
            "error: provide a device (auto|cpu|cuda|mps) or pass --clear",
            err=True,
        )
        raise typer.Exit(2)
    else:
        target = device.value

    try:
        set_device_override(model_id, target)
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if target is None:
        typer.echo(
            f"cleared device override for {model_id} "
            "(takes effect on next cold load)"
        )
    else:
        typer.echo(
            f"set {model_id} device override -> {target} "
            "(takes effect on next cold load)"
        )


@models_app.command("set-gpu-layers")
def models_set_gpu_layers(
    model_id: Annotated[str, typer.Argument()],
    n: Annotated[
        Optional[int],
        typer.Argument(
            help="llama.cpp n_gpu_layers: N >= 0 (0 = pure CPU, N > 0 = "
                 "first N layers on GPU); use --all for -1 (every layer on "
                 "GPU) or --clear to remove the pin",
        ),
    ] = None,
    all_layers: Annotated[
        bool,
        typer.Option(
            "--all",
            help="offload every layer to the GPU (writes -1); shells "
                 "reject a bare -1 positional (click has no negative-"
                 "number heuristic), so use this flag instead",
        ),
    ] = False,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="remove the pin (revert to manifest / runtime default)"),
    ] = False,
) -> None:
    """Pin a GGUF model's llama.cpp GPU/CPU layer split (operator override).

    Writes a per-model `gpu_layers_override` to the catalog. Precedence at
    load time: pin > manifest capabilities.n_gpu_layers > runtime default
    (-1, everything the GPU fits). GGUF-only: refuses models without
    capabilities.gguf_file, since other runtimes silently ignore the kwarg.

    Exactly one of a layer count, --all, or --clear is required:

        muse models set-gpu-layers <id> 30      split: first 30 layers on GPU
        muse models set-gpu-layers <id> 0       pure CPU
        muse models set-gpu-layers <id> --all   everything on GPU (writes -1)
        muse models set-gpu-layers <id> --clear remove the pin

    click has no negative-number heuristic for positional arguments, so a
    bare `-1` is parsed as an unrecognized option and the command dies at
    the shell before this code ever runs -- that is why -1 is not
    documented as a positional value above; `--all` is the supported way
    to request it. Shell die-hards can still pass a literal -1 via the
    `--` end-of-options marker (`muse models set-gpu-layers <id> -- -1`),
    which forces click to treat everything after it as positional.

    Catalog-only state: takes effect on the model's NEXT cold load. To
    apply it to an already-resident worker, evict it or restart the
    supervisor. Run `muse models probe <id>` after pinning so admission
    sizing measures the split's real (smaller) VRAM peak.
    """
    from muse.core.catalog import known_models, set_gpu_layers_override

    given = (n is not None, all_layers, clear)
    if sum(given) != 1:
        typer.echo(
            "error: provide exactly one of a layer count (int >= 0), "
            "--all, or --clear",
            err=True,
        )
        raise typer.Exit(2)

    if clear:
        target = None
    elif all_layers:
        target = -1
    else:
        target = n

    if target is not None:
        entry = known_models().get(model_id)
        # entry is None (unknown / never-pulled model): skip the GGUF
        # capability check here and fall through to set_gpu_layers_override
        # below, whose KeyError produces the uniform "not pulled" error
        # instead of the misleading "is not a GGUF model" refusal.
        if entry is not None:
            capabilities = entry.extra or {}
            if not capabilities.get("gguf_file"):
                typer.echo(
                    f"error: {model_id!r} is not a GGUF model (no "
                    "capabilities.gguf_file); n_gpu_layers only applies to "
                    "llama.cpp runtimes and would be silently ignored",
                    err=True,
                )
                raise typer.Exit(2)

    try:
        set_gpu_layers_override(model_id, target)
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if target is None:
        typer.echo(
            f"cleared gpu-layers pin for {model_id} "
            "(takes effect on next cold load)"
        )
    else:
        typer.echo(
            f"set {model_id} gpu layers -> {target} "
            "(takes effect on next cold load; run `muse models probe "
            f"{model_id}` to re-measure VRAM)"
        )


@models_app.command("warmup")
def models_warmup(
    model_id: Annotated[str, typer.Argument()],
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help=(
                "max seconds to wait for the cold load to complete "
                "(default: 300; bump for very large models)"
            ),
        ),
    ] = 300.0,
) -> None:
    """Pre-load a model via the running supervisor.

    Routes through the admin API's `POST /v1/admin/models/{id}/warmup`,
    which calls `LoadDirector.warmup` to bring the model into the loaded
    set without consuming a request slot. Subsequent inference requests
    skip the cold-load latency.

    Warmup is purely a runtime operation: there is no offline equivalent
    (catalog state alone cannot pre-load anything). When
    `MUSE_ADMIN_TOKEN` is unset OR the supervisor isn't reachable, this
    command exits non-zero with a clear error rather than silently
    succeeding.
    """
    from muse.core import config
    if not config.get("admin.token"):
        typer.echo(
            "error: warmup requires a running `muse serve` with MUSE_ADMIN_TOKEN set",
            err=True,
        )
        raise typer.Exit(2)

    try:
        from muse.admin.client import AdminClient, AdminClientError
    except Exception as e:  # noqa: BLE001
        typer.echo(f"error: admin client unavailable: {e}", err=True)
        raise typer.Exit(2)

    client = AdminClient(timeout=timeout)
    try:
        # Pass the per-call timeout explicitly: cold loads take 10-60s
        # for typical models and longer (5+ min) for large diffusion or
        # video models. The constructor default (30s) would routinely
        # truncate; the CLI knob lets operators tune up.
        out = client.warmup(model_id, timeout=timeout)
    except AdminClientError as e:
        if e.status == 503 and e.code == "admin_disabled":
            typer.echo(
                "error: warmup requires a running `muse serve` with MUSE_ADMIN_TOKEN set",
                err=True,
            )
            raise typer.Exit(2)
        typer.echo(f"error: {e.message}", err=True)
        raise typer.Exit(1)
    except Exception as e:  # noqa: BLE001
        typer.echo(
            f"error: warmup requires a running `muse serve`; could not reach supervisor ({e})",
            err=True,
        )
        raise typer.Exit(2)

    port = out.get("worker_port") if isinstance(out, dict) else None
    if port:
        typer.echo(f"warmed up {model_id} (worker port {port})")
    else:
        typer.echo(f"warmed up {model_id}")


@models_app.command("probe")
def models_probe(
    model_id: Annotated[
        Optional[str],
        typer.Argument(help="model to probe (omit to probe all enabled)"),
    ] = None,
    no_inference: Annotated[
        bool,
        typer.Option(
            "--no-inference",
            help="load only; skip representative inference (faster but undersells peak)",
        ),
    ] = False,
    device: Annotated[
        Optional[Device],
        typer.Option(help="override the model's device preference for this probe"),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="machine-readable output instead of human-readable summary",
        ),
    ] = False,
) -> None:
    """Measure VRAM/RAM by loading the model and running representative inference."""
    from muse.cli_impl.probe import run_probe, run_probe_all
    device_str = device.value if device is not None else None
    if model_id is None:
        rc = run_probe_all(
            no_inference=no_inference, device=device_str, as_json=as_json,
        )
    else:
        rc = run_probe(
            model_id=model_id, no_inference=no_inference,
            device=device_str, as_json=as_json,
        )
    raise typer.Exit(rc or 0)


@models_app.command("refresh")
def models_refresh(
    model_id: Annotated[
        Optional[str],
        typer.Argument(help="model to refresh (omit if using --all or --enabled)"),
    ] = None,
    all_: Annotated[
        bool,
        typer.Option("--all", help="refresh every pulled venv"),
    ] = False,
    enabled_only: Annotated[
        bool,
        typer.Option("--enabled", help="only refresh enabled venvs"),
    ] = False,
    no_extras: Annotated[
        bool,
        typer.Option(
            "--no-extras",
            help="only refresh museq[server]; skip the model's pip_extras",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="machine-readable output instead of human-readable summary",
        ),
    ] = False,
) -> None:
    """Re-install museq[server] + the model's pip_extras into per-model venvs."""
    from muse.cli_impl.refresh import run_refresh
    raise typer.Exit(run_refresh(
        model_id=model_id, all_=all_, enabled_only=enabled_only,
        no_extras=no_extras, as_json=as_json,
    ) or 0)


# `muse config <verb>` -------------------------------------------------------


@config_app.command("path")
def config_path() -> None:
    """Print the resolved config.yaml path."""
    from muse.cli_impl.config_cmd import run_path
    raise typer.Exit(run_path() or 0)


@config_app.command("generate")
def config_generate(
    force: Annotated[
        bool,
        typer.Option("--force", help="overwrite an existing config.yaml"),
    ] = False,
) -> None:
    """Write a fully-commented config.yaml template to the config path."""
    from muse.cli_impl.config_cmd import run_generate
    raise typer.Exit(run_generate(force) or 0)


@config_app.command("show")
def config_show(
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="machine-readable JSON instead of the human table",
        ),
    ] = False,
) -> None:
    """Show every setting's resolved value + source (env/file/default).

    `admin.token` is always redacted to `set`/`unset`.
    """
    from muse.cli_impl.config_cmd import run_show
    raise typer.Exit(run_show(as_json) or 0)


@config_app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="dotted setting key, e.g. limits.rerank_max_documents")],
) -> None:
    """Print one setting's resolved value."""
    from muse.cli_impl.config_cmd import run_get
    raise typer.Exit(run_get(key) or 0)


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="dotted setting key, e.g. server.gpu_headroom_gb")],
    value: Annotated[str, typer.Argument(help="raw value; coerced per the setting's type")],
) -> None:
    """Write one setting into config.yaml (strict type validation)."""
    from muse.cli_impl.config_cmd import run_set
    raise typer.Exit(run_set(key, value) or 0)


@config_app.command("unset")
def config_unset(
    key: Annotated[str, typer.Argument(help="dotted setting key to remove from config.yaml")],
) -> None:
    """Remove one setting from config.yaml so it falls back to env/default."""
    from muse.cli_impl.config_cmd import run_unset
    raise typer.Exit(run_unset(key) or 0)


# Helpers (preserved verbatim from the argparse era) -----------------------


def _probe_online_worker_status(model_id: str) -> dict | None:
    """Best-effort lookup of live worker status.

    Prefers the admin API (rich worker pid / uptime / restart detail)
    when MUSE_ADMIN_TOKEN is set and the supervisor is reachable. Falls
    back to the PUBLIC /v1/models endpoint, which reports loaded state
    without a token, so `muse models info` shows loaded/not-loaded for
    anyone who can reach the server. Returns None only when nothing is
    reachable; the caller then shows the offline "unreachable" view.

    Shapes:
      - admin:  the /v1/admin/models/{id}/status dict (loaded + worker_*).
      - public: {"loaded": bool, "detail_source": "public"}.
      - None:   nothing reachable.
    """
    admin = _probe_admin_worker_status(model_id)
    if admin is not None:
        return admin
    return _public_loaded_status(model_id)


def _probe_admin_worker_status(model_id: str) -> dict | None:
    """Admin-API worker status, or None without a token / on any failure."""
    from muse.core import config
    if not config.get("admin.token"):
        return None
    try:
        from muse.admin.client import AdminClient, AdminClientError
    except Exception:  # noqa: BLE001
        return None
    client = AdminClient(timeout=2.0)
    try:
        return client.status(model_id)
    except AdminClientError:
        return None
    except Exception:  # noqa: BLE001
        return None


def _public_loaded_status(model_id: str) -> dict | None:
    """Loaded state from the public /v1/models endpoint (no admin token).

    Returns {"loaded": bool, "detail_source": "public"} when the server
    is reachable, else None. A reachable server that does not list the
    model reports loaded=False (reachable, not resident).
    """
    from muse.cli_impl.runtime_state import fetch_public_models, loaded_ids
    data = fetch_public_models()
    if data is None:
        return None
    # Reuse loaded_ids so `muse models info` and `muse models list` answer
    # "is this loaded?" with the SAME predicate (strict `loaded is True`).
    return {"loaded": model_id in loaded_ids(data), "detail_source": "public"}


def _try_admin_action(action: str, model_id: str) -> AdminActionOutcome:
    """Try the admin-API path for enable/disable.

    Returns an `AdminActionOutcome` (see its docstring for the meaning
    of each value). Callers must NOT collapse this back to a bool: a
    FAILED or TIMEOUT outcome means the admin API WAS consulted and the
    action did not cleanly succeed, which is different from NOT_HANDLED
    (no admin token / unreachable / 503 admin_disabled), where falling
    through to the catalog-only path is correct.
    """
    from muse.core import config
    if not config.get("admin.token"):
        return AdminActionOutcome.NOT_HANDLED
    try:
        from muse.admin.client import AdminClient, AdminClientError
    except Exception:  # noqa: BLE001
        return AdminActionOutcome.NOT_HANDLED
    client = AdminClient(timeout=5.0)
    try:
        if action == "enable":
            out = client.enable(model_id)
            job_id = out.get("job_id")
            if not job_id:
                # client.enable() didn't raise, so the admin API already
                # accepted the request; a missing/empty job_id is a
                # degenerate but still-successful response shape. Treat
                # it as handled so the caller does NOT fall through to
                # the catalog-only fallback (which would both re-do the
                # mutation and print a misleading "catalog only" message).
                typer.echo(f"enabled {model_id} (admin API accepted)")
                return AdminActionOutcome.SUCCESS
            typer.echo(f"enable job submitted: {job_id}")
            try:
                final = client.wait(job_id, timeout=120.0, poll=0.5)
                if final.get("state") == "done":
                    port = (final.get("result") or {}).get("worker_port")
                    msg = f"enabled {model_id}"
                    if port:
                        msg += f" (worker port {port})"
                    typer.echo(msg)
                    return AdminActionOutcome.SUCCESS
                typer.echo(
                    f"error: enable failed: {final.get('error')}", err=True,
                )
                return AdminActionOutcome.FAILED
            except TimeoutError:
                typer.echo(
                    f"warning: enable still running; poll job {job_id} for status",
                    err=True,
                )
                return AdminActionOutcome.TIMEOUT
        elif action == "disable":
            out = client.disable(model_id)
            typer.echo(f"disabled {model_id}")
            if out.get("worker_terminated"):
                typer.echo(f"  worker on port {out.get('worker_port')} stopped")
            elif out.get("worker_port"):
                remaining = out.get("remaining_models_in_worker") or []
                typer.echo(
                    f"  worker on port {out['worker_port']} restarted; "
                    f"remaining models: {', '.join(remaining)}"
                )
            return AdminActionOutcome.SUCCESS
    except AdminClientError as e:
        if e.status == 503 or e.code == "admin_disabled":
            return AdminActionOutcome.NOT_HANDLED
        typer.echo(f"error: {e.message}", err=True)
        return AdminActionOutcome.FAILED
    except Exception as e:  # noqa: BLE001
        typer.echo(
            f"warning: admin API unreachable ({e}); falling back to catalog-only update",
            err=True,
        )
        return AdminActionOutcome.NOT_HANDLED
    return AdminActionOutcome.NOT_HANDLED


def main(argv: list[str] | None = None) -> int:
    """Programmatic entry point. Accepts argv override for tests.

    `standalone_mode=False` lets us translate `typer.Exit` into a
    return code instead of letting click call `sys.exit`. The price
    is that any other click exception (`NoArgsIsHelpError` when the
    user runs `muse` with no subcommand; `UsageError` for bad flags;
    `Abort` for SIGINT) also propagates here. We catch the click
    base class `ClickException` and use its `exit_code`.

    Critically, with `standalone_mode=False` neither click nor typer's
    own `_main` override ever calls `ClickException.show()` -- both
    just re-raise (see `click.core.BaseCommand.main` and
    `typer.core._main`'s `if not standalone_mode: raise` branches).
    Rendering the error is therefore entirely our job here; skipping
    `e.show()` (a past bug) meant every usage error (unknown
    command/option, missing argument, bad value) exited nonzero with
    completely empty stdout+stderr on the shipped `muse` binary.
    `NoArgsIsHelpError` is the one exception already visible before we
    get here: typer's rich help renderer prints as a side effect of
    building `ctx.get_help()` while the exception is constructed, so
    calling `.show()` on it too only adds a harmless blank line to
    stderr, not a duplicate help render.
    """
    import click
    _click_exceptions = [click.exceptions.ClickException]
    try:
        # typer >= 0.26 vendors click as `typer._click`; its exception
        # classes are DISTINCT from the installed click package's, so
        # catching only `click.exceptions.ClickException` lets every
        # usage error escape as an uncaught traceback on the shipped
        # `muse` binary. Catch both hierarchies.
        from typer import _click as _typer_click
        _click_exceptions.append(_typer_click.exceptions.ClickException)
    except ImportError:
        pass  # older typer: the installed click IS typer's click
    _click_exceptions = tuple(_click_exceptions)
    try:
        rv = app(args=argv, standalone_mode=False)
        # With standalone_mode=False, click catches `typer.Exit` itself and
        # *returns* its exit code as app()'s return value (it does not
        # re-raise). Honor that int so a command's `raise typer.Exit(2)`
        # actually exits 2 on the real `muse` binary; a normal command
        # returns None -> 0. (Subprocess tests use `python -m muse.cli` ->
        # app() in standalone mode, so they never exercised this path.)
        return rv if isinstance(rv, int) and not isinstance(rv, bool) else 0
    except typer.Exit as e:
        return e.exit_code
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    except _click_exceptions as e:
        # Render the error/usage message ourselves: standalone_mode=False
        # suppresses click's own show()+sys.exit() so nothing gets
        # printed otherwise (see docstring above).
        e.show()
        return e.exit_code


if __name__ == "__main__":
    app()
