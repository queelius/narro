"""`muse` CLI — admin commands only.

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
    - LLMs:   muse mcp (future — MCP server over HTTP)

Deliberate non-goals:
    - Per-modality CLI subcommands (e.g., `muse speak`, `muse audio ...`).
      They'd require hardcoded modality→verb mappings that grow every
      time a new modality lands. Keeping the CLI modality-agnostic means
      embeddings / transcriptions / video all work without CLI changes.

Heavy imports (torch, diffusers) are kept out of this module so
`muse --help` stays instant. Command implementations live in
`muse.cli_impl.*` and import what they need when invoked.
"""
from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger("muse")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="muse",
        description="Multi-modality generation server + admin CLI",
    )
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="cmd", required=False)

    # serve
    sp_serve = sub.add_parser("serve", help="start the HTTP gateway (spawns one worker per venv)")
    sp_serve.add_argument("--host", default="0.0.0.0")
    sp_serve.add_argument("--port", type=int, default=8000)
    sp_serve.add_argument("--device", default="auto",
                          choices=["auto", "cpu", "cuda", "mps"])
    sp_serve.set_defaults(func=_cmd_serve)

    # pull (accepts bare model_id OR resolver URI)
    sp_pull = sub.add_parser(
        "pull",
        help=(
            "download weights + install deps for a model "
            "(bundled id like `kokoro-82m` OR resolver URI like "
            "`hf://Qwen/Qwen3-8B-GGUF@q4_k_m`)"
        ),
    )
    sp_pull.add_argument(
        "identifier",
        help="bundled model_id OR resolver URI (e.g. hf://org/repo@variant)",
    )
    sp_pull.set_defaults(func=_cmd_pull)

    # search (HuggingFace + future resolvers)
    sp_search = sub.add_parser(
        "search",
        help="search resolvers (e.g. HuggingFace) for pullable models",
    )
    sp_search.add_argument("query", help="search query")
    sp_search.add_argument(
        "--modality",
        choices=["chat/completion", "embedding/text"],
        default=None,
        help="filter by modality (omit to search all supported)",
    )
    sp_search.add_argument("--limit", type=int, default=20)
    sp_search.add_argument(
        "--sort",
        choices=["downloads", "lastModified", "likes"],
        default="downloads",
    )
    sp_search.add_argument(
        "--max-size-gb",
        type=float,
        default=None,
        help="filter out rows whose size exceeds this (rows with unknown size pass through)",
    )
    sp_search.add_argument(
        "--backend",
        default=None,
        help="resolver backend to use (default: only-registered, or pick when ambiguous)",
    )
    sp_search.set_defaults(func=_cmd_search)

    # _worker (hidden; invoked by supervisor via subprocess)
    sp_worker = sub.add_parser("_worker", help="internal: run a single worker (invoked by muse serve)")
    sp_worker.add_argument("--host", default="127.0.0.1",
                           help="bind address (default: 127.0.0.1, workers are internal)")
    sp_worker.add_argument("--port", type=int, required=True)
    sp_worker.add_argument("--model", action="append", default=[], required=True,
                           help="model to load (repeatable; one worker can host multiple compatible models)")
    sp_worker.add_argument("--device", default="auto",
                           choices=["auto", "cpu", "cuda", "mps"])
    sp_worker.set_defaults(func=_cmd_worker)

    # models (catalog admin)
    sp_models = sub.add_parser("models", help="manage the model catalog")
    models_sub = sp_models.add_subparsers(dest="models_cmd", required=True)

    sp_list = models_sub.add_parser(
        "list",
        help="list known models (bundled scripts + curated recommendations + pulled)",
    )
    sp_list.add_argument("--modality", default=None,
                         help="filter by modality (e.g., audio/speech)")
    sp_list.add_argument(
        "--installed",
        action="store_true",
        help="only models with a catalog.json entry (enabled or disabled)",
    )
    sp_list.add_argument(
        "--available",
        action="store_true",
        help="only models you could install (recommended or available bundled)",
    )
    sp_list.set_defaults(func=_cmd_models_list)

    sp_info = models_sub.add_parser("info", help="show catalog entry for a model")
    sp_info.add_argument("model_id")
    sp_info.set_defaults(func=_cmd_models_info)

    sp_remove = models_sub.add_parser("remove", help="unregister a model from the catalog")
    sp_remove.add_argument("model_id")
    sp_remove.set_defaults(func=_cmd_models_remove)

    sp_enable = models_sub.add_parser("enable", help="enable a pulled model for serving")
    sp_enable.add_argument("model_id")
    sp_enable.set_defaults(func=_cmd_models_enable)

    sp_disable = models_sub.add_parser("disable", help="disable a pulled model (stays in catalog, not loaded by muse serve)")
    sp_disable.add_argument("model_id")
    sp_disable.set_defaults(func=_cmd_models_disable)

    return p


# Command implementations (deferred imports for fast startup)

def _cmd_serve(args):
    from muse.cli_impl.serve import run_serve
    return run_serve(host=args.host, port=args.port, device=args.device)


def _cmd_pull(args):
    from muse.core.catalog import pull
    # When the identifier is a resolver URI, ensure the matching
    # resolver is registered before pull() dispatches. Today only the
    # HF resolver exists; future schemes get their own lazy import.
    if "://" in args.identifier:
        scheme = args.identifier.split("://", 1)[0]
        if scheme == "hf":
            import muse.core.resolvers_hf  # noqa: F401  (registers HFResolver on import)
    try:
        pull(args.identifier)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"pulled {args.identifier}")
    return 0


def _cmd_search(args):
    from muse.cli_impl.search import run_search
    # Register resolver backends. Today only HF; future backends slot in
    # by importing their resolvers_<scheme> module here.
    import muse.core.resolvers_hf  # noqa: F401
    return run_search(
        query=args.query,
        modality=args.modality,
        limit=args.limit,
        sort=args.sort,
        max_size_gb=args.max_size_gb,
        backend=args.backend,
    )


def _cmd_worker(args):
    from muse.cli_impl.worker import run_worker
    return run_worker(
        host=args.host, port=args.port,
        models=args.model, device=args.device,
    )


def _cmd_models_list(args):
    """Print models from three sources: bundled scripts + curated + pulled.

    Status precedence:
      - pulled (in catalog.json) -> 'enabled' or 'disabled' (catalog wins)
      - curated and not pulled  -> 'recommended' (curated trumps bundled-available)
      - bundled and not pulled  -> 'available'

    Filters (mutually compatible):
      --modality M     : only entries whose modality == M
      --installed      : only entries with a catalog.json record
      --available      : only entries you could install (recommended/available)
    """
    from muse.core.catalog import is_enabled, is_pulled, list_known
    from muse.core.curated import load_curated

    bundled_entries = {e.model_id: e for e in list_known(None)}
    curated_entries = {c.id: c for c in load_curated()}

    # Build the unified row set keyed by id. Each row is a dict carrying
    # whatever metadata we have, plus the computed status.
    rows: list[dict] = []
    seen: set[str] = set()

    # 1. Bundled scripts and resolver-pulled entries (everything in known_models).
    for model_id, e in bundled_entries.items():
        seen.add(model_id)
        pulled = is_pulled(model_id)
        if pulled:
            status = "enabled" if is_enabled(model_id) else "disabled"
        elif model_id in curated_entries:
            status = "recommended"
        else:
            status = "available"
        rows.append({
            "id": model_id,
            "modality": e.modality,
            "description": e.description,
            "status": status,
        })

    # 2. Curated entries that aren't already covered by a bundled/pulled
    #    entry above (i.e. resolver-pulled curated aliases).
    for cid, c in curated_entries.items():
        if cid in seen:
            continue
        if is_pulled(cid):
            status = "enabled" if is_enabled(cid) else "disabled"
        else:
            status = "recommended"
        rows.append({
            "id": cid,
            "modality": c.modality or "?",
            "description": c.description or "",
            "status": status,
        })

    # Filters
    if args.modality:
        rows = [r for r in rows if r["modality"] == args.modality]
    if args.installed:
        rows = [r for r in rows if r["status"] in ("enabled", "disabled")]
    if args.available:
        rows = [r for r in rows if r["status"] in ("recommended", "available")]

    if not rows:
        suffixes = []
        if args.modality:
            suffixes.append(f"modality {args.modality!r}")
        if args.installed:
            suffixes.append("--installed")
        if args.available:
            suffixes.append("--available")
        suffix = (" matching " + ", ".join(suffixes)) if suffixes else ""
        print(f"no models{suffix}")
        return 0

    rows.sort(key=lambda r: (r["status"], r["id"]))
    for r in rows:
        print(
            f"  {r['id']:20s} [{r['status']:11s}] "
            f"{r['modality']:22s} {r['description']}"
        )
    return 0


def _cmd_models_info(args):
    from muse.core.catalog import known_models
    catalog = known_models()
    if args.model_id not in catalog:
        print(f"error: unknown model {args.model_id!r}", file=sys.stderr)
        return 2
    e = catalog[args.model_id]
    print(f"model_id:     {e.model_id}")
    print(f"modality:     {e.modality}")
    print(f"hf_repo:      {e.hf_repo}")
    print(f"backend:      {e.backend_path}")
    print(f"pip_extras:   {list(e.pip_extras)}")
    print(f"system_pkgs:  {list(e.system_packages)}")
    if e.extra:
        print(f"extra:        {e.extra}")
    return 0


def _cmd_models_remove(args):
    from muse.core.catalog import remove
    remove(args.model_id)
    print(f"removed {args.model_id} from catalog")
    return 0


def _cmd_models_enable(args):
    from muse.core.catalog import set_enabled
    try:
        set_enabled(args.model_id, True)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"enabled {args.model_id}")
    return 0


def _cmd_models_disable(args):
    from muse.core.catalog import set_enabled
    try:
        set_enabled(args.model_id, False)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"disabled {args.model_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
