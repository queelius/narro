"""`muse` CLI — admin commands only.

The CLI surface is deliberately minimal and modality-agnostic:

    muse serve                    start the HTTP server
    muse pull <model-id>          download weights + install deps
    muse models list              list known/pulled models (all modalities)
    muse models info <model-id>   show catalog entry
    muse models remove <model-id> unregister from catalog

Generation endpoints are reached via HTTP (the canonical interface):
    - Python: muse.audio.speech.SpeechClient,
              muse.images.generations.GenerationsClient
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

    # pull
    sp_pull = sub.add_parser("pull", help="download weights + install deps for a model")
    sp_pull.add_argument("model_id")
    sp_pull.set_defaults(func=_cmd_pull)

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

    sp_list = models_sub.add_parser("list", help="list known models across all modalities")
    sp_list.add_argument("--modality", default=None,
                         help="filter by modality (e.g., audio.speech)")
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
    try:
        pull(args.model_id)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"pulled {args.model_id}")
    return 0


def _cmd_worker(args):
    from muse.cli_impl.worker import run_worker
    return run_worker(
        host=args.host, port=args.port,
        models=args.model, device=args.device,
    )


def _cmd_models_list(args):
    from muse.core.catalog import is_pulled, list_known
    entries = list_known(args.modality)
    if not entries:
        msg = (f"no known models for modality {args.modality!r}"
               if args.modality else "no known models")
        print(msg)
        return 0
    for e in entries:
        if is_pulled(e.model_id):
            from muse.core.catalog import is_enabled
            status = "pulled" if is_enabled(e.model_id) else "disabled"
        else:
            status = "available"
        print(f"  {e.model_id:20s} [{status:9s}] {e.modality:22s} {e.description}")
    return 0


def _cmd_models_info(args):
    from muse.core.catalog import KNOWN_MODELS
    if args.model_id not in KNOWN_MODELS:
        print(f"error: unknown model {args.model_id!r}", file=sys.stderr)
        return 2
    e = KNOWN_MODELS[args.model_id]
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
