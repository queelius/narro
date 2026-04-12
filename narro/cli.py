#!/usr/bin/env python3
"""Narro TTS Command Line Interface.

``narro serve`` starts the TTS server.  ``narro models`` manages local
model downloads (pull, list, info, remove).
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def cmd_serve(args):
    """Start the TTS API server, loading all pulled models."""
    from narro.server import serve

    models = None
    if args.model:
        models = [m.strip() for m in args.model.split(",")]

    serve(
        host=args.host,
        port=args.port,
        models=models,
        device=args.device,
        compile=not args.no_compile,
        quantize=args.quantize,
    )


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def cmd_models_list(args):
    """List known models and their pull status."""
    from narro.catalog import KNOWN_MODELS, pulled_models

    pulled = pulled_models()
    for mid, entry in sorted(KNOWN_MODELS.items()):
        status = "pulled" if mid in pulled else "not pulled"
        voice_info = f"  ({len(entry.voices)} voices)" if entry.voices else ""
        print(f"  {mid:<20} {status:<14} {entry.size_mb:>4} MB  {entry.description}{voice_info}")


def cmd_models_pull(args):
    """Download a model's weights."""
    from narro.catalog import pull
    pull(args.name)
    print(f"Pulled {args.name}.")


def cmd_models_info(args):
    """Show detailed info about a model."""
    from narro.catalog import KNOWN_MODELS, is_pulled, voices_dir

    entry = KNOWN_MODELS.get(args.name)
    if entry is None:
        available = ", ".join(sorted(KNOWN_MODELS))
        print(f"Unknown model: {args.name!r}. Available: {available}", file=sys.stderr)
        sys.exit(1)

    status = "pulled" if is_pulled(args.name) else "not pulled"
    print(f"Model:       {entry.id}")
    print(f"Status:      {status}")
    print(f"Description: {entry.description}")
    print(f"HF repo:     {entry.hf_repo}")
    print(f"Backend:     {entry.backend}")
    print(f"Sample rate: {entry.sample_rate} Hz")
    print(f"Size:        ~{entry.size_mb} MB")

    if entry.voices:
        print(f"Voices:      {', '.join(entry.voices)}")

    # Show custom voices if any
    vdir = voices_dir(args.name)
    custom = [f.stem for f in vdir.iterdir() if f.suffix == ".npz"]
    if custom:
        print(f"Custom:      {', '.join(sorted(custom))}")


def cmd_models_remove(args):
    """Remove a model from the local catalog."""
    from narro.catalog import remove
    remove(args.name)
    print(f"Removed {args.name} from catalog.")


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Narro: model-agnostic text-to-speech server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  narro models pull soprano-80m           # download model
  narro models list                       # show available models
  narro serve                             # serve all pulled models
  narro serve --model soprano-80m         # serve specific model
""",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Start TTS API server")
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", "-p", type=int, default=8000, help="Bind port (default: 8000)"
    )
    serve_parser.add_argument(
        "--model",
        help="Model(s) to load, comma-separated (default: all pulled)",
    )
    serve_parser.add_argument(
        "--device",
        "-d",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)",
    )
    serve_parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile optimization",
    )
    serve_parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization (faster, lower quality)",
    )
    serve_parser.set_defaults(func=cmd_serve)

    # --- models ---
    models_parser = subparsers.add_parser("models", help="Manage TTS models")
    models_sub = models_parser.add_subparsers(dest="models_command")

    # models list
    list_parser = models_sub.add_parser("list", help="List available models")
    list_parser.set_defaults(func=cmd_models_list)

    # models pull
    pull_parser = models_sub.add_parser("pull", help="Download a model")
    pull_parser.add_argument("name", help="Model ID (e.g. soprano-80m)")
    pull_parser.set_defaults(func=cmd_models_pull)

    # models info
    info_parser = models_sub.add_parser("info", help="Show model details")
    info_parser.add_argument("name", help="Model ID")
    info_parser.set_defaults(func=cmd_models_info)

    # models remove
    rm_parser = models_sub.add_parser("remove", help="Remove a model from catalog")
    rm_parser.add_argument("name", help="Model ID")
    rm_parser.set_defaults(func=cmd_models_remove)

    # Default: show help for models subcommand
    models_parser.set_defaults(func=lambda a: models_parser.print_help())

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
