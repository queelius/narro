#!/usr/bin/env python3
"""Narro TTS Command Line Interface.

The CLI is a thin client against a Narro TTS server.  ``narro serve``
starts the server; ``narro speak`` talks to it.
"""

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _require_server(args) -> str:
    """Return the server URL from --server / NARRO_SERVER, or exit."""
    url = getattr(args, "server", None) or os.environ.get("NARRO_SERVER")
    if not url:
        print(
            "Error: No server URL configured.\n"
            "\n"
            "  Set NARRO_SERVER or pass --server:\n"
            "    export NARRO_SERVER=http://localhost:8000\n"
            "    narro speak \"Hello world\"\n"
            "\n"
            "  Or start a local server first:\n"
            "    narro serve",
            file=sys.stderr,
        )
        sys.exit(1)
    return url


# ---------------------------------------------------------------------------
# speak
# ---------------------------------------------------------------------------


def cmd_speak(args):
    """Synthesize speech via the TTS server."""
    from narro.client import NarroClient

    url = _require_server(args)
    client = NarroClient(url)

    if args.align:
        from narro.alignment import save_alignment

        audio_bytes, alignment = client.generate_with_alignment(
            [args.text], args.output,
        )
        save_alignment(alignment, args.align)
    else:
        client.infer(args.text, out_path=args.output)

    logger.info("Audio saved to: %s", args.output)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def cmd_serve(args):
    """Start the TTS API server."""
    from narro.server import serve

    serve(
        host=args.host,
        port=args.port,
        model=args.model,
        device=args.device,
        model_path=args.model_path,
        compile=not args.no_compile,
        quantize=args.quantize,
    )


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def main():
    if (
        len(sys.argv) > 1
        and sys.argv[1] not in {"speak", "serve", "-h", "--help"}
    ):
        sys.argv.insert(1, "speak")

    parser = argparse.ArgumentParser(
        description="Narro TTS — model-agnostic text-to-speech server and client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  narro serve                            # start the server
  narro serve --model soprano --device cuda
  narro "Hello world" -o output.wav      # synthesize (client)
""",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- speak ---
    speak_parser = subparsers.add_parser("speak", help="Synthesize speech (default)")
    speak_parser.add_argument("text", help="Text to synthesize")
    speak_parser.add_argument(
        "--output", "-o", default="output.wav", help="Output audio file path"
    )
    speak_parser.add_argument(
        "--align", "-a", help="Output word-alignment JSON file path"
    )
    speak_parser.add_argument(
        "--server", "-s", help="Server URL (or set NARRO_SERVER env var)"
    )
    speak_parser.set_defaults(func=cmd_speak)

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
        default="soprano",
        help="Model backend to load (default: soprano)",
    )
    serve_parser.add_argument(
        "--device",
        "-d",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)",
    )
    serve_parser.add_argument(
        "--model-path", "-m", help="Path to local model directory"
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

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
