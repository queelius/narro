"""`muse speak` / `muse audio speech create` implementation."""
from __future__ import annotations

import os
import sys


def run_speak(args) -> int:
    from muse.audio.speech.client import SpeechClient

    server = args.server or os.environ.get("MUSE_SERVER", "http://localhost:8000")
    client = SpeechClient(server_url=server, model=args.model)
    try:
        wav_bytes = client.infer(args.text, model=args.model)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    with open(args.output, "wb") as f:
        f.write(wav_bytes)
    print(f"wrote {args.output} ({len(wav_bytes)} bytes)")
    return 0
