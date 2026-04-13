"""`muse imagine` / `muse images generations create` implementation."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def run_imagine(args) -> int:
    from muse.images.generations.client import GenerationsClient

    server = args.server or os.environ.get("MUSE_SERVER", "http://localhost:8000")
    client = GenerationsClient(base_url=server)
    try:
        images = client.generate(
            args.prompt,
            model=args.model,
            n=args.n,
            size=args.size,
            negative_prompt=args.negative,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    out = Path(args.output)
    if args.n == 1:
        out.write_bytes(images[0])
        print(f"wrote {out}")
    else:
        stem = out.with_suffix("")
        suf = out.suffix
        for i, img in enumerate(images):
            p = Path(f"{stem}_{i}{suf}")
            p.write_bytes(img)
            print(f"wrote {p}")
    return 0
