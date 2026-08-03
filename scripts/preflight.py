#!/usr/bin/env python
"""Preflight guard: verify the dev venv can run the fast test lane, then run it.

The muse fast lane (`pytest -m "not slow"`) imports the heavy ML stack
(torch, transformers, diffusers, sentence-transformers) plus server deps in
many test modules; several import them at collection time (not behind a
mock), so a venv missing those deps does not merely skip - it errors at
collection or silently runs a partial suite. This script asserts the
required deps import BEFORE running pytest, so a release cannot be
"verified" in a drifted venv.

Usage:
    python scripts/preflight.py                  # check deps, then run the lane
    python scripts/preflight.py --check-only     # check deps only, no tests
    python scripts/preflight.py -- -k resolver   # forward args after -- to pytest
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import subprocess
import sys

try:
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion
except ImportError:  # handled as an actionable missing dev dependency below
    Requirement = None  # type: ignore[assignment,misc]
    InvalidVersion = ValueError  # type: ignore[assignment,misc]

# (import_name, extra, requirement). Import name != distribution name for
# several packages, hence the explicit mapping. Keep this in sync with
# INSTALL_CMD and pyproject optional-dependencies. Versioned requirements are
# checked against installed distribution metadata as well as imported: an old
# SDK can import successfully while lacking an API Muse calls.
REQUIRED: list[tuple[str, str, str]] = [
    ("huggingface_hub", "core", "huggingface_hub"),
    ("numpy", "core", "numpy"),
    ("yaml", "core", "pyyaml"),
    ("requests", "core", "requests"),
    ("typer", "core", "typer>=0.12.0"),
    ("rich", "core", "rich>=13.0.0"),
    ("torch", "audio", "torch>=2.1.0"),
    ("transformers", "audio", "transformers>=4.36.0"),
    ("scipy", "audio", "scipy"),
    ("inflect", "audio", "inflect"),
    ("unidecode", "audio", "unidecode"),
    ("diffusers", "images", "diffusers>=0.27.0"),
    ("accelerate", "images", "accelerate"),
    ("PIL", "images", "Pillow"),
    ("safetensors", "images", "safetensors"),
    ("imageio", "images", "imageio[ffmpeg]>=2.31.0"),
    ("imageio_ffmpeg", "images", "imageio-ffmpeg"),
    (
        "sentence_transformers",
        "embeddings",
        "sentence-transformers>=2.2.0",
    ),
    ("fastapi", "server", "fastapi"),
    ("uvicorn", "server", "uvicorn"),
    ("sse_starlette", "server", "sse-starlette>=3.0.0,<3.1.0"),
    ("httpx", "server", "httpx"),
    ("psutil", "server", "psutil>=5.9"),
    ("multipart", "server", "python-multipart"),
    ("pynvml", "server", "nvidia-ml-py>=12.0"),
    (
        "mcp.server.streamable_http_manager",
        "server",
        "mcp>=1.8.0,<2",
    ),
    ("pytest", "dev", "pytest"),
    ("pytest_cov", "dev", "pytest-cov"),
    ("pytest_asyncio", "dev", "pytest-asyncio"),
    ("pytest_timeout", "dev", "pytest-timeout"),
    ("packaging", "dev", "packaging>=23.0"),
]
if sys.version_info < (3, 11):
    REQUIRED.append(("tomli", "dev", "tomli"))

INSTALL_CMD = (
    'pip install -e ".[dev,server,audio,images,embeddings]" '
    "--extra-index-url https://download.pytorch.org/whl/cpu"
)


def _requirement_satisfied(requirement: str) -> bool:
    """Return whether installed metadata satisfies ``requirement``."""
    if Requirement is None:
        return False
    parsed = Requirement(requirement)
    if not parsed.specifier:
        return True
    try:
        installed = importlib.metadata.version(parsed.name)
        return parsed.specifier.contains(installed, prereleases=True)
    except (importlib.metadata.PackageNotFoundError, InvalidVersion):
        return False


def missing_deps() -> list[tuple[str, str, str]]:
    """Return dependencies that fail to import or violate their version."""
    if Requirement is None:
        return [("packaging", "dev", "packaging>=23.0")]
    missing: list[tuple[str, str, str]] = []
    for import_name, extra, requirement in REQUIRED:
        try:
            importlib.import_module(import_name)
        except Exception:  # noqa: BLE001 - any import failure means "missing"
            missing.append((import_name, extra, requirement))
            continue
        if not _requirement_satisfied(requirement):
            missing.append((import_name, extra, requirement))
    return missing


def report_missing(missing: list[tuple[str, str, str]]) -> None:
    """Print an actionable error naming each missing dep and the fix."""
    print("preflight: venv is not fast-lane ready; missing or incompatible:",
          file=sys.stderr)
    for import_name, extra, requirement in missing:
        print(f"  - {import_name}  (extra: {extra}, requires: {requirement})",
              file=sys.stderr)
    print(f"\nInstall the full dev stack:\n  {INSTALL_CMD}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="muse fast-lane preflight guard")
    parser.add_argument("--check-only", action="store_true",
                        help="verify deps only; do not run pytest")
    parser.add_argument("pytest_args", nargs="*",
                        help="args forwarded to pytest (use -- to separate)")
    args = parser.parse_args(argv)

    missing = missing_deps()
    if missing:
        report_missing(missing)
        return 1
    print(f"preflight: all {len(REQUIRED)} fast-lane deps present.")
    if args.check_only:
        return 0

    cmd = [sys.executable, "-m", "pytest", "-m", "not slow", *args.pytest_args]
    print(f"preflight: running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
