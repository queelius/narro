#!/usr/bin/env python
"""Validate a cleanly installed Muse distribution without importing Muse."""
from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


REQUIRED_RUNTIME_FILES = (
    "muse/curated.yaml",
    "muse/chat_formats.yaml",
)


def project_version(pyproject: Path) -> str:
    """Read the expected package version from ``pyproject.toml``."""
    with pyproject.open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


def validate_distribution(
    *,
    expected_version: str,
    distribution: Any | None = None,
) -> list[str]:
    """Return human-readable errors for an installed ``museq`` artifact."""
    try:
        dist = distribution or importlib.metadata.distribution("museq")
    except importlib.metadata.PackageNotFoundError:
        return ["museq distribution is not installed"]

    errors: list[str] = []
    if dist.version != expected_version:
        errors.append(
            f"installed version {dist.version!r} != expected {expected_version!r}"
        )

    for relative in REQUIRED_RUNTIME_FILES:
        target = Path(dist.locate_file(relative))
        if not target.is_file():
            errors.append(f"missing packaged runtime file: {relative}")

    console_scripts = {
        entry.name: entry.value
        for entry in dist.entry_points
        if entry.group == "console_scripts"
    }
    if console_scripts.get("muse") != "muse.cli:main":
        errors.append(
            "missing or incorrect console script: expected "
            "muse = muse.cli:main"
        )

    requires_python = dist.metadata.get("Requires-Python")
    if requires_python != ">=3.10":
        errors.append(
            f"unexpected Requires-Python metadata: {requires_python!r}"
        )

    requirements = dist.requires or []
    normalized = [requirement.replace(" ", "").lower() for requirement in requirements]
    if not any(
        requirement.startswith("mcp")
        and ">=1.8.0" in requirement
        and "<2" in requirement
        for requirement in normalized
    ):
        errors.append("artifact metadata lacks the required mcp>=1.8.0,<2 bound")
    if not any(
        requirement.startswith("sse-starlette")
        and ">=3.0.0" in requirement
        and "<3.1.0" in requirement
        for requirement in normalized
    ):
        errors.append(
            "artifact metadata lacks the required "
            "sse-starlette>=3.0.0,<3.1.0 bound"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="validate an installed Muse wheel or sdist-built wheel",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="project metadata used to determine the expected version",
    )
    args = parser.parse_args(argv)

    errors = validate_distribution(
        expected_version=project_version(args.pyproject),
    )
    if errors:
        for error in errors:
            print(f"artifact validation failed: {error}", file=sys.stderr)
        return 1
    print("artifact validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
