"""Regression checks on pyproject.toml's optional-dependencies.

These assert that deps known-required-but-easily-missed are actually
declared. Each entry has a comment naming the failure mode the test
guards against.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import pytest


PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _server_extras(pyproject: dict) -> list[str]:
    return pyproject["project"]["optional-dependencies"]["server"]


def test_server_extras_include_python_multipart(pyproject):
    """Every per-model venv installs museq[server]; multipart is required
    by FastAPI's Form/UploadFile decorators. The audio_transcription
    router uses both at import time, so a worker missing multipart
    crashes before the worker can serve anything.

    Regression: pre-v0.13.1 omitted this and every worker that mounted
    the new modality router (i.e., every worker) crashed at startup."""
    extras = _server_extras(pyproject)
    extras_str = " ".join(extras)
    assert "python-multipart" in extras_str, (
        f"python-multipart must be in server extras (FastAPI Form/UploadFile "
        f"requirement); got {extras}"
    )


def test_server_extras_include_fastapi_and_uvicorn(pyproject):
    """Sanity check on the load-bearing server runtime deps."""
    extras = _server_extras(pyproject)
    extras_str = " ".join(extras)
    assert "fastapi" in extras_str
    assert "uvicorn" in extras_str


def test_server_extras_require_multi_loop_safe_sse_starlette(pyproject):
    """sse-starlette 2.x keeps shutdown state in a global asyncio.Event.

    Reusing the app across event loops then raises "bound to a different
    event loop". v3.0 replaces that global with context-local state. Keep the
    upper bound below 3.1 because later 3.x releases require a newer Starlette
    compatibility window than Muse's currently supported FastAPI stack.
    """
    extras = _server_extras(pyproject)
    assert "sse-starlette>=3.0.0,<3.1.0" in extras


def test_kokoro_extra_uses_reviewed_adapter_version(pyproject):
    extras = pyproject["project"]["optional-dependencies"]["audio-kokoro"]
    assert "kokoro==0.9.4" in extras
    assert all(not requirement.startswith("misaki") for requirement in extras)
