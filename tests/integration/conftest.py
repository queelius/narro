"""Shared fixtures for opt-in integration tests against a real muse server.

How to run:
    MUSE_REMOTE_SERVER=http://192.168.0.225:8000 pytest tests/integration/

Default (no env var): all integration tests are skipped. They never run
in CI unless CI explicitly sets MUSE_REMOTE_SERVER.

The fixtures probe the server before each test and skip if:
  - MUSE_REMOTE_SERVER is unset
  - the server doesn't respond on /health
  - the server doesn't host the model the test requires

This keeps the tests purely additive: they augment unit tests without
ever blocking the fast suite.
"""
from __future__ import annotations

import os
from typing import Iterable

import pytest


def _server_url() -> str | None:
    """Return the muse server URL or None if integration tests are off."""
    return os.environ.get("MUSE_REMOTE_SERVER")


@pytest.fixture(scope="session")
def remote_url() -> str:
    """Skip the test if MUSE_REMOTE_SERVER is unset."""
    url = _server_url()
    if not url:
        pytest.skip("MUSE_REMOTE_SERVER not set; integration tests are opt-in")
    return url.rstrip("/")


@pytest.fixture(scope="session")
def remote_health(remote_url) -> dict:
    """Probe /health; skip if the server isn't reachable or isn't muse-shaped."""
    import httpx
    try:
        r = httpx.get(f"{remote_url}/health", timeout=5.0)
    except httpx.HTTPError as e:
        pytest.skip(f"muse server at {remote_url} not reachable: {e}")
    if r.status_code != 200:
        pytest.skip(f"muse server returned status {r.status_code} for /health")
    body = r.json()
    if "modalities" not in body or "models" not in body:
        pytest.skip(
            f"server at {remote_url} returned non-muse /health body: {body}"
        )
    return body


@pytest.fixture(scope="session")
def openai_client(remote_url):
    """OpenAI SDK client pointed at the muse server."""
    try:
        from openai import OpenAI
    except ImportError:
        pytest.skip("openai package not installed; pip install openai")
    return OpenAI(base_url=f"{remote_url}/v1", api_key="not-used")


def _require_model(remote_health: dict, model_id: str) -> None:
    """Skip if the loaded muse server doesn't have a specific model."""
    loaded = remote_health.get("models") or []
    if model_id not in loaded:
        pytest.skip(
            f"muse server doesn't have {model_id!r} loaded "
            f"(loaded: {loaded}); pull and restart to enable this test"
        )


def require_model_fixture(model_id: str):
    """Build a session-scoped fixture that skips if model_id isn't loaded."""
    @pytest.fixture(scope="session")
    def _fixture(remote_health):
        _require_model(remote_health, model_id)
        return model_id
    return _fixture


# Pre-built model gates for the ids we test against. Add new ones as needed.
qwen3_5_4b = require_model_fixture("qwen3.5-4b-q4")
qwen3_embedding = require_model_fixture("qwen3-embedding-0.6b")
kokoro_82m = require_model_fixture("kokoro-82m")
