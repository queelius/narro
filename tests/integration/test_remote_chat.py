"""End-to-end OpenAI-SDK tests against a running muse server.

Opt-in: set MUSE_REMOTE_SERVER to enable. See tests/integration/conftest.py.

Requires:
  - a muse server reachable on $MUSE_REMOTE_SERVER
  - the muse server must have qwen3.5-4b-q4 (or whichever model the
    test names) pulled and loaded

These tests exercise:
  - non-streaming chat completion via OpenAI SDK
  - streaming chat completion via OpenAI SDK
  - response_format=json_object structured output
  - the `model` field in responses (the v0.11.4 fix: must be the muse
    catalog id, not the GGUF filesystem path)

These tests are heavy (real GPU inference) and slow (5-30s each). The
pytest.mark.slow marker makes them skipped by default in the fast lane.
"""
from __future__ import annotations

import json

import pytest


pytestmark = pytest.mark.slow


def test_models_list_via_sdk(openai_client, remote_health):
    """Lists at least one model with id matching what /health reports."""
    models = openai_client.models.list()
    ids = {m.id for m in models.data}
    assert ids == set(remote_health["models"]), (
        f"SDK saw {ids}, /health reported {remote_health['models']}"
    )


def test_chat_non_streaming(openai_client, chat_model):
    """Basic chat works; finish_reason is sane; usage is populated."""
    r = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Reply with exactly the word 'pong'."}],
        max_tokens=20,
        temperature=0.0,
    )
    assert r.choices[0].finish_reason in ("stop", "length")
    assert r.choices[0].message.content
    assert r.usage.total_tokens > 0
    assert r.usage.prompt_tokens > 0
    # The catalog id wins over the GGUF filesystem path (v0.11.4 fix)
    assert r.model == chat_model
    assert "/" not in r.model.replace("/", "", 1)  # no extra slashes (no path)
    assert ".gguf" not in r.model


def test_chat_streaming(openai_client, chat_model):
    """Streaming yields multiple chunks; final chunk has finish_reason set;
    every chunk has model_id == catalog id (v0.11.4 fix per-chunk)."""
    chunks = list(openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Count 1 to 3."}],
        max_tokens=30,
        temperature=0.0,
        stream=True,
    ))
    assert len(chunks) >= 2
    # Per-chunk model field must equal the catalog id
    for c in chunks:
        assert c.model == chat_model, f"chunk model field is {c.model!r}, expected {chat_model!r}"
    # Some chunk somewhere along the way must terminate with a finish_reason
    finishers = [c for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finishers, "no chunk had finish_reason set"


def test_response_format_json_object(openai_client, chat_model):
    """response_format json_object produces parseable JSON content."""
    r = openai_client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "Respond in JSON only."},
            {"role": "user", "content": "Return {\"answer\": 42} verbatim."},
        ],
        response_format={"type": "json_object"},
        max_tokens=100,
        temperature=0.0,
    )
    content = r.choices[0].message.content
    assert content
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


def test_temperature_zero_is_deterministic(openai_client, chat_model):
    """Two identical calls at temperature=0 should produce identical content.

    Catches regressions where seed/temperature plumbing breaks. Llama-cpp's
    temperature=0 is greedy decoding, so output should match exactly.
    """
    def _call():
        return openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the digit."}],
            max_tokens=20,
            temperature=0.0,
            seed=12345,
        ).choices[0].message.content

    out1 = _call()
    out2 = _call()
    assert out1 == out2, f"Greedy decoding non-deterministic:\n  call 1: {out1!r}\n  call 2: {out2!r}"
