"""Slow e2e: the MODEL_OPTIONAL_PATHS default-model seam resolves through
the REAL director-driven gateway routing path (build_gateway(state=...)),
not just the legacy static-routes path already covered by
tests/cli_impl/test_gateway_default_model.py.

A fake translation model is registered as the sole enabled catalog entry
for modality text/translation. POST /translate (bare alias -- LT clients
never send `model`) must:

  1. resolve the default model via the REAL `model_optional_paths()` map
     (built by discover_modalities from the actual text_translation
     modality package, not mocked -- text_translation ships
     MODEL_OPTIONAL_PATHS = ("/v1/translate", "/translate", "/languages")),
  2. drive that model_id through director.acquire (the v0.40.0+ lazy-load
     path, mirroring tests/cli_impl/test_gateway_lazy.py's pattern),
  3. forward to the acquired worker port, and
  4. return the LibreTranslate-shape response body untouched.

Mirrors the fake-backend-registered-with-real-app style of
tests/cli_impl/test_e2e_summarization.py and the director/mocked-forward
style of tests/cli_impl/test_gateway_lazy.py + test_e2e_queueing.py.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from muse.cli_impl.gateway import build_gateway
from muse.cli_impl.supervisor import SupervisorState
from muse.core.catalog import CatalogEntry


pytestmark = pytest.mark.slow


_MODEL_ID = "fake-translator"
_WORKER_PORT = 9321
_MANIFEST = {
    "model_id": _MODEL_ID,
    "modality": "text/translation",
    "capabilities": {"memory_gb": 0.1, "device": "cpu"},
}


def _catalog_entry() -> CatalogEntry:
    return CatalogEntry(
        model_id=_MODEL_ID,
        modality="text/translation",
        backend_path="fake.module:FakeModel",
        hf_repo="fake/repo",
    )


def _make_state(acquire_port: int = _WORKER_PORT) -> SupervisorState:
    state = SupervisorState(workers=[], device="cpu")
    director = MagicMock()
    director.acquire.return_value = acquire_port
    state.director = director
    return state


def _wire_forward(mock_client_cls: MagicMock, body: bytes) -> MagicMock:
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.aclose = AsyncMock()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}

    def _aiter_raw(*, chunk_size=None):
        async def _chunks():
            yield body
        return _chunks()

    mock_response.aiter_raw = _aiter_raw

    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    stream_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(return_value=stream_ctx)

    mock_client_cls.return_value = mock_client
    return mock_client


@pytest.mark.timeout(10)
def test_translate_alias_without_model_resolves_via_seam_e2e():
    state = _make_state()
    app = build_gateway(state=state)
    client = TestClient(app)

    known = {_MODEL_ID: _catalog_entry()}
    with patch("muse.cli_impl.gateway.known_models", return_value=known), \
         patch("muse.cli_impl.gateway.is_enabled", return_value=True), \
         patch("muse.cli_impl.gateway.get_manifest", return_value=_MANIFEST), \
         patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
        _wire_forward(mock_cls, b'{"translatedText": "Hola mundo"}')

        r = client.post(
            "/translate",
            json={"q": "Hello world", "source": "en", "target": "es"},
        )

    assert r.status_code == 200, r.text
    assert r.json() == {"translatedText": "Hola mundo"}

    # The seam actually resolved the default model (no `model` field was
    # sent) and drove it through the REAL director-based routing path.
    state.director.acquire.assert_called_once_with(_MODEL_ID, manifest=_MANIFEST)
    state.director.release.assert_called_once_with(_MODEL_ID)
    forward_url = mock_cls.return_value.stream.call_args.kwargs["url"]
    assert forward_url == f"http://127.0.0.1:{_WORKER_PORT}/translate"


@pytest.mark.timeout(10)
def test_translate_list_q_roundtrips_via_seam():
    """List-in/list-out (LT batch shape) also flows through the seam."""
    state = _make_state()
    app = build_gateway(state=state)
    client = TestClient(app)

    known = {_MODEL_ID: _catalog_entry()}
    with patch("muse.cli_impl.gateway.known_models", return_value=known), \
         patch("muse.cli_impl.gateway.is_enabled", return_value=True), \
         patch("muse.cli_impl.gateway.get_manifest", return_value=_MANIFEST), \
         patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
        _wire_forward(mock_cls, b'{"translatedText": ["Hola", "mundo"]}')

        r = client.post(
            "/translate",
            json={"q": ["Hello", "world"], "source": "en", "target": "es"},
        )

    assert r.status_code == 200, r.text
    assert r.json() == {"translatedText": ["Hola", "mundo"]}


@pytest.mark.timeout(10)
def test_translate_with_explicit_model_still_works_via_seam_path():
    """An explicit `model` field bypasses default-resolution entirely but
    still routes through the same director-based proxy, unaffected by the
    MODEL_OPTIONAL_PATHS wiring."""
    state = _make_state()
    app = build_gateway(state=state)
    client = TestClient(app)

    with patch("muse.cli_impl.gateway.get_manifest", return_value=_MANIFEST), \
         patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
        _wire_forward(mock_cls, b'{"translatedText": "Hola"}')

        r = client.post(
            "/v1/translate",
            json={"q": "Hello", "source": "en", "target": "es", "model": _MODEL_ID},
        )

    assert r.status_code == 200, r.text
    state.director.acquire.assert_called_once_with(_MODEL_ID, manifest=_MANIFEST)


@pytest.mark.timeout(10)
def test_translate_without_model_and_no_enabled_translation_model_503s():
    """No enabled text/translation model anywhere -> 503 no_default_model,
    and the director is never touched (fails before acquire)."""
    state = _make_state()
    app = build_gateway(state=state)
    client = TestClient(app)

    with patch("muse.cli_impl.gateway.known_models", return_value={}), \
         patch("muse.cli_impl.gateway.is_enabled", return_value=False):
        r = client.post(
            "/translate", json={"q": "hello", "source": "en", "target": "es"},
        )

    assert r.status_code == 503
    assert r.json()["error"]["code"] == "no_default_model"
    state.director.acquire.assert_not_called()
