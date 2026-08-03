"""Route tests for POST /v1/images/embeddings.

Uses fully-mocked backends + real data URLs (constructed via PIL) so
the decode helper exercises the same code path as production.
"""
import base64
import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_embedding import (
    MODALITY,
    ImageEmbeddingResult,
    build_router,
)
from muse.modalities.image_embedding.codec import base64_to_embedding


pytest.importorskip("PIL.Image")


def _png_data_url(width=32, height=32, color=(0, 128, 255)):
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    return f"data:image/png;base64,{base64.b64encode(raw).decode()}"


def _make_client(backend, manifest=None):
    reg = ModalityRegistry()
    if manifest is None:
        manifest = {"model_id": "dinov2-small"}
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def _fake_backend(result):
    backend = MagicMock()
    backend.model_id = "dinov2-small"
    backend.embed.return_value = result
    return backend


def _fake_result(*, n_images=1, dim=384, model_id="dinov2-small"):
    return ImageEmbeddingResult(
        embeddings=[[0.1] * dim for _ in range(n_images)],
        dimensions=dim,
        model_id=model_id,
        n_images=n_images,
    )


def test_embeddings_returns_envelope_for_single_input():
    backend = _fake_backend(_fake_result(n_images=1, dim=4))
    client = _make_client(backend)

    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "dinov2-small"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == 4


def test_embeddings_returns_envelope_for_batched_input():
    backend = _fake_backend(_fake_result(n_images=3, dim=4))
    client = _make_client(backend)

    r = client.post("/v1/images/embeddings", json={
        "input": [_png_data_url(), _png_data_url(), _png_data_url()],
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 3
    indices = [entry["index"] for entry in body["data"]]
    assert indices == [0, 1, 2]


def test_embeddings_default_encoding_is_float():
    backend = _fake_backend(_fake_result(dim=4))
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
    })
    body = r.json()
    # When encoding is float, the embedding field is a list of numbers.
    assert isinstance(body["data"][0]["embedding"], list)
    assert all(isinstance(x, (int, float)) for x in body["data"][0]["embedding"])


def test_embeddings_base64_encoding_returns_string():
    backend = _fake_backend(_fake_result(dim=4))
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
        "encoding_format": "base64",
    })
    body = r.json()
    enc = body["data"][0]["embedding"]
    assert isinstance(enc, str)
    # Round-trip check
    decoded = base64_to_embedding(enc)
    assert len(decoded) == 4


def test_embeddings_400_on_empty_input_string():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={"input": ""})
    assert r.status_code == 422  # pydantic validation
    body = r.json()
    assert "error" in body


def test_embeddings_400_on_empty_input_list():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={"input": []})
    assert r.status_code == 422
    body = r.json()
    assert "error" in body


def test_embeddings_400_on_bad_data_url():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": "data:image/png;base64,!!!not-real-base64!!!",
    })
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
    assert "image decode failed" in body["error"]["message"].lower()


def test_embeddings_400_on_unsupported_input_format():
    """Strings not starting with data: or http(s):// fail decode."""
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": "just-a-plain-string",
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_batch_decode_closes_prior_images_on_unexpected_error(monkeypatch):
    from muse.modalities.image_embedding import routes

    decoded = MagicMock()
    decoded.size = (32, 32)
    calls = 0

    async def fail_second(_value):
        nonlocal calls
        calls += 1
        if calls == 1:
            return decoded
        raise RuntimeError("decode interrupted")

    monkeypatch.setattr(routes, "decode_image_input", fail_second)
    backend = _fake_backend(_fake_result(n_images=2))
    client = _make_client(backend)

    with pytest.raises(RuntimeError, match="decode interrupted"):
        client.post("/v1/images/embeddings", json={
            "input": ["data:first", "data:second"],
        })

    decoded.close.assert_called_once_with()


def test_embeddings_404_on_unknown_model():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
        "model": "nonexistent-embedder",
    })
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"


def test_embeddings_default_model_resolves_first_registered():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
    })
    assert r.status_code == 200
    assert r.json()["model"] == "dinov2-small"


def test_embeddings_passes_dimensions_to_backend():
    backend = _fake_backend(_fake_result(dim=128))
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
        "dimensions": 128,
    })
    assert r.status_code == 200
    _, kwargs = backend.embed.call_args
    assert kwargs.get("dimensions") == 128


def test_embeddings_no_dimensions_passes_none():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
    })
    _, kwargs = backend.embed.call_args
    assert kwargs.get("dimensions") is None


def test_embeddings_400_on_dimensions_too_large():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
        "dimensions": 100000,  # exceeds le=4096
    })
    assert r.status_code == 422


def test_embeddings_400_on_dimensions_zero():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
        "dimensions": 0,  # below ge=1
    })
    assert r.status_code == 422


def test_embeddings_user_field_accepted_and_ignored():
    """OpenAI compat: user field accepted, ignored."""
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
        "user": "alex@example.com",
    })
    assert r.status_code == 200


def test_embeddings_invalid_encoding_format_rejected():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
        "encoding_format": "WRONG",
    })
    assert r.status_code == 422


def test_embeddings_400_on_oversized_batch(monkeypatch):
    """Beyond MUSE_IMAGE_EMBEDDINGS_MAX_BATCH should 400 cleanly.

    Reloads the routes module so the new env var is picked up; restores
    it on teardown so subsequent tests in this file see the unpatched
    cap (otherwise the lowered cap leaks across tests).
    """
    monkeypatch.setenv("MUSE_IMAGE_EMBEDDINGS_MAX_BATCH", "2")
    import importlib
    import muse.modalities.image_embedding.routes as rmod
    importlib.reload(rmod)

    reg = ModalityRegistry()
    backend = _fake_backend(_fake_result())
    reg.register(MODALITY, backend, manifest={"model_id": "dinov2-small"})
    app = create_app(registry=reg, routers={MODALITY: rmod.build_router(reg)})
    client = TestClient(app)

    try:
        r = client.post("/v1/images/embeddings", json={
            "input": [_png_data_url(), _png_data_url(), _png_data_url()],
        })
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_parameter"
        assert "exceeds" in r.json()["error"]["message"].lower()
    finally:
        # Restore the production cap so later tests don't see the lower one.
        monkeypatch.delenv("MUSE_IMAGE_EMBEDDINGS_MAX_BATCH", raising=False)
        importlib.reload(rmod)


def test_embeddings_returns_model_id_from_backend_result():
    """Response 'model' field is the result.model_id, not the request body."""
    backend = _fake_backend(_fake_result(model_id="custom-id"))
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
    })
    body = r.json()
    assert body["model"] == "custom-id"


def test_embeddings_usage_zero_for_image_inputs():
    """Image embedding has no text tokenization; usage stays 0."""
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
    })
    body = r.json()
    assert body["usage"]["prompt_tokens"] == 0
    assert body["usage"]["total_tokens"] == 0


def test_embeddings_404_uses_error_envelope_not_detail():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": _png_data_url(),
        "model": "nonexistent",
    })
    body = r.json()
    assert "detail" not in body
    assert "error" in body


def test_embeddings_400_uses_error_envelope_not_detail():
    """muse error envelope is {"error": {...}}, not FastAPI's {"detail": ...}."""
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": "data:image/png;base64,!!!bogus!!!",
    })
    body = r.json()
    assert "detail" not in body
    assert "error" in body
    assert "code" in body["error"]
    assert "message" in body["error"]
    assert "type" in body["error"]


def test_embeddings_passes_decoded_pil_images_to_backend():
    """Backend receives a list of PIL.Image objects, not raw URL strings."""
    backend = _fake_backend(_fake_result(n_images=2))
    client = _make_client(backend)
    client.post("/v1/images/embeddings", json={
        "input": [_png_data_url(), _png_data_url()],
    })
    args, _ = backend.embed.call_args
    images = args[0]
    assert isinstance(images, list)
    assert len(images) == 2
    # Each image is a PIL.Image with a size attribute
    for img in images:
        assert hasattr(img, "size")
        assert img.size == (32, 32)


def test_embeddings_rejects_aggregate_decoded_pixel_budget(monkeypatch):
    from muse.core import config as cfg

    backend = _fake_backend(_fake_result(n_images=2))
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_TOTAL_PIXELS", "100")
    cfg.reset_config()
    try:
        client = _make_client(backend)
        response = client.post("/v1/images/embeddings", json={
            "input": [_png_data_url(8, 8), _png_data_url(8, 8)],
        })
        assert response.status_code == 400
        assert "aggregate limit 100" in response.json()["error"]["message"]
        backend.embed.assert_not_called()
    finally:
        cfg.reset_config()


def test_embeddings_closes_decoded_inputs_after_backend_call():
    from unittest.mock import AsyncMock, patch

    backend = _fake_backend(_fake_result())
    decoded = MagicMock()
    decoded.size = (8, 8)
    with patch(
        "muse.modalities.image_embedding.routes.decode_image_input",
        new=AsyncMock(return_value=decoded),
    ):
        response = _make_client(backend).post(
            "/v1/images/embeddings", json={"input": _png_data_url()},
        )
    assert response.status_code == 200
    decoded.close.assert_called_once_with()


def test_embeddings_data_object_marker():
    """Each entry in `data` carries object='embedding'."""
    backend = _fake_backend(_fake_result(n_images=2))
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": [_png_data_url(), _png_data_url()],
    })
    body = r.json()
    for entry in body["data"]:
        assert entry["object"] == "embedding"


def test_embeddings_indices_are_zero_based_and_ordered():
    backend = _fake_backend(_fake_result(n_images=3))
    client = _make_client(backend)
    r = client.post("/v1/images/embeddings", json={
        "input": [_png_data_url(), _png_data_url(), _png_data_url()],
    })
    body = r.json()
    for i, entry in enumerate(body["data"]):
        assert entry["index"] == i


def test_backend_error_returns_openai_envelope():
    """Finding 2 (v0.58.1 review): routes.py ~117 calls backend.embed()
    with no local try/except, so a backend exception used to escape as
    FastAPI's bare `{"detail": "Internal Server Error"}`. create_app's
    global exception handler must catch it and return the documented
    OpenAI envelope instead."""
    backend = MagicMock()
    backend.model_id = "dinov2-small"
    backend.embed.side_effect = RuntimeError("secret /internal/weights/path")
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "dinov2-small"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/images/embeddings", json={"input": [_png_data_url()]})
    assert r.status_code == 500
    body = r.json()
    assert "detail" not in body
    assert body["error"]["code"] == "internal_error"
    assert "secret /internal/weights/path" not in r.text
