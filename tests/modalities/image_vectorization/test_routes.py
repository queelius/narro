import io
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_vectorization import (
    MODALITY,
    VectorizationOutputError,
    VectorizationResult,
    build_router,
)


def _png_bytes(width=64, height=32):
    image = Image.new("RGB", (width, height), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeVectorizer:
    def __init__(self, model_id="vectorizer"):
        self.model_id = model_id
        self.last_image = None
        self.last_kwargs = None

    def vectorize(self, image, **kwargs):
        self.last_image = image
        self.last_kwargs = kwargs
        return VectorizationResult(
            svg=(
                '<svg xmlns="http://www.w3.org/2000/svg" '
                'viewBox="0 0 64 32"><rect width="64" height="32"/></svg>'
            ),
            model_id="upstream-name",
            source_width=image.width,
            source_height=image.height,
            completion_tokens=20,
            seed=kwargs.get("seed") if kwargs.get("seed") is not None else 99,
            width=64,
            height=32,
            view_box=(0, 0, 64, 32),
        )


def _client(backend):
    registry = ModalityRegistry()
    registry.register(
        MODALITY, backend,
        manifest={
            "model_id": backend.model_id,
            "modality": MODALITY,
            "capabilities": {"supports_image_to_svg": True},
        },
    )
    app = create_app(
        registry=registry,
        routers={MODALITY: build_router(registry)},
    )
    return TestClient(app)


def test_json_response_and_generation_controls():
    backend = _FakeVectorizer("starvector")
    response = _client(backend).post(
        "/v1/images/vectorize",
        files={"image": ("diagram.png", _png_bytes(), "image/png")},
        data={
            "model": "starvector",
            "max_new_tokens": "512",
            "temperature": "0.5",
            "top_p": "0.8",
            "num_beams": "3",
            "seed": "7",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "image.vectorization"
    assert body["model"] == "starvector"
    assert body["mime_type"] == "image/svg+xml"
    assert body["source_size"] == [64, 32]
    assert body["view_box"] == [0, 0, 64, 32]
    assert body["usage"]["completion_tokens"] == 20
    assert backend.last_kwargs == {
        "max_new_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.8,
        "num_beams": 3,
        "seed": 7,
    }


def test_raw_svg_response():
    response = _client(_FakeVectorizer("starvector")).post(
        "/v1/images/vectorize",
        files={"image": ("diagram.png", _png_bytes(), "image/png")},
        data={"response_format": "svg", "seed": "42"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.headers["x-muse-model"] == "starvector"
    assert response.headers["x-muse-seed"] == "42"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in (
        response.headers["content-security-policy"]
    )
    assert response.text.startswith("<svg")


def test_default_model_is_optional():
    response = _client(_FakeVectorizer("default-vectorizer")).post(
        "/v1/images/vectorize",
        files={"image": ("diagram.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "default-vectorizer"


def test_unknown_model_returns_404_before_decode(monkeypatch):
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "2")
    response = _client(_FakeVectorizer("real")).post(
        "/v1/images/vectorize",
        files={"image": ("bad.png", b"too-large-and-invalid", "image/png")},
        data={"model": "ghost"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_corrupt_image_returns_400():
    response = _client(_FakeVectorizer()).post(
        "/v1/images/vectorize",
        files={"image": ("bad.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_parameter"


def test_input_side_cap(monkeypatch):
    monkeypatch.setenv("MUSE_VECTORIZATION_MAX_INPUT_SIDE", "32")
    response = _client(_FakeVectorizer()).post(
        "/v1/images/vectorize",
        files={"image": ("large.png", _png_bytes(64, 32), "image/png")},
    )
    assert response.status_code == 400
    assert "MUSE_VECTORIZATION_MAX_INPUT_SIDE" in (
        response.json()["error"]["message"]
    )


def test_nonpositive_input_side_cap_falls_back(monkeypatch):
    from muse.core import config
    from muse.modalities.image_vectorization.routes import _max_input_side

    monkeypatch.setenv("MUSE_VECTORIZATION_MAX_INPUT_SIDE", "0")
    assert _max_input_side() == config.SETTINGS_BY_KEY[
        "limits.vectorization_max_input_side"
    ].default


def test_parameter_validation():
    client = _client(_FakeVectorizer())
    cases = [
        ({"max_new_tokens": "0"}, "max_new_tokens"),
        ({"max_new_tokens": "7681"}, "max_new_tokens"),
        ({"temperature": "2.1"}, "temperature"),
        ({"top_p": "0"}, "top_p"),
        ({"num_beams": "9"}, "num_beams"),
        ({"seed": "-1"}, "seed"),
        ({"seed": str(2**63)}, "seed"),
        ({"response_format": "base64"}, "response_format"),
    ]
    for data, field in cases:
        response = client.post(
            "/v1/images/vectorize",
            files={"image": ("diagram.png", _png_bytes(), "image/png")},
            data=data,
        )
        assert response.status_code == 400
        assert field in response.json()["error"]["message"]


def test_invalid_model_svg_returns_sanitized_502():
    backend = _FakeVectorizer()
    backend.vectorize = MagicMock(
        side_effect=VectorizationOutputError("<script>secret</script>")
    )
    response = _client(backend).post(
        "/v1/images/vectorize",
        files={"image": ("diagram.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"
    assert "secret" not in response.text


def test_backend_cannot_bypass_svg_validation():
    backend = _FakeVectorizer()

    def unsafe(image, **kwargs):
        return VectorizationResult(
            svg="<svg><script>secret()</script></svg>",
            model_id=backend.model_id,
            source_width=image.width,
            source_height=image.height,
        )

    backend.vectorize = unsafe
    response = _client(backend).post(
        "/v1/images/vectorize",
        files={"image": ("diagram.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"
    assert "secret" not in response.text


def test_runtime_error_returns_sanitized_500():
    backend = _FakeVectorizer()
    backend.vectorize = MagicMock(
        side_effect=RuntimeError("/secret/path CUDA crash")
    )
    response = _client(backend).post(
        "/v1/images/vectorize",
        files={"image": ("diagram.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "/secret/path" not in response.text


def test_route_closes_decoded_source_after_backend():
    backend = _FakeVectorizer()
    decoded = MagicMock()
    decoded.size = (64, 32)
    decoded.width = 64
    decoded.height = 32

    with patch(
        "muse.modalities.image_vectorization.routes.decode_image_file",
        new=AsyncMock(return_value=decoded),
    ):
        response = _client(backend).post(
            "/v1/images/vectorize",
            files={"image": ("diagram.png", b"placeholder", "image/png")},
        )

    assert response.status_code == 200, response.text
    decoded.close.assert_called_once_with()


def test_registry_attaches_inference_lock():
    backend = _FakeVectorizer()
    client = _client(backend)
    for _ in range(2):
        response = client.post(
            "/v1/images/vectorize",
            files={"image": ("diagram.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 200
    assert hasattr(backend, "_inference_lock")
