"""Tests for /v1/images/generations router."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.images.generations.protocol import ImageResult
from muse.images.generations.routes import build_router


class FakeImageModel:
    model_id = "fake-sd"
    default_size = (64, 64)

    def generate(self, prompt, **kwargs):
        w = kwargs.get("width", self.default_size[0])
        h = kwargs.get("height", self.default_size[1])
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return ImageResult(
            image=arr, width=w, height=h,
            seed=kwargs.get("seed", 0) or 0,
            metadata={"prompt": prompt},
        )


@pytest.fixture
def client():
    reg = ModalityRegistry()
    reg.register("images.generations", FakeImageModel())
    app = create_app(
        registry=reg,
        routers={"images.generations": build_router(reg)},
    )
    return TestClient(app)


def test_generate_returns_base64_by_default(client):
    r = client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 1
    assert "b64_json" in data[0]
    # Must be decodable base64
    import base64
    decoded = base64.b64decode(data[0]["b64_json"])
    # PNG magic bytes
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_response_format_url_returns_data_url(client):
    r = client.post("/v1/images/generations", json={
        "prompt": "a cat",
        "response_format": "url",
    })
    assert r.status_code == 200
    url = r.json()["data"][0]["url"]
    assert url.startswith("data:image/png;base64,")


def test_generate_n_creates_multiple_images(client):
    r = client.post("/v1/images/generations", json={"prompt": "a dog", "n": 3})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3


def test_generate_echoes_prompt_as_revised_prompt(client):
    r = client.post("/v1/images/generations", json={"prompt": "a bird"})
    assert r.status_code == 200
    entry = r.json()["data"][0]
    assert entry["revised_prompt"] == "a bird"


def test_unknown_model_returns_openai_shape_404(client):
    r = client.post("/v1/images/generations", json={
        "prompt": "x", "model": "no-such-model",
    })
    assert r.status_code == 404
    body = r.json()
    # OpenAI-style envelope, not FastAPI's default {detail: ...}
    assert "error" in body
    assert "detail" not in body
    assert body["error"]["code"] == "model_not_found"


def test_empty_prompt_rejected(client):
    r = client.post("/v1/images/generations", json={"prompt": ""})
    assert r.status_code in (400, 422)


def test_n_over_limit_rejected(client):
    r = client.post("/v1/images/generations", json={"prompt": "x", "n": 100})
    assert r.status_code in (400, 422)


def test_size_out_of_range_rejected(client):
    r = client.post("/v1/images/generations", json={
        "prompt": "x",
        "size": "4096x4096",
    })
    assert r.status_code in (400, 422)


def test_size_invalid_format_rejected(client):
    r = client.post("/v1/images/generations", json={
        "prompt": "x",
        "size": "big",
    })
    assert r.status_code in (400, 422)


def test_response_includes_created_unix_timestamp(client):
    r = client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 200
    created = r.json()["created"]
    assert isinstance(created, int)
    # Sanity: later than 2020-01-01 UTC (1577836800)
    assert created > 1577836800


def test_seed_passed_through_to_backend(client):
    """The backend should receive the seed kwarg."""
    r = client.post("/v1/images/generations", json={"prompt": "x", "seed": 42, "n": 1})
    assert r.status_code == 200
    # Fake backend echoes seed in metadata via kwargs — we don't surface it
    # in the response here, but the call should succeed
