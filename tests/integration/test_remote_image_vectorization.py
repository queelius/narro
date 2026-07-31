"""Opt-in real-server tests for StarVector raster-to-SVG."""
import io
import os

import httpx
import pytest
from PIL import Image, ImageDraw

from muse.modalities.image_vectorization.svg import validate_static_svg


pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def vectorization_model(remote_health):
    model_id = os.environ.get(
        "MUSE_VECTORIZATION_MODEL_ID", "starvector-1b-im2svg",
    )
    loaded = remote_health.get("models") or []
    if model_id not in loaded:
        pytest.skip(
            f"muse server does not have {model_id!r} loaded; loaded: {loaded}"
        )
    return model_id


def _diagram_png():
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 24, 112, 104), fill="#2878d0")
    draw.ellipse((44, 44, 84, 84), fill="#f5c542")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def test_protocol_vectorization_returns_valid_static_svg(
    remote_url, vectorization_model,
):
    response = httpx.post(
        f"{remote_url}/v1/images/vectorize",
        files={"image": ("diagram.png", _diagram_png(), "image/png")},
        data={
            "model": vectorization_model,
            "seed": "0",
            "max_new_tokens": "2048",
            "response_format": "json",
        },
        timeout=900,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == vectorization_model
    assert body["mime_type"] == "image/svg+xml"
    assert body["source_size"] == [128, 128]
    assert body["usage"]["completion_tokens"] > 0
    validate_static_svg(body["svg"])


def test_protocol_raw_svg_response(remote_url, vectorization_model):
    response = httpx.post(
        f"{remote_url}/v1/images/vectorize",
        files={"image": ("diagram.png", _diagram_png(), "image/png")},
        data={
            "model": vectorization_model,
            "seed": "0",
            "max_new_tokens": "2048",
            "response_format": "svg",
        },
        timeout=900,
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/svg+xml")
    validate_static_svg(response.text)
