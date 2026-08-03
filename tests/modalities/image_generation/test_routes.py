"""Tests for /v1/images/generations router."""
import asyncio
import threading
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_generation import routes as routes_mod
from muse.modalities.image_generation.protocol import ImageResult
from muse.modalities.image_generation.routes import build_router


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


class RecordingImageModel:
    """Like FakeImageModel but records the kwargs each generate() received.

    Also records inpaint/vary calls so the /edits and /variations route
    tests can assert routing reached the right method.
    """

    def __init__(self, model_id="fake-i2i-model"):
        self.model_id = model_id
        self.default_size = (64, 64)
        self.calls: list[dict] = []
        self.inpaint_calls: list[dict] = []
        self.vary_calls: list[dict] = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        w = kwargs.get("width", self.default_size[0])
        h = kwargs.get("height", self.default_size[1])
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return ImageResult(
            image=arr, width=w, height=h,
            seed=kwargs.get("seed", 0) or 0,
            metadata={"prompt": prompt},
        )

    def inpaint(self, prompt, **kwargs):
        self.inpaint_calls.append({"prompt": prompt, **kwargs})
        w = kwargs.get("width", self.default_size[0])
        h = kwargs.get("height", self.default_size[1])
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return ImageResult(
            image=arr, width=w, height=h,
            seed=kwargs.get("seed", 0) or 0,
            metadata={"prompt": prompt, "mode": "inpaint"},
        )

    def vary(self, **kwargs):
        self.vary_calls.append(kwargs)
        w = kwargs.get("width", self.default_size[0])
        h = kwargs.get("height", self.default_size[1])
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return ImageResult(
            image=arr, width=w, height=h,
            seed=kwargs.get("seed", 0) or 0,
            metadata={"mode": "variations"},
        )


@pytest.fixture
def client():
    reg = ModalityRegistry()
    reg.register("image/generation", FakeImageModel())
    app = create_app(
        registry=reg,
        routers={"image/generation": build_router(reg)},
    )
    return TestClient(app)


@pytest.fixture
def capable_model():
    """Recording model registered with supports_img2img: True."""
    return RecordingImageModel(model_id="fake-i2i-model")


@pytest.fixture
def client_with_capable_model(capable_model):
    """Client whose registered model declares supports_img2img: True."""
    reg = ModalityRegistry()
    reg.register(
        "image/generation",
        capable_model,
        manifest={
            "model_id": capable_model.model_id,
            "modality": "image/generation",
            "capabilities": {"supports_img2img": True},
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/generation": build_router(reg)},
    )
    return TestClient(app)


@pytest.fixture
def client_with_uncapable_model():
    """Client whose registered model declares supports_img2img: False."""
    reg = ModalityRegistry()
    model = RecordingImageModel(model_id="fake-t2i-only")
    reg.register(
        "image/generation",
        model,
        manifest={
            "model_id": model.model_id,
            "modality": "image/generation",
            "capabilities": {"supports_img2img": False},
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/generation": build_router(reg)},
    )
    return TestClient(app)


@pytest.fixture
def edits_model():
    """Recording model registered with supports_inpainting: True."""
    return RecordingImageModel(model_id="fake-edits-model")


@pytest.fixture
def client_with_edits_model(edits_model):
    """Client whose model advertises inpainting + variations support."""
    reg = ModalityRegistry()
    reg.register(
        "image/generation",
        edits_model,
        manifest={
            "model_id": edits_model.model_id,
            "modality": "image/generation",
            "capabilities": {
                "supports_inpainting": True,
                "supports_variations": True,
            },
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/generation": build_router(reg)},
    )
    return TestClient(app)


@pytest.fixture
def client_without_edits_capability():
    """Client whose model does NOT advertise supports_inpainting/supports_variations."""
    reg = ModalityRegistry()
    model = RecordingImageModel(model_id="fake-edits-blocked")
    reg.register(
        "image/generation",
        model,
        manifest={
            "model_id": model.model_id,
            "modality": "image/generation",
            "capabilities": {
                "supports_inpainting": False,
                "supports_variations": False,
            },
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/generation": build_router(reg)},
    )
    return TestClient(app)


@pytest.fixture
def lora_model():
    """Recording model registered with capabilities.lora_adapter: True."""
    return RecordingImageModel(model_id="fake-lora-model")


@pytest.fixture
def client_with_lora_model(lora_model):
    """Client whose registered model declares capabilities.lora_adapter: True."""
    reg = ModalityRegistry()
    reg.register(
        "image/generation",
        lora_model,
        manifest={
            "model_id": lora_model.model_id,
            "modality": "image/generation",
            "capabilities": {"lora_adapter": True},
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/generation": build_router(reg)},
    )
    return TestClient(app)


def _png_bytes(width=32, height=32, color=(0, 128, 255)):
    """Helper: minimal PNG bytes via PIL."""
    import io
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mask_bytes(width=32, height=32, white=True):
    """Helper: minimal grayscale-mask PNG (white = regenerate)."""
    import io
    from PIL import Image
    fill = 255 if white else 0
    img = Image.new("L", (width, height), fill)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
    # Fake backend echoes seed in metadata via kwargs; we don't surface it
    # in the response here, but the call should succeed


def test_post_with_image_data_url_routes_through_img2img(
    client_with_capable_model, capable_model,
):
    """When image is supplied to a supports_img2img model, it reaches generate(init_image=...)."""
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", (64, 64), (0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "make it red",
        "model": "fake-i2i-model",
        "image": data_url,
        "strength": 0.7,
    })
    assert r.status_code == 200
    body = r.json()
    assert "data" in body
    # Verify the model received init_image (not None) and strength
    assert len(capable_model.calls) == 1
    call = capable_model.calls[0]
    assert call["init_image"] is not None
    assert call["strength"] == 0.7


def test_post_strength_without_image_is_ignored(
    client_with_capable_model, capable_model,
):
    """strength alone (no image) does not error; falls through to text-to-image."""
    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "a cat",
        "model": "fake-i2i-model",
        "strength": 0.5,
    })
    assert r.status_code == 200
    # No image was sent, so init_image should be None on the call
    assert len(capable_model.calls) == 1
    assert capable_model.calls[0]["init_image"] is None


def test_post_image_with_unsupported_model_returns_400(client_with_uncapable_model):
    """A model whose supports_img2img is False rejects requests with image."""
    r = client_with_uncapable_model.post("/v1/images/generations", json={
        "prompt": "x",
        "model": "fake-t2i-only",
        "image": "data:image/png;base64,iVBORw0KGgo=",
    })
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert "img2img" in body["error"]["message"].lower()


def test_post_malformed_data_url_returns_400_image_decode_error(
    client_with_capable_model,
):
    """A bad data URL returns 400 with image_decode-flavored message."""
    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "x",
        "model": "fake-i2i-model",
        "image": "data:image/png;base64,!!!not_base64!!!",
    })
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    err = body["error"]
    assert err["code"] == "invalid_parameter"
    # Type or message should reference image
    assert "image" in err["message"].lower() or "decode" in err["message"].lower()


def test_post_strength_out_of_range_returns_400(client_with_capable_model):
    """strength=1.5 is rejected by pydantic ge/le validation."""
    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "x",
        "model": "fake-i2i-model",
        "image": "data:image/png;base64,iVBORw0KGgo=",
        "strength": 1.5,  # out of [0, 1]
    })
    assert r.status_code in (400, 422)


@pytest.mark.asyncio
async def test_cancelled_generation_defers_input_and_eventual_output_cleanup(
    monkeypatch,
):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    input_cleaned = asyncio.Event()
    output_cleaned = asyncio.Event()
    release = threading.Event()
    backend_exited = threading.Event()

    class _Resource:
        size = (64, 64)

        def __init__(self, cleaned):
            self.closed = False
            self.close_calls = 0
            self._cleaned = cleaned

        def close(self):
            assert backend_exited.is_set()
            self.close_calls += 1
            self.closed = True
            loop.call_soon_threadsafe(self._cleaned.set)

    init_image = _Resource(input_cleaned)
    output_image = _Resource(output_cleaned)

    class _BlockingImageModel:
        model_id = "blocking-image-model"
        default_size = (64, 64)

        def __init__(self):
            self._inference_lock = threading.Lock()
            self.calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            assert kwargs["init_image"] is init_image
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5)
            assert not init_image.closed
            backend_exited.set()
            return ImageResult(
                image=output_image,
                width=64,
                height=64,
                seed=0,
                metadata={"prompt": prompt},
            )

    model = _BlockingImageModel()
    registry = ModalityRegistry()
    registry.register(
        "image/generation",
        model,
        manifest={
            "model_id": model.model_id,
            "capabilities": {"supports_img2img": True},
        },
    )
    endpoint = next(
        route.endpoint
        for route in build_router(registry).routes
        if route.path == "/v1/images/generations"
    )
    monkeypatch.setattr(
        routes_mod,
        "decode_image_input",
        AsyncMock(return_value=init_image),
    )

    pending = asyncio.create_task(endpoint(routes_mod.GenerationRequest(
        prompt="draw",
        model=model.model_id,
        image="data:image/png;base64,unused",
        n=3,
    )))
    await asyncio.wait_for(started.wait(), timeout=1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert not init_image.closed
    assert not output_image.closed
    release.set()
    await asyncio.wait_for(input_cleaned.wait(), timeout=1)
    await asyncio.wait_for(output_cleaned.wait(), timeout=1)
    assert init_image.close_calls == 1
    assert output_image.close_calls == 1
    assert model.calls == 1


# ---------------- /v1/images/edits (inpainting, multipart) ----------------


def test_post_edits_multipart_returns_envelope(
    client_with_edits_model, edits_model,
):
    """Inpainting: image + mask + prompt yields the OpenAI-shape envelope."""
    import base64

    r = client_with_edits_model.post(
        "/v1/images/edits",
        files={
            "image": ("scene.png", _png_bytes(64, 64), "image/png"),
            "mask": ("mask.png", _mask_bytes(64, 64), "image/png"),
        },
        data={
            "prompt": "add a moon to the sky",
            "model": "fake-edits-model",
            "n": "1",
            "size": "64x64",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert "b64_json" in entry
    decoded = base64.b64decode(entry["b64_json"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    # revised_prompt is the inpainting echo (matches /v1/images/generations)
    assert entry["revised_prompt"] == "add a moon to the sky"
    # Backend got init_image and mask_image kwargs
    assert len(edits_model.inpaint_calls) == 1
    call = edits_model.inpaint_calls[0]
    assert call["prompt"] == "add a moon to the sky"
    assert call["init_image"] is not None
    assert call["mask_image"] is not None


def test_post_edits_with_uncapable_model_returns_400(
    client_without_edits_capability,
):
    """Inpainting on a model without supports_inpainting returns 400."""
    r = client_without_edits_capability.post(
        "/v1/images/edits",
        files={
            "image": ("s.png", _png_bytes(), "image/png"),
            "mask": ("m.png", _mask_bytes(), "image/png"),
        },
        data={
            "prompt": "x",
            "model": "fake-edits-blocked",
            "size": "64x64",
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert "inpainting" in body["error"]["message"].lower()


def test_post_edits_empty_image_returns_400(client_with_edits_model):
    """Empty image upload returns 400 (decode failure)."""
    r = client_with_edits_model.post(
        "/v1/images/edits",
        files={
            "image": ("s.png", b"", "image/png"),
            "mask": ("m.png", _mask_bytes(), "image/png"),
        },
        data={
            "prompt": "x",
            "model": "fake-edits-model",
            "size": "64x64",
        },
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"


def test_post_edits_malformed_image_returns_400(client_with_edits_model):
    """Undecodable image bytes return 400."""
    r = client_with_edits_model.post(
        "/v1/images/edits",
        files={
            "image": ("s.png", b"not really an image", "image/png"),
            "mask": ("m.png", _mask_bytes(), "image/png"),
        },
        data={
            "prompt": "x",
            "model": "fake-edits-model",
            "size": "64x64",
        },
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"


def test_post_edits_rejects_aggregate_decoded_pixels_and_closes_inputs(
    client_with_edits_model, edits_model,
):
    from unittest.mock import AsyncMock, MagicMock, patch

    decoded_image = MagicMock(size=(20_000_000, 1))
    decoded_mask = MagicMock(size=(20_000_000, 1))
    with patch(
        "muse.modalities.image_generation.routes.decode_image_file",
        new=AsyncMock(side_effect=[decoded_image, decoded_mask]),
    ):
        response = client_with_edits_model.post(
            "/v1/images/edits",
            files={
                "image": ("s.png", b"placeholder", "image/png"),
                "mask": ("m.png", b"placeholder", "image/png"),
            },
            data={
                "prompt": "x",
                "model": "fake-edits-model",
                "size": "64x64",
            },
        )

    assert response.status_code == 400
    assert "aggregate limit" in response.json()["error"]["message"]
    assert edits_model.inpaint_calls == []
    decoded_image.close.assert_called_once_with()
    decoded_mask.close.assert_called_once_with()


def test_post_edits_n_creates_multiple_images(
    client_with_edits_model, edits_model,
):
    """n=3 yields 3 entries; backend sees 3 inpaint() calls."""
    r = client_with_edits_model.post(
        "/v1/images/edits",
        files={
            "image": ("s.png", _png_bytes(), "image/png"),
            "mask": ("m.png", _mask_bytes(), "image/png"),
        },
        data={
            "prompt": "y",
            "model": "fake-edits-model",
            "n": "3",
            "size": "64x64",
        },
    )
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3
    assert len(edits_model.inpaint_calls) == 3


def test_post_edits_response_format_url_returns_data_url(
    client_with_edits_model,
):
    r = client_with_edits_model.post(
        "/v1/images/edits",
        files={
            "image": ("s.png", _png_bytes(), "image/png"),
            "mask": ("m.png", _mask_bytes(), "image/png"),
        },
        data={
            "prompt": "y",
            "model": "fake-edits-model",
            "size": "64x64",
            "response_format": "url",
        },
    )
    assert r.status_code == 200
    entry = r.json()["data"][0]
    assert entry["url"].startswith("data:image/png;base64,")


def test_post_edits_unknown_model_returns_404(client_with_edits_model):
    r = client_with_edits_model.post(
        "/v1/images/edits",
        files={
            "image": ("s.png", _png_bytes(), "image/png"),
            "mask": ("m.png", _mask_bytes(), "image/png"),
        },
        data={
            "prompt": "y",
            "model": "no-such-model",
            "size": "64x64",
        },
    )
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "model_not_found"


def test_post_edits_size_out_of_range_returns_400(client_with_edits_model):
    r = client_with_edits_model.post(
        "/v1/images/edits",
        files={
            "image": ("s.png", _png_bytes(), "image/png"),
            "mask": ("m.png", _mask_bytes(), "image/png"),
        },
        data={
            "prompt": "y",
            "model": "fake-edits-model",
            "size": "4096x4096",
        },
    )
    assert r.status_code == 400


# ---------------- /v1/images/variations (multipart) ----------------


def test_post_variations_multipart_returns_envelope(
    client_with_edits_model, edits_model,
):
    """Variations: image only (no prompt). Envelope has b64_json, no revised_prompt."""
    import base64

    r = client_with_edits_model.post(
        "/v1/images/variations",
        files={"image": ("s.png", _png_bytes(64, 64), "image/png")},
        data={
            "model": "fake-edits-model",
            "n": "1",
            "size": "64x64",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body and len(body["data"]) == 1
    entry = body["data"][0]
    assert "b64_json" in entry
    # No prompt -> no revised_prompt
    assert "revised_prompt" not in entry
    decoded = base64.b64decode(entry["b64_json"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    # Backend's vary() got the init_image
    assert len(edits_model.vary_calls) == 1
    assert edits_model.vary_calls[0]["init_image"] is not None


def test_post_variations_with_uncapable_model_returns_400(
    client_without_edits_capability,
):
    r = client_without_edits_capability.post(
        "/v1/images/variations",
        files={"image": ("s.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-edits-blocked",
            "size": "64x64",
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert "variations" in body["error"]["message"].lower()


def test_post_variations_n_creates_multiple_images(
    client_with_edits_model, edits_model,
):
    r = client_with_edits_model.post(
        "/v1/images/variations",
        files={"image": ("s.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-edits-model",
            "n": "2",
            "size": "64x64",
        },
    )
    assert r.status_code == 200
    assert len(r.json()["data"]) == 2
    assert len(edits_model.vary_calls) == 2


def test_post_variations_unknown_model_returns_404(client_with_edits_model):
    r = client_with_edits_model.post(
        "/v1/images/variations",
        files={"image": ("s.png", _png_bytes(), "image/png")},
        data={
            "model": "no-such-model",
            "size": "64x64",
        },
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"


def test_post_variations_empty_image_returns_400(client_with_edits_model):
    r = client_with_edits_model.post(
        "/v1/images/variations",
        files={"image": ("s.png", b"", "image/png")},
        data={
            "model": "fake-edits-model",
            "size": "64x64",
        },
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"


def test_post_variations_response_format_url_returns_data_url(
    client_with_edits_model,
):
    r = client_with_edits_model.post(
        "/v1/images/variations",
        files={"image": ("s.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-edits-model",
            "size": "64x64",
            "response_format": "url",
        },
    )
    assert r.status_code == 200
    entry = r.json()["data"][0]
    assert entry["url"].startswith("data:image/png;base64,")
    assert "revised_prompt" not in entry


# ---------------- lora_scale (capability gate) ----------------


def test_lora_scale_on_non_lora_model_returns_400(client_with_uncapable_model):
    """lora_scale against a model without capabilities.lora_adapter is a
    capability mismatch, mirroring the img2img gate."""
    r = client_with_uncapable_model.post("/v1/images/generations", json={
        "prompt": "a cat",
        "model": "fake-t2i-only",
        "lora_scale": 0.8,
    })
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "lora_not_supported"


def test_lora_scale_forwarded_to_lora_model(client_with_lora_model, lora_model):
    """A model whose manifest declares lora_adapter receives the value."""
    r = client_with_lora_model.post("/v1/images/generations", json={
        "prompt": "pixel art, a knight",
        "model": "fake-lora-model",
        "lora_scale": 0.7,
    })
    assert r.status_code == 200, r.text
    assert len(lora_model.calls) == 1
    assert lora_model.calls[0]["lora_scale"] == 0.7


def test_lora_scale_out_of_range_is_422(client_with_lora_model):
    r = client_with_lora_model.post("/v1/images/generations", json={
        "prompt": "a cat",
        "model": "fake-lora-model",
        "lora_scale": 3.5,
    })
    assert r.status_code == 422


# ---------------- Finding 2 (v0.58.1 review): enveloped 500 ----------------


def test_backend_error_in_generations_returns_openai_envelope():
    """The main /v1/images/generations path (routes.py ~129-131) calls
    model.generate() with no local try/except, so a backend exception used
    to escape as FastAPI's bare `{"detail": "Internal Server Error"}`.
    create_app's global exception handler must catch it and return the
    documented OpenAI envelope instead."""

    class _RaisingModel:
        model_id = "fake-raising-sd"
        default_size = (64, 64)

        def generate(self, prompt, **kwargs):
            raise RuntimeError("secret /internal/weights/path")

    reg = ModalityRegistry()
    reg.register("image/generation", _RaisingModel())
    app = create_app(
        registry=reg, routers={"image/generation": build_router(reg)},
    )
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/images/generations", json={
        "prompt": "a cat", "model": "fake-raising-sd",
    })
    assert r.status_code == 500
    body = r.json()
    assert "detail" not in body
    assert body["error"]["code"] == "internal_error"
    assert "secret /internal/weights/path" not in r.text
