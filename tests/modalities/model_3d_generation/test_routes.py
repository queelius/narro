"""Route tests for /v1/3d/generations and /v1/3d/from-image.

Two routes share the `3d/generation` modality. Capability flags
`supports_text_to_3d` and `supports_image_to_3d` on each manifest gate
which route a model accepts; mismatch returns 400 `unsupported_route`
before the runtime is invoked.
"""
from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.model_3d_generation import (
    Generation3DResult,
    MODALITY,
    build_router,
)


class _FakeBackend:
    def __init__(
        self,
        model_id: str = "3d-fake",
        glb_bytes: bytes = b"fake-glb-bytes",
    ) -> None:
        self.model_id = model_id
        self._glb_bytes = glb_bytes
        self.last_input = None
        self.last_kwargs: dict | None = None
        self.image_called = 0
        self.text_called = 0

    def text_to_3d(self, prompt, **kwargs):
        self.last_input = prompt
        self.last_kwargs = kwargs
        self.text_called += 1
        n = kwargs.get("n", 1) or 1
        return [
            Generation3DResult(
                glb_bytes=self._glb_bytes, model_id=self.model_id,
            )
            for _ in range(n)
        ]

    def image_to_3d(self, image, **kwargs):
        self.last_input = image
        self.last_kwargs = kwargs
        self.image_called += 1
        n = kwargs.get("n", 1) or 1
        return [
            Generation3DResult(
                glb_bytes=self._glb_bytes, model_id=self.model_id,
            )
            for _ in range(n)
        ]


def _fake_pil_image():
    """Return a mock PIL.Image that satisfies routes._PIL_Image checks."""
    img = MagicMock(name="PIL.Image")
    img.convert = MagicMock(return_value=img)
    return img


def _valid_png_bytes():
    """Return bytes for a minimal 1x1 RGB PNG that PIL can decode."""
    from PIL import Image as PILImage
    import io as _io
    buf = _io.BytesIO()
    PILImage.new("RGB", (1, 1), color=(0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _client(backend, manifest=None):
    if manifest is None:
        manifest = {
            "model_id": backend.model_id,
            "capabilities": {
                "supports_text_to_3d": True,
                "supports_image_to_3d": True,
            },
        }
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


# ---------------- text route ----------------


def test_text_route_returns_envelope():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/generations",
        json={"prompt": "a small wooden chair"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == backend.model_id
    assert body["id"].startswith("3d-")
    assert isinstance(body["created"], int)
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["format"] == "glb"
    assert "b64_json" in entry
    assert base64.b64decode(entry["b64_json"]) == b"fake-glb-bytes"
    assert backend.text_called == 1
    assert backend.image_called == 0


def test_text_route_url_response_format():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/generations",
        json={"prompt": "x", "response_format": "url"},
    )
    assert r.status_code == 200, r.text
    entry = r.json()["data"][0]
    assert entry["url"].startswith("data:model/gltf-binary;base64,")
    assert "b64_json" not in entry


def test_text_route_unsupported_returns_400():
    backend = _FakeBackend()
    manifest = {
        "model_id": backend.model_id,
        "capabilities": {
            "supports_text_to_3d": False,
            "supports_image_to_3d": True,
        },
    }
    client = _client(backend, manifest=manifest)
    r = client.post(
        "/v1/3d/generations",
        json={"prompt": "x"},
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["code"] == "unsupported_route"
    msg = body["error"]["message"]
    assert "text-to-3d" in msg
    assert "/v1/3d/from-image" in msg
    # Backend not invoked.
    assert backend.text_called == 0


def test_text_route_missing_capability_treated_as_unsupported():
    backend = _FakeBackend()
    manifest = {"model_id": backend.model_id, "capabilities": {}}
    client = _client(backend, manifest=manifest)
    r = client.post(
        "/v1/3d/generations",
        json={"prompt": "x"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_route"
    assert backend.text_called == 0


def test_text_route_unknown_model_returns_404():
    backend = _FakeBackend(model_id="real")
    client = _client(backend)
    r = client.post(
        "/v1/3d/generations",
        json={"prompt": "x", "model": "ghost"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_text_route_invalid_response_format_returns_400():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/generations",
        json={"prompt": "x", "response_format": "json"},
    )
    assert r.status_code == 400, r.text
    assert "response_format" in r.json()["error"]["message"]


def test_text_route_n_too_high_returns_400():
    """Pydantic validates ge=1, le=2 on n."""
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/generations",
        json={"prompt": "x", "n": 5},
    )
    # Pydantic validation failures surface as 422 via the global
    # validation handler; n must be in [1, 2] either way.
    assert r.status_code in (400, 422), r.text


def test_text_route_n_zero_returns_400():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/generations",
        json={"prompt": "x", "n": 0},
    )
    assert r.status_code in (400, 422), r.text


def test_text_route_forwards_n_and_seed_kwargs():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/generations",
        json={"prompt": "panda", "n": 2, "seed": 42},
    )
    assert r.status_code == 200, r.text
    assert backend.last_input == "panda"
    assert backend.last_kwargs is not None
    assert backend.last_kwargs.get("n") == 2
    assert backend.last_kwargs.get("seed") == 42
    # n=2 -> envelope carries 2 entries.
    assert len(r.json()["data"]) == 2


def test_text_route_backend_exception_returns_500():
    backend = MagicMock()
    backend.model_id = "broken"
    backend.text_to_3d = MagicMock(side_effect=RuntimeError("boom"))
    manifest = {
        "model_id": "broken",
        "capabilities": {
            "supports_text_to_3d": True,
            "supports_image_to_3d": False,
        },
    }
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    c = TestClient(app)
    r = c.post("/v1/3d/generations", json={"prompt": "x"})
    assert r.status_code == 500, r.text
    body = r.json()
    assert body["error"]["code"] == "internal_error"
    # Finding 1 (v0.58.1 review): the backend exception text must NOT
    # reach the client body; only a generic message does.
    assert "boom" not in body["error"]["message"]


# ---------------- image route ----------------


def test_image_route_returns_envelope():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", _valid_png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == backend.model_id
    assert body["id"].startswith("3d-")
    assert len(body["data"]) == 1
    assert body["data"][0]["format"] == "glb"
    assert "b64_json" in body["data"][0]
    assert backend.image_called == 1
    assert backend.text_called == 0


def test_image_route_url_response_format():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", _valid_png_bytes(), "image/png")},
        data={"response_format": "url"},
    )
    assert r.status_code == 200, r.text
    entry = r.json()["data"][0]
    assert entry["url"].startswith("data:model/gltf-binary;base64,")


def test_image_route_unsupported_returns_400():
    backend = _FakeBackend()
    manifest = {
        "model_id": backend.model_id,
        "capabilities": {
            "supports_text_to_3d": True,
            "supports_image_to_3d": False,
        },
    }
    client = _client(backend, manifest=manifest)
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", b"PNG", "image/png")},
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["code"] == "unsupported_route"
    msg = body["error"]["message"]
    assert "image-to-3d" in msg
    assert "/v1/3d/generations" in msg
    assert backend.image_called == 0


def test_image_route_missing_capability_treated_as_unsupported():
    backend = _FakeBackend()
    manifest = {"model_id": backend.model_id, "capabilities": {}}
    client = _client(backend, manifest=manifest)
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", b"PNG", "image/png")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_route"
    assert backend.image_called == 0


def test_image_route_unknown_model_returns_404():
    backend = _FakeBackend(model_id="real")
    client = _client(backend)
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", b"PNG", "image/png")},
        data={"model": "ghost"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_image_route_empty_image_returns_400():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", b"", "image/png")},
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["code"] == "invalid_image"


def test_image_route_oversized_returns_400(monkeypatch):
    """MUSE_3D_INPUT_MAX_BYTES env cap rejects oversized uploads."""
    backend = _FakeBackend()
    client = _client(backend)
    monkeypatch.setenv("MUSE_3D_INPUT_MAX_BYTES", "100")
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", b"x" * 200, "image/png")},
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["code"] == "invalid_image"
    assert "MUSE_3D_INPUT_MAX_BYTES" in body["error"]["message"]


def test_image_route_invalid_response_format_returns_400():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", b"PNG", "image/png")},
        data={"response_format": "json"},
    )
    assert r.status_code == 400, r.text
    assert "response_format" in r.json()["error"]["message"]


def test_image_route_invalid_n_returns_400():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", b"PNG", "image/png")},
        data={"n": "5"},
    )
    assert r.status_code == 400, r.text
    assert "n" in r.json()["error"]["message"]


def test_image_route_n_zero_returns_400():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", b"PNG", "image/png")},
        data={"n": "0"},
    )
    assert r.status_code == 400, r.text


def test_image_route_forwards_n_and_seed_kwargs():
    backend = _FakeBackend()
    client = _client(backend)
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", _valid_png_bytes(), "image/png")},
        data={"n": "2", "seed": "99"},
    )
    assert r.status_code == 200, r.text
    assert backend.last_kwargs is not None
    assert backend.last_kwargs.get("n") == 2
    assert backend.last_kwargs.get("seed") == 99
    assert len(r.json()["data"]) == 2


def test_image_route_passes_pil_image_to_backend():
    """Route decodes the upload to a PIL Image; backend.image_to_3d receives
    a PIL.Image.Image (not a str path). This is the regression guard for H1."""
    from PIL import Image as PILImage
    backend = _FakeBackend()
    client = _client(backend)
    # Use a valid 1x1 PNG so PIL can actually decode it.
    import io as _io
    buf = _io.BytesIO()
    PILImage.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    png_bytes = buf.getvalue()
    r = client.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", png_bytes, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert backend.last_input is not None
    # Must be a PIL Image instance, NOT a str path.
    assert isinstance(backend.last_input, PILImage.Image), (
        f"expected PIL.Image.Image, got {type(backend.last_input)}"
    )


def test_image_route_closes_decoded_image_after_backend():
    backend = _FakeBackend()
    decoded = MagicMock()
    decoded.mode = "RGB"
    decoded.size = (1, 1)

    with patch(
        "muse.modalities.model_3d_generation.routes.decode_image_file",
        return_value=decoded,
    ):
        response = _client(backend).post(
            "/v1/3d/from-image",
            files={"image": ("a.png", b"placeholder", "image/png")},
        )

    assert response.status_code == 200, response.text
    decoded.close.assert_called_once_with()


def test_image_route_enforces_shared_decoded_pixel_cap(monkeypatch):
    from PIL import Image as PILImage
    import io as _io

    backend = _FakeBackend()
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_PIXELS", "1")
    buf = _io.BytesIO()
    PILImage.new("RGB", (2, 1)).save(buf, format="PNG")

    response = _client(backend).post(
        "/v1/3d/from-image",
        files={"image": ("wide.png", buf.getvalue(), "image/png")},
    )

    assert response.status_code == 400
    assert "max input pixels" in response.json()["error"]["message"]
    assert backend.image_called == 0


def test_image_route_backend_exception_no_temp_file_left():
    """Even when the backend raises, no temp files should be left (no temp file
    is created in the new PIL-based flow)."""
    import os
    import glob
    import tempfile
    inputs_seen: list = []

    backend = MagicMock()
    backend.model_id = "broken"

    def _raise(image, **kwargs):
        inputs_seen.append(image)
        raise RuntimeError("boom")

    backend.image_to_3d = MagicMock(side_effect=_raise)

    manifest = {
        "model_id": "broken",
        "capabilities": {
            "supports_text_to_3d": False,
            "supports_image_to_3d": True,
        },
    }
    from PIL import Image as PILImage
    import io as _io
    buf = _io.BytesIO()
    PILImage.new("RGB", (1, 1)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    c = TestClient(app)
    r = c.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", png_bytes, "image/png")},
    )
    assert r.status_code == 500, r.text
    assert r.json()["error"]["code"] == "internal_error"
    # Finding 1 (v0.58.1 review): the backend exception text must NOT
    # reach the client body; only a generic message does.
    assert "boom" not in r.json()["error"]["message"]
    # Backend was invoked with a PIL image.
    assert inputs_seen, "backend.image_to_3d was never called"


def test_model_3d_cap_zero_env_falls_back(monkeypatch):
    """MUSE_3D_INPUT_MAX_BYTES=0 must not turn into "reject everything".

    config.get() parses "0" to the literal int 0 (a valid int, not an
    error), so the accessor must guard non-positive values itself and
    fall back to the registry default, mirroring image_input.py's
    _default_max_bytes."""
    from muse.core import config
    monkeypatch.setenv("MUSE_3D_INPUT_MAX_BYTES", "0")
    config.reset_config()
    from muse.modalities.model_3d_generation.routes import _max_bytes
    assert _max_bytes() == config.SETTINGS_BY_KEY["limits.model_3d_input_max_bytes"].default
    config.reset_config()


def test_model_3d_cap_empty_env_falls_back(monkeypatch):
    """MUSE_3D_INPUT_MAX_BYTES="" coerces to None (opt_int); the
    accessor must also fall back to the registry default rather than
    returning None or crashing on the len(raw) > cap comparison."""
    from muse.core import config
    monkeypatch.setenv("MUSE_3D_INPUT_MAX_BYTES", "")
    config.reset_config()
    from muse.modalities.model_3d_generation.routes import _max_bytes
    assert _max_bytes() == config.SETTINGS_BY_KEY["limits.model_3d_input_max_bytes"].default
    config.reset_config()


def test_image_route_backend_exception_returns_500():
    from PIL import Image as PILImage
    import io as _io
    buf = _io.BytesIO()
    PILImage.new("RGB", (1, 1)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    backend = MagicMock()
    backend.model_id = "broken"
    backend.image_to_3d = MagicMock(side_effect=RuntimeError("boom"))
    manifest = {
        "model_id": "broken",
        "capabilities": {
            "supports_text_to_3d": False,
            "supports_image_to_3d": True,
        },
    }
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    c = TestClient(app)
    r = c.post(
        "/v1/3d/from-image",
        files={"image": ("a.png", png_bytes, "image/png")},
    )
    assert r.status_code == 500, r.text
    body = r.json()
    assert body["error"]["code"] == "internal_error"
    # Finding 1 (v0.58.1 review): the backend exception text must NOT
    # reach the client body; only a generic message does.
    assert "boom" not in body["error"]["message"]
