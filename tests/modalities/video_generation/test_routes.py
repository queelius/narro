"""Tests for POST /v1/video/generations."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.video_generation import (
    MODALITY,
    VideoResult,
    build_router,
)


class RecordingModel:
    """Captures the kwargs each generate() received."""

    model_id = "fake-vid"

    def __init__(self):
        self.last_kwargs = None

    def generate(self, prompt, **kwargs):
        self.last_kwargs = kwargs
        # Return a 5-frame "video" at 720x480.
        frames = [
            Image.new("RGB", (720, 480), (i * 40, 100, 150)) for i in range(5)
        ]
        return VideoResult(
            frames=frames, fps=5, width=720, height=480,
            duration_seconds=1.0, seed=-1, metadata={"prompt": prompt},
        )


def _fake_imageio_factory(format_marker: bytes):
    """Build a fake imageio module that writes deterministic bytes."""
    fake_imageio = type("ii", (), {})()

    def fake_mimwrite(buf, frames, *, fps, codec, format, **_):
        buf.write(format_marker + f":{codec}:{fps}:{len(frames)}".encode())

    fake_imageio.mimwrite = fake_mimwrite
    return fake_imageio


@pytest.fixture
def client_default():
    backend = RecordingModel()
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={
        "model_id": "fake-vid",
        "capabilities": {
            "default_duration_seconds": 1.0,
            "default_fps": 5,
            "default_size": (720, 480),
        },
    })
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app), backend


def test_post_returns_mp4_by_default(client_default):
    client, _ = client_default
    fake = _fake_imageio_factory(b"FAKEMP4")
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "a cat", "model": "fake-vid"},
        )
    assert r.status_code == 200
    body = r.json()
    assert "data" in body
    assert len(body["data"]) == 1
    assert body["model"] == "fake-vid"
    assert body["metadata"]["format"] == "mp4"
    assert body["metadata"]["fps"] == 5
    import base64
    asset = base64.b64decode(body["data"][0]["b64_json"])
    assert asset.startswith(b"FAKEMP4")
    assert b"h264" in asset


def test_post_response_format_webm(client_default):
    client, _ = client_default
    fake = _fake_imageio_factory(b"FAKEWEBM")
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={
                "prompt": "x", "model": "fake-vid",
                "response_format": "webm",
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["metadata"]["format"] == "webm"
    import base64
    asset = base64.b64decode(body["data"][0]["b64_json"])
    assert asset.startswith(b"FAKEWEBM")
    assert b"vp9" in asset


def test_post_response_format_frames_b64(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={
            "prompt": "x", "model": "fake-vid",
            "response_format": "frames_b64",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 5  # 5 frames
    for entry in body["data"]:
        import base64
        png = base64.b64decode(entry["b64_json"])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert body["metadata"]["format"] == "frames_b64"


def test_frames_b64_exceeding_cap_returns_400(client_default, monkeypatch):
    # RecordingModel emits 5 frames; lower the cap to 3 so frames_b64 trips
    # the payload guard. mp4/webm are uncapped, so the cap is format-specific.
    # The cap is read per-request via config, not a module constant, so an
    # env var set at test time is honored without a reload.
    monkeypatch.setenv("MUSE_VIDEO_MAX_FRAMES_B64", "3")

    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "fake-vid",
              "response_format": "frames_b64"},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
    assert "frames_b64" in body["error"]["message"]
    assert "MUSE_VIDEO_MAX_FRAMES_B64" in body["error"]["message"]


def test_mp4_not_subject_to_frames_b64_cap(client_default, monkeypatch):
    # The same 5-frame result is fine for mp4 even with the frame cap at 3:
    # the cap only protects the inline-JSON frames_b64 path.
    monkeypatch.setenv("MUSE_VIDEO_MAX_FRAMES_B64", "3")

    client, _ = client_default
    fake = _fake_imageio_factory(b"FAKEMP4")
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "x", "model": "fake-vid", "response_format": "mp4"},
        )
    assert r.status_code == 200


def test_max_frames_b64_reads_config_live(monkeypatch):
    """MUSE_VIDEO_MAX_FRAMES_B64 set after import is reflected by the
    accessor once config.reset_config() clears the cached Config
    singleton. The cap must be a per-request function reading
    muse.core.config, not a module-level constant frozen at import
    time, so an operator's env change takes effect on the next
    request rather than requiring a server restart. Matches the
    MUSE_IMAGE_INPUT_MAX_BYTES / MUSE_MODERATIONS_MAX_BATCH pattern."""
    from muse.core import config as cfg
    import muse.modalities.video_generation.routes as vroutes

    monkeypatch.setenv("MUSE_VIDEO_MAX_FRAMES_B64", "42")
    cfg.reset_config()
    assert vroutes._max_frames_b64() == 42
    cfg.reset_config()


def test_post_n_2_returns_two_videos_for_mp4(client_default):
    client, _ = client_default
    fake = _fake_imageio_factory(b"FAKEMP4")
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "x", "model": "fake-vid", "n": 2},
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 2


def test_post_unknown_model_returns_404(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "nonexistent"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_post_missing_prompt_returns_422(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"model": "fake-vid"},
    )
    assert r.status_code == 422


def test_post_duration_out_of_range_returns_422(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "fake-vid", "duration_seconds": 100.0},
    )
    assert r.status_code in (400, 422)


def test_post_invalid_response_format_returns_422(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "fake-vid", "response_format": "avi"},
    )
    assert r.status_code in (400, 422)


def test_post_malformed_size_returns_422(client_default):
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "fake-vid", "size": "garbage"},
    )
    assert r.status_code in (400, 422)


def test_post_n_too_large_returns_422(client_default):
    """n is capped at 2; n=3 should fail validation."""
    client, _ = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "fake-vid", "n": 3},
    )
    assert r.status_code in (400, 422)


def test_post_size_parsed_into_width_and_height(client_default):
    """size='WxH' is split and forwarded as width + height to backend."""
    client, backend = client_default
    fake = _fake_imageio_factory(b"FAKEMP4")
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "x", "model": "fake-vid", "size": "832x480"},
        )
    assert r.status_code == 200
    assert backend.last_kwargs["width"] == 832
    assert backend.last_kwargs["height"] == 480


def test_post_oversized_dimensions_rejected_before_backend(client_default):
    client, backend = client_default
    r = client.post(
        "/v1/video/generations",
        json={"prompt": "x", "model": "fake-vid", "size": "4096x4096"},
    )
    assert r.status_code == 422
    assert backend.last_kwargs is None


def test_post_excessive_pixel_frame_workload_rejected(client_default):
    client, backend = client_default
    r = client.post(
        "/v1/video/generations",
        json={
            "prompt": "x", "model": "fake-vid", "size": "1920x1080",
            "duration_seconds": 30, "fps": 60, "n": 2,
        },
    )
    assert r.status_code == 422
    assert backend.last_kwargs is None


def test_post_seed_offset_per_n(client_default):
    """For n>1, each result gets seed + offset."""
    backend = RecordingModel()
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "fake-vid"})
    app = create_app(
        registry=reg, routers={MODALITY: build_router(reg)},
    )
    client = TestClient(app)
    fake = _fake_imageio_factory(b"FAKEMP4")
    seeds_seen = []

    original_generate = backend.generate

    def tracked_generate(prompt, **kwargs):
        seeds_seen.append(kwargs.get("seed"))
        return original_generate(prompt, **kwargs)

    backend.generate = tracked_generate

    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "x", "model": "fake-vid", "n": 2, "seed": 100},
        )
    assert r.status_code == 200
    assert seeds_seen == [100, 101]


def test_post_unsupported_format_returns_400_when_imageio_absent(
    client_default,
):
    """If imageio isn't installed, mp4/webm requests get a clean 400."""
    client, _ = client_default
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=None,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "x", "model": "fake-vid", "response_format": "mp4"},
        )
    assert r.status_code == 400
    assert "imageio" in r.json()["error"]["message"]


def test_video_backend_error_returns_500_envelope():
    """L14: a backend exception surfaces as the OpenAI 500 envelope."""
    class _RaisingModel:
        model_id = "fake-vid"

        def generate(self, prompt, **kwargs):
            raise RuntimeError("model exploded")

    reg = ModalityRegistry()
    reg.register(MODALITY, _RaisingModel(), manifest={"model_id": "fake-vid"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/video/generations", json={"prompt": "a cat", "model": "fake-vid"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"


def test_empty_frames_returns_openai_error_envelope():
    """A backend that returns zero frames trips encode_mp4/encode_webm's
    plain `ValueError("...frames list is empty")`. The route must catch
    it and return the OpenAI-shape error envelope, not let it escape as
    FastAPI's bare default 500 handler."""

    class _EmptyFramesModel:
        model_id = "fake-vid"

        def generate(self, prompt, **kwargs):
            return VideoResult(
                frames=[], fps=5, width=720, height=480,
                duration_seconds=1.0, seed=-1, metadata={"prompt": prompt},
            )

    reg = ModalityRegistry()
    reg.register(MODALITY, _EmptyFramesModel(), manifest={"model_id": "fake-vid"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app, raise_server_exceptions=False)

    fake = _fake_imageio_factory(b"FAKEMP4")
    with patch(
        "muse.modalities.video_generation.codec._try_import_imageio",
        return_value=fake,
    ):
        r = client.post(
            "/v1/video/generations",
            json={"prompt": "a cat", "model": "fake-vid"},
        )
    assert r.status_code in (422, 500)
    body = r.json()
    assert "error" in body
    assert body["error"]["code"]
    # Finding 1 (v0.58.1 review): the backend/codec exception text
    # ("...frames list is empty") must NOT reach the client body; only a
    # generic message does.
    assert "empty" not in body["error"]["message"]
