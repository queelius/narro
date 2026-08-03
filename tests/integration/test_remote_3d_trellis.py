# tests/integration/test_remote_3d_trellis.py
"""Integration tests for TRELLIS image-to-3D against a real muse server.

Opt-in via MUSE_REMOTE_SERVER. Skips when env unset or trellis-image
isn't enabled on the server.
"""
import base64
import os
from io import BytesIO

import pytest
import requests


pytestmark = pytest.mark.skipif(
    not os.environ.get("MUSE_REMOTE_SERVER"),
    reason="MUSE_REMOTE_SERVER not set; integration tests skipped",
)


TRELLIS_MODEL_ID = os.environ.get("MUSE_TRELLIS_MODEL_ID", "trellis-image")


@pytest.fixture(scope="module")
def base_url():
    return os.environ["MUSE_REMOTE_SERVER"].rstrip("/")


@pytest.fixture(scope="module")
def trellis_loaded(base_url):
    """Skip if the configured TRELLIS model isn't enabled on the server."""
    r = requests.get(f"{base_url}/v1/models", timeout=10)
    r.raise_for_status()
    ids = {m["id"] for m in r.json()["data"]}
    if TRELLIS_MODEL_ID not in ids:
        pytest.skip(f"trellis model {TRELLIS_MODEL_ID} not on server")


def _red_square_png_bytes(side: int = 256) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (side, side), "red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_protocol_image_to_3d_returns_glb(base_url, trellis_loaded):
    """Hard claim: an image yields a non-empty GLB blob."""
    png_bytes = _red_square_png_bytes()
    r = requests.post(
        f"{base_url}/v1/3d/from-image",
        files={"image": ("input.png", png_bytes, "image/png")},
        data={"model": TRELLIS_MODEL_ID},
        timeout=600,
    )
    r.raise_for_status()
    body = r.json()
    assert body["model"] == TRELLIS_MODEL_ID
    assert len(body["data"]) >= 1
    item = body["data"][0]
    assert "b64_json" in item or "url" in item


def test_protocol_text_to_3d_rejected(base_url, trellis_loaded):
    """TRELLIS-image declares supports_text_to_3d: False, so the
    text-to-3D route should reject with 400."""
    r = requests.post(
        f"{base_url}/v1/3d/generations",
        json={"model": TRELLIS_MODEL_ID, "prompt": "a chair"},
        timeout=30,
    )
    assert r.status_code == 400


def test_protocol_capability_advertised_in_models(base_url, trellis_loaded):
    """The /v1/models endpoint advertises supports_image_to_3d=True
    and supports_text_to_3d=False for trellis-image."""
    r = requests.get(f"{base_url}/v1/models", timeout=10)
    r.raise_for_status()
    entries = [m for m in r.json()["data"] if m["id"] == TRELLIS_MODEL_ID]
    assert len(entries) == 1
    caps = entries[0].get("capabilities") or {}
    assert caps.get("supports_image_to_3d") is True
    assert caps.get("supports_text_to_3d") is False
    assert "trust_remote_code" not in caps


def test_protocol_glb_magic_bytes(base_url, trellis_loaded):
    """Returned bytes are a valid GLB (magic bytes b'glTF')."""
    png_bytes = _red_square_png_bytes()
    r = requests.post(
        f"{base_url}/v1/3d/from-image",
        files={"image": ("input.png", png_bytes, "image/png")},
        data={"model": TRELLIS_MODEL_ID},
        timeout=600,
    )
    r.raise_for_status()
    glb_b64 = r.json()["data"][0].get("b64_json")
    assert glb_b64
    glb_bytes = base64.b64decode(glb_b64)
    assert glb_bytes[:4] == b"glTF"


def test_observe_trellis_describes_simple_image(base_url, trellis_loaded):
    """Watchdog: not a hard claim. Logs the GLB blob size so a human
    can spot regressions in mesh complexity across runs."""
    png_bytes = _red_square_png_bytes(side=512)
    r = requests.post(
        f"{base_url}/v1/3d/from-image",
        files={"image": ("input.png", png_bytes, "image/png")},
        data={"model": TRELLIS_MODEL_ID},
        timeout=600,
    )
    r.raise_for_status()
    glb_b64 = r.json()["data"][0].get("b64_json")
    print(f"\n[observed trellis GLB blob size]: {len(glb_b64)} b64 chars")
