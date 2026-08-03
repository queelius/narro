# tests/integration/test_remote_3d_hunyuan3d.py
"""Integration tests for Hunyuan3D-2 dual-direction against a real muse server.

Opt-in via MUSE_REMOTE_SERVER. Skips when env unset or hunyuan3d-2
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


HUNYUAN3D_MODEL_ID = os.environ.get("MUSE_HUNYUAN3D_MODEL_ID", "hunyuan3d-2")


@pytest.fixture(scope="module")
def base_url():
    return os.environ["MUSE_REMOTE_SERVER"].rstrip("/")


@pytest.fixture(scope="module")
def hunyuan_loaded(base_url):
    r = requests.get(f"{base_url}/v1/models", timeout=10)
    r.raise_for_status()
    ids = {m["id"] for m in r.json()["data"]}
    if HUNYUAN3D_MODEL_ID not in ids:
        pytest.skip(f"hunyuan3d model {HUNYUAN3D_MODEL_ID} not on server")


def _red_png_bytes(side: int = 256) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (side, side), "red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_protocol_image_to_3d_returns_glb(base_url, hunyuan_loaded):
    """Hard claim: image input yields a non-empty GLB blob."""
    r = requests.post(
        f"{base_url}/v1/3d/from-image",
        files={"image": ("input.png", _red_png_bytes(), "image/png")},
        data={"model": HUNYUAN3D_MODEL_ID},
        timeout=600,
    )
    r.raise_for_status()
    body = r.json()
    assert body["model"] == HUNYUAN3D_MODEL_ID
    assert len(body["data"]) >= 1
    item = body["data"][0]
    assert "b64_json" in item or "url" in item


def test_protocol_text_to_3d_returns_glb(base_url, hunyuan_loaded):
    """Hard claim: text prompt yields a non-empty GLB blob."""
    r = requests.post(
        f"{base_url}/v1/3d/generations",
        json={"model": HUNYUAN3D_MODEL_ID, "prompt": "a small cube"},
        timeout=600,
    )
    r.raise_for_status()
    body = r.json()
    assert body["model"] == HUNYUAN3D_MODEL_ID
    assert len(body["data"]) >= 1


def test_protocol_image_to_3d_glb_magic_bytes(base_url, hunyuan_loaded):
    r = requests.post(
        f"{base_url}/v1/3d/from-image",
        files={"image": ("input.png", _red_png_bytes(), "image/png")},
        data={"model": HUNYUAN3D_MODEL_ID},
        timeout=600,
    )
    r.raise_for_status()
    glb_b64 = r.json()["data"][0].get("b64_json")
    assert glb_b64
    glb_bytes = base64.b64decode(glb_b64)
    assert glb_bytes[:4] == b"glTF"


def test_protocol_text_to_3d_glb_magic_bytes(base_url, hunyuan_loaded):
    r = requests.post(
        f"{base_url}/v1/3d/generations",
        json={"model": HUNYUAN3D_MODEL_ID, "prompt": "a sphere"},
        timeout=600,
    )
    r.raise_for_status()
    glb_b64 = r.json()["data"][0].get("b64_json")
    assert glb_b64
    glb_bytes = base64.b64decode(glb_b64)
    assert glb_bytes[:4] == b"glTF"


def test_protocol_capability_advertised_in_models(base_url, hunyuan_loaded):
    """The endpoint advertises both SDK-backed generation directions."""
    r = requests.get(f"{base_url}/v1/models", timeout=10)
    r.raise_for_status()
    entries = [m for m in r.json()["data"] if m["id"] == HUNYUAN3D_MODEL_ID]
    assert len(entries) == 1
    caps = entries[0].get("capabilities") or {}
    assert caps.get("supports_image_to_3d") is True
    assert caps.get("supports_text_to_3d") is True
    assert "trust_remote_code" not in caps


def test_protocol_legacy_string_prompt_unaffected(base_url, hunyuan_loaded):
    """Existing prompt-string shape works (regression watchdog)."""
    r = requests.post(
        f"{base_url}/v1/3d/generations",
        json={"model": HUNYUAN3D_MODEL_ID, "prompt": "a chair"},
        timeout=600,
    )
    assert r.status_code == 200


def test_observe_hunyuan3d_describes_simple_image(base_url, hunyuan_loaded):
    """Watchdog: log blob size for regression spotting across SDK updates."""
    r = requests.post(
        f"{base_url}/v1/3d/from-image",
        files={"image": ("input.png", _red_png_bytes(side=512), "image/png")},
        data={"model": HUNYUAN3D_MODEL_ID},
        timeout=600,
    )
    r.raise_for_status()
    glb_b64 = r.json()["data"][0].get("b64_json")
    print(f"\n[observed hunyuan3d-2 GLB blob size]: {len(glb_b64)} b64 chars")
