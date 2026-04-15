"""Tests for the core FastAPI app factory."""
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app


def test_create_app_returns_fastapi():
    app = create_app(registry=ModalityRegistry(), routers={})
    assert isinstance(app, FastAPI)


def test_root_health_endpoint():
    app = create_app(registry=ModalityRegistry(), routers={})
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["modalities"] == []


def test_health_reports_registered_modalities():
    reg = ModalityRegistry()

    class Fake:
        model_id = "fake"
    reg.register("audio/speech", Fake())
    reg.register("image/generation", Fake())

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/health")
    assert set(r.json()["modalities"]) == {"audio/speech", "image/generation"}


def test_routers_are_mounted():
    router = APIRouter()

    @router.get("/v1/test/ping")
    def ping():
        return {"ok": True}

    app = create_app(registry=ModalityRegistry(), routers={"test": router})
    client = TestClient(app)
    r = client.get("/v1/test/ping")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_global_v1_models_endpoint_aggregates():
    reg = ModalityRegistry()

    class FakeAudio:
        model_id = "fake-tts"
        sample_rate = 16000
    reg.register("audio/speech", FakeAudio())

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert any(m["id"] == "fake-tts" and m["modality"] == "audio/speech" for m in data)


def test_registry_stored_on_app_state():
    reg = ModalityRegistry()
    app = create_app(registry=reg, routers={})
    assert app.state.registry is reg


def test_v1_models_registry_fields_win_over_manifest():
    """Authoritative fields (id, modality, object) must never be clobbered by manifest."""
    reg = ModalityRegistry()

    class HostileModel:
        model_id = "real-id"

    # A manifest whose top-level + capabilities both try to override the
    # authoritative fields. The server must ignore all three overrides.
    hostile_manifest = {
        "model_id": "real-id",
        "modality": "audio/speech",
        "id": "IMPOSTOR",           # collides with authoritative "id"
        "object": "evil",           # collides with authoritative "object"
        "capabilities": {
            "id": "ALSO-IMPOSTOR",
            "modality": "wrong",
            "object": "evil-cap",
        },
    }
    reg.register("audio/speech", HostileModel(), manifest=hostile_manifest)

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/v1/models")
    data = r.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == "real-id"
    assert data[0]["modality"] == "audio/speech"
    assert data[0]["object"] == "model"


def test_v1_models_exposes_capabilities_and_metadata_from_manifest():
    """Capabilities, description, license, hf_repo from the manifest flow to /v1/models."""
    reg = ModalityRegistry()

    class Fake:
        model_id = "kokoro-82m"

    manifest = {
        "model_id": "kokoro-82m",
        "modality": "audio/speech",
        "hf_repo": "hexgrad/Kokoro-82M",
        "description": "Lightweight TTS, 54 voices, 24kHz",
        "license": "Apache 2.0",
        "capabilities": {
            "sample_rate": 24000,
            "voices": ["af_heart", "am_adam"],
        },
    }
    reg.register("audio/speech", Fake(), manifest=manifest)

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/v1/models")
    entry = r.json()["data"][0]
    # Top-level metadata
    assert entry["hf_repo"] == "hexgrad/Kokoro-82M"
    assert entry["description"] == "Lightweight TTS, 54 voices, 24kHz"
    assert entry["license"] == "Apache 2.0"
    # Capabilities projected to top level
    assert entry["sample_rate"] == 24000
    assert entry["voices"] == ["af_heart", "am_adam"]


def test_model_not_found_error_serializes_openai_shape():
    """ModelNotFoundError must produce {error: {...}} not {detail: {error: {...}}}."""
    from muse.core.errors import ModelNotFoundError

    reg = ModalityRegistry()
    app = create_app(registry=reg, routers={})

    @app.get("/boom")
    def boom():
        raise ModelNotFoundError(model_id="missing-model", modality="audio/speech")

    client = TestClient(app)
    r = client.get("/boom")
    assert r.status_code == 404
    body = r.json()
    # Top-level key must be "error", not "detail"
    assert "error" in body
    assert "detail" not in body
    err = body["error"]
    assert err["code"] == "model_not_found"
    assert err["type"] == "invalid_request_error"
    assert "missing-model" in err["message"]
    assert "audio/speech" in err["message"]


def test_speech_route_with_empty_registry_returns_openai_404():
    """When no audio/speech model is registered, POST /v1/audio/speech must
    return a 404 with the OpenAI error envelope (not FastAPI's generic 404).

    This requires the audio/speech router to be mounted even when no models are
    loaded — otherwise FastAPI returns {detail: 'Not Found'} for an unknown path.
    """
    from muse.modalities.audio_speech.routes import build_router as build_audio_router

    reg = ModalityRegistry()
    router = build_audio_router(reg)
    app = create_app(registry=reg, routers={"audio/speech": router})
    client = TestClient(app)

    r = client.post("/v1/audio/speech", json={"input": "hello"})
    assert r.status_code == 404
    body = r.json()
    assert "error" in body, f"Expected OpenAI envelope, got: {body}"
    assert "detail" not in body
    assert body["error"]["code"] == "model_not_found"


def test_images_route_with_empty_registry_returns_openai_404():
    """Same contract for images.generations."""
    from muse.modalities.image_generation.routes import build_router as build_images_router

    reg = ModalityRegistry()
    router = build_images_router(reg)
    app = create_app(registry=reg, routers={"image/generation": router})
    client = TestClient(app)

    r = client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 404
    body = r.json()
    assert "error" in body, f"Expected OpenAI envelope, got: {body}"
    assert "detail" not in body
    assert body["error"]["code"] == "model_not_found"


def test_validation_error_uses_openai_envelope():
    """422 for invalid input must use OpenAI shape, not FastAPI default."""
    from pydantic import BaseModel, Field

    router = APIRouter()

    class Req(BaseModel):
        value: int = Field(..., ge=0, le=10)

    @router.post("/validate")
    def validate(r: Req):
        return {"ok": True}

    app = create_app(registry=ModalityRegistry(), routers={"v": router})
    client = TestClient(app)
    r = client.post("/validate", json={"value": 999})
    assert r.status_code == 422
    body = r.json()
    assert "error" in body, f"Expected OpenAI envelope, got: {body}"
    assert "detail" not in body
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_request"
