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
    reg.register("audio.speech", Fake())
    reg.register("images.generations", Fake())

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/health")
    assert set(r.json()["modalities"]) == {"audio.speech", "images.generations"}


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
    reg.register("audio.speech", FakeAudio())

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert any(m["id"] == "fake-tts" and m["modality"] == "audio.speech" for m in data)


def test_registry_stored_on_app_state():
    reg = ModalityRegistry()
    app = create_app(registry=reg, routers={})
    assert app.state.registry is reg


def test_v1_models_registry_fields_win_over_extra():
    """Authoritative fields (id, modality, object) must never be clobbered by extra."""
    reg = ModalityRegistry()

    class HostileModel:
        model_id = "real-id"
        # Attributes that, if harvested into extra, would collide with authoritative fields.
        # None of these names are in registry._extra's current allowlist, but the server
        # must be defensive regardless.
        pass

    reg.register("audio.speech", HostileModel())
    # Manually inject collision keys into the ModelInfo.extra to simulate a future
    # backend that exposes them.
    info = reg.list_all()[0]
    info.extra["id"] = "IMPOSTOR"
    info.extra["modality"] = "wrong"
    info.extra["object"] = "evil"

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/v1/models")
    data = r.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == "real-id"
    assert data[0]["modality"] == "audio.speech"
    assert data[0]["object"] == "model"


def test_model_not_found_error_serializes_openai_shape():
    """ModelNotFoundError must produce {error: {...}} not {detail: {error: {...}}}."""
    from muse.core.errors import ModelNotFoundError

    reg = ModalityRegistry()
    app = create_app(registry=reg, routers={})

    @app.get("/boom")
    def boom():
        raise ModelNotFoundError(model_id="missing-model", modality="audio.speech")

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
    assert "audio.speech" in err["message"]


def test_speech_route_with_empty_registry_returns_openai_404():
    """When no audio.speech model is registered, POST /v1/audio/speech must
    return a 404 with the OpenAI error envelope (not FastAPI's generic 404).

    This requires the audio.speech router to be mounted even when no models are
    loaded — otherwise FastAPI returns {detail: 'Not Found'} for an unknown path.
    """
    from muse.audio.speech.routes import build_router as build_audio_router

    reg = ModalityRegistry()
    router = build_audio_router(reg)
    app = create_app(registry=reg, routers={"audio.speech": router})
    client = TestClient(app)

    r = client.post("/v1/audio/speech", json={"input": "hello"})
    assert r.status_code == 404
    body = r.json()
    assert "error" in body, f"Expected OpenAI envelope, got: {body}"
    assert "detail" not in body
    assert body["error"]["code"] == "model_not_found"


def test_images_route_with_empty_registry_returns_openai_404():
    """Same contract for images.generations."""
    from muse.images.generations.routes import build_router as build_images_router

    reg = ModalityRegistry()
    router = build_images_router(reg)
    app = create_app(registry=reg, routers={"images.generations": router})
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
