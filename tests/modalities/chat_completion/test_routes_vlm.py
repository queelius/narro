"""VLM-extended chat/completions route tests.

Exercises the new pre-dispatch step: capability gating, image decoding,
content-shape validation. Text-only requests must remain byte-identical
to v0.41.x behavior (regression watchdog).
"""
import asyncio
import base64
import threading
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.modalities.chat_completion import routes as routes_mod
from muse.modalities.chat_completion.routes import build_router


def _make_data_url(size=(8, 8), color="red"):
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class _FakeChatModel:
    def __init__(self, model_id, supports_vision=False, supports_multi_image=False):
        self.model_id = model_id
        self.supports_vision = supports_vision
        self.supports_multi_image = supports_multi_image
        self.received_messages = None

    def chat(self, messages, **kwargs):
        self.received_messages = messages
        from muse.modalities.chat_completion.protocol import (
            ChatChoice, ChatResult,
        )
        return ChatResult(
            id="chatcmpl-test",
            model_id=self.model_id,
            created=0,
            choices=[ChatChoice(
                index=0,
                message={"role": "assistant", "content": "ok"},
                finish_reason="stop",
            )],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


def _client_with_model(model):
    registry = ModalityRegistry()
    # Build a manifest from the model's capability attrs so the route's
    # manifest-based capability gating (fix #4) sees the right flags.
    manifest = {
        "model_id": model.model_id,
        "capabilities": {
            "supports_vision": getattr(model, "supports_vision", False),
            "supports_multi_image": getattr(model, "supports_multi_image", False),
        },
    }
    registry.register("chat/completion", model, manifest=manifest)
    app = FastAPI()
    app.include_router(build_router(registry))
    return TestClient(app)


def test_text_only_request_byte_identical():
    """v0.41.x regression watchdog: pure-text requests do not trip the
    new pre-dispatch step, do not touch decode_image_input, and reach
    the backend unchanged."""
    model = _FakeChatModel("text-only", supports_vision=False)
    client = _client_with_model(model)
    r = client.post("/v1/chat/completions", json={
        "model": "text-only",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert model.received_messages == [{"role": "user", "content": "hi"}]


def test_vision_capability_mismatch_returns_400():
    model = _FakeChatModel("text-only", supports_vision=False)
    client = _client_with_model(model)
    data_url = _make_data_url()
    r = client.post("/v1/chat/completions", json={
        "model": "text-only",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    })
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "vision_not_supported"


def test_multi_image_capability_mismatch_returns_400():
    model = _FakeChatModel(
        "vlm-single", supports_vision=True, supports_multi_image=False,
    )
    client = _client_with_model(model)
    data_url = _make_data_url()
    r = client.post("/v1/chat/completions", json={
        "model": "vlm-single",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "too_many_images"


def test_too_many_images_counts_per_conversation_not_per_message():
    # A single-image model must reject 2 images even when split 1-per-message
    # across two messages: per-message counting would admit them (each <= 1),
    # but the backend still receives 2 images total. (M5)
    model = _FakeChatModel(
        "vlm-single", supports_vision=True, supports_multi_image=False,
    )
    client = _client_with_model(model)
    data_url = _make_data_url()
    r = client.post("/v1/chat/completions", json={
        "model": "vlm-single",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
    })
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "too_many_images"
    assert "conversation" in body["error"]["message"]


def test_one_image_per_message_ok_for_multi_image_model():
    # The same 1-per-message split is fine for a multi-image model.
    model = _FakeChatModel(
        "vlm-multi", supports_vision=True, supports_multi_image=True,
    )
    client = _client_with_model(model)
    data_url = _make_data_url()
    r = client.post("/v1/chat/completions", json={
        "model": "vlm-multi",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
    })
    assert r.status_code == 200


def test_supports_tools_read_from_manifest_not_instance(caplog):
    # Resolver-pulled GGUFs carry supports_tools in the synthesized manifest,
    # not as an instance attr. The route must read it from the manifest: a
    # manifest-declared supports_tools=True suppresses the "tool support
    # unknown" warning even though the model instance lacks the attribute. (M6)
    import logging

    model = _FakeChatModel("gguf-tools", supports_vision=False)
    assert not hasattr(model, "supports_tools")  # capability lives in manifest only

    registry = ModalityRegistry()
    registry.register(
        "chat/completion", model,
        manifest={"model_id": "gguf-tools",
                  "capabilities": {"supports_tools": True}},
    )
    app = FastAPI()
    app.include_router(build_router(registry))
    client = TestClient(app)

    with caplog.at_level(logging.WARNING):
        r = client.post("/v1/chat/completions", json={
            "model": "gguf-tools",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function",
                       "function": {"name": "f", "parameters": {}}}],
        })
    assert r.status_code == 200
    assert not any("tool" in rec.message.lower() for rec in caplog.records)


def test_invalid_content_part_missing_url_returns_400():
    model = _FakeChatModel("vlm", supports_vision=True, supports_multi_image=True)
    client = _client_with_model(model)
    r = client.post("/v1/chat/completions", json={
        "model": "vlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {}},  # missing url
            ],
        }],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_content_part"


@pytest.mark.parametrize(
    "image_url",
    ["not-an-object", ["not", "an", "object"], 7, {"url": 7}],
)
def test_invalid_nested_image_url_shape_returns_structured_400(image_url):
    model = _FakeChatModel("vlm", supports_vision=True, supports_multi_image=True)
    response = _client_with_model(model).post(
        "/v1/chat/completions",
        json={
            "model": "vlm",
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": image_url}],
            }],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_content_part"


def test_invalid_later_image_url_closes_an_earlier_decoded_image():
    model = _FakeChatModel("vlm", supports_vision=True, supports_multi_image=True)
    decoded = MagicMock(size=(8, 8))
    with patch(
        "muse.modalities.chat_completion.routes.decode_image_input",
        new=AsyncMock(return_value=decoded),
    ) as decode:
        response = _client_with_model(model).post(
            "/v1/chat/completions",
            json={
                "model": "vlm",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,unused"},
                        },
                        {"type": "image_url", "image_url": "invalid"},
                    ],
                }],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_content_part"
    assert decode.await_count == 1
    decoded.close.assert_called_once_with()


def test_unsupported_content_type_returns_400():
    model = _FakeChatModel("vlm", supports_vision=True, supports_multi_image=True)
    client = _client_with_model(model)
    r = client.post("/v1/chat/completions", json={
        "model": "vlm",
        "messages": [{
            "role": "user",
            "content": [{"type": "video_url", "video_url": {"url": "x"}}],
        }],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_content_type"


def test_malformed_data_url_returns_400_invalid_image():
    model = _FakeChatModel("vlm", supports_vision=True, supports_multi_image=True)
    client = _client_with_model(model)
    r = client.post("/v1/chat/completions", json={
        "model": "vlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:notavalidurl"}},
            ],
        }],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_image"


def test_valid_image_decoded_and_forwarded():
    """Happy path: image_url part is decoded, rewritten to {type: image,
    image: <PIL>}, and the backend sees the rewritten messages."""
    model = _FakeChatModel("vlm", supports_vision=True, supports_multi_image=True)
    client = _client_with_model(model)
    data_url = _make_data_url((16, 16), "blue")
    r = client.post("/v1/chat/completions", json={
        "model": "vlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what?"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    })
    assert r.status_code == 200
    msg = model.received_messages[0]
    assert msg["content"][0] == {"type": "text", "text": "what?"}
    assert msg["content"][1]["type"] == "image"
    img = msg["content"][1]["image"]
    assert img.size == (16, 16)


def test_multi_image_supported_when_capability_true():
    model = _FakeChatModel(
        "multi-vlm", supports_vision=True, supports_multi_image=True,
    )
    client = _client_with_model(model)
    data_url = _make_data_url()
    r = client.post("/v1/chat/completions", json={
        "model": "multi-vlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    })
    assert r.status_code == 200
    images = [p for p in model.received_messages[0]["content"] if p["type"] == "image"]
    assert len(images) == 2


def test_multi_image_rejects_aggregate_decoded_pixel_budget(monkeypatch):
    from muse.core import config as cfg

    model = _FakeChatModel(
        "multi-vlm", supports_vision=True, supports_multi_image=True,
    )
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_TOTAL_PIXELS", "100")
    cfg.reset_config()
    try:
        data_url = _make_data_url((8, 8))
        response = _client_with_model(model).post(
            "/v1/chat/completions",
            json={
                "model": "multi-vlm",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "image_budget_exceeded"
        assert model.received_messages is None
    finally:
        cfg.reset_config()


def test_non_streaming_chat_closes_decoded_inputs():
    model = _FakeChatModel("vlm", supports_vision=True)
    decoded = MagicMock()
    decoded.size = (8, 8)
    with patch(
        "muse.modalities.chat_completion.routes.decode_image_input",
        new=AsyncMock(return_value=decoded),
    ):
        response = _client_with_model(model).post(
            "/v1/chat/completions",
            json={
                "model": "vlm",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,unused"},
                    }],
                }],
            },
        )
    assert response.status_code == 200
    decoded.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_cancelled_non_stream_chat_keeps_decoded_image_until_backend_exits(
    monkeypatch,
):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    cleaned = asyncio.Event()
    release = threading.Event()
    backend_exited = threading.Event()

    class _DecodedImage:
        size = (8, 8)

        def __init__(self):
            self.closed = False
            self.close_calls = 0

        def close(self):
            assert backend_exited.is_set()
            self.close_calls += 1
            self.closed = True
            loop.call_soon_threadsafe(cleaned.set)

    decoded = _DecodedImage()

    class _BlockingChatModel(_FakeChatModel):
        def __init__(self):
            super().__init__("blocking-vlm", supports_vision=True)
            self._inference_lock = threading.Lock()

        def chat(self, messages, **kwargs):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5)
            assert not decoded.closed
            backend_exited.set()
            return super().chat(messages, **kwargs)

    model = _BlockingChatModel()
    registry = ModalityRegistry()
    registry.register(
        "chat/completion",
        model,
        manifest={
            "model_id": model.model_id,
            "capabilities": {"supports_vision": True},
        },
    )
    endpoint = next(
        route.endpoint
        for route in build_router(registry).routes
        if route.path == "/v1/chat/completions"
    )
    monkeypatch.setattr(
        routes_mod,
        "decode_image_input",
        AsyncMock(return_value=decoded),
    )

    pending = asyncio.create_task(endpoint(routes_mod.ChatCompletionRequest(
        model=model.model_id,
        messages=[{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,unused"},
            }],
        }],
    )))
    await asyncio.wait_for(started.wait(), timeout=1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert not decoded.closed
    release.set()
    await asyncio.wait_for(cleaned.wait(), timeout=1)
    assert decoded.close_calls == 1


def test_legacy_string_content_unaffected():
    """OpenAI's older `content: "string"` shape: pass through unchanged."""
    model = _FakeChatModel("text", supports_vision=False)
    client = _client_with_model(model)
    r = client.post("/v1/chat/completions", json={
        "model": "text",
        "messages": [{"role": "user", "content": "plain text"}],
    })
    assert r.status_code == 200
    assert model.received_messages == [{"role": "user", "content": "plain text"}]


def test_resolver_pulled_vlm_capability_via_manifest():
    """Regression for the IMPORTANT #4 fix: when a model is registered
    WITHOUT supports_vision as an instance attr (as resolver-pulled
    VLMs are), the route still allows vision requests because the
    manifest declares the capability."""

    class _BareModel:
        # No class-level supports_vision attribute. Mirrors a resolver-
        # pulled model whose capabilities come from the synthesized
        # manifest, not the class.
        model_id = "resolver-pulled-vlm"

        def chat(self, messages, **_):
            from muse.modalities.chat_completion.protocol import (
                ChatChoice, ChatResult,
            )
            return ChatResult(
                id="x", model_id=self.model_id, created=0,
                choices=[ChatChoice(
                    index=0,
                    message={"role": "assistant", "content": "ok"},
                    finish_reason="stop",
                )],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

    registry = ModalityRegistry()
    registry.register(
        "chat/completion",
        _BareModel(),
        manifest={
            "model_id": "resolver-pulled-vlm",
            "capabilities": {
                "supports_vision": True,
                "supports_multi_image": True,
            },
        },
    )
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from muse.modalities.chat_completion.routes import build_router
    app = FastAPI()
    app.include_router(build_router(registry))
    client = TestClient(app)
    data_url = _make_data_url()
    r = client.post("/v1/chat/completions", json={
        "model": "resolver-pulled-vlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "?"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    })
    assert r.status_code == 200, r.text


def test_capability_error_fires_before_sse_stream_opens():
    """When stream=true is set, capability errors MUST surface as a
    pre-stream 400, never as an event-error mid-stream."""
    model = _FakeChatModel("text-only", supports_vision=False)
    client = _client_with_model(model)
    data_url = _make_data_url()
    r = client.post("/v1/chat/completions", json={
        "model": "text-only",
        "stream": True,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    })
    # Pre-stream 400 (not text/event-stream).
    assert r.status_code == 400
    assert "text/event-stream" not in r.headers.get("content-type", "")
