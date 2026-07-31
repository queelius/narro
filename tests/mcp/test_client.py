"""Tests for muse.mcp.client.MuseClient.

Each method is covered by a mocked httpx.Client patched at the call
site; no network is touched.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from muse.mcp.client import MuseClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    return MuseClient(server_url="http://test.example.com", admin_token="tok")


@pytest.fixture
def mock_httpx_client():
    """Patch httpx.Client to a context-managed MagicMock returning the mock."""
    with patch("muse.mcp.client.httpx.Client") as cls:
        ctx = MagicMock()
        cls.return_value = ctx
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=None)
        yield ctx


def _ok_json(body: dict, status: int = 200, content_type: str = "application/json"):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    r.headers = {"content-type": content_type}
    r.content = b""
    r.raise_for_status = MagicMock()
    return r


def _ok_bytes(data: bytes, status: int = 200, content_type: str = "audio/wav"):
    r = MagicMock()
    r.status_code = status
    r.content = data
    r.headers = {"content-type": content_type}
    r.raise_for_status = MagicMock()
    return r


class TestProbes:
    def test_health(self, client, mock_httpx_client):
        mock_httpx_client.get.return_value = _ok_json({"status": "ok"})
        out = client.health()
        assert out == {"status": "ok"}
        assert mock_httpx_client.get.call_args.args[0].endswith("/health")

    def test_list_models(self, client, mock_httpx_client):
        mock_httpx_client.get.return_value = _ok_json(
            {"data": [{"id": "kokoro-82m"}]},
        )
        out = client.list_models()
        assert out["data"][0]["id"] == "kokoro-82m"


class TestTextRoutes:
    def test_chat(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json(
            {"choices": [{"message": {"content": "hi"}}]},
        )
        out = client.chat(model="qwen", messages=[{"role": "user", "content": "x"}])
        assert out["choices"][0]["message"]["content"] == "hi"
        call = mock_httpx_client.post.call_args
        assert call.args[0].endswith("/v1/chat/completions")
        assert call.kwargs["json"]["model"] == "qwen"

    def test_summarize(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json({"summary": "tldr"})
        out = client.summarize(text="long text", length="short")
        assert out["summary"] == "tldr"
        assert mock_httpx_client.post.call_args.args[0].endswith("/v1/summarize")

    def test_rerank(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json(
            {"results": [{"index": 0, "relevance_score": 0.9}]},
        )
        out = client.rerank(query="q", documents=["a", "b"])
        assert out["results"][0]["index"] == 0

    def test_classify(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json({"results": [{"flagged": False}]})
        out = client.classify(input="hello")
        assert out["results"][0]["flagged"] is False

    def test_embed_text(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json(
            {"data": [{"embedding": [0.1, 0.2]}]},
        )
        out = client.embed_text(input=["x"])
        assert out["data"][0]["embedding"] == [0.1, 0.2]

    def test_post_drops_none_values(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json({"x": 1})
        client.chat(model="m", messages=[{"role": "user", "content": "x"}], temperature=None)
        call = mock_httpx_client.post.call_args
        assert "temperature" not in call.kwargs["json"]


class TestImageRoutes:
    def test_generate_image_defaults_b64(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json(
            {"data": [{"b64_json": "AAAA"}]},
        )
        out = client.generate_image(prompt="cat")
        assert out["data"][0]["b64_json"] == "AAAA"
        call = mock_httpx_client.post.call_args
        assert call.kwargs["json"]["response_format"] == "b64_json"

    def test_edit_image_multipart(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json({"data": [{"b64_json": "X"}]})
        out = client.edit_image(image=b"img", mask=b"msk", prompt="paint blue")
        assert out["data"][0]["b64_json"] == "X"
        call = mock_httpx_client.post.call_args
        assert call.args[0].endswith("/v1/images/edits")
        files = call.kwargs["files"]
        assert "image" in files and "mask" in files

    def test_vary_image_multipart(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json({"data": [{"b64_json": "Y"}]})
        out = client.vary_image(image=b"img")
        assert out["data"][0]["b64_json"] == "Y"
        call = mock_httpx_client.post.call_args
        assert call.args[0].endswith("/v1/images/variations")

    def test_upscale_image_multipart(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json({"data": [{"b64_json": "Z"}]})
        client.upscale_image(image=b"img", scale=4)
        call = mock_httpx_client.post.call_args
        assert call.args[0].endswith("/v1/images/upscale")
        assert call.kwargs["data"]["scale"] == "4"

    def test_segment_image_multipart(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json({"masks": []})
        client.segment_image(image=b"img", mode="auto")
        call = mock_httpx_client.post.call_args
        assert call.args[0].endswith("/v1/images/segment")

    def test_vectorize_image_multipart(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json({
            "mime_type": "image/svg+xml", "svg": "<svg/>",
        })
        out = client.vectorize_image(
            image=b"img", model="starvector", seed=7,
        )
        assert out["mime_type"] == "image/svg+xml"
        call = mock_httpx_client.post.call_args
        assert call.args[0].endswith("/v1/images/vectorize")
        assert call.kwargs["data"]["model"] == "starvector"
        assert call.kwargs["data"]["seed"] == "7"
        assert call.kwargs["data"]["response_format"] == "json"

    def test_generate_animation(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json({"data": [{"b64": "..."}]})
        client.generate_animation(prompt="bouncing ball")
        call = mock_httpx_client.post.call_args
        assert call.args[0].endswith("/v1/images/animations")

    def test_generate_video(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json({"data": [{"b64": "..."}]})
        client.generate_video(prompt="a sunset over the ocean")
        call = mock_httpx_client.post.call_args
        assert call.args[0].endswith("/v1/video/generations")

    def test_embed_image(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json(
            {"data": [{"embedding": [0.1, 0.2]}]},
        )
        client.embed_image(input=["data:image/png;base64,AAAA"])
        call = mock_httpx_client.post.call_args
        assert call.args[0].endswith("/v1/images/embeddings")


class TestAudioRoutes:
    def test_speak_returns_bytes(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_bytes(b"WAVE")
        out = client.speak(input="hello", model="kokoro-82m")
        assert out == b"WAVE"

    def test_generate_music_returns_bytes(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_bytes(b"WAVE")
        out = client.generate_music(prompt="ambient pad")
        assert out == b"WAVE"

    def test_generate_sfx_returns_bytes(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_bytes(b"WAVE")
        out = client.generate_sfx(prompt="thunder")
        assert out == b"WAVE"

    def test_transcribe_multipart(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json({"text": "hello world"})
        out = client.transcribe(audio=b"riff", model="whisper-tiny", language="en")
        assert out["text"] == "hello world"
        call = mock_httpx_client.post.call_args
        assert call.args[0].endswith("/v1/audio/transcriptions")
        assert "file" in call.kwargs["files"]

    def test_embed_audio_multipart(self, client, mock_httpx_client):
        mock_httpx_client.post.return_value = _ok_json(
            {"data": [{"embedding": [0.1]}]},
        )
        client.embed_audio(audio=b"riff", model="mert-v1")
        call = mock_httpx_client.post.call_args
        assert call.args[0].endswith("/v1/audio/embeddings")


class TestServerUrlResolution:
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MUSE_SERVER", "http://env-host:8080")
        c = MuseClient(server_url="http://ctor-host:9000")
        assert c.server_url == "http://ctor-host:9000"

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("MUSE_SERVER", "http://env-host:8080")
        c = MuseClient()
        assert c.server_url == "http://env-host:8080"

    def test_default_is_localhost(self, monkeypatch):
        monkeypatch.delenv("MUSE_SERVER", raising=False)
        c = MuseClient()
        assert c.server_url == "http://localhost:8000"

    def test_admin_token_passes_through(self, monkeypatch):
        monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
        c = MuseClient(server_url="http://x", admin_token="t")
        assert c.admin.token == "t"


class TestSearchModels:
    def test_search_uses_resolver(self, client, monkeypatch):
        from types import SimpleNamespace

        def fake_search(query, *, backend, modality, limit):  # noqa: ARG001
            return [
                SimpleNamespace(
                    uri="hf://x/y@q4",
                    modality="chat/completion",
                    size_gb=1.2,
                    downloads=5,
                    license="apache",
                    description="d",
                ),
            ]
        monkeypatch.setattr(
            "muse.core.resolvers.search", fake_search,
        )
        # also patch resolvers_hf import side-effect to a no-op
        monkeypatch.setattr(
            "muse.core.resolvers_hf.HFResolver", object, raising=False,
        )
        out = client.search_models(query="qwen", modality="chat/completion")
        assert out["results"][0]["uri"] == "hf://x/y@q4"

    def test_search_filters_by_max_size(self, client, monkeypatch):
        from types import SimpleNamespace

        def fake_search(query, *, backend, modality, limit):  # noqa: ARG001
            return [
                SimpleNamespace(uri="a", modality="x", size_gb=0.5,
                                downloads=1, license=None, description=""),
                SimpleNamespace(uri="b", modality="x", size_gb=10.0,
                                downloads=1, license=None, description=""),
                SimpleNamespace(uri="c", modality="x", size_gb=None,
                                downloads=1, license=None, description=""),
            ]
        monkeypatch.setattr("muse.core.resolvers.search", fake_search)
        out = client.search_models(query="q", max_size_gb=1.0)
        uris = [r["uri"] for r in out["results"]]
        assert "a" in uris and "c" in uris
        assert "b" not in uris  # filtered out by size cap
