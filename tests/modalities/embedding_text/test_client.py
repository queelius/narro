"""Tests for EmbeddingsClient HTTP client."""
import base64
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from muse.modalities.embedding_text.client import EmbeddingsClient


def _make_response(embeddings, encoding_format="float", model="fake-embed", tokens=3):
    """Build a fake OpenAI-shape response body."""
    data = []
    for i, vec in enumerate(embeddings):
        if encoding_format == "base64":
            arr = np.asarray(vec, dtype="<f4")
            embedding_field = base64.b64encode(arr.tobytes()).decode()
        else:
            embedding_field = vec
        data.append({"object": "embedding", "embedding": embedding_field, "index": i})
    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


def test_default_server_url():
    c = EmbeddingsClient()
    assert c.server_url == "http://localhost:8000"


def test_trailing_slash_stripped():
    c = EmbeddingsClient(server_url="http://lan:8000/")
    assert c.server_url == "http://lan:8000"


def test_muse_server_env_fallback(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://custom:9999")
    c = EmbeddingsClient()
    assert c.server_url == "http://custom:9999"


def test_embed_single_string_returns_list_of_vectors():
    fake_body = _make_response([[0.1, 0.2, 0.3]])
    with patch("muse.modalities.embedding_text.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: fake_body)
        c = EmbeddingsClient()
        vectors = c.embed("hello")
        assert len(vectors) == 1
        assert vectors[0] == [0.1, 0.2, 0.3]
        body = mock_post.call_args.kwargs["json"]
        assert body["input"] == "hello"
        assert body["encoding_format"] == "float"


def test_embed_list_input_returns_list_of_vectors():
    fake_body = _make_response([[0.1], [0.2], [0.3]])
    with patch("muse.modalities.embedding_text.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: fake_body)
        c = EmbeddingsClient()
        vectors = c.embed(["a", "b", "c"])
        assert len(vectors) == 3
        body = mock_post.call_args.kwargs["json"]
        assert body["input"] == ["a", "b", "c"]


def test_embed_base64_decodes_to_floats():
    original = [1.0, 2.0, 3.0, 4.0]
    fake_body = _make_response([original], encoding_format="base64")
    with patch("muse.modalities.embedding_text.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: fake_body)
        c = EmbeddingsClient()
        # Client requests base64 from the server, decodes to floats for caller
        vectors = c.embed("hi", encoding_format="base64")
        assert len(vectors) == 1
        for a, b in zip(vectors[0], original):
            assert abs(a - b) < 1e-6


def test_embed_sends_optional_fields_when_provided():
    fake_body = _make_response([[0.1]])
    with patch("muse.modalities.embedding_text.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: fake_body)
        c = EmbeddingsClient()
        c.embed("hi", model="all-minilm-l6-v2", dimensions=128)
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "all-minilm-l6-v2"
        assert body["dimensions"] == 128


def test_embed_omits_none_optional_fields():
    fake_body = _make_response([[0.1]])
    with patch("muse.modalities.embedding_text.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: fake_body)
        c = EmbeddingsClient()
        c.embed("hi")
        body = mock_post.call_args.kwargs["json"]
        for field in ("model", "dimensions"):
            assert field not in body


def test_embed_raises_on_http_error():
    with patch("muse.modalities.embedding_text.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=500, text="boom")
        c = EmbeddingsClient()
        with pytest.raises(RuntimeError, match="500"):
            c.embed("x")


def test_embed_rejects_invalid_encoding_format():
    c = EmbeddingsClient()
    with pytest.raises(ValueError, match="encoding_format"):
        c.embed("hi", encoding_format="bogus")
