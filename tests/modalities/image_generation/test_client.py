"""Tests for GenerationsClient HTTP client."""
import base64
from unittest.mock import MagicMock, patch

import pytest

from muse.images.generations.client import GenerationsClient


def test_default_base_url():
    c = GenerationsClient()
    assert c.base_url == "http://localhost:8000"


def test_custom_base_url_strips_trailing_slash():
    c = GenerationsClient(base_url="http://lan:8000/")
    assert c.base_url == "http://lan:8000"


def test_muse_server_env_var_used_when_base_url_unset(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://custom-host:9999")
    c = GenerationsClient()
    assert c.base_url == "http://custom-host:9999"


def test_generate_sends_prompt_and_returns_decoded_bytes():
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch("muse.images.generations.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"b64_json": base64.b64encode(fake_png).decode()}]},
        )
        c = GenerationsClient()
        images = c.generate("a cat", n=1)

        assert len(images) == 1
        assert images[0] == fake_png

        body = mock_post.call_args.kwargs["json"]
        assert body["prompt"] == "a cat"
        assert body["response_format"] == "b64_json"
        assert body["n"] == 1


def test_generate_sends_optional_kwargs_when_provided():
    with patch("muse.images.generations.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"b64_json": base64.b64encode(b"x").decode()}]},
        )
        c = GenerationsClient()
        c.generate(
            "a bird",
            model="sd-turbo",
            n=2,
            size="256x256",
            negative_prompt="blurry",
            steps=4,
            guidance=1.5,
            seed=7,
        )
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "sd-turbo"
        assert body["n"] == 2
        assert body["size"] == "256x256"
        assert body["negative_prompt"] == "blurry"
        assert body["steps"] == 4
        assert body["guidance"] == 1.5
        assert body["seed"] == 7


def test_generate_omits_none_optional_fields():
    """Unsupplied optionals must not appear in the request body."""
    with patch("muse.images.generations.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"b64_json": base64.b64encode(b"x").decode()}]},
        )
        c = GenerationsClient()
        c.generate("hi")
        body = mock_post.call_args.kwargs["json"]
        # These were not passed as kwargs — shouldn't leak as null keys
        for field in ("model", "negative_prompt", "steps", "guidance", "seed"):
            assert field not in body


def test_generate_raises_on_http_error():
    with patch("muse.images.generations.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=500, text="boom")
        c = GenerationsClient()
        with pytest.raises(RuntimeError, match="500"):
            c.generate("x")


def test_generate_returns_list_of_bytes_for_n_greater_than_1():
    fake_a = b"\x89PNG" + b"A" * 10
    fake_b = b"\x89PNG" + b"B" * 10
    with patch("muse.images.generations.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [
                {"b64_json": base64.b64encode(fake_a).decode()},
                {"b64_json": base64.b64encode(fake_b).decode()},
            ]},
        )
        c = GenerationsClient()
        images = c.generate("x", n=2)
        assert images == [fake_a, fake_b]
