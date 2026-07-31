import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.image_vectorization import VectorizationClient


def _response(body=None, *, text="", content_type="application/json"):
    response = MagicMock()
    response.headers = {"content-type": content_type}
    response.json.return_value = body
    response.text = text or json.dumps(body)
    return response


def test_client_posts_bytes_and_returns_json():
    body = {"model": "starvector", "svg": "<svg/>"}
    with patch(
        "muse.modalities.image_vectorization.client.requests.post"
    ) as post:
        post.return_value = _response(body)
        result = VectorizationClient("http://muse/").vectorize(
            b"png", model="starvector", seed=7, max_new_tokens=512,
        )
    assert result == body
    assert post.call_args.args[0] == "http://muse/v1/images/vectorize"
    assert post.call_args.kwargs["files"]["image"][1] == b"png"
    assert post.call_args.kwargs["data"]["seed"] == "7"
    assert post.call_args.kwargs["data"]["max_new_tokens"] == "512"


def test_client_raw_svg_response():
    with patch(
        "muse.modalities.image_vectorization.client.requests.post"
    ) as post:
        post.return_value = _response(
            text="<svg/>", content_type="image/svg+xml",
        )
        result = VectorizationClient().vectorize(
            b"png", response_format="svg",
        )
    assert result == "<svg/>"
    post.return_value.raise_for_status.assert_called_once()


def test_client_accepts_path_file_and_pil(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"path-bytes")
    with patch(
        "muse.modalities.image_vectorization.client.requests.post"
    ) as post:
        post.return_value = _response({})
        VectorizationClient().vectorize(path)
        assert post.call_args.kwargs["files"]["image"][1] == b"path-bytes"

        VectorizationClient().vectorize(io.BytesIO(b"file-bytes"))
        assert post.call_args.kwargs["files"]["image"][1] == b"file-bytes"

        class FakePIL:
            def save(self, buffer, format):
                buffer.write(b"pil-bytes")

        VectorizationClient().vectorize(FakePIL())
        assert post.call_args.kwargs["files"]["image"][1] == b"pil-bytes"


def test_client_rejects_unknown_input_type():
    with pytest.raises(TypeError, match="unsupported image type"):
        VectorizationClient().vectorize(123)
