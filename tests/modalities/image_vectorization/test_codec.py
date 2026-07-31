from unittest.mock import patch

from muse.modalities.image_vectorization.codec import encode_vectorization
from muse.modalities.image_vectorization.protocol import VectorizationResult


def test_encode_vectorization_envelope():
    result = VectorizationResult(
        svg='<svg xmlns="http://www.w3.org/2000/svg"/>',
        model_id="starvector",
        source_width=640,
        source_height=480,
        completion_tokens=123,
        seed=7,
        width=100,
        height=50,
        view_box=(0, 0, 100, 50),
        metadata={"num_beams": 2},
    )
    with patch(
        "muse.modalities.image_vectorization.codec.time.time",
        return_value=1234,
    ):
        body = encode_vectorization(result)

    assert body["id"].startswith("vec-")
    assert body["object"] == "image.vectorization"
    assert body["created"] == 1234
    assert body["mime_type"] == "image/svg+xml"
    assert body["source_size"] == [640, 480]
    assert body["view_box"] == [0, 0, 100, 50]
    assert body["usage"] == {"completion_tokens": 123}
    assert body["metadata"] == {"num_beams": 2}
