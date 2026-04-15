"""Tests for image codec."""
import base64

import numpy as np
import pytest

from muse.images.generations.codec import (
    ImageFormatError,
    to_bytes,
    to_data_url,
    to_pil,
)

PIL = pytest.importorskip("PIL.Image")


def test_to_pil_accepts_pil():
    img = PIL.new("RGB", (10, 10))
    assert to_pil(img) is img


def test_to_pil_accepts_numpy_uint8():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    img = to_pil(arr)
    assert img.size == (10, 10)


def test_to_pil_accepts_numpy_float_in_unit_range():
    arr = np.ones((4, 4, 3), dtype=np.float32) * 0.5
    img = to_pil(arr)
    assert img.size == (4, 4)


def test_to_pil_accepts_numpy_grayscale():
    arr = np.zeros((8, 8), dtype=np.uint8)
    img = to_pil(arr)
    assert img.size == (8, 8)
    assert img.mode == "L"


def test_to_pil_accepts_numpy_rgba():
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    img = to_pil(arr)
    assert img.mode == "RGBA"


def test_to_pil_rejects_1d():
    with pytest.raises(ImageFormatError):
        to_pil(np.zeros((10,), dtype=np.uint8))


def test_to_pil_rejects_unsupported_type():
    with pytest.raises(ImageFormatError):
        to_pil("not an image")


def test_to_bytes_png():
    img = PIL.new("RGB", (4, 4))
    data = to_bytes(img, fmt="png")
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_to_bytes_jpeg():
    img = PIL.new("RGB", (4, 4))
    data = to_bytes(img, fmt="jpeg")
    assert data[:3] == b"\xff\xd8\xff"


def test_to_bytes_jpg_alias_for_jpeg():
    img = PIL.new("RGB", (4, 4))
    data = to_bytes(img, fmt="jpg")
    assert data[:3] == b"\xff\xd8\xff"


def test_to_bytes_rejects_unknown_format():
    img = PIL.new("RGB", (4, 4))
    with pytest.raises(ImageFormatError):
        to_bytes(img, fmt="bmp")


def test_to_data_url_format():
    img = PIL.new("RGB", (4, 4))
    url = to_data_url(img, fmt="png")
    assert url.startswith("data:image/png;base64,")
    payload = url.split(",", 1)[1]
    base64.b64decode(payload)  # no error


def test_to_data_url_jpeg():
    img = PIL.new("RGB", (4, 4))
    url = to_data_url(img, fmt="jpeg")
    assert url.startswith("data:image/jpeg;base64,")
