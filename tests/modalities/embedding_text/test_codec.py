"""Tests for embeddings base64 codec."""
import base64
import struct

import pytest

from muse.modalities.embedding_text.codec import embedding_to_base64, base64_to_embedding


def test_round_trip_preserves_values_approximately():
    original = [0.1, -0.5, 3.14159, 0.0, 1.0]
    encoded = embedding_to_base64(original)
    decoded = base64_to_embedding(encoded)
    assert len(decoded) == len(original)
    # float32 round-trip is lossy; use approx equality
    for a, b in zip(original, decoded):
        assert abs(a - b) < 1e-6


def test_format_is_little_endian_float32():
    """OpenAI's format is little-endian float32 bytes -> base64.

    The openai-python SDK decodes with `np.frombuffer(bytes, dtype='<f4')`
    so our encoding must match that exactly.
    """
    vec = [1.0, 2.0]
    encoded = embedding_to_base64(vec)
    raw = base64.b64decode(encoded)
    # Two little-endian float32 values = 8 bytes
    assert len(raw) == 8
    # Unpack with struct to confirm byte order
    v0 = struct.unpack("<f", raw[0:4])[0]
    v1 = struct.unpack("<f", raw[4:8])[0]
    assert abs(v0 - 1.0) < 1e-6
    assert abs(v1 - 2.0) < 1e-6


def test_empty_embedding_encodes_to_empty_base64():
    assert embedding_to_base64([]) == ""
    assert base64_to_embedding("") == []


def test_large_embedding_round_trips():
    # 384-dim (MiniLM size) vector
    original = [i * 0.01 for i in range(384)]
    encoded = embedding_to_base64(original)
    decoded = base64_to_embedding(encoded)
    assert len(decoded) == 384
    for a, b in zip(original, decoded):
        assert abs(a - b) < 1e-5


def test_base64_to_embedding_rejects_invalid_length():
    """Raw bytes must be a multiple of 4 (float32 size)."""
    # 5 bytes = not a whole number of float32s
    bad_bytes = base64.b64encode(b"\x00" * 5).decode()
    with pytest.raises(ValueError, match="multiple of 4"):
        base64_to_embedding(bad_bytes)


def test_encoded_is_pure_ascii_base64():
    encoded = embedding_to_base64([1.0, 2.0, 3.0])
    # Should be valid base64 string (ascii-safe)
    encoded.encode("ascii")  # raises if not ascii
    # And should round-trip through base64.b64decode
    base64.b64decode(encoded)
