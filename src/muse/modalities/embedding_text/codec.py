"""Base64 encoding for embeddings.

OpenAI's /v1/embeddings with encoding_format='base64' returns
little-endian float32 bytes encoded as base64. Match that format
exactly so openai-python SDK clients round-trip cleanly via
`np.frombuffer(decoded_bytes, dtype='<f4')`.
"""
from __future__ import annotations

import base64

import numpy as np


def embedding_to_base64(embedding: list[float]) -> str:
    """Encode a float vector as little-endian float32 bytes, then base64.

    Empty input yields empty string.
    """
    if not embedding:
        return ""
    arr = np.asarray(embedding, dtype="<f4")  # little-endian float32
    return base64.b64encode(arr.tobytes()).decode("ascii")


def base64_to_embedding(encoded: str) -> list[float]:
    """Decode base64, then little-endian float32 bytes, then list[float].

    Empty input yields empty list. Raises ValueError if the decoded
    byte length is not a multiple of 4 (float32 size).
    """
    if not encoded:
        return []
    raw = base64.b64decode(encoded)
    if len(raw) % 4 != 0:
        raise ValueError(
            f"decoded byte length {len(raw)} is not a multiple of 4 "
            f"(float32 size); data is corrupt"
        )
    arr = np.frombuffer(raw, dtype="<f4")
    return arr.tolist()
