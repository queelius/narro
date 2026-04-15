"""HTTP client for /v1/embeddings.

Mirrors the shape of SpeechClient and GenerationsClient: server_url
constructor param with MUSE_SERVER env fallback, synchronous POST,
returns the essential payload (list[list[float]] of vectors).

Consumers who want the full OpenAI-shape response (with usage, model,
etc.) can POST directly or use the openai-python SDK against muse.
"""
from __future__ import annotations

import os
from typing import Any, Union

import requests

from muse.modalities.embedding_text.codec import base64_to_embedding


class EmbeddingsClient:
    """Thin HTTP client against muse's /v1/embeddings endpoint."""

    def __init__(self, server_url: str | None = None, timeout: float = 60.0) -> None:
        server_url = server_url or os.environ.get(
            "MUSE_SERVER", "http://localhost:8000",
        )
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def embed(
        self,
        input: Union[str, list[str]],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: str = "float",
    ) -> list[list[float]]:
        """Return embeddings as list[list[float]] regardless of wire format.

        If encoding_format='base64' is requested (to save bandwidth on
        the wire), the server returns base64 strings which the client
        decodes back to float lists before returning to the caller.
        """
        if encoding_format not in ("float", "base64"):
            raise ValueError(
                f"encoding_format must be 'float' or 'base64', got {encoding_format!r}"
            )

        body: dict[str, Any] = {
            "input": input,
            "encoding_format": encoding_format,
        }
        if model is not None:
            body["model"] = model
        if dimensions is not None:
            body["dimensions"] = dimensions

        r = requests.post(
            f"{self.server_url}/v1/embeddings",
            json=body, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")

        payload = r.json()
        entries = payload["data"]
        if encoding_format == "base64":
            return [base64_to_embedding(e["embedding"]) for e in entries]
        return [e["embedding"] for e in entries]
