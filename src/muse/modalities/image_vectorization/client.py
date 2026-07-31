"""HTTP client for ``POST /v1/images/vectorize``."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import requests

from muse.core import config


class VectorizationClient:
    """Small requests-based client for raster-to-SVG conversion."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        timeout: float = 600.0,
    ) -> None:
        self.server_url = (
            server_url or config.get("client.server_url")
        ).rstrip("/")
        self._timeout = timeout

    def vectorize(
        self,
        image: bytes | str | Path | Any,
        *,
        model: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        num_beams: int | None = None,
        seed: int | None = None,
        response_format: str = "json",
    ) -> dict[str, Any] | str:
        """Vectorize bytes, a path, file-like object, or PIL image.

        ``response_format="json"`` returns the full response envelope.
        ``response_format="svg"`` returns the raw SVG string.
        """
        files = {"image": ("image.png", _to_bytes(image), "image/png")}
        fields = {
            "model": model,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "seed": seed,
            "response_format": response_format,
        }
        data = {key: str(value) for key, value in fields.items() if value is not None}
        response = requests.post(
            f"{self.server_url}/v1/images/vectorize",
            files=files,
            data=data,
            timeout=self._timeout,
        )
        response.raise_for_status()
        if response_format == "svg":
            return response.text
        return response.json()


def _to_bytes(image: Any) -> bytes:
    if isinstance(image, bytes):
        return image
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    if hasattr(image, "read"):
        return image.read()
    if hasattr(image, "save"):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    raise TypeError(f"unsupported image type: {type(image).__name__}")
