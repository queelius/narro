"""HTTP client for /v1/images/generations."""
from __future__ import annotations

import base64
import os
from typing import Any

import requests


class GenerationsClient:
    """Thin HTTP client against the muse images.generations endpoint."""

    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        base = base_url or os.environ.get("MUSE_SERVER", "http://localhost:8000")
        self.base_url = base.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str = "512x512",
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
    ) -> list[bytes]:
        """Generate n PNG images. Returns raw PNG bytes per image."""
        body: dict[str, Any] = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json",
        }
        if model is not None:
            body["model"] = model
        if negative_prompt is not None:
            body["negative_prompt"] = negative_prompt
        if steps is not None:
            body["steps"] = steps
        if guidance is not None:
            body["guidance"] = guidance
        if seed is not None:
            body["seed"] = seed

        r = requests.post(
            f"{self.base_url}/v1/images/generations",
            json=body,
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")

        data = r.json()["data"]
        return [base64.b64decode(entry["b64_json"]) for entry in data]
