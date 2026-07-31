"""MuseClient: thin httpx wrapper aggregating muse's HTTP routes.

Wraps /v1/chat/completions, /v1/audio/*, /v1/images/*, /v1/embeddings,
/v1/rerank, /v1/summarize, /v1/moderations, /v1/translate, /health,
/v1/models, plus delegates admin operations to ``AdminClient``.

Each method returns either parsed JSON (for non-binary modalities) or
raw bytes (for /v1/audio/speech, /v1/audio/music, /v1/audio/sfx).
Image bytes come back as base64 strings inside the b64_json field of
/v1/images/* JSON responses; the MCP layer unwraps once when packing
into MCP ImageContent.

This client is intentionally thin: the MCP server stays stateless and
imports muse's per-modality client modules lazily on demand.
"""
from __future__ import annotations

from typing import Any

import httpx

from muse.admin.client import AdminClient
from muse.core import config


class MuseClient:
    """HTTP aggregator over muse's /v1/* surface.

    Construct with the muse server URL; each method maps to one route.
    Tests inject an instance with mocked httpx, or stub specific
    methods with ``unittest.mock``.
    """

    def __init__(
        self,
        server_url: str | None = None,
        admin_token: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.server_url = (server_url or config.get("client.server_url")).rstrip("/")
        self.admin_token = admin_token or config.get("admin.token")
        self.timeout = timeout
        self.admin = AdminClient(
            base_url=self.server_url,
            token=self.admin_token,
            timeout=30.0,
        )

    # Probes

    def health(self) -> dict:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(f"{self.server_url}/health")
        r.raise_for_status()
        return r.json()

    def list_models(self) -> dict:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(f"{self.server_url}/v1/models")
        r.raise_for_status()
        return r.json()

    # JSON in / JSON out

    def _post_json(self, path: str, body: dict) -> dict:
        cleaned = {k: v for k, v in body.items() if v is not None}
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(f"{self.server_url}{path}", json=cleaned)
        r.raise_for_status()
        return r.json()

    def _post_bytes(self, path: str, body: dict) -> bytes:
        cleaned = {k: v for k, v in body.items() if v is not None}
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(f"{self.server_url}{path}", json=cleaned)
        r.raise_for_status()
        return r.content

    def _post_multipart(
        self,
        path: str,
        *,
        files: dict,
        data: dict | None = None,
    ) -> Any:
        """POST multipart/form-data; return JSON if response is JSON, else bytes."""
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(f"{self.server_url}{path}", files=files, data=data or {})
        r.raise_for_status()
        ct = r.headers.get("content-type", "").lower()
        if ct.startswith("application/json"):
            return r.json()
        return r.content

    # Text routes

    def chat(self, **body: Any) -> dict:
        return self._post_json("/v1/chat/completions", body)

    def summarize(self, **body: Any) -> dict:
        return self._post_json("/v1/summarize", body)

    def rerank(self, **body: Any) -> dict:
        return self._post_json("/v1/rerank", body)

    def classify(self, **body: Any) -> dict:
        return self._post_json("/v1/moderations", body)

    def embed_text(self, **body: Any) -> dict:
        return self._post_json("/v1/embeddings", body)

    def translate(self, **body: Any) -> dict:
        return self._post_json("/v1/translate", body)

    # Image routes

    def embed_image(self, **body: Any) -> dict:
        return self._post_json("/v1/images/embeddings", body)

    def generate_image(self, **body: Any) -> dict:
        body.setdefault("response_format", "b64_json")
        return self._post_json("/v1/images/generations", body)

    def edit_image(
        self,
        *,
        image: bytes,
        mask: bytes,
        **body: Any,
    ) -> dict:
        files = {
            "image": ("image.png", image, "image/png"),
            "mask": ("mask.png", mask, "image/png"),
        }
        body.setdefault("response_format", "b64_json")
        data = {k: str(v) for k, v in body.items() if v is not None}
        return self._post_multipart("/v1/images/edits", files=files, data=data)

    def vary_image(self, *, image: bytes, **body: Any) -> dict:
        files = {"image": ("image.png", image, "image/png")}
        body.setdefault("response_format", "b64_json")
        data = {k: str(v) for k, v in body.items() if v is not None}
        return self._post_multipart("/v1/images/variations", files=files, data=data)

    def upscale_image(self, *, image: bytes, **body: Any) -> dict:
        files = {"image": ("image.png", image, "image/png")}
        body.setdefault("response_format", "b64_json")
        data = {k: str(v) for k, v in body.items() if v is not None}
        return self._post_multipart("/v1/images/upscale", files=files, data=data)

    def segment_image(self, *, image: bytes, **body: Any) -> dict:
        files = {"image": ("image.png", image, "image/png")}
        data = {k: str(v) for k, v in body.items() if v is not None}
        return self._post_multipart("/v1/images/segment", files=files, data=data)

    def vectorize_image(self, *, image: bytes, **body: Any) -> dict:
        files = {"image": ("image.png", image, "image/png")}
        body.setdefault("response_format", "json")
        data = {k: str(v) for k, v in body.items() if v is not None}
        return self._post_multipart(
            "/v1/images/vectorize", files=files, data=data,
        )

    def generate_animation(self, **body: Any) -> dict:
        return self._post_json("/v1/images/animations", body)

    def generate_video(self, **body: Any) -> dict:
        return self._post_json("/v1/video/generations", body)

    # Audio routes

    def generate_music(self, **body: Any) -> bytes:
        return self._post_bytes("/v1/audio/music", body)

    def generate_sfx(self, **body: Any) -> bytes:
        return self._post_bytes("/v1/audio/sfx", body)

    def speak(self, **body: Any) -> bytes:
        return self._post_bytes("/v1/audio/speech", body)

    def transcribe(
        self,
        *,
        audio: bytes,
        filename: str = "audio.wav",
        **body: Any,
    ) -> dict:
        files = {"file": (filename, audio, "application/octet-stream")}
        data = {k: str(v) for k, v in body.items() if v is not None}
        return self._post_multipart(
            "/v1/audio/transcriptions", files=files, data=data,
        )

    def embed_audio(
        self,
        *,
        audio: bytes,
        filename: str = "audio.wav",
        **body: Any,
    ) -> dict:
        files = {"file": (filename, audio, "application/octet-stream")}
        data = {k: str(v) for k, v in body.items() if v is not None}
        return self._post_multipart(
            "/v1/audio/embeddings", files=files, data=data,
        )

    # Resolver search (admin-style discovery)

    def search_models(
        self,
        *,
        query: str,
        modality: str | None = None,
        max_size_gb: float | None = None,
        limit: int = 20,
    ) -> dict:
        """Run resolver.search() locally and return rows as plain dicts.

        Pinning resolver search to local execution (rather than going
        through muse) keeps this useful when the LLM is connected to a
        muse server that doesn't yet have the desired model. The HF
        resolver registers itself on import.
        """
        import muse.core.resolvers_hf  # noqa: F401  (registers HFResolver)
        from muse.core.resolvers import search as _resolver_search

        rows = list(_resolver_search(
            query, backend=None, modality=modality, limit=limit,
        ))
        if max_size_gb is not None:
            rows = [r for r in rows if r.size_gb is None or r.size_gb <= max_size_gb]
        return {
            "results": [
                {
                    "uri": r.uri,
                    "modality": r.modality,
                    "size_gb": r.size_gb,
                    "downloads": r.downloads,
                    "license": r.license,
                    "description": r.description,
                }
                for r in rows
            ],
        }
