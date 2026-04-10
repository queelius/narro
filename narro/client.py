"""HTTP client for a Narro-compatible TTS server.

Codes against the standard API contract -- works with any server that
implements ``/v1/audio/speech`` and ``/v1/models``.
"""

import json
import logging

import requests

logger = logging.getLogger(__name__)


class NarroClient:
    """Thin client for the Narro TTS HTTP API.

    Args:
        server_url: Base URL of the server (e.g. ``http://localhost:8000``).
        model: Default model ID to use for synthesis requests.
    """

    def __init__(self, server_url: str, model: str | None = None):
        self.server_url = server_url.rstrip("/")
        self.model = model

    # -- Probes ------------------------------------------------------------

    def health(self) -> dict:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Cannot reach server at {self.server_url}: {e}"
            ) from e

    def list_models(self) -> list[dict]:
        """Return available models from ``/v1/models``."""
        resp = requests.get(f"{self.server_url}/v1/models", timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", [])

    # -- Synthesis ---------------------------------------------------------

    def _speech(
        self,
        payload: dict,
        out_path: str | None = None,
        model: str | None = None,
    ) -> requests.Response:
        """POST to ``/v1/audio/speech``; optionally write audio to disk."""
        effective_model = model or self.model
        if effective_model:
            payload["model"] = effective_model

        resp = requests.post(
            f"{self.server_url}/v1/audio/speech",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        if out_path:
            with open(out_path, "wb") as f:
                f.write(resp.content)
        return resp

    def infer(
        self,
        text: str,
        out_path: str | None = None,
        response_format: str = "wav",
        model: str | None = None,
    ) -> bytes:
        """Synthesize *text* and return raw audio bytes."""
        resp = self._speech(
            {"input": text, "response_format": response_format},
            out_path=out_path,
            model=model,
        )
        return resp.content

    def generate_with_alignment(
        self,
        paragraphs: list[str],
        out_path: str,
        response_format: str = "wav",
        model: str | None = None,
    ) -> tuple[bytes, list]:
        """Synthesize paragraphs with alignment metadata.

        Returns:
            Tuple of (audio_bytes, alignment_list).
        """
        resp = self._speech(
            {"input": "\n\n".join(paragraphs), "response_format": response_format, "align": True},
            out_path=out_path,
            model=model,
        )
        alignment = (
            json.loads(resp.headers["x-alignment"])
            if "x-alignment" in resp.headers
            else []
        )
        return resp.content, alignment
