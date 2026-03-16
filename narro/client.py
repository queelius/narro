"""HTTP client for a remote narro TTS server."""
import json
import logging
import requests

logger = logging.getLogger(__name__)


class NarroClient:
    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')

    def health(self):
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError as e:
            raise ConnectionError(f"Cannot reach narro server at {self.server_url}: {e}") from e

    def infer(self, text, out_path=None, response_format='wav'):
        resp = requests.post(
            f"{self.server_url}/v1/audio/speech",
            json={"input": text, "response_format": response_format},
            timeout=300,
        )
        resp.raise_for_status()
        audio_bytes = resp.content
        if out_path:
            with open(out_path, 'wb') as f:
                f.write(audio_bytes)
        return audio_bytes

    def generate_with_alignment(self, paragraphs, out_path, response_format='wav'):
        resp = requests.post(
            f"{self.server_url}/v1/audio/speech",
            json={"input": "\n\n".join(paragraphs), "response_format": response_format, "align": True},
            timeout=300,
        )
        resp.raise_for_status()
        audio_bytes = resp.content
        if out_path:
            with open(out_path, 'wb') as f:
                f.write(audio_bytes)
        alignment = []
        if "x-alignment" in resp.headers:
            alignment = json.loads(resp.headers["x-alignment"])
        return audio_bytes, alignment
