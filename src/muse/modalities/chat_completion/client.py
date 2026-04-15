"""HTTP client for /v1/chat/completions. Mirrors the OpenAI SDK shape."""
from __future__ import annotations

import json
import logging
import os
from typing import Iterator

import httpx


logger = logging.getLogger(__name__)


class ChatClient:
    """Client for the chat/completion modality.

    Non-streaming: `chat(model, messages, **kwargs)` returns the full
    OpenAI ChatCompletion dict.

    Streaming: `chat_stream(model, messages, **kwargs)` yields each
    ChatCompletionChunk dict as it arrives, stopping at the [DONE]
    sentinel.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = (
            base_url
            or os.environ.get("MUSE_SERVER")
            or "http://localhost:8000"
        ).rstrip("/")
        self.timeout = timeout

    def chat(self, *, model: str | None = None, messages: list[dict], **kwargs) -> dict:
        body = {"model": model, "messages": messages, "stream": False, **kwargs}
        body = {k: v for k, v in body.items() if v is not None}
        r = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def chat_stream(
        self,
        *,
        model: str | None = None,
        messages: list[dict],
        **kwargs,
    ) -> Iterator[dict]:
        body = {"model": model, "messages": messages, "stream": True, **kwargs}
        body = {k: v for k, v in body.items() if v is not None}
        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=body,
            timeout=self.timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError as e:
                    logger.warning("malformed SSE chunk: %s (%s)", payload, e)
