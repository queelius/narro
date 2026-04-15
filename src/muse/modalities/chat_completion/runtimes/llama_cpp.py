"""LlamaCppModel: GGUF chat-completion runtime via llama-cpp-python.

One runtime, many models: a GGUF-addressable model's MANIFEST supplies
`gguf_file`, `chat_template` (optional), `context_length` (optional),
and any capabilities. The class is named `Model` when aliased from a
model script; `LlamaCppModel` here for direct import. Tests patch
`Llama` at module level via `unittest.mock.patch`.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatResult,
)


logger = logging.getLogger(__name__)

# Module-level sentinel; real import deferred so discovery tolerates
# missing deps. Tests patch this attribute directly via the module path.
Llama: Any = None


def _ensure_deps() -> None:
    global Llama
    if Llama is not None:
        return
    try:
        from llama_cpp import Llama as _Llama
        Llama = _Llama
    except Exception as e:  # noqa: BLE001
        logger.debug("llama-cpp-python unavailable: %s", e)


class LlamaCppModel:
    """Chat-completion backend wrapping llama_cpp.Llama.

    Constructor kwargs mirror MANIFEST capability keys:
      - gguf_file (required): filename inside local_dir
      - context_length (default 8192)
      - chat_template (default None; uses template embedded in GGUF)
      - n_gpu_layers (default -1; offload everything GPU can fit)
      - additional llama.cpp options absorbed by **_
    """

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        gguf_file: str,
        context_length: int = 8192,
        chat_format: str | None = None,
        chat_template: str | None = None,  # deprecated alias for chat_format
        n_gpu_layers: int = -1,
        device: str = "auto",
        supports_tools: bool | None = None,  # consumed; not forwarded
        **_: Any,
    ) -> None:
        _ensure_deps()
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed; run `muse pull` or "
                "install `llama-cpp-python` into this venv"
            )
        self.model_id = model_id
        # `chat_format` is the canonical kwarg (matches llama-cpp-python's API);
        # `chat_template` is kept as a deprecated alias for back-compat with
        # earlier muse manifests that used the wrong name.
        effective_chat_format = chat_format if chat_format is not None else chat_template
        # Stash for the chat-time tools warning (see routes.py).
        self.supports_tools = supports_tools
        if not local_dir:
            raise RuntimeError("local_dir is required; the GGUF file must be on disk")
        base = Path(local_dir)
        gguf_path = base / gguf_file
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
        logger.info(
            "loading GGUF %s (ctx=%d, gpu_layers=%d, chat_format=%s)",
            gguf_path, context_length, n_gpu_layers, effective_chat_format,
        )
        self._llama = Llama(
            model_path=str(gguf_path),
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
            chat_format=effective_chat_format,
            verbose=False,
        )

    def chat(self, messages: list[dict], **kwargs) -> ChatResult:
        resp = self._llama.create_chat_completion(
            messages=messages,
            stream=False,
            **_filter_kwargs(kwargs),
        )
        return _dict_to_chat_result(resp, model_id=self.model_id)

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[ChatChunk]:
        for chunk in self._llama.create_chat_completion(
            messages=messages,
            stream=True,
            **_filter_kwargs(kwargs),
        ):
            yield _dict_to_chat_chunk(chunk, model_id=self.model_id)


_FORWARDED_KWARGS = frozenset({
    "temperature", "top_p", "max_tokens", "stop", "seed",
    "tools", "tool_choice", "response_format", "logprobs",
    "top_logprobs", "frequency_penalty", "presence_penalty",
    "repeat_penalty", "grammar", "logit_bias",
})


def _filter_kwargs(kwargs: dict) -> dict:
    """Only forward kwargs llama-cpp-python understands; silently drop the rest."""
    return {k: v for k, v in kwargs.items() if k in _FORWARDED_KWARGS}


def _dict_to_chat_result(resp: dict, *, model_id: str) -> ChatResult:
    """Translate llama-cpp's OpenAI-shape response to our ChatResult.

    `model_id` is authoritative: llama-cpp sets the response's `model`
    field to the GGUF filesystem path (e.g. /home/.../model.gguf),
    which is useless to clients. We always override with our catalog id.
    """
    return ChatResult(
        id=resp.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model_id=model_id,
        created=resp.get("created") or int(time.time()),
        choices=[
            ChatChoice(
                index=c["index"],
                message=c["message"],
                finish_reason=c.get("finish_reason"),
            )
            for c in resp.get("choices", [])
        ],
        usage=resp.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


def _dict_to_chat_chunk(chunk: dict, *, model_id: str) -> ChatChunk:
    """Translate a llama-cpp stream chunk to our ChatChunk.

    Same model_id-override rationale as _dict_to_chat_result.
    """
    choices = chunk.get("choices") or []
    c = choices[0] if choices else {"index": 0, "delta": {}, "finish_reason": None}
    return ChatChunk(
        id=chunk.get("id") or "chatcmpl-stream",
        model_id=model_id,
        created=chunk.get("created") or int(time.time()),
        choice_index=c.get("index", 0),
        delta=c.get("delta") or {},
        finish_reason=c.get("finish_reason"),
    )
