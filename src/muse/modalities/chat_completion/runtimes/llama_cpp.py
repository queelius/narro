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
        chat_template: str | None = None,
        n_gpu_layers: int = -1,
        device: str = "auto",
        **_: Any,
    ) -> None:
        _ensure_deps()
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed; run `muse pull` or "
                "install `llama-cpp-python` into this venv"
            )
        self.model_id = model_id
        if not local_dir:
            raise RuntimeError("local_dir is required; the GGUF file must be on disk")
        base = Path(local_dir)
        gguf_path = base / gguf_file
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
        logger.info(
            "loading GGUF %s (ctx=%d, gpu_layers=%d, chat_template=%s)",
            gguf_path, context_length, n_gpu_layers, chat_template,
        )
        self._llama = Llama(
            model_path=str(gguf_path),
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_template,
            verbose=False,
        )

    def chat(self, messages: list[dict], **kwargs) -> ChatResult:
        resp = self._llama.create_chat_completion(
            messages=messages,
            stream=False,
            **_filter_kwargs(kwargs),
        )
        return _dict_to_chat_result(resp, fallback_model_id=self.model_id)

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[ChatChunk]:
        for chunk in self._llama.create_chat_completion(
            messages=messages,
            stream=True,
            **_filter_kwargs(kwargs),
        ):
            yield _dict_to_chat_chunk(chunk, fallback_model_id=self.model_id)


_FORWARDED_KWARGS = frozenset({
    "temperature", "top_p", "max_tokens", "stop", "seed",
    "tools", "tool_choice", "response_format", "logprobs",
    "top_logprobs", "frequency_penalty", "presence_penalty",
    "repeat_penalty", "grammar", "logit_bias",
})


def _filter_kwargs(kwargs: dict) -> dict:
    """Only forward kwargs llama-cpp-python understands; silently drop the rest."""
    return {k: v for k, v in kwargs.items() if k in _FORWARDED_KWARGS}


def _dict_to_chat_result(resp: dict, *, fallback_model_id: str) -> ChatResult:
    return ChatResult(
        id=resp.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model_id=resp.get("model") or fallback_model_id,
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


def _dict_to_chat_chunk(chunk: dict, *, fallback_model_id: str) -> ChatChunk:
    choices = chunk.get("choices") or []
    c = choices[0] if choices else {"index": 0, "delta": {}, "finish_reason": None}
    return ChatChunk(
        id=chunk.get("id") or "chatcmpl-stream",
        model_id=chunk.get("model") or fallback_model_id,
        created=chunk.get("created") or int(time.time()),
        choice_index=c.get("index", 0),
        delta=c.get("delta") or {},
        finish_reason=c.get("finish_reason"),
    )
