"""Qwen3-Embedding-0.6B backend.

Qwen/Qwen3-Embedding-0.6B: 1024 dims with matryoshka support, 32K max
context, Apache 2.0. Based on the Qwen3 backbone, drop-in compatible
with sentence-transformers (trust_remote_code required because the
architecture is published inside the HF repo).

Qwen3-Embedding accepts optional "instruction" prefixes for task
specialization (retrieval, classification, etc.), but works fine
without. We forward whatever the caller passes and do not prepend
anything by default (unlike Nomic, which requires a prefix).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from muse.modalities.embedding_text.protocol import EmbeddingResult

logger = logging.getLogger(__name__)

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    SentenceTransformer = None  # type: ignore


class Qwen3Embedding06BBackend:
    model_id = "qwen3-embedding-0.6b"
    dimensions = 1024

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        **_: Any,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed; "
                "run `muse pull qwen3-embedding-0.6b`"
            )
        self._device = _select_device(device)
        src = local_dir or hf_repo
        logger.info("loading Qwen3-Embedding-0.6B from %s (device=%s)", src, self._device)
        # trust_remote_code=True: architecture lives in the HF repo
        self._model = SentenceTransformer(
            src, device=self._device, trust_remote_code=True,
        )

    def embed(
        self,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        **_: Any,
    ) -> EmbeddingResult:
        texts = [input] if isinstance(input, str) else list(input)

        raw = self._model.encode(texts, convert_to_numpy=True)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        prompt_tokens = _count_tokens(self._model, texts)

        # Qwen3-Embedding is matryoshka-trained: truncating dimensions
        # retains most of the quality. Re-normalize to unit length since
        # the full-dim vectors are L2-normalized by sentence-transformers.
        if dimensions is not None and dimensions < arr.shape[1]:
            arr = arr[:, :dimensions]
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms
            out_dim = dimensions
        else:
            out_dim = arr.shape[1]

        return EmbeddingResult(
            embeddings=arr.tolist(),
            dimensions=out_dim,
            model_id=self.model_id,
            prompt_tokens=prompt_tokens,
            metadata={"source": "qwen3-embedding-0.6b"},
        )


def _count_tokens(model: Any, texts: list[str]) -> int:
    """Sum attention_mask entries for prompt-token accounting."""
    try:
        tok = model.tokenize(texts)
        attn = tok.get("attention_mask")
        if attn is None:
            return 0
        if hasattr(attn, "sum"):
            total = attn.sum()
            if hasattr(total, "item"):
                return int(total.item())
            return int(total)
        return sum(sum(row) for row in attn)
    except Exception as e:
        logger.debug("token counting failed: %s", e)
        return 0


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
