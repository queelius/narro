"""MiniLM sentence embeddings: 384 dims, 22MB, CPU-friendly.

Wraps ``sentence-transformers/all-MiniLM-L6-v2`` to implement the
:class:`~muse.modalities.embedding_text.protocol.EmbeddingsModel` protocol.

MiniLM isn't matryoshka-trained, so naive dimension truncation is lossy
but OpenAI-SDK-compatible; re-normalization is applied after truncation
so the vectors stay unit-length.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from muse.modalities.embedding_text import EmbeddingResult

logger = logging.getLogger(__name__)

# Heavy imports are NOT done at module import time. Discovery must be
# robust to torch / sentence-transformers being absent OR broken on the
# host python (they land in the per-model venv via `muse pull`, not the
# supervisor env). Sentinels stay None until `_ensure_deps()` runs
# inside Model.__init__. Tests that patch these module attrs see their
# mocks preserved: `_ensure_deps` short-circuits on non-None.
torch: Any = None
SentenceTransformer: Any = None


def _ensure_deps() -> None:
    """Lazy-import torch + sentence-transformers (per-symbol; test-safe)."""
    global torch, SentenceTransformer
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("all-minilm-l6-v2 torch unavailable: %s", e)
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _s
            SentenceTransformer = _s
        except Exception as e:  # noqa: BLE001
            logger.debug("all-minilm-l6-v2 sentence-transformers unavailable: %s", e)


MANIFEST = {
    "model_id": "all-minilm-l6-v2",
    "modality": "embedding/text",
    "hf_repo": "sentence-transformers/all-MiniLM-L6-v2",
    "description": "MiniLM sentence embeddings: 384 dims, 22MB, CPU-friendly",
    "license": "Apache 2.0",
    "pip_extras": ("torch>=2.1.0", "sentence-transformers>=2.2.0"),
    "system_packages": (),
    "capabilities": {
        "dimensions": 384,
        "max_sequence_length": 512,
    },
}


class Model:
    """MiniLM embedding backend.

    Named ``Model`` per muse discovery convention. Tests alias
    ``from muse.models.all_minilm_l6_v2 import Model as MiniLMBackend``
    for readability.
    """

    model_id = "all-minilm-l6-v2"
    dimensions = 384

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        **_: Any,
    ) -> None:
        _ensure_deps()
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed; "
                "run `muse pull all-minilm-l6-v2`"
            )
        self._device = _select_device(device)
        src = local_dir or hf_repo
        logger.info("loading MiniLM from %s (device=%s)", src, self._device)
        self._model = SentenceTransformer(src, device=self._device)

    def embed(
        self,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        **_: Any,
    ) -> EmbeddingResult:
        texts = [input] if isinstance(input, str) else list(input)

        # sentence-transformers encode returns np.ndarray (N, dim)
        raw = self._model.encode(texts, convert_to_numpy=True)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Count tokens via the tokenizer's attention_mask (matches OpenAI
        # spirit of "tokens used" without needing their exact tokenizer)
        prompt_tokens = _count_tokens(self._model, texts)

        # Dimensions truncation. MiniLM isn't matryoshka-trained, so
        # naively truncating + re-normalizing is lossy but OpenAI-SDK-
        # compatible. Skip work if dimensions is None or >= full size.
        if dimensions is not None and dimensions < arr.shape[1]:
            arr = arr[:, :dimensions]
            # L2 re-normalize so the vectors remain unit-length
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
            metadata={"source": "sentence-transformers"},
        )


def _count_tokens(model: Any, texts: list[str]) -> int:
    """Sum attention_mask entries to get actual (non-padding) token count."""
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
        # list-of-lists fallback
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
