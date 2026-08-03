"""BAAI bge-reranker-v2-m3 (multilingual cross-encoder reranker).

Curated default for `text/rerank`. ~568MB on disk; works on CPU.
8192-token context window (the m3 lineage). Multilingual.

License: Apache 2.0.

Wraps `sentence_transformers.CrossEncoder`; lazy imports so muse pull
+ muse --help work without ML deps installed.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import select_device
from muse.modalities.text_rerank.protocol import RerankResult


logger = logging.getLogger(__name__)


# Sentinels (lazy-import pattern matches sd_turbo, soprano_80m, etc.).
torch: Any = None
CrossEncoder: Any = None


def _ensure_deps() -> None:
    global torch, CrossEncoder
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("bge_reranker_v2_m3: torch unavailable: %s", e)
    if CrossEncoder is None:
        try:
            from sentence_transformers import CrossEncoder as _ce
            CrossEncoder = _ce
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "bge_reranker_v2_m3: sentence_transformers unavailable: %s", e,
            )


MANIFEST = {
    "model_id": "bge-reranker-v2-m3",
    "modality": "text/rerank",
    "hf_repo": "BAAI/bge-reranker-v2-m3",
    "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "description": (
        "BAAI bge-reranker-v2-m3: multilingual cross-encoder reranker, "
        "8192-token context, ~568MB"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "sentence-transformers>=2.2.0",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "auto",
        "max_length": 8192,
        # Measured peak fp16 inference, query + 32 documents @ 8192 ctx.
        "memory_gb": 1.2,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "tokenizer*", "spiece.model",
    ],
}


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


class Model:
    """bge-reranker-v2-m3 backend (cross-encoder)."""

    model_id = MANIFEST["model_id"]

    def __init__(
        self,
        *,
        hf_repo: str = MANIFEST["hf_repo"],
        local_dir: str | None = None,
        device: str = "auto",
        max_length: int = 8192,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if CrossEncoder is None:
            raise RuntimeError(
                "sentence-transformers is not installed; run "
                "`muse pull bge-reranker-v2-m3` or install "
                "`sentence-transformers` into this venv"
            )
        self._device = _select_device(device)
        self._max_length = max_length

        src = local_dir or hf_repo
        logger.info(
            "loading bge-reranker-v2-m3 from %s (device=%s, max_length=%d)",
            src, self._device, max_length,
        )
        self._model = CrossEncoder(
            src, max_length=max_length, device=self._device,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        scored = sorted(
            [(i, float(s)) for i, s in enumerate(scores)],
            key=lambda kv: kv[1], reverse=True,
        )
        if top_n is not None:
            scored = scored[:top_n]
        return [
            RerankResult(
                index=i,
                relevance_score=s,
                document_text=documents[i],
            )
            for i, s in scored
        ]
