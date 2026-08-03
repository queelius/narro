"""CrossEncoderRuntime: generic runtime over any HF cross-encoder reranker.

One class wraps `sentence_transformers.CrossEncoder` for any repo on
HuggingFace that ships a cross-encoder. Pulled via the HF resolver:
`muse pull hf://BAAI/bge-reranker-v2-m3` synthesizes a manifest
pointing at this class.

Deferred imports follow the muse pattern: torch + CrossEncoder stay
as module-top sentinels (None) until _ensure_deps() lazy-imports them.
Tests patch the sentinels directly; _ensure_deps short-circuits on
non-None.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import select_device
from muse.modalities.text_rerank.protocol import RerankResult


logger = logging.getLogger(__name__)


torch: Any = None
CrossEncoder: Any = None


def _ensure_deps() -> None:
    global torch, CrossEncoder
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("CrossEncoderRuntime: torch unavailable: %s", e)
    if CrossEncoder is None:
        try:
            from sentence_transformers import CrossEncoder as _ce
            CrossEncoder = _ce
        except Exception as e:  # noqa: BLE001
            logger.debug("CrossEncoderRuntime: sentence_transformers unavailable: %s", e)


class CrossEncoderRuntime:
    """Generic cross-encoder reranker runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        max_length: int = 512,
        trust_remote_code: bool = False,
        code_revision: str | None = None,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if CrossEncoder is None:
            raise RuntimeError(
                "sentence-transformers is not installed; run `muse pull` "
                "or install `sentence-transformers` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._max_length = max_length

        src = local_dir or hf_repo
        logger.info(
            "loading cross-encoder reranker from %s (device=%s, max_length=%d)",
            src, self._device, max_length,
        )
        remote_code_kwargs: dict[str, Any] = {}
        if code_revision is not None:
            remote_code_kwargs["model_kwargs"] = {
                "code_revision": code_revision,
            }
            remote_code_kwargs["config_kwargs"] = {
                "code_revision": code_revision,
            }
        self._model = CrossEncoder(
            src,
            max_length=max_length,
            device=self._device,
            trust_remote_code=trust_remote_code,
            **remote_code_kwargs,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Score query against each document; return sorted descending.

        Empty `documents` returns []. Otherwise builds [(query, doc), ...]
        pairs, runs `predict`, sorts by score descending, slices to
        `top_n` if set, and emits one RerankResult per surviving row.
        """
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


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)
