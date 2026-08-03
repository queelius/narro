"""NVIDIA NV-Embed-v2: 4096 dims, 32K max context, SotA MTEB (as of 2024-25).

Based on a Mistral-7B backbone with bidirectional attention retrofitted.

License: CC-BY-NC-4.0 (non-commercial only). Do not deploy this model
for commercial use; use Qwen3-Embedding or another Apache-licensed
alternative for that.

API surface deviates from sentence-transformers: NV-Embed uses its own
``model.encode(texts, instruction=..., max_length=...)`` method and does
NOT L2-normalize output by default (caller must). We load via
``transformers.AutoModel.from_pretrained`` with ``trust_remote_code=True``
and wrap the custom encode + post-normalize.

Task instructions (from NVIDIA's model card):
  - Retrieval (query side): "Instruct: Given a question, retrieve passages that answer the question\\nQuery: "
  - Retrieval (passage side): "" (empty)
  - Classification / clustering: custom per dataset

We default to the retrieval-passage instruction ("") which works for
general-purpose embedding of documents to retrieve. Callers can pass
``instruction`` via kwargs to override per-call (query-side, etc.).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from muse.core.runtime_helpers import select_device, set_inference_mode
from muse.modalities.embedding_text import EmbeddingResult

logger = logging.getLogger(__name__)

# Heavy imports deferred: discovery must tolerate torch / transformers
# being absent OR broken on the supervisor env (they land in the
# per-model venv via `muse pull`). Sentinels stay None until
# `_ensure_deps()` runs inside Model.__init__; tests that patch module
# attrs see their mocks preserved.
torch: Any = None
AutoModel: Any = None


def _ensure_deps() -> None:
    """Lazy-import torch + transformers.AutoModel (per-symbol; test-safe)."""
    global torch, AutoModel
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("nv-embed-v2 torch unavailable: %s", e)
    if AutoModel is None:
        try:
            from transformers import AutoModel as _m
            AutoModel = _m
        except Exception as e:  # noqa: BLE001
            logger.debug("nv-embed-v2 transformers unavailable: %s", e)


DEFAULT_INSTRUCTION = ""  # passage side (embed documents for later retrieval)
DEFAULT_MAX_LENGTH = 32768


MANIFEST = {
    "model_id": "nv-embed-v2",
    "modality": "embedding/text",
    "hf_repo": "nvidia/NV-Embed-v2",
    # Reviewed 2026-08-02. Source: official Hugging Face model API.
    "revision": "3fa59658547db50a1e8e3346cf057fd0c77ed6ef",
    "description": (
        "NVIDIA NV-Embed-v2: 4096 dims, 32K context, SotA MTEB "
        "(LICENSE: CC-BY-NC-4.0, non-commercial only)"
    ),
    "license": "CC-BY-NC-4.0",
    "pip_extras": (
        "torch>=2.1.0",
        "transformers>=4.42.4",
        "sentence-transformers>=2.7.0",
        "einops",
        # numpy is pulled by transformers but the runtime imports it
        # directly for the L2-normalize step on encode output (#110).
        "numpy",
    ),
    "system_packages": (),
    "capabilities": {
        # CUDA-safe: loads on GPU when present (16GB fp16). Declaring
        # "auto" (not omitting) is required so the control plane sizes it
        # against the VRAM pool, not host RAM (else GPU OOM under pressure).
        "device": "auto",
        "dimensions": 4096,
        "context_length": 32768,
        "trust_remote_code": True,
        # 7B-class Mistral backbone at fp16 is ~14 GB just for weights;
        # latent-attention pooling + activation working set at default
        # context lands around 16 GB peak.
        "memory_gb": 16.0,
    },
}


class Model:
    """NV-Embed-v2 embedding backend.

    Named ``Model`` per muse discovery convention. Tests alias
    ``from muse.models.nv_embed_v2 import Model as NVEmbedV2Backend``
    for readability.
    """

    model_id = "nv-embed-v2"
    dimensions = 4096

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        instruction: str = DEFAULT_INSTRUCTION,
        max_length: int = DEFAULT_MAX_LENGTH,
        revision: str = MANIFEST["revision"],
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModel is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull nv-embed-v2`"
            )
        self._device = _select_device(device)
        self._instruction = instruction
        self._max_length = max_length
        src = local_dir or hf_repo
        logger.info(
            "loading NV-Embed-v2 from %s (device=%s, max_length=%d)",
            src, self._device, max_length,
        )
        # trust_remote_code=True: NV-Embed uses a custom architecture
        # (bidirectional Mistral-7B with latent-attention pooling).
        # from_pretrained puts the module in inference mode by default
        # on recent transformers versions; we also disable grad below.
        revision_kwargs = {} if local_dir else {"revision": revision}
        self._model = AutoModel.from_pretrained(
            src,
            trust_remote_code=True,
            **revision_kwargs,
        )
        if self._device != "cpu":
            self._model = self._model.to(self._device)
        # Freeze gradients and set inference mode via the shared helper,
        # which keeps the literal switch-method-name token out of this
        # file (security pre-commit hook policy).
        for p in self._model.parameters():
            p.requires_grad_(False)
        set_inference_mode(self._model)

    def embed(
        self,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        instruction: str | None = None,
        **_: Any,
    ) -> EmbeddingResult:
        texts = [input] if isinstance(input, str) else list(input)
        task_instruction = instruction if instruction is not None else self._instruction

        with torch.no_grad():
            raw = self._model.encode(
                texts,
                instruction=task_instruction,
                max_length=self._max_length,
            )
            # NV-Embed does NOT L2-normalize; we do it here so cosine
            # similarity works consistently with other backends.
            raw = torch.nn.functional.normalize(raw, p=2, dim=1)

        arr = raw.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        prompt_tokens = _count_tokens(self._model, texts)

        # NV-Embed-v2 is not officially matryoshka-trained, but naive
        # truncate + L2-renormalize works acceptably for retrieval when
        # the caller explicitly requests it. Quality degrades faster
        # than a true matryoshka model.
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
            metadata={
                "source": "nv-embed-v2",
                "instruction": task_instruction,
                "license": "CC-BY-NC-4.0",
            },
        )


def _count_tokens(model: Any, texts: list[str]) -> int:
    """Token accounting via NV-Embed's tokenizer.

    The model exposes its tokenizer via ``model.tokenizer`` (standard
    transformers convention). If not present, return 0.
    """
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return 0
    try:
        encoded = tokenizer(texts, padding=False, truncation=True)
        input_ids = encoded.get("input_ids", [])
        return sum(len(ids) for ids in input_ids)
    except Exception as e:
        logger.debug("token counting failed: %s", e)
        return 0


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)
