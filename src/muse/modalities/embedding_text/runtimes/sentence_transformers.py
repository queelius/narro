"""SentenceTransformerModel: generic runtime for any sentence-transformers repo.

One class serves any repo that exposes the sentence-transformers API
(`SentenceTransformer(repo_id).encode(texts)`). Unlike a scripted model
where the class is paired 1-to-1 with a specific HF repo, this runtime
takes `model_id` as a constructor kwarg and can back any of thousands
of HF sentence-transformers repos.

Pulled via the HF resolver (muse.core.resolvers_hf): `muse pull
hf://sentence-transformers/all-MiniLM-L6-v2` synthesizes a manifest
that references this class by path, and `load_backend()` instantiates
it with the manifest's capabilities merged into kwargs.

Dimensions auto-detected from the loaded model via
`get_sentence_embedding_dimension()`. trust_remote_code is forwarded
for repos that ship custom architectures (Qwen3-Embedding, Nomic,
Instruct variants).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from muse.core.runtime_helpers import select_device
from muse.modalities.embedding_text import EmbeddingResult


logger = logging.getLogger(__name__)

# Deferred imports: discovery must tolerate torch / sentence-transformers
# being absent OR broken on the supervisor env. Sentinels stay None until
# `_ensure_deps()` runs inside Model.__init__; tests that patch these
# module attrs see their mocks preserved.
torch: Any = None
SentenceTransformer: Any = None


class _PinnedCodeRevisionKwargs(dict[str, Any]):
    """Keep a code pin when SentenceTransformers probes module classes.

    SentenceTransformers 5 consumes ``model_kwargs['code_revision']`` with
    ``pop`` while resolving a modules.json class before it constructs the
    underlying Transformers AutoModel.  An external auto_map (Nomic) needs
    the same pin at both boundaries. Returning the value without deleting it
    preserves that invariant and remains an ordinary dict for downstream
    ``**kwargs`` expansion.
    """

    def pop(self, key: str, *default: Any) -> Any:
        if key == "code_revision":
            if key in self:
                return self[key]
            if default:
                return default[0]
            raise KeyError(key)
        return super().pop(key, *default)


def _ensure_deps() -> None:
    """Lazy-import torch + sentence-transformers (per-symbol; test-safe)."""
    global torch, SentenceTransformer
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("SentenceTransformerModel torch unavailable: %s", e)
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _s
            SentenceTransformer = _s
        except Exception as e:  # noqa: BLE001
            logger.debug("SentenceTransformerModel sentence-transformers unavailable: %s", e)


class SentenceTransformerModel:
    """Generic sentence-transformers runtime.

    Constructor kwargs (from a resolver-synthesized manifest's capabilities):
      - model_id (required, passed by load_backend)
      - hf_repo (required, fallback weight source)
      - local_dir (optional, preferred over hf_repo)
      - device ("auto" | "cpu" | "cuda" | "mps")
      - trust_remote_code (default False; set True for reviewed custom models)
      - code_revision (optional immutable commit for an external auto_map repo)
      - other kwargs absorbed by **_
    """

    model_id: str
    dimensions: int

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        trust_remote_code: bool = False,
        code_revision: str | None = None,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed; run `muse pull` or "
                "install `sentence-transformers` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        src = local_dir or hf_repo
        logger.info(
            "loading SentenceTransformer from %s (device=%s, trust_remote_code=%s)",
            src, self._device, trust_remote_code,
        )
        remote_code_kwargs: dict[str, Any] = {}
        if code_revision is not None:
            # Transformers distinguishes the model revision from code hosted
            # in a different repository named by config.auto_map.
            remote_code_kwargs["model_kwargs"] = _PinnedCodeRevisionKwargs(
                code_revision=code_revision,
            )
            remote_code_kwargs["config_kwargs"] = {
                "code_revision": code_revision,
            }
        self._model = SentenceTransformer(
            src,
            device=self._device,
            trust_remote_code=trust_remote_code,
            **remote_code_kwargs,
        )
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

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

        # Optional matryoshka-style truncation + re-normalization.
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
        return sum(sum(row) for row in attn)
    except Exception as e:
        logger.debug("token counting failed: %s", e)
        return 0


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)
