"""facebook/dinov2-small (DINOv2 small self-supervised image embedder).

Curated default for `image/embedding`. ~88MB on disk; CPU-friendly.
384-dim embeddings via the CLS token of the last hidden state. No
text alignment (DINOv2 is purely self-supervised on image data).

License: Apache 2.0.

Wraps `transformers.AutoModel` + `AutoImageProcessor`; lazy imports so
muse pull + muse --help work without ML deps installed.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import select_device, set_inference_mode
from muse.modalities.image_embedding.protocol import ImageEmbeddingResult


logger = logging.getLogger(__name__)


# Sentinels (lazy-import pattern matches sd_turbo, soprano_80m, etc.).
torch: Any = None
AutoModel: Any = None
AutoImageProcessor: Any = None


def _ensure_deps() -> None:
    global torch, AutoModel, AutoImageProcessor
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("dinov2_small: torch unavailable: %s", e)
    if AutoModel is None:
        try:
            from transformers import (
                AutoImageProcessor as _aip,
                AutoModel as _am,
            )
            AutoModel = _am
            AutoImageProcessor = _aip
        except Exception as e:  # noqa: BLE001
            logger.debug("dinov2_small: transformers unavailable: %s", e)


MANIFEST = {
    "model_id": "dinov2-small",
    "modality": "image/embedding",
    "hf_repo": "facebook/dinov2-small",
    "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "description": (
        "DINOv2 small: 88MB, 384-dim self-supervised image features, Apache 2.0"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "Pillow>=9.1.0",
        # numpy is pulled by transformers but the runtime imports it
        # directly for embedding postprocessing (#110).
        "numpy",
    ),
    "system_packages": (),
    "capabilities": {
        # CPU-friendly default; "auto" lets the runtime pick GPU when available.
        "device": "auto",
        "dimensions": 384,
        "image_size": 224,
        "supports_text_embeddings_too": False,
        # Measured peak inference, DINOv2-small at fp32, single 224x224 image.
        "memory_gb": 0.4,
    },
    # DINOv2 ships a preprocessor_config.json + safetensors. The wide
    # *.json + *.txt patterns cover the full tokenizer/preprocessor set
    # without our having to enumerate every file.
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "preprocessor_config.json",
    ],
}


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


def _set_inference_mode(model: Any) -> None:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    set_inference_mode(model)


def _extract_cls_token(outputs: Any) -> Any:
    """DINOv2's CLS token is row 0 of last_hidden_state along dim 1.

    Kept in a helper so tests can monkeypatch it without going through
    the full forward pass.
    """
    return outputs.last_hidden_state[:, 0]


class Model:
    """dinov2-small backend (self-supervised image embedder)."""

    model_id = MANIFEST["model_id"]
    dimensions = 384

    def __init__(
        self,
        *,
        hf_repo: str = MANIFEST["hf_repo"],
        local_dir: str | None = None,
        device: str = "auto",
        image_size: int = 224,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModel is None or AutoImageProcessor is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull dinov2-small` "
                "or install `transformers` into this venv"
            )
        self._device = _select_device(device)
        self._image_size = image_size

        src = local_dir or hf_repo
        logger.info(
            "loading dinov2-small from %s (device=%s, image_size=%d)",
            src, self._device, image_size,
        )
        self._processor = AutoImageProcessor.from_pretrained(src)
        self._model = AutoModel.from_pretrained(src)
        self._model = self._model.to(self._device)
        _set_inference_mode(self._model)

    def embed(
        self,
        images: list,
        *,
        dimensions: int | None = None,
    ) -> ImageEmbeddingResult:
        """Embed a list of PIL images into 384-dim vectors.

        DINOv2 is not instruction-tuned for cross-modal alignment;
        embeddings are pure visual features pooled via the CLS token.
        Optional `dimensions` truncates and re-normalizes per-row.
        """
        import numpy as np

        if not isinstance(images, list):
            images = [images]
        n_images = len(images)

        inputs = self._processor(images=images, return_tensors="pt")
        inputs = _move_to_device(inputs, self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)
        embeddings = _extract_cls_token(outputs)
        arr = embeddings.detach().to("cpu").float().numpy().astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if dimensions is not None and dimensions < arr.shape[1]:
            arr = arr[:, :dimensions]
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms
            out_dim = dimensions
        else:
            out_dim = int(arr.shape[1])

        return ImageEmbeddingResult(
            embeddings=arr.tolist(),
            dimensions=out_dim,
            model_id=self.model_id,
            n_images=n_images,
            metadata={"source": "dinov2"},
        )


def _move_to_device(inputs: Any, device: str) -> Any:
    """Best-effort move of a processor's BatchEncoding to a device."""
    to_method = getattr(inputs, "to", None)
    if callable(to_method):
        return to_method(device)
    if isinstance(inputs, dict):
        return {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
    return inputs
