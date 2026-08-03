"""m-a-p/MERT-v1-95M (MERT music understanding audio embedder).

Curated default for `audio/embedding`. ~95MB on disk; CPU-friendly.
768-dim embeddings via mean-pool over the time dimension of the
last hidden state. Trained on music with masked acoustic modeling
(MERT v1).

License: MIT.

MERT ships custom model code in the repo, so loading requires
`trust_remote_code=True`; Muse pins and downloads that code with the weights.

Wraps `transformers.AutoModel` + `AutoFeatureExtractor` with librosa
audio decoding; lazy imports so muse pull + muse --help work without
ML deps installed.
"""
from __future__ import annotations

import io
import logging
from typing import Any

from muse.core.runtime_helpers import select_device, set_inference_mode
from muse.modalities.audio_embedding.protocol import AudioEmbeddingResult


logger = logging.getLogger(__name__)


# Sentinels (lazy-import pattern matches dinov2_small, sd_turbo, etc.).
torch: Any = None
AutoModel: Any = None
AutoFeatureExtractor: Any = None
librosa: Any = None


def _ensure_deps() -> None:
    global torch, AutoModel, AutoFeatureExtractor, librosa
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("mert_v1_95m: torch unavailable: %s", e)
    if AutoModel is None:
        try:
            from transformers import (
                AutoFeatureExtractor as _afe,
                AutoModel as _am,
            )
            AutoModel = _am
            AutoFeatureExtractor = _afe
        except Exception as e:  # noqa: BLE001
            logger.debug("mert_v1_95m: transformers unavailable: %s", e)
    if librosa is None:
        try:
            import librosa as _l
            librosa = _l
        except Exception as e:  # noqa: BLE001
            logger.debug("mert_v1_95m: librosa unavailable: %s", e)


MANIFEST = {
    "model_id": "mert-v1-95m",
    "modality": "audio/embedding",
    "hf_repo": "m-a-p/MERT-v1-95M",
    # Reviewed 2026-08-02. Source: official Hugging Face model API.
    "revision": "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
    "description": (
        "MERT v1 95M: music understanding, 768-dim audio embeddings, MIT"
    ),
    "license": "MIT",
    "pip_extras": (
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "librosa>=0.10.0",
        # numpy is pulled by librosa but the runtime imports it directly;
        # listing it keeps the manifest self-describing (#110).
        "numpy",
    ),
    "system_packages": (),
    "capabilities": {
        # CPU-friendly default; "auto" lets the runtime pick GPU when available.
        "device": "auto",
        "dimensions": 768,
        # MERT v1 was trained at 24kHz; resample inputs to that rate.
        "sample_rate": 24000,
        "max_duration_seconds": 60.0,
        "supports_text_embeddings_too": False,
        # MERT ships a custom feature extractor in the repo.
        "trust_remote_code": True,
        # Measured peak inference, MERT v1 95M at fp32, single 1s 24kHz clip.
        "memory_gb": 0.5,
    },
    # MERT ships preprocessor_config.json + safetensors + custom .py for
    # the feature extractor (required by trust_remote_code path).
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "*.py",
        "preprocessor_config.json",
    ],
}


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


def _set_inference_mode(model: Any) -> None:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    set_inference_mode(model)


def _mean_pool_time(outputs: Any) -> Any:
    """Mean-pool last_hidden_state over the time dimension (dim 1).

    last_hidden_state shape is [B, T, H]; mean across T leaves [B, H].
    Kept in a helper so tests can monkeypatch it without going through
    the full forward pass.
    """
    return outputs.last_hidden_state.mean(dim=1)


def _decode_audio(
    raw: bytes,
    *,
    sample_rate: int,
    max_seconds: float,
) -> Any:
    """Decode raw audio bytes into a mono float32 numpy array.

    Resamples to `sample_rate`, mono, truncates to `max_seconds *
    sample_rate` samples to bound memory. librosa's BytesIO path
    handles wav/mp3/flac/ogg/etc. via its underlying audioread /
    soundfile chain.
    """
    import numpy as np

    audio, _ = librosa.load(
        io.BytesIO(raw), sr=sample_rate, mono=True,
    )
    max_samples = int(max_seconds * sample_rate)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio.astype(np.float32)


class Model:
    """mert-v1-95m backend (music understanding audio embedder)."""

    model_id = MANIFEST["model_id"]
    dimensions = 768

    def __init__(
        self,
        *,
        hf_repo: str = MANIFEST["hf_repo"],
        local_dir: str | None = None,
        device: str = "auto",
        sample_rate: int = 24000,
        max_duration_seconds: float = 60.0,
        trust_remote_code: bool = True,
        revision: str = MANIFEST["revision"],
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModel is None or AutoFeatureExtractor is None:
            raise RuntimeError(
                "transformers is not installed; run `muse pull mert-v1-95m` "
                "or install `transformers` into this venv"
            )
        if librosa is None:
            raise RuntimeError(
                "librosa is not installed; run `muse pull mert-v1-95m` "
                "or install `librosa` into this venv"
            )
        self._device = _select_device(device)
        self._sample_rate = int(sample_rate)
        self._max_duration_seconds = float(max_duration_seconds)
        self._trust_remote_code = bool(trust_remote_code)

        src = local_dir or hf_repo
        revision_kwargs = {} if local_dir else {"revision": revision}
        logger.info(
            "loading mert-v1-95m from %s (device=%s, sr=%d)",
            src, self._device, self._sample_rate,
        )
        self._processor = AutoFeatureExtractor.from_pretrained(
            src,
            trust_remote_code=self._trust_remote_code,
            **revision_kwargs,
        )
        self._model = AutoModel.from_pretrained(
            src,
            trust_remote_code=self._trust_remote_code,
            **revision_kwargs,
        )
        self._model = self._model.to(self._device)
        _set_inference_mode(self._model)

    def embed(
        self,
        audio_bytes_list: list[bytes],
    ) -> AudioEmbeddingResult:
        """Embed a list of raw audio byte strings into 768-dim vectors.

        MERT is a music understanding model; embeddings are pooled via
        mean over the time dimension of the last hidden state.
        """
        import numpy as np

        if not isinstance(audio_bytes_list, list):
            audio_bytes_list = [audio_bytes_list]
        n_audio_clips = len(audio_bytes_list)

        decoded = [
            _decode_audio(
                raw,
                sample_rate=self._sample_rate,
                max_seconds=self._max_duration_seconds,
            )
            for raw in audio_bytes_list
        ]

        inputs = self._processor(
            decoded,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
        )
        inputs = _move_to_device(inputs, self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)
        embeddings = _mean_pool_time(outputs)
        arr = embeddings.detach().to("cpu").float().numpy().astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out_dim = int(arr.shape[1])

        return AudioEmbeddingResult(
            embeddings=arr.tolist(),
            dimensions=out_dim,
            model_id=self.model_id,
            n_audio_clips=n_audio_clips,
            metadata={
                "source": "mert",
                "sample_rate_used": self._sample_rate,
            },
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
