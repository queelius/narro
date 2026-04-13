"""Soprano TTS model backend.

Wraps the Soprano-1.1-80M model (Qwen3 LLM + Vocos decoder) to
implement the :class:`~muse.audio.speech.protocol.TTSModel` protocol.

The encode/decode split, hidden-state IR (``.soprano`` files), and
Vocos decoder are all internal to this backend -- the server and
clients never see them.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from muse.audio.speech.protocol import AudioChunk, AudioResult

logger = logging.getLogger(__name__)


class SopranoModel:
    """Soprano-1.1-80M TTS backend.

    Args:
        hf_repo: HuggingFace repo id (used when local_dir is not set).
        local_dir: Local model directory (takes precedence over hf_repo).
        compile: Enable torch.compile optimisation.
        quantize: Enable INT8 quantization (faster, lower quality).
        decoder_batch_size: Batch size for Vocos decoder.
        num_threads: CPU thread count (None = auto).
        device: ``'auto'``, ``'cpu'``, ``'cuda'``, or ``'mps'``.
    """

    MODEL_ID = "soprano-80m"

    def __init__(
        self,
        *,
        hf_repo: str = "ekwek/Soprano-1.1-80M",
        local_dir: str | None = None,
        compile: bool = True,
        quantize: bool = False,
        decoder_batch_size: int = 4,
        num_threads: int | None = None,
        device: str = "auto",
        **_: Any,
    ) -> None:
        from muse.audio.speech.tts import Narro, SAMPLE_RATE

        self._tts = Narro(
            model_path=local_dir or hf_repo,
            compile=compile,
            quantize=quantize,
            decoder_batch_size=decoder_batch_size,
            num_threads=num_threads,
            device=device,
        )
        self._sample_rate = SAMPLE_RATE

    # -- TTSModel protocol -------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str, **kwargs) -> AudioResult:
        """Synthesize speech from *text*.

        Soprano-specific kwargs:
            top_p, temperature, repetition_penalty, retries
            align (bool): include paragraph-level alignment in metadata.
        """
        align = kwargs.pop("align", False)
        gen_kwargs = _pick(kwargs, ("top_p", "temperature", "repetition_penalty", "retries"))

        if not align:
            audio_tensor = self._tts.infer(text, **gen_kwargs)
            return AudioResult(
                audio=audio_tensor.cpu().numpy().astype(np.float32),
                sample_rate=self._sample_rate,
            )

        from muse.audio.speech.alignment import extract_paragraph_alignment
        import torch

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        encoded = self._tts.encode_batch(paragraphs, **gen_kwargs)
        audio_list = self._tts.decode(encoded)
        audio = torch.cat(audio_list) if audio_list else torch.zeros(0)

        return AudioResult(
            audio=audio.cpu().numpy().astype(np.float32),
            sample_rate=self._sample_rate,
            metadata={"alignment": extract_paragraph_alignment(encoded)},
        )

    def synthesize_stream(self, text: str, **kwargs) -> Iterator[AudioChunk]:
        """Streaming synthesis -- yields audio chunks as they are decoded."""
        gen_kwargs = _pick(kwargs, ("top_p", "temperature", "repetition_penalty", "chunk_size"))

        for chunk_tensor in self._tts.infer_stream(text, **gen_kwargs):
            yield AudioChunk(
                audio=chunk_tensor.cpu().numpy().astype(np.float32),
                sample_rate=self._sample_rate,
            )


def _pick(src: dict, keys: tuple[str, ...]) -> dict:
    """Return the subset of *src* whose keys are in *keys*."""
    return {k: v for k, v in src.items() if k in keys}
