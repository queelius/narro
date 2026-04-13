"""Kokoro TTS model backend.

Wraps hexgrad/Kokoro-82M via the ``kokoro`` package to implement
the :class:`~muse.audio.speech.protocol.TTSModel` protocol.  Lightweight (82M),
fast, with 54 voices across 6 languages. Sentence-level streaming.

Requires: ``pip install kokoro soundfile`` and system ``espeak-ng``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from muse.audio.speech.protocol import AudioChunk, AudioResult

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24000

KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa",
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora",
    "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta",
    "hm_omega", "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
]


class KokoroModel:
    """Kokoro 82M TTS backend.

    Args:
        model_path: Unused (Kokoro loads from HuggingFace repo_id).
        device: ``'auto'``, ``'cpu'``, or ``'cuda'``.
        lang_code: Default language code (``'a'`` = US English).
        compile: Unused.
        quantize: Unused.
    """

    MODEL_ID = "kokoro-82m"
    VOICES = KOKORO_VOICES

    @property
    def voices(self) -> list[str]:
        """Lowercase alias so registry / routes see the voice list."""
        return self.VOICES

    def __init__(
        self,
        *,
        hf_repo: str = "hexgrad/Kokoro-82M",
        local_dir: str | None = None,
        device: str = "auto",
        lang_code: str = "a",
        **_: Any,
    ) -> None:
        import torch
        from kokoro import KPipeline

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading Kokoro (lang=%s, device=%s)", lang_code, device)
        self._pipeline = KPipeline(
            lang_code=lang_code,
            repo_id=local_dir or hf_repo,
            device=device,
        )
        self._device = device

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def sample_rate(self) -> int:
        return KOKORO_SAMPLE_RATE

    def synthesize(self, text: str, **kwargs) -> AudioResult:
        """Synthesize speech from *text*.

        Kokoro-specific kwargs:
            voice (str): voice name (default ``'af_heart'``).
            speed (float): speaking speed multiplier (default 1.0).
        """
        voice = kwargs.get("voice", "af_heart")
        speed = kwargs.get("speed", 1.0)

        chunks = []
        for result in self._pipeline(text, voice=voice, speed=speed):
            if result.audio is not None:
                chunks.append(result.audio.numpy())

        if chunks:
            audio = np.concatenate(chunks).astype(np.float32)
        else:
            audio = np.zeros(0, dtype=np.float32)

        return AudioResult(
            audio=audio,
            sample_rate=KOKORO_SAMPLE_RATE,
            metadata={"voice": voice} if voice else {},
        )

    def synthesize_stream(self, text: str, **kwargs) -> Iterator[AudioChunk]:
        """Sentence-level streaming: yields one chunk per sentence."""
        voice = kwargs.get("voice", "af_heart")
        speed = kwargs.get("speed", 1.0)

        for result in self._pipeline(text, voice=voice, speed=speed):
            if result.audio is not None:
                audio = result.audio.numpy().astype(np.float32)
                yield AudioChunk(audio=audio, sample_rate=KOKORO_SAMPLE_RATE)
