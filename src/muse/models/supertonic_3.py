"""Supertonic 3 TTS: lightweight on-device multilingual TTS, ONNX (CPU).

Wraps Supertone/supertonic-3 via the `supertonic` package to implement the
audio/speech TTSModel protocol. ~99M params, ONNX Runtime, 31 languages,
fixed preset voice styles. License: OpenRAIL.

SDK API (TTS.get_voice_style + TTS.synthesize), verified at B1:
  from supertonic import TTS
  tts = TTS(auto_download=True)
  style = tts.get_voice_style(voice_name="M1")
  wav, _duration = tts.synthesize(text, voice_style=style, lang="en")
  # wav has shape (1, N); flatten to (N,) before returning.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from muse.modalities.audio_speech import AudioChunk, AudioResult

logger = logging.getLogger(__name__)

# Verified at B1 (Task 1 Step 2):
SUPERTONIC_SAMPLE_RATE = 44100
SUPERTONIC_VOICES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
DEFAULT_VOICE = "M1"

SUPERTONIC_LANGUAGES = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi",
    "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro",
    "ru", "sk", "sl", "sv", "tr", "uk", "vi",
]

MANIFEST = {
    "model_id": "supertonic-3",
    "modality": "audio/speech",
    "hf_repo": "Supertone/supertonic-3",
    "revision": "3cadd1ee6394adea1bd021217a0e650ede09a323",
    "description": "Supertonic 3: lightweight on-device TTS, 31 languages, ONNX (CPU)",
    "license": "OpenRAIL",
    "pip_extras": (
        "supertonic",
        "onnxruntime",
        "numpy",
        "soundfile",
    ),
    "system_packages": (),
    "capabilities": {
        "sample_rate": SUPERTONIC_SAMPLE_RATE,
        "voices": SUPERTONIC_VOICES,
        "languages": SUPERTONIC_LANGUAGES,
        "device": "cpu",
        # ~99M ONNX params; conservative peak-inference estimate.
        "memory_gb": 0.5,
    },
}


class Model:
    """Supertonic 3 TTS backend. Named ``Model`` per muse discovery convention.

    Args:
        hf_repo: HuggingFace repo id (informational; the SDK manages its
            own cache at ~/.cache/supertonic3/).
        local_dir: Accepted for catalog-loader compatibility. Passed to the
            SDK as model_dir so that muse pull can pre-populate the assets.
        device: Ignored; Supertonic 3 is CPU-only (ONNX Runtime).
    """

    MODEL_ID = "supertonic-3"
    VOICES = SUPERTONIC_VOICES

    @property
    def voices(self) -> list[str]:
        return self.VOICES

    def __init__(
        self,
        *,
        hf_repo: str = "Supertone/supertonic-3",
        local_dir: str | None = None,
        device: str = "auto",
        **_: Any,
    ) -> None:
        from supertonic import TTS  # deferred: keep top-level imports ML-free

        # A Muse worker always receives its pulled local directory and must
        # fail closed if that snapshot is incomplete. Keep SDK auto-download
        # only for explicit direct construction without Muse-managed assets.
        logger.info("Loading Supertonic 3 (device=cpu, local_dir=%s)", local_dir)
        self._tts = TTS(model_dir=local_dir, auto_download=local_dir is None)
        self._device = "cpu"

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def sample_rate(self) -> int:
        return SUPERTONIC_SAMPLE_RATE

    def synthesize(self, text: str, **kwargs: Any) -> AudioResult:
        """Synthesize speech from *text*.

        Supertonic-specific kwargs:
            voice (str): preset voice name (default ``DEFAULT_VOICE``).
            lang (str): ISO-639-1 language code (default ``'en'``).

        Unknown kwargs are silently ignored per the TTSModel protocol.
        """
        # `or` (not the dict-default) so an explicit None defaults too: the
        # /v1/audio/speech route declares `voice: str | None = None` and always
        # forwards `voice=req.voice`, so an omitted voice arrives present-but-None
        # (the same trap fixed in kokoro_82m.py, commit d28c82a).
        voice = kwargs.get("voice") or DEFAULT_VOICE
        lang = kwargs.get("lang") or "en"
        style = self._tts.get_voice_style(voice_name=voice)
        # SDK returns (wav, duration); wav has shape (1, N) float32 in [-1, 1].
        # Flatten to (N,) to satisfy the AudioResult contract.
        wav, _duration = self._tts.synthesize(text, voice_style=style, lang=lang)
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        return AudioResult(
            audio=audio,
            sample_rate=SUPERTONIC_SAMPLE_RATE,
            metadata={"voice": voice, "lang": lang},
        )

    def synthesize_stream(self, text: str, **kwargs: Any) -> Iterator[AudioChunk]:
        """Supertonic has no native streaming: yield the full result as one chunk."""
        result = self.synthesize(text, **kwargs)
        yield AudioChunk(audio=result.audio, sample_rate=SUPERTONIC_SAMPLE_RATE)
