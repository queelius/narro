"""Bark TTS model backend.

Wraps Suno's Bark model (via HuggingFace transformers) to implement
the :class:`~muse.audio.speech.protocol.TTSModel` protocol.  Supports voice
presets and voice cloning from ``.npz`` history prompts.

Bark generates complete audio in one shot (no native streaming).
``synthesize_stream`` yields a single chunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from muse.audio.speech.backends.base import voices_dir
from muse.audio.speech.protocol import AudioChunk, AudioResult

logger = logging.getLogger(__name__)

BARK_SAMPLE_RATE = 24000

VOICE_PRESETS = [
    f"v2/{lang}_speaker_{i}"
    for lang in ("en", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", "zh")
    for i in range(10)
]


class BarkModel:
    """Bark TTS backend.

    Args:
        model_path: Local model directory (None = download ``suno/bark`` from HF).
        small: Use ``suno/bark-small`` (~600MB) instead of full (~1.5GB).
        device: ``'auto'``, ``'cpu'``, ``'cuda'``, or ``'mps'``.
        compile: Unused (kept for API compatibility with catalog loader).
        quantize: Unused.
    """

    VOICES = VOICE_PRESETS

    @property
    def voices(self) -> list[str]:
        """Lowercase alias so registry / routes see the voice list."""
        return self.VOICES

    def __init__(
        self,
        *,
        hf_repo: str = "suno/bark",
        local_dir: str | None = None,
        small: bool = False,
        device: str = "auto",
        **_: Any,
    ) -> None:
        import torch
        from transformers import AutoProcessor, AutoModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        repo = local_dir or hf_repo
        self._is_small = small or "small" in repo
        dtype = torch.float16 if device != "cpu" else torch.float32

        logger.info("Loading Bark from %s (device=%s, dtype=%s)", repo, device, dtype)
        self._processor = AutoProcessor.from_pretrained(repo)
        self._model = AutoModel.from_pretrained(repo, torch_dtype=dtype).to(device)
        self._device = device

    @property
    def model_id(self) -> str:
        return "bark-small" if self._is_small else "bark"

    @property
    def sample_rate(self) -> int:
        return BARK_SAMPLE_RATE

    def list_voices(self) -> list[str]:
        """Return presets plus any custom cloned voices."""
        custom = self._custom_voice_names()
        return list(VOICE_PRESETS) + sorted(custom)

    def create_voice(self, name: str, npz_path: str) -> None:
        """Import a voice cloning .npz file as a named voice."""
        src = Path(npz_path)
        if not src.exists():
            raise FileNotFoundError(f"Voice file not found: {npz_path}")
        data = np.load(str(src))
        required = {"semantic_prompt", "coarse_prompt", "fine_prompt"}
        if not required.issubset(data.files):
            raise ValueError(
                f"Voice .npz must contain {required}, got {set(data.files)}"
            )
        dest = voices_dir(self.model_id) / f"{name}.npz"
        import shutil
        shutil.copy2(str(src), str(dest))
        logger.info("Saved voice %r to %s", name, dest)

    def synthesize(self, text: str, **kwargs) -> AudioResult:
        """Synthesize speech from *text*.

        Bark-specific kwargs:
            voice (str): preset name or custom voice name.
            text_temp (float): semantic token temperature (default 0.7).
            waveform_temp (float): waveform temperature (default 0.7).
        """
        voice = kwargs.get("voice")
        text_temp = kwargs.get("text_temp", 0.7)
        waveform_temp = kwargs.get("waveform_temp", 0.7)

        voice_preset = self._resolve_voice(voice)

        inputs = self._processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        import warnings
        import torch
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
            warnings.filterwarnings("ignore", message=".*attention_mask.*pad_token_id.*")
            output = self._model.generate(
                **inputs,
                do_sample=True,
                semantic_temperature=text_temp,
                coarse_temperature=waveform_temp,
                fine_temperature=0.5,
            )

        audio = output.cpu().numpy().squeeze().astype(np.float32)
        if audio.ndim == 0:
            audio = np.zeros(0, dtype=np.float32)

        return AudioResult(
            audio=audio,
            sample_rate=BARK_SAMPLE_RATE,
            metadata={"voice": voice} if voice else {},
        )

    def synthesize_stream(self, text: str, **kwargs) -> Iterator[AudioChunk]:
        """Bark has no native streaming; yields one chunk."""
        result = self.synthesize(text, **kwargs)
        yield AudioChunk(audio=result.audio, sample_rate=result.sample_rate)

    def _resolve_voice(self, voice: str | None):
        """Resolve a voice name to a preset string or .npz path."""
        if voice is None:
            return None
        if voice in VOICE_PRESETS or voice.startswith("v2/"):
            return voice
        npz = voices_dir(self.model_id) / f"{voice}.npz"
        if npz.exists():
            return str(npz)
        available = ", ".join(self._custom_voice_names()[:5])
        raise ValueError(
            f"Unknown voice: {voice!r}. Use a preset (v2/en_speaker_0) "
            f"or a custom voice name. Custom voices: {available or '(none)'}"
        )

    def _custom_voice_names(self) -> list[str]:
        vdir = voices_dir(self.model_id)
        return [f.stem for f in vdir.iterdir() if f.suffix == ".npz"]
