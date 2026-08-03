"""Bark Small TTS: multilingual + voice cloning, 24kHz.

Wraps Suno's Bark model (via HuggingFace transformers) to implement
the :class:`~muse.modalities.audio_speech.protocol.TTSModel` protocol.
Supports voice presets and voice cloning from ``.npz`` history prompts.

Bark generates complete audio in one shot (no native streaming).
``synthesize_stream`` yields a single chunk.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from muse.modalities.audio_speech import AudioChunk, AudioResult

# NOTE: voices_dir lives in audio_speech.backends.base, which has
# unconditional module-top imports of torch + transformers. Importing
# it at muse.models.bark_small module-top breaks discovery in the
# supervisor process (which has no ML deps) with "No module named
# 'transformers'". We defer the import inside _voices_dir() so it only
# fires when Bark is actually instantiated (in its own venv where
# transformers is installed).

logger = logging.getLogger(__name__)


def _voices_dir(model_id: str):
    """Deferred import shim for audio_speech.backends.base.voices_dir."""
    from muse.modalities.audio_speech.backends.base import voices_dir
    return voices_dir(model_id)

BARK_SAMPLE_RATE = 24000

VOICE_PRESETS = [
    f"v2/{lang}_speaker_{i}"
    for lang in ("en", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", "zh")
    for i in range(10)
]


MANIFEST = {
    "model_id": "bark-small",
    "modality": "audio/speech",
    "hf_repo": "suno/bark-small",
    "revision": "1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd",
    "description": "Multilingual TTS with voice cloning, 24kHz",
    "license": "MIT",
    # transformers pulls torch + numpy transitively, but declaring them
    # explicitly keeps the contract self-contained for fresh-venv
    # installs (#110).
    "pip_extras": (
        "torch>=2.1.0",
        "numpy",
        "transformers>=4.36.0",
        "scipy",
    ),
    "system_packages": (),
    "capabilities": {
        # CUDA-safe: loads on GPU when present (~3GB fp16). "auto" (not
        # omitted) so the control plane sizes it against the VRAM pool,
        # not host RAM (else GPU OOM under pressure).
        "device": "auto",
        "sample_rate": BARK_SAMPLE_RATE,
        "voices": VOICE_PRESETS,
        "voice_cloning": True,
        # Bark small is ~600 MB on disk; weights at fp16 plus generation
        # buffers (semantic, coarse, fine) push peak inference to ~3 GB.
        "memory_gb": 3.0,
    },
}


class Model:
    """Bark TTS backend. Named ``Model`` per muse discovery convention.

    Args:
        hf_repo: HuggingFace repo id (default ``suno/bark-small``).
        local_dir: Local model directory (overrides hf_repo if set).
        small: Use ``suno/bark-small`` (~600MB) instead of full (~1.5GB).
        device: ``'auto'``, ``'cpu'``, ``'cuda'``, or ``'mps'``.
    """

    model_id = "bark-small"
    VOICES = VOICE_PRESETS

    @property
    def voices(self) -> list[str]:
        """Lowercase alias so registry / routes see the voice list."""
        return self.VOICES

    def __init__(
        self,
        *,
        hf_repo: str = "suno/bark-small",
        local_dir: str | None = None,
        small: bool = False,
        device: str = "auto",
        **_: Any,
    ) -> None:
        import torch
        from transformers import AutoProcessor, AutoModel

        from muse.core.runtime_helpers import select_device

        device = select_device(device, torch_module=torch)

        repo = local_dir or hf_repo
        self._is_small = small or "small" in repo
        dtype = torch.float16 if device != "cpu" else torch.float32

        logger.info("Loading Bark from %s (device=%s, dtype=%s)", repo, device, dtype)
        self._processor = AutoProcessor.from_pretrained(repo)
        self._model = AutoModel.from_pretrained(repo, torch_dtype=dtype).to(device)
        self._device = device

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
        dest = _voices_dir(self.model_id) / f"{name}.npz"
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
        npz = _voices_dir(self.model_id) / f"{voice}.npz"
        if npz.exists():
            return str(npz)
        available = ", ".join(self._custom_voice_names()[:5])
        raise ValueError(
            f"Unknown voice: {voice!r}. Use a preset (v2/en_speaker_0) "
            f"or a custom voice name. Custom voices: {available or '(none)'}"
        )

    def _custom_voice_names(self) -> list[str]:
        vdir = _voices_dir(self.model_id)
        return [f.stem for f in vdir.iterdir() if f.suffix == ".npz"]
