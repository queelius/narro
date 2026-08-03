"""Kokoro 82M TTS: lightweight multi-voice text-to-speech, 24kHz.

Wraps hexgrad/Kokoro-82M via the ``kokoro`` package to implement
the :class:`~muse.modalities.audio_speech.protocol.TTSModel` protocol.  Lightweight (82M),
fast, with 54 voices across 6 languages. Sentence-level streaming.

Requires: ``pip install kokoro soundfile`` and system ``espeak-ng``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from muse.modalities.audio_speech import AudioChunk, AudioResult

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


MANIFEST = {
    "model_id": "kokoro-82m",
    "modality": "audio/speech",
    "hf_repo": "hexgrad/Kokoro-82M",
    "revision": "f3ff3571791e39611d31c381e3a41a3af07b4987",
    "description": "Lightweight TTS, 54 voices, 24kHz",
    "license": "Apache 2.0",
    # kokoro pulls torch + numpy transitively; declared explicitly so
    # the manifest stays self-describing for fresh-venv installs (#110).
    "pip_extras": (
        "torch>=2.1.0",
        "numpy",
        "kokoro==0.9.4",
        "soundfile",
    ),
    "allow_patterns": (
        "config.json",
        "kokoro-v1_0.pth",
        "voices/*.pt",
    ),
    "system_packages": ("espeak-ng",),
    "capabilities": {
        "sample_rate": KOKORO_SAMPLE_RATE,
        "voices": KOKORO_VOICES,
        "device": "auto",
        # Conservative architecture-derived peak-inference estimate.
        # ~82M params at fp32 is ~330 MB; with phonemizer + activation
        # working set, ~0.5 GB total fits this 82M-param model on CPU.
        "memory_gb": 0.5,
    },
}


class Model:
    """Kokoro 82M TTS backend. Named ``Model`` per muse discovery convention.

    Args:
        hf_repo: HuggingFace repo id (used when local_dir is not set).
        local_dir: Pulled snapshot root. When supplied, all model and voice
            artifacts are loaded exclusively from this directory.
        device: ``'auto'``, ``'cpu'``, or ``'cuda'``.
        lang_code: Default language code (``'a'`` = US English).
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
        local_artifacts = (
            self._validate_local_artifacts(local_dir)
            if local_dir is not None
            else None
        )

        import torch
        from kokoro import KModel, KPipeline

        from muse.core.runtime_helpers import select_device

        device = select_device(device, torch_module=torch)

        logger.info("Loading Kokoro (lang=%s, device=%s)", lang_code, device)
        if local_artifacts is None:
            self._pipeline = KPipeline(
                lang_code=lang_code,
                repo_id=hf_repo,
                device=device,
            )
            self._local_voice_paths: dict[str, str] | None = None
        else:
            config_path, weights_path, voice_paths = local_artifacts
            local_model = KModel(
                config=str(config_path),
                model=str(weights_path),
            )
            self._pipeline = KPipeline(
                lang_code=lang_code,
                model=local_model,
                device=device,
            )
            self._local_voice_paths = {
                voice: str(path) for voice, path in voice_paths.items()
            }
        self._device = device

    @staticmethod
    def _validate_local_artifacts(
        local_dir: str,
    ) -> tuple[Path, Path, dict[str, Path]]:
        """Return a complete local Kokoro bundle or fail before construction."""
        root = Path(local_dir).expanduser().resolve()
        config_path = root / "config.json"
        weights_path = root / "kokoro-v1_0.pth"
        voice_paths = {
            voice: root / "voices" / f"{voice}.pt" for voice in KOKORO_VOICES
        }
        required_paths = (config_path, weights_path, *voice_paths.values())
        missing = [path for path in required_paths if not path.is_file()]
        if missing:
            missing_list = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Kokoro snapshot {root} is incomplete; missing required "
                f"artifact(s): {missing_list}. Re-run `muse pull kokoro-82m` "
                "to repair the local model bundle."
            )
        # Keep the snapshot-relative filenames even when they are Hugging Face
        # cache symlinks. Kokoro recognizes local voices by the `.pt` suffix;
        # resolving a symlink to its extensionless blob would trigger repo lookup.
        return config_path, weights_path, voice_paths

    def _pipeline_voice(self, voice: str) -> str:
        """Resolve a logical voice name without allowing local-mode downloads."""
        local_voice_paths = getattr(self, "_local_voice_paths", None)
        if local_voice_paths is None:
            return voice
        try:
            return local_voice_paths[voice]
        except KeyError as exc:
            raise ValueError(
                f"Unknown bundled Kokoro voice {voice!r}; choose one of "
                f"{', '.join(self.VOICES)}"
            ) from exc

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
        voice = kwargs.get("voice") or "af_heart"
        speed = kwargs.get("speed", 1.0)
        pipeline_voice = self._pipeline_voice(voice)

        chunks = []
        for result in self._pipeline(text, voice=pipeline_voice, speed=speed):
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
        voice = kwargs.get("voice") or "af_heart"
        speed = kwargs.get("speed", 1.0)
        pipeline_voice = self._pipeline_voice(voice)

        for result in self._pipeline(text, voice=pipeline_voice, speed=speed):
            if result.audio is not None:
                audio = result.audio.numpy().astype(np.float32)
                yield AudioChunk(audio=audio, sample_rate=KOKORO_SAMPLE_RATE)
