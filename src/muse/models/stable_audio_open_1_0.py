"""Stable Audio Open 1.0: 47s music + SFX, 44.1kHz stereo.

Model by Stability AI (Apache 2.0). Diffusion-based latent audio
generation via diffusers.StableAudioPipeline. ~3.4GB at fp16.

Defaults align with the official model card recommendations:
50 inference steps, guidance scale 7.0, 10s default duration,
44.1kHz stereo output. Native max ~47s.

Mirrors sd_turbo.py: lazy-import sentinels, `_ensure_deps()`,
no-op-on-import safety so `muse --help` and `muse pull` work
without diffusers installed.

The class is named `Model` per the discovery convention.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from muse.core.runtime_helpers import dtype_for_name, select_device
from muse.modalities.audio_generation.protocol import AudioGenerationResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
StableAudioPipeline: Any = None


def _ensure_deps() -> None:
    """Lazy-import torch + diffusers.StableAudioPipeline.

    Each symbol imported independently so tests can patch one without
    forcing the other to load.
    """
    global torch, StableAudioPipeline
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("stable_audio_open_1_0: torch unavailable: %s", e)
    if StableAudioPipeline is None:
        try:
            from diffusers import StableAudioPipeline as _p
            StableAudioPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "stable_audio_open_1_0: diffusers.StableAudioPipeline unavailable: %s",
                e,
            )


MANIFEST = {
    "model_id": "stable-audio-open-1.0",
    "modality": "audio/generation",
    "hf_repo": "stabilityai/stable-audio-open-1.0",
    "revision": "f21265c1e2710b3bd2386596943f0007f55f802e",
    "description": (
        "Stable Audio Open 1.0: 47s music + SFX, 44.1kHz stereo, "
        "Apache 2.0"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        # transformers required by StableAudioPipeline's text encoder
        # (T5). diffusers lists it as an optional extra; declare it
        # explicitly so per-model venvs aren't broken on transitive
        # availability.
        "transformers>=4.36.0",
        "accelerate",
        "soundfile",  # FLAC encoding in the codec layer.
        # numpy is pulled by torch but the runtime imports it directly
        # for waveform shaping (#110).
        "numpy",
        # StableAudioPipeline's default CosineDPMSolverMultistepScheduler
        # needs torchsde for its stochastic SDE solver; diffusers imports
        # it lazily at load time, so a venv without it fails at model load
        # (not at import). Declare it explicitly.
        "torchsde",
    ),
    # ffmpeg is optional; only needed for mp3/opus response_format.
    # Declaring it here lets `muse pull` ensure it's available, but
    # the server doesn't crash without it (codec returns 400 instead).
    "system_packages": ("ffmpeg",),
    "capabilities": {
        # cuda by default: Stable Audio is impractical on CPU.
        # Users without a GPU can override via curated capability
        # overlay.
        "device": "cuda",
        "supports_music": True,
        "supports_sfx": True,
        "default_duration": 10.0,
        "min_duration": 1.0,
        "max_duration": 47.0,
        "default_sample_rate": 44100,
        "default_steps": 50,
        "default_guidance": 7.0,
        # Conservative VRAM/RAM peak estimate (fp16, 10s clips).
        # Annotation feeds `muse models list`; `muse models probe`
        # measures the real number.
        "memory_gb": 6.0,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "scheduler/*.json",
        "transformer/*.fp16.safetensors", "transformer/*.json",
        "vae/*.fp16.safetensors", "vae/*.json",
        "text_encoder/*.fp16.safetensors", "text_encoder/*.json",
        "tokenizer/*",
    ],
}


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


def _normalize_pipeline_output(audio: Any) -> tuple[np.ndarray, int]:
    """Normalize diffusers StableAudioPipeline output to numpy
    + (samples,) mono or (samples, channels) multi-channel.

    Mirror StableAudioRuntime._normalize_pipeline_output. Kept local to
    the bundled script so this file doesn't depend on the runtime
    module at import time (which would force diffusers as a hard dep
    of discovery).
    """
    if isinstance(audio, list):
        audio = np.asarray(audio)
    if not isinstance(audio, np.ndarray):
        if torch is not None and isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()
        else:
            audio = np.asarray(audio)
    audio = audio.astype(np.float32, copy=False)
    if audio.ndim == 1:
        return audio, 1
    if audio.ndim == 2:
        if audio.shape[0] <= 2 and audio.shape[1] > 2:
            return audio.T.copy(), audio.shape[0]
        return audio, audio.shape[1]
    raise ValueError(
        f"unsupported StableAudio output shape: {audio.shape}"
    )


class Model:
    """Stable Audio Open 1.0 backend (diffusers StableAudioPipeline)."""

    model_id = MANIFEST["model_id"]

    def __init__(
        self,
        *,
        hf_repo: str = MANIFEST["hf_repo"],
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        **_: Any,
    ) -> None:
        _ensure_deps()
        if StableAudioPipeline is None:
            raise RuntimeError(
                "diffusers.StableAudioPipeline is not installed; run "
                "`muse pull stable-audio-open-1.0` to install deps "
                "into this venv"
            )
        self._device = _select_device(device)

        # Access the module-level torch so tests can patch it.
        import muse.models.stable_audio_open_1_0 as _self_mod
        _torch = _self_mod.torch
        torch_dtype = dtype_for_name(dtype, _torch)
        self._dtype = dtype
        self._torch_dtype = torch_dtype

        src = local_dir or hf_repo
        logger.info(
            "loading Stable Audio Open 1.0 from %s (device=%s, dtype=%s)",
            src, self._device, dtype,
        )
        self._pipe = StableAudioPipeline.from_pretrained(
            src,
            torch_dtype=torch_dtype,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

    def generate(
        self,
        prompt: str,
        *,
        duration: float | None = None,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        **_: Any,
    ) -> AudioGenerationResult:
        caps = MANIFEST["capabilities"]
        eff_duration = duration if duration is not None else caps["default_duration"]
        eff_duration = max(
            caps["min_duration"], min(caps["max_duration"], eff_duration),
        )
        n_steps = steps if steps is not None else caps["default_steps"]
        cfg = guidance if guidance is not None else caps["default_guidance"]

        gen = None
        if seed is not None:
            import muse.models.stable_audio_open_1_0 as _self_mod
            _torch = _self_mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "audio_end_in_s": float(eff_duration),
            "num_inference_steps": int(n_steps),
            "guidance_scale": float(cfg),
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        audios = getattr(out, "audios", None)
        if audios is None:
            audios = out[0] if isinstance(out, (list, tuple)) else out
        if isinstance(audios, (list, tuple)):
            audio = audios[0]
        else:
            audio = audios
            # Real diffusers StableAudioPipeline returns a bare
            # (batch, channels, samples) tensor/ndarray on .audios (not a
            # list); strip the leading batch dim before normalizing (H2).
            if getattr(audio, "ndim", None) == 3:
                audio = audio[0]

        normalized, channels = _normalize_pipeline_output(audio)
        sample_rate = getattr(self._pipe, "sample_rate", None)
        if sample_rate is None:
            sample_rate = caps["default_sample_rate"]
        n_samples = normalized.shape[0]
        duration_seconds = n_samples / float(sample_rate)

        return AudioGenerationResult(
            audio=normalized,
            sample_rate=int(sample_rate),
            channels=int(channels),
            duration_seconds=duration_seconds,
            metadata={
                "prompt": prompt,
                "steps": int(n_steps),
                "guidance": float(cfg),
                "seed": seed,
                "model": self.model_id,
                "duration_requested": float(eff_duration),
            },
        )
