"""WAV and Opus encoding for audio.speech responses.

Extracted from narro/server.py to separate modality-specific encoding
from the modality-agnostic server framework.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import wave

import numpy as np


class AudioFormatError(ValueError):
    """Raised when audio data cannot be encoded to the requested format."""


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 [-1, 1] audio to a 16-bit PCM WAV bytestring."""
    if audio.ndim != 1:
        raise AudioFormatError(f"expected 1-D audio, got shape {audio.shape}")
    # Scale by 32768 so that -1.0 → -32768 and +1.0 → +32768, then clip to
    # int16 range [-32768, 32767] before casting.  Using 32767 as the
    # multiplier would leave -32768 unreachable (broken clipping test).
    scaled = np.clip(audio, -1.0, 1.0) * 32768.0
    pcm = np.clip(scaled, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_bytes_to_opus(wav_data: bytes) -> bytes:
    """Transcode WAV -> Opus via system ffmpeg.

    Raises AudioFormatError if ffmpeg is unavailable or conversion fails.
    """
    if shutil.which("ffmpeg") is None:
        raise AudioFormatError("ffmpeg not found; cannot encode opus")
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "wav", "-i", "pipe:0",
            "-c:a", "libopus", "-b:a", "64k",
            "-f", "ogg", "pipe:1",
        ],
        input=wav_data,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise AudioFormatError(f"ffmpeg failed: {proc.stderr.decode()[:200]}")
    return proc.stdout
