"""Tests for audio codec conversion."""
import io
import wave

import numpy as np
import pytest

from muse.audio.speech.codec import (
    AudioFormatError,
    audio_to_wav_bytes,
    wav_bytes_to_opus,
)


def test_audio_to_wav_bytes_produces_valid_wav():
    audio = np.zeros(16000, dtype=np.float32)
    data = audio_to_wav_bytes(audio, sample_rate=16000)
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2  # int16


def test_audio_to_wav_bytes_clips_to_int16_range():
    audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)  # out of [-1,1]
    data = audio_to_wav_bytes(audio, sample_rate=16000)
    with wave.open(io.BytesIO(data), "rb") as w:
        frames = w.readframes(w.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
    assert samples.max() == 32767
    assert samples.min() == -32768


def test_audio_to_wav_bytes_rejects_non_1d():
    audio = np.zeros((2, 1000), dtype=np.float32)
    with pytest.raises(AudioFormatError):
        audio_to_wav_bytes(audio, sample_rate=16000)


def test_wav_bytes_to_opus_produces_bytes_when_ffmpeg_present():
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    audio = np.zeros(16000, dtype=np.float32)
    wav_data = audio_to_wav_bytes(audio, sample_rate=16000)
    opus_data = wav_bytes_to_opus(wav_data)
    assert isinstance(opus_data, bytes)
    assert len(opus_data) > 0


def test_wav_bytes_to_opus_raises_when_ffmpeg_missing(monkeypatch):
    import muse.audio.speech.codec as codec_module
    monkeypatch.setattr(codec_module.shutil, "which", lambda x: None)
    with pytest.raises(AudioFormatError, match="ffmpeg not found"):
        wav_bytes_to_opus(b"fake wav data")
