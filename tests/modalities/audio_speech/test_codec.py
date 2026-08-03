"""Tests for audio codec conversion."""
import io
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from muse.modalities.audio_speech.codec import (
    AudioFormatError,
    audio_to_wav_bytes,
    float_to_pcm16,
    wav_bytes_to_opus,
)


def test_float_to_pcm16_reaches_full_negative_range():
    # The shared helper must scale by 32768 (not 32767): -1.0 has to reach
    # the int16 floor -32768. The streaming route used *32767 before it was
    # unified onto this helper, which left -32768 unreachable.
    pcm = float_to_pcm16(np.array([-1.0, 1.0, 0.0], dtype=np.float32))
    assert pcm.dtype == np.int16
    assert pcm[0] == -32768
    assert pcm[1] == 32767  # +1.0 * 32768 = 32768, clipped down to 32767
    assert pcm[2] == 0


def test_float_to_pcm16_clips_out_of_range_input():
    pcm = float_to_pcm16(np.array([5.0, -5.0], dtype=np.float32))
    assert pcm[0] == 32767
    assert pcm[1] == -32768


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


def test_wav_bytes_to_opus_uses_owned_bounded_ffmpeg():
    audio = np.zeros(16000, dtype=np.float32)
    wav_data = audio_to_wav_bytes(audio, sample_rate=16000)
    work_dirs = []

    def fake_run(cmd, **kwargs):
        work_dirs.append(Path(cmd[-1]).parent)
        Path(cmd[-1]).write_bytes(b"OggS-test")
        assert kwargs["timeout"] == 60.0
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch(
        "muse.modalities.audio_speech.codec.shutil.which",
        return_value="/usr/bin/ffmpeg",
    ), patch(
        "muse.modalities.audio_speech.codec.run_owned_command",
        side_effect=fake_run,
    ):
        opus_data = wav_bytes_to_opus(wav_data)
    assert isinstance(opus_data, bytes)
    assert opus_data == b"OggS-test"
    assert work_dirs and not work_dirs[0].exists()


def test_wav_bytes_to_opus_raises_when_ffmpeg_missing(monkeypatch):
    import muse.modalities.audio_speech.codec as codec_module
    monkeypatch.setattr(codec_module.shutil, "which", lambda x: None)
    with pytest.raises(AudioFormatError, match="ffmpeg not found"):
        wav_bytes_to_opus(b"fake wav data")


def test_wav_bytes_to_opus_maps_owned_timeout():
    with patch(
        "muse.modalities.audio_speech.codec.shutil.which",
        return_value="/usr/bin/ffmpeg",
    ), patch(
        "muse.modalities.audio_speech.codec.run_owned_command",
        side_effect=subprocess.TimeoutExpired("ffmpeg", 60),
    ):
        with pytest.raises(AudioFormatError, match="timed out after 60s"):
            wav_bytes_to_opus(b"fake wav data")


def test_wav_bytes_to_opus_maps_nonzero_exit():
    failed = subprocess.CompletedProcess(
        ["ffmpeg"], 1, "", "bad codec /private/path",
    )
    with patch(
        "muse.modalities.audio_speech.codec.shutil.which",
        return_value="/usr/bin/ffmpeg",
    ), patch(
        "muse.modalities.audio_speech.codec.run_owned_command",
        return_value=failed,
    ):
        with pytest.raises(AudioFormatError, match="ffmpeg failed: bad codec"):
            wav_bytes_to_opus(b"fake wav data")
