from types import SimpleNamespace

import pytest
import torch

from muse.modalities.audio_quality import AudioDurationExceededError
from muse.modalities.audio_quality import decoding


def _decoder(
    monkeypatch,
    *,
    actual_seconds,
    reported_seconds=None,
    channels=1,
    raise_at_eof=False,
    first_range_pts_offset=0.0,
):
    instances = []

    class _Decoder:
        def __init__(self, path, *, sample_rate, num_channels):
            self.path = path
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.metadata = SimpleNamespace(
                duration_seconds=(
                    actual_seconds
                    if reported_seconds is None
                    else reported_seconds
                ),
            )
            self.calls = []
            instances.append(self)

        def get_samples_played_in_range(self, *, start_seconds, stop_seconds):
            self.calls.append((start_seconds, stop_seconds))
            if raise_at_eof and start_seconds >= actual_seconds:
                raise RuntimeError(
                    "No audio frames were decoded. This is probably because "
                    "start_seconds is too high"
                )
            sample_start = start_seconds
            if len(self.calls) == 1:
                sample_start += first_range_pts_offset
            duration = max(
                0.0,
                min(stop_seconds, actual_seconds) - sample_start,
            )
            sample_count = round(duration * self.sample_rate)
            return SimpleNamespace(
                data=torch.ones(channels, sample_count),
                pts_seconds=sample_start,
                duration_seconds=duration,
            )

    monkeypatch.setattr(decoding, "AudioDecoder", _Decoder)
    return instances


def test_decodes_fixed_ranges_and_downmixes(monkeypatch):
    instances = _decoder(
        monkeypatch,
        actual_seconds=25,
        channels=2,
    )
    reader = decoding.WindowedAudio(
        "chapter.mp3",
        sample_rate=16000,
        window_seconds=10,
        max_duration_seconds=60,
    )
    windows = list(reader)
    assert [(w.start_seconds, w.end_seconds) for w in windows] == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 25.0),
    ]
    assert all(window.waveform.shape[0] == 1 for window in windows)
    assert instances[0].num_channels == 1
    assert instances[0].calls == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]


def test_short_compressed_interior_range_does_not_end_decode(monkeypatch):
    instances = _decoder(
        monkeypatch,
        actual_seconds=28,
        first_range_pts_offset=0.023,
    )
    reader = decoding.WindowedAudio(
        "chapter.mp3",
        sample_rate=16000,
        window_seconds=10,
        max_duration_seconds=60,
    )

    windows = list(reader)

    assert [(w.start_seconds, w.end_seconds) for w in windows] == [
        (0.023, 10.0),
        (10.0, 20.0),
        (20.0, 28.0),
    ]
    assert instances[0].calls == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 30.0),
    ]


def test_overreported_duration_does_not_extend_partial_eof(monkeypatch):
    instances = _decoder(
        monkeypatch,
        actual_seconds=28,
        reported_seconds=38,
        raise_at_eof=True,
    )
    reader = decoding.WindowedAudio(
        "overreported.mp3",
        sample_rate=16000,
        window_seconds=10,
        max_duration_seconds=60,
    )

    windows = list(reader)

    assert [(w.start_seconds, w.end_seconds) for w in windows] == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 28.0),
    ]
    assert instances[0].calls == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 30.0),
    ]


def test_overreported_duration_treats_no_frames_as_eof(monkeypatch):
    instances = _decoder(
        monkeypatch,
        actual_seconds=20,
        reported_seconds=38,
        raise_at_eof=True,
    )
    reader = decoding.WindowedAudio(
        "overreported.mp3",
        sample_rate=16000,
        window_seconds=10,
        max_duration_seconds=60,
    )

    windows = list(reader)

    assert [(w.start_seconds, w.end_seconds) for w in windows] == [
        (0.0, 10.0),
        (10.0, 20.0),
    ]
    assert instances[0].calls[-1] == (20.0, 30.0)


def test_rejects_duration_from_metadata_before_decoding(monkeypatch):
    instances = _decoder(monkeypatch, actual_seconds=601)
    with pytest.raises(AudioDurationExceededError, match="600.000s"):
        decoding.WindowedAudio(
            "long.mp3",
            max_duration_seconds=600,
        )
    assert instances[0].calls == []


def test_probes_past_limit_when_metadata_underreports(monkeypatch):
    instances = _decoder(
        monkeypatch,
        actual_seconds=21,
        reported_seconds=20,
    )
    reader = decoding.WindowedAudio(
        "misleading.mp3",
        window_seconds=10,
        max_duration_seconds=20,
    )
    with pytest.raises(AudioDurationExceededError):
        list(reader)
    assert instances[0].calls[-1] == (20.0, 20.1)


def test_empty_decode_fails_clearly(monkeypatch):
    _decoder(monkeypatch, actual_seconds=0)
    reader = decoding.WindowedAudio("empty.wav")
    with pytest.raises(RuntimeError, match="no samples"):
        list(reader)


def test_exact_window_multiple_treats_torchcodec_no_frames_as_eof(monkeypatch):
    instances = _decoder(
        monkeypatch,
        actual_seconds=20,
        raise_at_eof=True,
    )
    reader = decoding.WindowedAudio(
        "twenty-seconds.wav",
        window_seconds=10,
        max_duration_seconds=60,
    )
    windows = list(reader)
    assert [(window.start_seconds, window.end_seconds) for window in windows] == [
        (0.0, 10.0),
        (10.0, 20.0),
    ]
    assert instances[0].calls[-1] == (20.0, 30.0)
