"""Bounded, range-based decoding shared by audio-quality runtimes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from muse.modalities.audio_quality.protocol import AudioDurationExceededError


AudioDecoder: Any = None


def _ensure_decoder() -> None:
    global AudioDecoder
    if AudioDecoder is not None:
        return
    try:
        from torchcodec.decoders import AudioDecoder as _AudioDecoder
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "TorchCodec audio decoder is unavailable; run `muse pull` and "
            "ensure FFmpeg shared libraries are installed"
        ) from exc
    AudioDecoder = _AudioDecoder


def _metadata_duration(metadata: Any) -> float | None:
    """Read the best duration exposed across TorchCodec releases."""
    for name in (
        "duration_seconds",
        "duration_seconds_from_content",
        "duration_seconds_from_header",
    ):
        value = getattr(metadata, name, None)
        if value is not None and float(value) >= 0:
            return float(value)
    begin = getattr(metadata, "begin_stream_seconds", None)
    end = getattr(metadata, "end_stream_seconds", None)
    if begin is not None and end is not None and float(end) >= float(begin):
        return float(end) - float(begin)
    return None


@dataclass(frozen=True)
class AudioWindow:
    """One decoded mono range, retained on CPU until inference."""

    waveform: Any
    sample_rate: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class WindowedAudio:
    """Decode fixed ranges without ever materializing the complete waveform."""

    def __init__(
        self,
        audio_path: str,
        *,
        sample_rate: int = 16000,
        window_seconds: float = 10.0,
        max_duration_seconds: float = 600.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        _ensure_decoder()
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.max_duration_seconds = float(max_duration_seconds)
        self._decoder = AudioDecoder(
            audio_path,
            sample_rate=self.sample_rate,
            num_channels=1,
        )
        self.source_duration_seconds = _metadata_duration(
            self._decoder.metadata
        )
        if (
            self.source_duration_seconds is not None
            and self.source_duration_seconds > self.max_duration_seconds + 1e-6
        ):
            raise AudioDurationExceededError(
                maximum_seconds=self.max_duration_seconds,
                actual_seconds=self.source_duration_seconds,
            )

    def _decode_range(
        self,
        *,
        start_seconds: float,
        stop_seconds: float,
        allow_eof: bool,
    ) -> Any | None:
        """Decode one range, normalizing TorchCodec's EOF exception."""
        try:
            return self._decoder.get_samples_played_in_range(
                start_seconds=start_seconds,
                stop_seconds=stop_seconds,
            )
        except RuntimeError as exc:
            no_frames = "no audio frames were decoded" in str(exc).lower()
            if allow_eof and no_frames:
                return None
            raise

    def __iter__(self) -> Iterator[AudioWindow]:
        start = 0.0
        saw_samples = False
        reached_limit = False
        one_sample = 1.0 / self.sample_rate

        while start < self.max_duration_seconds - one_sample:
            stop = min(
                start + self.window_seconds,
                self.max_duration_seconds,
            )
            samples = self._decode_range(
                start_seconds=start,
                stop_seconds=stop,
                allow_eof=saw_samples,
            )
            if samples is None:
                break
            waveform = samples.data
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            sample_count = int(waveform.shape[-1])
            if sample_count <= 0:
                break

            saw_samples = True
            decoded_seconds = sample_count / self.sample_rate
            sample_pts = getattr(samples, "pts_seconds", None)
            sample_start = (
                start if sample_pts is None else float(sample_pts)
            )
            sample_end = min(sample_start + decoded_seconds, stop)
            yield AudioWindow(
                waveform=waveform,
                sample_rate=self.sample_rate,
                start_seconds=sample_start,
                end_seconds=sample_end,
            )

            # Encoder delay can make the first compressed range shorter than
            # requested even though its samples still reach ``stop``. Use the
            # decoded presentation range, rather than fallible container
            # duration metadata, to distinguish that case from a true EOF.
            if sample_end + one_sample < stop:
                break
            start = stop
            reached_limit = (
                start >= self.max_duration_seconds - one_sample
            )

        if not saw_samples:
            raise RuntimeError("audio decoder returned no samples")

        # Do not trust container metadata as the sole guard. If every allowed
        # range was full, probe just beyond the boundary and reject any sample.
        if reached_limit:
            overflow = self._decode_range(
                start_seconds=self.max_duration_seconds,
                stop_seconds=self.max_duration_seconds + min(
                    self.window_seconds, 0.1,
                ),
                allow_eof=True,
            )
            if overflow is not None and int(overflow.data.shape[-1]) > 0:
                raise AudioDurationExceededError(
                    maximum_seconds=self.max_duration_seconds,
                )
