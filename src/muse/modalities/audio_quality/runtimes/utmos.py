"""UTMOS TorchScript runtime for speech-naturalness MOS prediction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from muse.core.runtime_helpers import LoadTimer, select_device, set_inference_mode
from muse.modalities.audio_quality.decoding import WindowedAudio
from muse.modalities.audio_quality.protocol import (
    AudioQualityResult,
    AudioQualityScore,
)


logger = logging.getLogger(__name__)
torch: Any = None
MINIMUM_REVIEW_WINDOW_SECONDS = 1.0


def _ensure_deps() -> None:
    global torch
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except Exception as exc:  # noqa: BLE001
            logger.debug("UTMOS torch unavailable: %s", exc)


def _checkpoint(source: str) -> Path:
    path = Path(source)
    candidate = path if path.is_file() else path / "utmos_scripted.pt"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"UTMOS checkpoint not found at {candidate}; pull the model first"
        )
    return candidate


class UTMOSRuntime:
    """Run the fairseq-free UTMOS TorchScript checkpoint."""

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        window_seconds: float = 10.0,
        max_duration_seconds: float = 600.0,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "UTMOS requires torch; run `muse pull` to "
                "install the model dependencies"
            )
        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        self._window_seconds = float(window_seconds)
        self._max_duration_seconds = float(max_duration_seconds)
        checkpoint = _checkpoint(local_dir or hf_repo)
        with LoadTimer(f"loading UTMOS from {checkpoint}", logger):
            self._model = torch.jit.load(
                str(checkpoint), map_location=self._device,
            )
        set_inference_mode(self._model)

    def assess(
        self,
        audio_path: str,
        *,
        max_duration_seconds: float | None = None,
    ) -> AudioQualityResult:
        limit = (
            self._max_duration_seconds
            if max_duration_seconds is None
            else float(max_duration_seconds)
        )
        reader = WindowedAudio(
            audio_path,
            sample_rate=16000,
            window_seconds=self._window_seconds,
            max_duration_seconds=limit,
        )
        weighted_total = 0.0
        total_duration = 0.0
        segments: list[dict[str, Any]] = []
        for window in reader:
            waveform = window.waveform.to(self._device)
            with torch.inference_mode():
                # The raw TorchScript artifact exports only ``forward``;
                # the Python package's convenience wrapper adds ``score``.
                raw = self._model(waveform)
            value = float(raw.reshape(-1)[0].detach().cpu().item())
            duration = window.duration_seconds
            weighted_total += value * duration
            total_duration += duration
            segments.append({
                "start_seconds": round(window.start_seconds, 6),
                "end_seconds": round(window.end_seconds, 6),
                "scores": {"naturalness": value},
            })

        value = weighted_total / total_duration
        eligible_review_segments = [
            segment for segment in segments
            if (
                segment["end_seconds"] - segment["start_seconds"]
                >= MINIMUM_REVIEW_WINDOW_SECONDS
            )
        ]
        review_segments = eligible_review_segments or segments
        worst = min(
            segments,
            key=lambda segment: segment["scores"]["naturalness"],
        )
        worst_review = min(
            review_segments,
            key=lambda segment: segment["scores"]["naturalness"],
        )
        return AudioQualityResult(
            scores={
                "naturalness": AudioQualityScore(
                    value=value,
                    minimum=1.0,
                    maximum=5.0,
                    direction="higher_is_better",
                ),
            },
            primary_score="naturalness",
            metadata={
                "family": "utmos",
                "sample_rate": 16000,
                "duration_seconds": round(total_duration, 6),
                "window_seconds": self._window_seconds,
                "window_count": len(segments),
                "aggregation": "duration_weighted_mean",
                "segments": segments,
                "worst_segment": worst,
                "worst_review_segment": worst_review,
                "minimum_review_window_seconds": (
                    MINIMUM_REVIEW_WINDOW_SECONDS
                ),
                "short_review_windows_excluded": (
                    len(segments) - len(review_segments)
                ),
            },
        )
