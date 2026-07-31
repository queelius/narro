"""Meta Audiobox Aesthetics runtime for four-axis audio assessment."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from muse.core.runtime_helpers import LoadTimer, select_device, set_inference_mode
from muse.modalities.audio_quality.decoding import AudioWindow, WindowedAudio
from muse.modalities.audio_quality.protocol import (
    AudioQualityResult,
    AudioQualityScore,
)


logger = logging.getLogger(__name__)
torch: Any = None
AesMultiOutput: Any = None
MINIMUM_REVIEW_WINDOW_SECONDS = 1.0


_AXES = (
    ("CE", "content_enjoyment", "higher_is_better"),
    ("CU", "content_usefulness", "higher_is_better"),
    ("PC", "production_complexity", "descriptive"),
    ("PQ", "production_quality", "higher_is_better"),
)


def _ensure_deps() -> None:
    global torch, AesMultiOutput
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except Exception as exc:  # noqa: BLE001
            logger.debug("Audiobox torch unavailable: %s", exc)
    if AesMultiOutput is None:
        try:
            from audiobox_aesthetics.model.aes import (
                AesMultiOutput as _AesMultiOutput,
            )
            AesMultiOutput = _AesMultiOutput
        except Exception as exc:  # noqa: BLE001
            logger.debug("audiobox_aesthetics unavailable: %s", exc)


def _model_source(source: str) -> Path:
    path = Path(source)
    directory = path.parent if path.is_file() else path
    missing = [
        name for name in ("config.json", "model.safetensors")
        if not (directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Audiobox model files missing from {directory}: {missing}; "
            "pull the model first"
        )
    return directory


def _inverse_transform(raw_value: float, transform: Any) -> float:
    if isinstance(transform, dict):
        mean = transform["mean"]
        std = transform["std"]
    else:
        mean = transform.mean
        std = transform.std
    return raw_value * float(std) + float(mean)


class AudioboxAestheticsRuntime:
    """Load official safetensors and score bounded windows sequentially."""

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
        if torch is None or AesMultiOutput is None:
            raise RuntimeError(
                "Audiobox Aesthetics requires torch and audiobox_aesthetics; "
                "run `muse pull` to install the model dependencies"
            )
        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        self._window_seconds = float(window_seconds)
        self._max_duration_seconds = float(max_duration_seconds)
        source = _model_source(local_dir or hf_repo)
        with LoadTimer(f"loading Audiobox Aesthetics from {source}", logger):
            self._model = AesMultiOutput.from_pretrained(str(source))
        moved = self._model.to(torch.device(self._device))
        if moved is not None:
            self._model = moved
        set_inference_mode(self._model)
        self._target_transform = self._model.target_transform
        missing = [
            source_axis for source_axis, _, _ in _AXES
            if source_axis not in self._target_transform
        ]
        if missing:
            raise RuntimeError(
                f"Audiobox Aesthetics model missing target transforms: {missing}"
            )

    def _score_window(self, window: AudioWindow) -> dict[str, float]:
        waveform = window.waveform
        target_samples = int(round(self._window_seconds * 16000))
        valid_samples = min(int(waveform.shape[-1]), target_samples)
        waveform = waveform[..., :target_samples]
        if valid_samples < target_samples:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, target_samples - valid_samples),
            )
        mask = torch.zeros_like(waveform, dtype=torch.bool)
        mask[..., :valid_samples] = True
        batch = {
            "wav": waveform.unsqueeze(0).to(self._device),
            "mask": mask.unsqueeze(0).to(self._device),
        }
        with torch.inference_mode():
            raw = self._model(batch)
        missing = [source for source, _, _ in _AXES if source not in raw]
        if missing:
            raise RuntimeError(
                f"Audiobox Aesthetics result missing axes: {missing}"
            )
        return {
            name: _inverse_transform(
                float(raw[source].reshape(-1)[0].detach().cpu().item()),
                self._target_transform[source],
            )
            for source, name, _ in _AXES
        }

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
        totals = {name: 0.0 for _, name, _ in _AXES}
        total_duration = 0.0
        segments: list[dict[str, Any]] = []
        for window in reader:
            values = self._score_window(window)
            duration = window.duration_seconds
            total_duration += duration
            for name, value in values.items():
                totals[name] += value * duration
            segments.append({
                "start_seconds": round(window.start_seconds, 6),
                "end_seconds": round(window.end_seconds, 6),
                "scores": values,
            })

        values = {
            name: total / total_duration
            for name, total in totals.items()
        }
        directions = {name: direction for _, name, direction in _AXES}
        scores = {
            name: AudioQualityScore(
                value=value,
                minimum=1.0,
                maximum=10.0,
                direction=directions[name],
            )
            for name, value in values.items()
        }
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
            key=lambda segment: segment["scores"]["production_quality"],
        )
        worst_review = min(
            review_segments,
            key=lambda segment: segment["scores"]["production_quality"],
        )
        return AudioQualityResult(
            scores=scores,
            primary_score="production_quality",
            metadata={
                "family": "audiobox_aesthetics",
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
