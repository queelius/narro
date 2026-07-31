from unittest.mock import MagicMock

import pytest
import torch

from muse.modalities.audio_quality.decoding import AudioWindow
from muse.modalities.audio_quality.runtimes import utmos as runtime_module


def _runtime(tmp_path, monkeypatch, *, scores=(4.2,), windows=None):
    checkpoint = tmp_path / "utmos_scripted.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = MagicMock()
    model.side_effect = [torch.tensor([score]) for score in scores]
    load = MagicMock(return_value=model)
    monkeypatch.setattr(torch.jit, "load", load)
    monkeypatch.setattr(runtime_module, "torch", torch)
    if windows is None:
        windows = [AudioWindow(torch.ones(1, 16000), 16000, 0, 1)]
    reader = MagicMock(return_value=windows)
    monkeypatch.setattr(runtime_module, "WindowedAudio", reader)
    runtime = runtime_module.UTMOSRuntime(
        model_id="utmos",
        hf_repo="unused",
        local_dir=str(tmp_path),
        device="cpu",
    )
    return runtime, model, reader, load


def test_loads_local_torchscript_checkpoint(tmp_path, monkeypatch):
    _, _, _, load = _runtime(tmp_path, monkeypatch)
    assert load.call_args.args[0].endswith("utmos_scripted.pt")
    assert load.call_args.kwargs["map_location"] == "cpu"


def test_assess_aggregates_windows_and_reports_worst_segment(
    tmp_path, monkeypatch,
):
    windows = [
        AudioWindow(torch.ones(1, 160000), 16000, 0, 10),
        AudioWindow(torch.ones(1, 80000), 16000, 10, 15),
    ]
    runtime, model, reader, _ = _runtime(
        tmp_path,
        monkeypatch,
        scores=(4.0, 2.0),
        windows=windows,
    )
    result = runtime.assess("clip.wav", max_duration_seconds=30)
    assert model.call_count == 2
    assert result.primary_score == "naturalness"
    assert result.scores["naturalness"].value == pytest.approx(10 / 3)
    assert result.metadata["window_count"] == 2
    assert result.metadata["duration_seconds"] == 15
    assert result.metadata["worst_segment"]["start_seconds"] == 10
    assert reader.call_args.kwargs["max_duration_seconds"] == 30


def test_subsecond_tail_is_not_used_as_worst_review_window(
    tmp_path, monkeypatch,
):
    windows = [
        AudioWindow(torch.ones(1, 160000), 16000, 0, 10),
        AudioWindow(torch.ones(1, 4800), 16000, 10, 10.3),
    ]
    runtime, _, _, _ = _runtime(
        tmp_path,
        monkeypatch,
        scores=(4.0, 2.0),
        windows=windows,
    )

    result = runtime.assess("clip.wav")

    assert result.metadata["worst_segment"]["start_seconds"] == 10
    assert result.metadata["worst_review_segment"]["start_seconds"] == 0
    assert result.metadata["short_review_windows_excluded"] == 1
    assert result.scores["naturalness"].value < 4.0


def test_missing_checkpoint_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "torch", torch)
    with pytest.raises(FileNotFoundError, match="pull the model first"):
        runtime_module.UTMOSRuntime(
            model_id="utmos", hf_repo="unused", local_dir=str(tmp_path),
        )
