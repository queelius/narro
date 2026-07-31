from unittest.mock import MagicMock

import pytest
import torch

from muse.modalities.audio_quality.decoding import AudioWindow
from muse.modalities.audio_quality.runtimes import (
    audiobox_aesthetics as runtime_module,
)


def _runtime(tmp_path, monkeypatch, rows, *, windows=None, device="cpu"):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    model = MagicMock()
    model.to.return_value = model
    model.target_transform = {
        axis: {"mean": 0.0, "std": 1.0}
        for axis in ("CE", "CU", "PC", "PQ")
    }
    model.side_effect = [
        {axis: torch.tensor([value]) for axis, value in row.items()}
        for row in rows
    ]
    model_class = MagicMock()
    model_class.from_pretrained.return_value = model
    monkeypatch.setattr(runtime_module, "torch", torch)
    monkeypatch.setattr(runtime_module, "AesMultiOutput", model_class)
    if windows is None:
        windows = [AudioWindow(torch.ones(1, 16000), 16000, 0, 1)]
    reader = MagicMock(return_value=windows)
    monkeypatch.setattr(runtime_module, "WindowedAudio", reader)
    runtime = runtime_module.AudioboxAestheticsRuntime(
        model_id="audiobox-aesthetics",
        hf_repo="unused",
        local_dir=str(tmp_path),
        device=device,
    )
    return runtime, model, model_class, reader


def test_loads_safetensors_directly_on_requested_device(tmp_path, monkeypatch):
    _, model, model_class, _ = _runtime(
        tmp_path,
        monkeypatch,
        [{"CE": 7, "CU": 6, "PC": 2, "PQ": 8}],
    )
    model_class.from_pretrained.assert_called_once_with(str(tmp_path))
    model.to.assert_called_once_with(torch.device("cpu"))


def test_assess_aggregates_axes_pads_tail_and_reports_worst(
    tmp_path, monkeypatch,
):
    windows = [
        AudioWindow(torch.ones(1, 160000), 16000, 0, 10),
        AudioWindow(torch.ones(1, 80000), 16000, 10, 15),
    ]
    runtime, model, _, reader = _runtime(
        tmp_path,
        monkeypatch,
        [
            {"CE": 7.0, "CU": 6.0, "PC": 2.0, "PQ": 8.0},
            {"CE": 4.0, "CU": 3.0, "PC": 1.0, "PQ": 2.0},
        ],
        windows=windows,
    )
    result = runtime.assess("clip.wav", max_duration_seconds=25)
    assert result.primary_score == "production_quality"
    assert result.scores["production_quality"].value == pytest.approx(6.0)
    assert result.scores["production_complexity"].direction == "descriptive"
    assert result.metadata["window_count"] == 2
    assert result.metadata["worst_segment"]["start_seconds"] == 10
    assert reader.call_args.kwargs["max_duration_seconds"] == 25
    second_batch = model.call_args_list[1].args[0]
    assert second_batch["wav"].shape == (1, 1, 160000)
    assert int(second_batch["mask"].sum().item()) == 80000


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
        [
            {"CE": 7.0, "CU": 6.0, "PC": 2.0, "PQ": 8.0},
            {"CE": 4.0, "CU": 3.0, "PC": 1.0, "PQ": 2.0},
        ],
        windows=windows,
    )

    result = runtime.assess("clip.wav")

    assert result.metadata["worst_segment"]["start_seconds"] == 10
    assert result.metadata["worst_review_segment"]["start_seconds"] == 0
    assert result.metadata["short_review_windows_excluded"] == 1
    assert result.scores["production_quality"].value < 8.0


def test_missing_axis_fails_loudly(tmp_path, monkeypatch):
    runtime, _, _, _ = _runtime(
        tmp_path,
        monkeypatch,
        [{"CE": 7, "CU": 6, "PQ": 8}],
    )
    with pytest.raises(RuntimeError, match="missing axes"):
        runtime.assess("clip.wav")


def test_missing_safetensors_fails_clearly(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(runtime_module, "torch", torch)
    monkeypatch.setattr(runtime_module, "AesMultiOutput", MagicMock())
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        runtime_module.AudioboxAestheticsRuntime(
            model_id="audiobox-aesthetics",
            hf_repo="unused",
            local_dir=str(tmp_path),
        )
