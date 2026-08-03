"""Tests for muse.models.kokoro_82m: Kokoro TTS adapter."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from muse.modalities.audio_speech.protocol import AudioChunk, AudioResult, TTSModel


def _write_local_bundle(root):
    from muse.models.kokoro_82m import KOKORO_VOICES

    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "kokoro-v1_0.pth").write_bytes(b"weights")
    voices_dir = root / "voices"
    voices_dir.mkdir()
    for voice in KOKORO_VOICES:
        (voices_dir / f"{voice}.pt").write_bytes(b"voice")
    return root


def _install_fake_kokoro(monkeypatch):
    fake_kokoro = MagicMock()
    fake_kokoro.KModel = MagicMock(name="KModel")
    fake_kokoro.KPipeline = MagicMock(name="KPipeline")
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)
    return fake_kokoro


class TestKokoroModel:
    def _make_adapter(self):
        from muse.models.kokoro_82m import Model as KokoroModel

        mock_pipeline = MagicMock()
        result = MagicMock()
        result.audio = torch.randn(24000)
        mock_pipeline.return_value = [result]

        adapter = object.__new__(KokoroModel)
        adapter._pipeline = mock_pipeline
        adapter._device = "cpu"
        adapter._local_voice_paths = None
        return adapter

    def test_protocol_conformance(self):
        assert isinstance(self._make_adapter(), TTSModel)

    def test_model_id(self):
        assert self._make_adapter().model_id == "kokoro-82m"

    def test_sample_rate(self):
        assert self._make_adapter().sample_rate == 24000

    def test_synthesize_returns_audio_result(self):
        result = self._make_adapter().synthesize("Hello")
        assert isinstance(result, AudioResult)
        assert result.sample_rate == 24000
        assert len(result.audio) == 24000

    def test_synthesize_passes_voice(self):
        adapter = self._make_adapter()
        adapter.synthesize("Hello", voice="am_adam", speed=1.2)
        adapter._pipeline.assert_called_once_with("Hello", voice="am_adam", speed=1.2)

    def test_synthesize_defaults_voice_when_none(self):
        # The /v1/audio/speech route declares `voice: str | None = None` and
        # always forwards `voice=req.voice`, so an omitted voice arrives as an
        # explicit None. The default must still apply (kwargs.get default only
        # fires when the key is absent, not when it is present-but-None).
        adapter = self._make_adapter()
        adapter.synthesize("Hello", voice=None)
        adapter._pipeline.assert_called_once_with("Hello", voice="af_heart", speed=1.0)

    def test_stream_defaults_voice_when_none(self):
        adapter = self._make_adapter()
        list(adapter.synthesize_stream("Hello", voice=None))
        adapter._pipeline.assert_called_once_with("Hello", voice="af_heart", speed=1.0)

    def test_stream_yields_chunks(self):
        adapter = self._make_adapter()
        chunks = list(adapter.synthesize_stream("Hello"))
        assert len(chunks) == 1
        assert isinstance(chunks[0], AudioChunk)

    def test_voices_list(self):
        from muse.models.kokoro_82m import KOKORO_VOICES
        assert "af_heart" in KOKORO_VOICES
        assert "am_adam" in KOKORO_VOICES
        assert len(KOKORO_VOICES) > 50


def test_kokoro_has_lowercase_voices_property():
    """routes.py + registry look for `voices` (lowercase); KokoroModel must satisfy."""
    from muse.models.kokoro_82m import Model as KokoroModel

    assert "voices" in dir(KokoroModel), "KokoroModel must expose a `voices` attribute/property"

    # Verify via an instance (bypassing __init__) that it returns the VOICES list
    adapter = object.__new__(KokoroModel)
    assert hasattr(adapter, "voices")
    assert isinstance(adapter.voices, list)
    assert len(adapter.voices) > 0
    assert adapter.voices is KokoroModel.VOICES


def test_kokoro_local_snapshot_never_uses_repo_fallback(monkeypatch, tmp_path):
    """A pulled model must construct both Kokoro objects from absolute files."""
    local_dir = _write_local_bundle(tmp_path / "snapshot")
    voice_blob = tmp_path / "voice-blob-without-extension"
    voice_blob.write_bytes(b"voice")
    local_voice_path = local_dir / "voices" / "am_adam.pt"
    local_voice_path.unlink()
    local_voice_path.symlink_to(voice_blob)
    fake_kokoro = _install_fake_kokoro(monkeypatch)
    local_model = fake_kokoro.KModel.return_value

    from muse.models.kokoro_82m import Model as KokoroModel

    adapter = KokoroModel(
        hf_repo="hexgrad/Kokoro-82M",
        local_dir=str(local_dir),
        device="cpu",
    )

    fake_kokoro.KModel.assert_called_once_with(
        config=str((local_dir / "config.json").absolute()),
        model=str((local_dir / "kokoro-v1_0.pth").absolute()),
    )
    model_kwargs = fake_kokoro.KModel.call_args.kwargs
    pipeline_kwargs = fake_kokoro.KPipeline.call_args.kwargs
    assert "repo_id" not in model_kwargs
    assert "repo_id" not in pipeline_kwargs
    assert pipeline_kwargs == {
        "lang_code": "a",
        "model": local_model,
        "device": "cpu",
    }

    result = MagicMock()
    result.audio = torch.ones(8)
    adapter._pipeline.return_value = [result]
    audio = adapter.synthesize("Hello", voice="am_adam")
    adapter._pipeline.assert_called_once_with(
        "Hello",
        voice=str(local_voice_path.absolute()),
        speed=1.0,
    )
    assert adapter._pipeline.call_args.kwargs["voice"].endswith(".pt")
    assert audio.metadata["voice"] == "am_adam"


def test_kokoro_local_stream_uses_absolute_voice_path(monkeypatch, tmp_path):
    local_dir = _write_local_bundle(tmp_path / "snapshot")
    fake_kokoro = _install_fake_kokoro(monkeypatch)

    from muse.models.kokoro_82m import Model as KokoroModel

    adapter = KokoroModel(local_dir=str(local_dir), device="cpu")
    result = MagicMock()
    result.audio = torch.ones(8)
    adapter._pipeline.return_value = [result]

    list(adapter.synthesize_stream("Hello", voice="bf_emma"))

    adapter._pipeline.assert_called_once_with(
        "Hello",
        voice=str((local_dir / "voices" / "bf_emma.pt").absolute()),
        speed=1.0,
    )
    assert fake_kokoro.KModel.called


def test_kokoro_missing_local_artifact_fails_before_construction(monkeypatch, tmp_path):
    local_dir = _write_local_bundle(tmp_path / "snapshot")
    missing_voice = local_dir / "voices" / "af_heart.pt"
    missing_voice.unlink()
    fake_kokoro = _install_fake_kokoro(monkeypatch)

    from muse.models.kokoro_82m import Model as KokoroModel

    with pytest.raises(FileNotFoundError) as exc_info:
        KokoroModel(local_dir=str(local_dir), device="cpu")

    message = str(exc_info.value)
    assert str(missing_voice) in message
    assert "muse pull kokoro-82m" in message
    fake_kokoro.KModel.assert_not_called()
    fake_kokoro.KPipeline.assert_not_called()


def test_kokoro_unknown_local_voice_fails_without_pipeline_call(monkeypatch, tmp_path):
    local_dir = _write_local_bundle(tmp_path / "snapshot")
    _install_fake_kokoro(monkeypatch)

    from muse.models.kokoro_82m import Model as KokoroModel

    adapter = KokoroModel(local_dir=str(local_dir), device="cpu")
    with pytest.raises(ValueError, match="Unknown bundled Kokoro voice"):
        adapter.synthesize("Hello", voice="download_me")
    adapter._pipeline.assert_not_called()


def test_kokoro_without_local_snapshot_can_use_repo_id(monkeypatch):
    fake_kokoro = _install_fake_kokoro(monkeypatch)

    from muse.models.kokoro_82m import Model as KokoroModel

    KokoroModel(hf_repo="org/repo", device="cpu")

    fake_kokoro.KModel.assert_not_called()
    fake_kokoro.KPipeline.assert_called_once_with(
        lang_code="a",
        repo_id="org/repo",
        device="cpu",
    )


def test_manifest_has_required_fields():
    from muse.models.kokoro_82m import MANIFEST
    assert MANIFEST["model_id"] == "kokoro-82m"
    assert MANIFEST["modality"] == "audio/speech"
    assert "hf_repo" in MANIFEST
    assert "pip_extras" in MANIFEST
    assert MANIFEST["revision"] == "f3ff3571791e39611d31c381e3a41a3af07b4987"
    assert "kokoro==0.9.4" in MANIFEST["pip_extras"]
    assert all(not requirement.startswith("misaki") for requirement in MANIFEST["pip_extras"])
