"""Smoke tests for muse.models.soprano_80m.Model (mocked; no real weights)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.models.soprano_80m import Model as SopranoModel


# Narro is imported inside SopranoModel.__init__, so patch at the source module.
_NARRO_PATH = "muse.modalities.audio_speech.tts.Narro"
_SAMPLE_RATE_PATH = "muse.modalities.audio_speech.tts.SAMPLE_RATE"


def _make_model(**kwargs):
    """Construct a SopranoModel with Narro fully mocked."""
    with patch(_NARRO_PATH) as mock_narro, \
         patch(_SAMPLE_RATE_PATH, 32000):
        mock_narro.return_value = MagicMock(sample_rate=32000)
        defaults = dict(hf_repo="ekwek/Soprano-1.1-80M", local_dir=None)
        defaults.update(kwargs)
        return SopranoModel(**defaults)


def test_soprano_model_id():
    m = _make_model()
    assert m.model_id == "soprano-80m"


def test_soprano_sample_rate():
    m = _make_model()
    assert m.sample_rate == 32000


def test_soprano_accepts_local_dir_kwarg():
    """Catalog.load_backend passes local_dir; constructor must accept it."""
    # Should not raise TypeError
    _make_model(hf_repo="ekwek/Soprano-1.1-80M", local_dir="/some/path")


def test_soprano_accepts_unknown_kwargs_gracefully():
    """Future catalog kwargs (**_) should be absorbed without error."""
    _make_model(
        hf_repo="fake/repo",
        local_dir="/some/path",
        device="cpu",
        future_param="ignored",
    )


def test_soprano_prefers_local_dir_over_hf_repo():
    """When local_dir is set, Narro should receive it rather than hf_repo."""
    with patch("muse.modalities.audio_speech.tts.Narro") as mock_narro, \
         patch("muse.modalities.audio_speech.tts.SAMPLE_RATE", 32000):
        mock_narro.return_value = MagicMock(sample_rate=32000)
        SopranoModel(hf_repo="ekwek/Soprano-1.1-80M", local_dir="/local/weights")
        call_kwargs = mock_narro.call_args[1]
        assert call_kwargs["model_path"] == "/local/weights"


def test_soprano_falls_back_to_hf_repo_when_no_local_dir():
    """When local_dir is None, Narro should receive hf_repo."""
    with patch("muse.modalities.audio_speech.tts.Narro") as mock_narro, \
         patch("muse.modalities.audio_speech.tts.SAMPLE_RATE", 32000):
        mock_narro.return_value = MagicMock(sample_rate=32000)
        SopranoModel(hf_repo="ekwek/Soprano-1.1-80M", local_dir=None)
        call_kwargs = mock_narro.call_args[1]
        assert call_kwargs["model_path"] == "ekwek/Soprano-1.1-80M"


def test_narro_forwards_selected_source_to_encoder_and_decoder():
    """The managed source must not be swallowed by encoder ``**kwargs``."""
    with (
        patch(
            "muse.modalities.audio_speech.backends.transformers.TransformersModel"
        ) as mock_transformers,
        patch("muse.modalities.audio_speech.decode_only.load_decoder") as mock_decoder,
        patch("muse.modalities.audio_speech.tts.torch.set_num_threads"),
        patch("muse.modalities.audio_speech.tts.torch.set_num_interop_threads"),
        patch("muse.modalities.audio_speech.tts.torch.set_float32_matmul_precision"),
    ):
        mock_decoder.return_value.parameters.return_value = iter(())

        from muse.modalities.audio_speech.tts import Narro

        Narro(
            model_path="/managed/soprano-snapshot",
            compile=False,
            quantize=False,
            num_threads=1,
            device="cpu",
        )

    mock_transformers.assert_called_once_with(
        local_dir="/managed/soprano-snapshot",
        compile=False,
        quantize=False,
        device="cpu",
    )
    mock_decoder.assert_called_once_with(
        model_path="/managed/soprano-snapshot",
        compile=False,
        device="cpu",
    )


def test_manifest_has_required_fields():
    from muse.models.soprano_80m import MANIFEST
    assert MANIFEST["model_id"] == "soprano-80m"
    assert MANIFEST["modality"] == "audio/speech"
    assert "hf_repo" in MANIFEST
    assert "pip_extras" in MANIFEST
