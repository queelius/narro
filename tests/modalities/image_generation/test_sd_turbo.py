"""Tests for SDTurboModel (fully mocked; no weights loaded)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.images.generations.protocol import ImageResult


def _mock_pipe_return():
    """Return a fake pipeline output that matches diffusers' shape."""
    mock_img = MagicMock()
    mock_img.size = (512, 512)
    return MagicMock(images=[mock_img])


def test_sd_turbo_model_id_and_default_size():
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        from muse.images.generations.backends.sd_turbo import SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        assert m.model_id == "sd-turbo"
        assert m.default_size == (512, 512)


def test_sd_turbo_generate_returns_imageresult():
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_pipe = MagicMock(return_value=_mock_pipe_return())
        mock_cls.from_pretrained.return_value = mock_pipe
        from muse.images.generations.backends.sd_turbo import SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        result = m.generate("a cat on mars", width=512, height=512, seed=42)
        assert isinstance(result, ImageResult)
        assert result.width == 512
        assert result.height == 512
        assert result.seed == 42


def test_sd_turbo_passes_prompt_to_pipeline():
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_pipe = MagicMock(return_value=_mock_pipe_return())
        mock_cls.from_pretrained.return_value = mock_pipe
        from muse.images.generations.backends.sd_turbo import SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        m.generate("a red balloon")
        mock_pipe.assert_called_once()
        assert mock_pipe.call_args.kwargs["prompt"] == "a red balloon"


def test_sd_turbo_defaults_steps_to_1():
    """SD-Turbo is 1-step distilled; default num_inference_steps = 1."""
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_pipe = MagicMock(return_value=_mock_pipe_return())
        mock_cls.from_pretrained.return_value = mock_pipe
        from muse.images.generations.backends.sd_turbo import SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        m.generate("prompt")
        assert mock_pipe.call_args.kwargs["num_inference_steps"] == 1


def test_sd_turbo_defaults_guidance_to_0():
    """SD-Turbo: guidance_scale should default to 0.0."""
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_pipe = MagicMock(return_value=_mock_pipe_return())
        mock_cls.from_pretrained.return_value = mock_pipe
        from muse.images.generations.backends.sd_turbo import SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        m.generate("prompt")
        assert mock_pipe.call_args.kwargs["guidance_scale"] == 0.0


def test_sd_turbo_uses_seeded_generator():
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls, \
         patch("muse.images.generations.backends.sd_turbo.torch") as mock_torch:
        mock_pipe = MagicMock(return_value=_mock_pipe_return())
        mock_cls.from_pretrained.return_value = mock_pipe
        from muse.images.generations.backends.sd_turbo import SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        m.generate("prompt", seed=7)
        mock_torch.Generator.return_value.manual_seed.assert_called_with(7)


def test_sd_turbo_uses_local_dir_over_hf_repo():
    """When local_dir is provided, it should be the load path."""
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        from muse.images.generations.backends.sd_turbo import SDTurboModel
        SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/real/local/path")
        # The first positional arg to from_pretrained should be the local_dir
        assert mock_cls.from_pretrained.call_args.args[0] == "/real/local/path"


def test_sd_turbo_falls_back_to_hf_repo_when_no_local_dir():
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        from muse.images.generations.backends.sd_turbo import SDTurboModel
        SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir=None)
        assert mock_cls.from_pretrained.call_args.args[0] == "stabilityai/sd-turbo"


def test_sd_turbo_accepts_unknown_kwargs():
    """Future catalog kwargs should be absorbed by **_."""
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        from muse.images.generations.backends.sd_turbo import SDTurboModel
        # Should not TypeError
        SDTurboModel(
            hf_repo="stabilityai/sd-turbo",
            local_dir="/fake",
            device="cpu",
            extra_future_param="ignored",
        )
