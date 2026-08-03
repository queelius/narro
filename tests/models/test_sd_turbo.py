"""Tests for the SD-Turbo model script (fully mocked; no weights loaded)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.image_generation.protocol import ImageResult


def _mock_pipe_return():
    """Return a fake pipeline output that matches diffusers' shape."""
    mock_img = MagicMock()
    mock_img.size = (512, 512)
    return MagicMock(images=[mock_img])


def test_manifest_has_required_fields():
    from muse.models.sd_turbo import MANIFEST
    assert MANIFEST["model_id"] == "sd-turbo"
    assert MANIFEST["modality"] == "image/generation"
    assert "hf_repo" in MANIFEST
    assert "pip_extras" in MANIFEST


def test_manifest_pins_cuda_device_not_auto():
    """Finding 2 (Medium): sd-turbo's allow_patterns download ONLY
    *.fp16.safetensors -- no fp32 weights exist on disk. A device:auto
    resolution on a CPU-only host would load an fp16 pipeline that
    errors at inference (no CPU kernel for half-precision matmul).
    Pin device to 'cuda' (matching stable-audio-open / animatediff /
    x4-upscaler precedent) so a CPU-only host correctly reports this
    model unservable instead of silently 500ing on first request."""
    from muse.models.sd_turbo import MANIFEST
    assert MANIFEST["capabilities"]["device"] == "cuda"


def test_manifest_pip_extras_declares_transformers_and_torch():
    """sd-turbo's pipeline internally needs transformers (CLIP text
    encoder) and torch; both must be explicitly declared so per-model
    venvs have them without relying on transitive installs.

    Regression: v0.12.1 caught a live box where a venv pulled before
    this declaration was added had diffusers + accelerate but no
    transformers, so the worker exited at load time.
    """
    from muse.models.sd_turbo import MANIFEST
    extras_str = " ".join(MANIFEST["pip_extras"])
    assert "transformers" in extras_str, (
        f"transformers must be in pip_extras (StableDiffusionPipeline "
        f"requires it for the CLIP text encoder); got {MANIFEST['pip_extras']}"
    )
    assert "torch" in extras_str, (
        f"torch must be explicit in pip_extras; got {MANIFEST['pip_extras']}"
    )


def test_sd_turbo_model_id_and_default_size():
    with patch("muse.models.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        assert m.model_id == "sd-turbo"
        assert m.default_size == (512, 512)


def test_sd_turbo_retains_pipeline_when_device_move_returns_other_object():
    with patch("muse.models.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = MagicMock(name="unrelated-fluent-result")
        mock_cls.from_pretrained.return_value = mock_pipe
        from muse.models.sd_turbo import Model as SDTurboModel

        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo",
            local_dir="/fake",
            device="cuda",
        )

        assert m._pipe is mock_pipe
        mock_pipe.to.assert_called_once_with("cuda")


def test_sd_turbo_generate_returns_imageresult():
    with patch("muse.models.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_pipe = MagicMock(return_value=_mock_pipe_return())
        mock_cls.from_pretrained.return_value = mock_pipe
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        result = m.generate("a cat on mars", width=512, height=512, seed=42)
        assert isinstance(result, ImageResult)
        assert result.width == 512
        assert result.height == 512
        assert result.seed == 42


def test_sd_turbo_passes_prompt_to_pipeline():
    with patch("muse.models.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_pipe = MagicMock(return_value=_mock_pipe_return())
        mock_cls.from_pretrained.return_value = mock_pipe
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        m.generate("a red balloon")
        mock_pipe.assert_called_once()
        assert mock_pipe.call_args.kwargs["prompt"] == "a red balloon"


def test_sd_turbo_defaults_steps_to_1():
    """SD-Turbo is 1-step distilled; default num_inference_steps = 1."""
    with patch("muse.models.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_pipe = MagicMock(return_value=_mock_pipe_return())
        mock_cls.from_pretrained.return_value = mock_pipe
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        m.generate("prompt")
        assert mock_pipe.call_args.kwargs["num_inference_steps"] == 1


def test_sd_turbo_defaults_guidance_to_0():
    """SD-Turbo: guidance_scale should default to 0.0."""
    with patch("muse.models.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_pipe = MagicMock(return_value=_mock_pipe_return())
        mock_cls.from_pretrained.return_value = mock_pipe
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        m.generate("prompt")
        assert mock_pipe.call_args.kwargs["guidance_scale"] == 0.0


def test_sd_turbo_uses_seeded_generator():
    with patch("muse.models.sd_turbo.AutoPipelineForText2Image") as mock_cls, \
         patch("muse.models.sd_turbo.torch") as mock_torch:
        mock_pipe = MagicMock(return_value=_mock_pipe_return())
        mock_cls.from_pretrained.return_value = mock_pipe
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        m.generate("prompt", seed=7)
        mock_torch.Generator.return_value.manual_seed.assert_called_with(7)


def test_sd_turbo_uses_local_dir_over_hf_repo():
    """When local_dir is provided, it should be the load path."""
    with patch("muse.models.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        from muse.models.sd_turbo import Model as SDTurboModel
        SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/real/local/path")
        # The first positional arg to from_pretrained should be the local_dir
        assert mock_cls.from_pretrained.call_args.args[0] == "/real/local/path"


def test_sd_turbo_falls_back_to_hf_repo_when_no_local_dir():
    with patch("muse.models.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        from muse.models.sd_turbo import Model as SDTurboModel
        SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir=None)
        assert mock_cls.from_pretrained.call_args.args[0] == "stabilityai/sd-turbo"


def test_sd_turbo_accepts_unknown_kwargs():
    """Future catalog kwargs should be absorbed by **_."""
    with patch("muse.models.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        from muse.models.sd_turbo import Model as SDTurboModel
        # Should not TypeError
        SDTurboModel(
            hf_repo="stabilityai/sd-turbo",
            local_dir="/fake",
            device="cpu",
            extra_future_param="ignored",
        )


def test_manifest_advertises_supports_img2img():
    """sd-turbo's MANIFEST capabilities advertise img2img support."""
    from muse.models.sd_turbo import MANIFEST
    assert MANIFEST["capabilities"].get("supports_img2img") is True


def _patched_pipe():
    """Return a fake pipeline whose .from_pretrained yields a mock that
    returns one PIL-shaped image when called. Mirrors the helper in
    tests/modalities/image_generation/runtimes/test_diffusers.py."""
    fake_pipe = MagicMock()
    fake_image = MagicMock()
    fake_image.size = (512, 512)
    fake_pipe.return_value.images = [fake_image]
    return fake_pipe


def test_sd_turbo_generate_img2img_branch():
    """sd_turbo's bundled Model honors init_image by loading the i2i pipeline."""
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_i2i = MagicMock()
    fake_i2i.from_pipe.return_value = _patched_pipe()

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForImage2Image", fake_i2i), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("repaint", init_image=init_img, strength=0.6)

    fake_i2i.from_pipe.assert_called_once()


def test_sd_turbo_generate_without_init_image_keeps_t2i_path():
    """init_image=None keeps the existing text-to-image path (no regression)."""
    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_i2i = MagicMock()

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForImage2Image", fake_i2i), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        m.generate("a fox")

    fake_i2i.from_pipe.assert_not_called()


def test_sd_turbo_generate_img2img_caches_pipeline():
    """Second img2img call reuses the cached pipeline."""
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_i2i = MagicMock()
    fake_i2i.from_pipe.return_value = _patched_pipe()

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForImage2Image", fake_i2i), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("a", init_image=init_img)
        m.generate("b", init_image=init_img)

    assert fake_i2i.from_pipe.call_count == 1


def test_sd_turbo_img2img_lazy_build_is_thread_safe():
    """Two concurrent first-access img2img calls must build the lazy
    i2i pipeline exactly once. Without a lock guarding the
    check-then-assign gap in _generate_img2img, two threads racing on
    the first call can both see self._i2i_pipe is None and both call
    from_pipe(...), briefly doubling VRAM and leaking one pipeline
    instance. A slow from_pipe widens the race window so the bug
    reproduces reliably when unguarded."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_i2i = MagicMock()

    def _slow_from_pipe(*_a, **_kw):
        time.sleep(0.05)
        return _patched_pipe()

    fake_i2i.from_pipe.side_effect = _slow_from_pipe

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForImage2Image", fake_i2i), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        init_img = Image.new("RGB", (64, 64))

        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [
                ex.submit(m.generate, "repaint", init_image=init_img, strength=0.6)
                for _ in range(2)
            ]
            for f in futures:
                f.result()

    assert fake_i2i.from_pipe.call_count == 1


def test_sd_turbo_img2img_bumps_steps_to_satisfy_strength_contract():
    """sd-turbo defaults to 1 step; with strength=0.4 must bump to >= ceil(1/0.4)=3."""
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_i2i = MagicMock()
    fake_i2i_pipe = _patched_pipe()
    fake_i2i.from_pipe.return_value = fake_i2i_pipe

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForImage2Image", fake_i2i), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu")
        init_img = Image.new("RGB", (64, 64))
        m.generate("repaint", init_image=init_img, strength=0.4)

    assert fake_i2i_pipe.call_args.kwargs["num_inference_steps"] == 3


def test_sd_turbo_img2img_uses_from_pipe_not_from_pretrained_to_share_vram():
    """from_pipe shares weights; from_pretrained would OOM on small GPUs.

    Regression for v0.17.2: SDXL-Turbo on a 12GB GPU crashed loading a
    second pipeline because from_pretrained allocated a fresh copy of
    all weights. from_pipe reuses the loaded UNet/VAE/text-encoders.
    """
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i_pipe = _patched_pipe()
    fake_t2i.from_pretrained.return_value = fake_t2i_pipe
    fake_i2i = MagicMock()
    fake_i2i.from_pipe.return_value = _patched_pipe()

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForImage2Image", fake_i2i), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("repaint", init_image=init_img, strength=0.4)

    # Critical: from_pipe was called (shares weights). from_pretrained on
    # the i2i class was NOT called (would have allocated a fresh copy).
    fake_i2i.from_pipe.assert_called_once_with(fake_t2i_pipe)
    fake_i2i.from_pretrained.assert_not_called()


# ---------------- inpaint() / vary() (#100, v0.21.0) ----------------


def test_manifest_advertises_supports_inpainting():
    """sd-turbo MANIFEST capabilities advertise inpainting support."""
    from muse.models.sd_turbo import MANIFEST
    assert MANIFEST["capabilities"].get("supports_inpainting") is True


def test_manifest_advertises_supports_variations():
    """sd-turbo MANIFEST capabilities advertise variations support."""
    from muse.models.sd_turbo import MANIFEST
    assert MANIFEST["capabilities"].get("supports_variations") is True


def test_sd_turbo_inpaint_uses_from_pipe():
    """sd_turbo's bundled Model loads inpaint pipeline via from_pipe."""
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_inp = MagicMock()
    fake_inp.from_pipe.return_value = _patched_pipe()

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForInpainting", fake_inp), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        init_img = Image.new("RGB", (64, 64))
        mask = Image.new("L", (64, 64), 255)
        m.inpaint("add a moon", init_image=init_img, mask_image=mask)

    fake_inp.from_pipe.assert_called_once()
    fake_inp.from_pretrained.assert_not_called()


def test_sd_turbo_inpaint_caches_pipeline():
    """Second inpaint call reuses the cached pipeline."""
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_inp = MagicMock()
    fake_inp.from_pipe.return_value = _patched_pipe()

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForInpainting", fake_inp), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        init_img = Image.new("RGB", (64, 64))
        mask = Image.new("L", (64, 64), 255)
        m.inpaint("a", init_image=init_img, mask_image=mask)
        m.inpaint("b", init_image=init_img, mask_image=mask)

    assert fake_inp.from_pipe.call_count == 1


def test_sd_turbo_inpaint_lazy_build_is_thread_safe():
    """Same double-build race as img2img, guarded by self._inp_lock."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_inp = MagicMock()

    def _slow_from_pipe(*_a, **_kw):
        time.sleep(0.05)
        return _patched_pipe()

    fake_inp.from_pipe.side_effect = _slow_from_pipe

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForInpainting", fake_inp), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        init_img = Image.new("RGB", (64, 64))
        mask = Image.new("L", (64, 64), 255)

        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [
                ex.submit(m.inpaint, "a", init_image=init_img, mask_image=mask)
                for _ in range(2)
            ]
            for f in futures:
                f.result()

    assert fake_inp.from_pipe.call_count == 1


def test_sd_turbo_inpaint_normalizes_rgba_mask_to_grayscale():
    """A non-L mask is converted before being passed to the inpaint pipe."""
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_inp = MagicMock()
    fake_inp_pipe = _patched_pipe()
    fake_inp.from_pipe.return_value = fake_inp_pipe

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForInpainting", fake_inp), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        init_img = Image.new("RGB", (64, 64))
        rgba_mask = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
        m.inpaint("repaint", init_image=init_img, mask_image=rgba_mask)

    passed_mask = fake_inp_pipe.call_args.kwargs["mask_image"]
    assert passed_mask.mode == "L"


def test_sd_turbo_inpaint_returns_imageresult_with_mode_inpaint():
    """ImageResult.metadata.mode should be 'inpaint' for the inpaint path."""
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_inp = MagicMock()
    fake_inp.from_pipe.return_value = _patched_pipe()

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForInpainting", fake_inp), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        init_img = Image.new("RGB", (64, 64))
        mask = Image.new("L", (64, 64), 255)
        result = m.inpaint("paint", init_image=init_img, mask_image=mask)

    assert isinstance(result, ImageResult)
    assert result.metadata["mode"] == "inpaint"


def test_sd_turbo_vary_delegates_to_img2img_with_empty_prompt():
    """vary() reuses the img2img path; passes prompt='' and strength=0.85."""
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_i2i = MagicMock()
    fake_i2i_pipe = _patched_pipe()
    fake_i2i.from_pipe.return_value = fake_i2i_pipe

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForImage2Image", fake_i2i), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        init_img = Image.new("RGB", (64, 64))
        m.vary(init_image=init_img)

    call_kwargs = fake_i2i_pipe.call_args.kwargs
    assert call_kwargs["prompt"] == ""
    assert call_kwargs["strength"] == 0.85


def test_sd_turbo_vary_returns_imageresult_with_mode_variations():
    """ImageResult.metadata.mode should be 'variations' on vary()."""
    from PIL import Image

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_i2i = MagicMock()
    fake_i2i.from_pipe.return_value = _patched_pipe()

    with patch("muse.models.sd_turbo.AutoPipelineForText2Image", fake_t2i), \
         patch("muse.models.sd_turbo.AutoPipelineForImage2Image", fake_i2i), \
         patch("muse.models.sd_turbo.torch", MagicMock()):
        from muse.models.sd_turbo import Model as SDTurboModel
        m = SDTurboModel(
            hf_repo="stabilityai/sd-turbo", local_dir="/tmp/fake", device="cpu",
        )
        init_img = Image.new("RGB", (64, 64))
        result = m.vary(init_image=init_img)

    assert isinstance(result, ImageResult)
    assert result.metadata["mode"] == "variations"
