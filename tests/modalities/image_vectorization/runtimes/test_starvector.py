from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from muse.modalities.image_vectorization.runtimes import starvector as mod


@pytest.fixture(autouse=True)
def _restore_sentinels():
    names = (
        "torch", "np", "AutoTokenizer", "StoppingCriteriaList",
        "StarVectorConfig", "StarVectorForCausalLM", "_IMPORT_ERROR",
    )
    original = {name: getattr(mod, name) for name in names}
    yield
    for name, value in original.items():
        setattr(mod, name, value)


def _wire_constructor():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    fake_torch.float16 = "fp16"
    fake_torch.float32 = "fp32"
    mod.torch = fake_torch
    mod.np = MagicMock()
    mod.StoppingCriteriaList = MagicMock()

    config = SimpleNamespace(
        model_type="starvector",
        image_encoder_type="clip",
        starcoder_model_name="bigcode/starcoderbase-1b",
        hidden_size=2048,
        num_hidden_layers=24,
        adapter_norm="batch_norm",
        image_size=224,
        vocab_size=49156,
        num_attention_heads=16,
        multi_query=True,
        max_length_train=8192,
    )
    config_factory = MagicMock()
    config_factory.from_pretrained.return_value = config
    mod.StarVectorConfig = config_factory

    tokenizer_factory = MagicMock()
    tokenizer_factory.from_pretrained.return_value = MagicMock()
    mod.AutoTokenizer = tokenizer_factory

    model = MagicMock()
    model.to.return_value = model
    model_factory = MagicMock()
    model_factory.from_pretrained.return_value = model
    mod.StarVectorForCausalLM = model_factory
    return config_factory, tokenizer_factory, model_factory, model


def test_runtime_refuses_unreviewed_repo_before_import():
    with pytest.raises(RuntimeError, match="exact-only"):
        mod.StarVectorRuntime(model_id="x", hf_repo="attacker/model")


def test_runtime_refuses_unreviewed_revision_before_import():
    with pytest.raises(RuntimeError, match="reviewed revision"):
        mod.StarVectorRuntime(
            model_id="x",
            revision="0000000000000000000000000000000000000000",
        )


def test_constructor_uses_local_data_without_remote_code():
    config_factory, tokenizer_factory, model_factory, model = _wire_constructor()
    runtime = mod.StarVectorRuntime(
        model_id="starvector",
        local_dir="/models/pinned-snapshot",
        device="cpu",
        dtype="fp32",
    )
    assert runtime.model_id == "starvector"
    config_factory.from_pretrained.assert_called_once_with(
        "/models/pinned-snapshot", local_files_only=True,
    )
    tokenizer_kwargs = tokenizer_factory.from_pretrained.call_args.kwargs
    assert tokenizer_kwargs["local_files_only"] is True
    assert "trust_remote_code" not in tokenizer_kwargs
    model_kwargs = model_factory.from_pretrained.call_args.kwargs
    assert model_kwargs["local_files_only"] is True
    assert model_kwargs["low_cpu_mem_usage"] is True
    assert "trust_remote_code" not in model_kwargs
    model.to.assert_called_once_with("cpu")


def test_remote_fallback_is_revision_pinned():
    config_factory, tokenizer_factory, model_factory, _ = _wire_constructor()
    mod.StarVectorRuntime(
        model_id="starvector", device="cpu", dtype="fp32",
    )
    for factory in (config_factory, tokenizer_factory, model_factory):
        kwargs = factory.from_pretrained.call_args.kwargs
        assert kwargs["revision"] == mod.PINNED_REVISION
        assert kwargs["local_files_only"] is False


@pytest.mark.parametrize("change,message", [
    ({"model_type": "other"}, "not a StarVector"),
    ({"image_encoder_type": "siglip"}, "CLIP"),
    ({"starcoder_model_name": "bigcode/starcoder2-3b"}, "StarCoder v1"),
    ({"hidden_size": 1024}, "hidden size"),
    ({"num_hidden_layers": 12}, "layer count"),
    ({"adapter_norm": "layer_norm"}, "adapter_norm"),
    ({"image_size": 384}, "image_size"),
    ({"vocab_size": 49152}, "vocab_size"),
    ({"num_attention_heads": 8}, "num_attention_heads"),
    ({"multi_query": False}, "multi_query"),
    ({"max_length_train": 4096}, "max_length_train"),
])
def test_config_shape_guard(change, message):
    config = SimpleNamespace(
        model_type="starvector",
        image_encoder_type="clip",
        starcoder_model_name="bigcode/starcoderbase-1b",
        hidden_size=2048,
        num_hidden_layers=24,
        adapter_norm="batch_norm",
        image_size=224,
        vocab_size=49156,
        num_attention_heads=16,
        multi_query=True,
        max_length_train=8192,
    )
    for key, value in change.items():
        setattr(config, key, value)
    with pytest.raises(RuntimeError, match=message):
        mod._validate_config(config)


def test_stop_on_svg_suffix():
    torch = pytest.importorskip("torch")
    stop = mod._StopOnSuffix([4, 5])
    assert stop(torch.tensor([[1, 2, 4, 5]]), None) is True
    assert stop(torch.tensor([[1, 2, 4]]), None) is False


def test_preprocess_image_matches_expected_shape():
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    mod.torch = torch
    mod.np = numpy

    image = Image.new("RGBA", (20, 10), (255, 0, 0, 128))
    tensor = mod._preprocess_image(image, size=224)
    assert tuple(tensor.shape) == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_preprocess_image_composites_palette_transparency_on_white():
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    mod.torch = torch
    mod.np = numpy

    image = Image.new("P", (1, 1))
    image.putpalette([255, 0, 0] + [0, 0, 0] * 255)
    image.info["transparency"] = 0
    tensor = mod._preprocess_image(image, size=1)

    expected = torch.tensor([
        (1.0 - 0.48145466) / 0.26862954,
        (1.0 - 0.4578275) / 0.26130258,
        (1.0 - 0.40821073) / 0.27577711,
    ])
    assert torch.allclose(tensor[0, :, 0, 0], expected)
