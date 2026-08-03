"""Tests for the bundled dinov2_small script (fully mocked).

Module-level imports of `muse.models.dinov2_small` are DELIBERATELY
avoided: another test in the suite (test_discovery_robust_to_broken_deps)
pops `muse.models.*` from sys.modules and re-imports them, which means
a top-level `import muse.models.dinov2_small` captures a stale module
reference once that test has run. We re-resolve the live module inside
helpers / each test instead.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from muse.modalities.image_embedding import ImageEmbeddingResult


def _dinov2_script():
    """Resolve the live module each call so test_discovery's sys.modules
    eviction doesn't leave us holding a stale reference."""
    import importlib
    return importlib.import_module("muse.models.dinov2_small")


def _manifest():
    return _dinov2_script().MANIFEST


def _np_tensor(arr):
    arr = np.asarray(arr, dtype=np.float32)
    t = MagicMock()
    t.detach.return_value = t
    t.to.return_value = t
    t.float.return_value = t
    t.numpy.return_value = arr
    t.shape = arr.shape
    return t


def _patched_setup(*, embedding_arr=None, with_cuda=False):
    """Install fake transformers + torch on the live dinov2_small module.

    Returns (mod, fake_processor, fake_model, fake_processor_class,
    fake_model_class, fake_torch).
    """
    if embedding_arr is None:
        embedding_arr = np.array([[0.1] * 384], dtype=np.float32)

    mod = _dinov2_script()

    # Fake processor: __call__(images=..., return_tensors="pt") -> inputs
    fake_inputs = MagicMock()
    fake_inputs.to.return_value = fake_inputs
    fake_processor = MagicMock(return_value=fake_inputs)
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.return_value = fake_processor

    # Fake model: forward returns outputs with last_hidden_state.
    cls_tensor = _np_tensor(embedding_arr)
    seq = MagicMock()
    seq.__getitem__ = lambda self, idx: cls_tensor

    class FakeOutputs:
        pass
    fake_outputs = FakeOutputs()
    fake_outputs.last_hidden_state = seq

    fake_model = MagicMock(return_value=fake_outputs)
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = None
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = with_cuda
    fake_torch.backends.mps.is_available.return_value = False
    fake_torch.inference_mode.return_value.__enter__ = MagicMock(return_value=None)
    fake_torch.inference_mode.return_value.__exit__ = MagicMock(return_value=None)

    mod.AutoModel = fake_model_class
    mod.AutoImageProcessor = fake_processor_class
    mod.torch = fake_torch
    return mod, fake_processor, fake_model, fake_processor_class, fake_model_class, fake_torch


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    yield
    mod = _dinov2_script()
    mod.torch = None
    mod.AutoModel = None
    mod.AutoImageProcessor = None


def test_manifest_required_fields():
    m = _manifest()
    assert m["model_id"] == "dinov2-small"
    assert m["modality"] == "image/embedding"
    assert m["hf_repo"] == "facebook/dinov2-small"
    assert "pip_extras" in m
    assert "torch>=2.1.0" in m["pip_extras"]
    assert any("transformers" in x for x in m["pip_extras"])
    assert any("Pillow" in x for x in m["pip_extras"])


def test_manifest_includes_transformers_5_image_processor_dependency():
    assert any(x.startswith("torchvision") for x in _manifest()["pip_extras"])


def test_manifest_capabilities_shape():
    caps = _manifest()["capabilities"]
    assert caps["device"] == "auto"
    assert caps["dimensions"] == 384
    assert caps["image_size"] == 224
    assert caps["supports_text_embeddings_too"] is False
    assert "memory_gb" in caps


def test_manifest_allow_patterns_includes_preprocessor_config():
    """preprocessor_config.json is essential for image processor load."""
    patterns = _manifest()["allow_patterns"]
    assert any("preprocessor_config" in p for p in patterns)
    assert any(".safetensors" in p for p in patterns)


def test_manifest_license_is_apache_2_0():
    assert _manifest()["license"] == "Apache 2.0"


def test_model_class_exists():
    mod = _dinov2_script()
    assert hasattr(mod, "Model")
    assert mod.Model.model_id == "dinov2-small"
    assert mod.Model.dimensions == 384


def test_model_construction_lazy_imports():
    """Model.__init__ should call _ensure_deps and read transformers
    via the module's sentinels so tests can patch them."""
    mod, _, _, fake_proc_class, fake_model_class, _ = _patched_setup()
    m = mod.Model(hf_repo="facebook/dinov2-small", local_dir=None, device="cpu")
    assert m._device == "cpu"
    fake_proc_class.from_pretrained.assert_called_once()
    fake_model_class.from_pretrained.assert_called_once()


def test_model_prefers_local_dir():
    mod, _, _, fake_proc_class, fake_model_class, _ = _patched_setup()
    mod.Model(
        hf_repo="facebook/dinov2-small",
        local_dir="/tmp/dinov2",
        device="cpu",
    )
    args, _ = fake_proc_class.from_pretrained.call_args
    assert args[0] == "/tmp/dinov2"
    args2, _ = fake_model_class.from_pretrained.call_args
    assert args2[0] == "/tmp/dinov2"


def test_model_falls_back_to_hf_repo_when_no_local_dir():
    mod, _, _, fake_proc_class, _, _ = _patched_setup()
    mod.Model(hf_repo="facebook/dinov2-small", local_dir=None, device="cpu")
    args, _ = fake_proc_class.from_pretrained.call_args
    assert args[0] == "facebook/dinov2-small"


def test_embed_returns_image_embedding_result():
    mod, _, _, _, _, _ = _patched_setup()
    m = mod.Model(hf_repo="facebook/dinov2-small", device="cpu")
    out = m.embed(["pretend-pil-image"])
    assert isinstance(out, ImageEmbeddingResult)
    assert out.model_id == "dinov2-small"
    assert out.n_images == 1


def test_embed_returns_384_dim_by_default():
    mod, _, _, _, _, _ = _patched_setup(
        embedding_arr=np.tile(np.arange(384, dtype=np.float32) / 384, (1, 1)),
    )
    m = mod.Model(hf_repo="facebook/dinov2-small", device="cpu")
    out = m.embed(["x"])
    assert out.dimensions == 384
    assert len(out.embeddings[0]) == 384


def test_embed_truncates_to_requested_dimensions():
    mod, _, _, _, _, _ = _patched_setup(
        embedding_arr=np.tile(np.arange(384, dtype=np.float32) / 384, (1, 1)),
    )
    m = mod.Model(hf_repo="facebook/dinov2-small", device="cpu")
    out = m.embed(["x"], dimensions=128)
    assert out.dimensions == 128
    assert len(out.embeddings[0]) == 128


def test_embed_no_truncation_when_request_dim_geq_native():
    mod, _, _, _, _, _ = _patched_setup(
        embedding_arr=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    )
    m = mod.Model(hf_repo="facebook/dinov2-small", device="cpu")
    out = m.embed(["x"], dimensions=10)
    assert out.dimensions == 3


def test_embed_passes_return_tensors_pt():
    mod, fake_processor, _, _, _, _ = _patched_setup()
    m = mod.Model(hf_repo="facebook/dinov2-small", device="cpu")
    m.embed(["x"])
    _, kwargs = fake_processor.call_args
    assert kwargs.get("return_tensors") == "pt"


def test_embed_passes_image_list_to_processor():
    mod, fake_processor, _, _, _, _ = _patched_setup()
    m = mod.Model(hf_repo="facebook/dinov2-small", device="cpu")
    m.embed(["a", "b", "c"])
    _, kwargs = fake_processor.call_args
    assert kwargs.get("images") == ["a", "b", "c"]


def test_embed_wraps_single_image_into_list():
    mod, fake_processor, _, _, _, _ = _patched_setup()
    m = mod.Model(hf_repo="facebook/dinov2-small", device="cpu")
    m.embed("single-not-in-list")
    _, kwargs = fake_processor.call_args
    assert kwargs.get("images") == ["single-not-in-list"]


def test_embed_n_images_matches_input_length():
    arr = np.array([[0.1] * 384, [0.2] * 384, [0.3] * 384], dtype=np.float32)
    mod, _, _, _, _, _ = _patched_setup(embedding_arr=arr)
    m = mod.Model(hf_repo="facebook/dinov2-small", device="cpu")
    out = m.embed(["a", "b", "c"])
    assert out.n_images == 3


def test_embed_metadata_marks_source_as_dinov2():
    mod, _, _, _, _, _ = _patched_setup()
    m = mod.Model(hf_repo="facebook/dinov2-small", device="cpu")
    out = m.embed(["x"])
    assert out.metadata.get("source") == "dinov2"


def test_model_raises_when_transformers_missing():
    """Stub _ensure_deps so it leaves AutoModel as None
    (simulates transformers not being installed in the venv)."""
    mod = _dinov2_script()
    fake_torch = MagicMock()
    with patch.object(mod, "AutoModel", None), \
            patch.object(mod, "AutoImageProcessor", None), \
            patch.object(mod, "_ensure_deps", lambda: None), \
            patch.object(mod, "torch", fake_torch):
        with pytest.raises(RuntimeError, match="transformers"):
            mod.Model(hf_repo="facebook/dinov2-small", local_dir=None)


def test_select_device_auto_falls_back_to_cpu_with_no_torch():
    mod = _dinov2_script()
    with patch.object(mod, "torch", None):
        assert mod._select_device("auto") == "cpu"


def test_select_device_auto_picks_cuda():
    mod = _dinov2_script()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.object(mod, "torch", fake_torch):
        assert mod._select_device("auto") == "cuda"


def test_select_device_auto_picks_mps_when_cuda_absent():
    mod = _dinov2_script()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = True
    with patch.object(mod, "torch", fake_torch):
        assert mod._select_device("auto") == "mps"


def test_select_device_explicit_passes_through():
    mod = _dinov2_script()
    with patch.object(mod, "torch", MagicMock()):
        assert mod._select_device("cuda:1") == "cuda:1"


def test_set_inference_mode_invokes_method_when_present():
    mod = _dinov2_script()
    m = MagicMock()
    mod._set_inference_mode(m)
    assert m.eval.call_count == 1


def test_set_inference_mode_safe_when_method_missing():
    mod = _dinov2_script()

    class NoEvalMethod:
        pass

    mod._set_inference_mode(NoEvalMethod())  # no exception


def test_extract_cls_token_returns_first_row():
    """The CLS token slice is [:, 0] of last_hidden_state."""
    mod = _dinov2_script()
    expected = MagicMock()
    seq = MagicMock()
    seq.__getitem__ = lambda self, idx: expected
    outputs = MagicMock()
    outputs.last_hidden_state = seq
    assert mod._extract_cls_token(outputs) is expected


def test_dinov2_capability_supports_text_embeddings_too_is_false():
    """DINOv2 is purely image-side; no text tower."""
    caps = _manifest()["capabilities"]
    assert caps["supports_text_embeddings_too"] is False


def test_model_class_dimensions_class_attr_is_384():
    """Class-level dimensions attribute matches manifest value."""
    mod = _dinov2_script()
    assert mod.Model.dimensions == 384
