"""Tests for the bundled mert_v1_95m script (fully mocked).

Module-level imports of `muse.models.mert_v1_95m` are DELIBERATELY
avoided: another test in the suite (test_discovery_robust_to_broken_deps)
pops `muse.models.*` from sys.modules and re-imports them, which means
a top-level `import muse.models.mert_v1_95m` captures a stale module
reference once that test has run. We re-resolve the live module inside
helpers / each test instead.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from muse.modalities.audio_embedding import AudioEmbeddingResult


def _mert_script():
    """Resolve the live module each call so test_discovery's sys.modules
    eviction doesn't leave us holding a stale reference."""
    import importlib
    return importlib.import_module("muse.models.mert_v1_95m")


def _manifest():
    return _mert_script().MANIFEST


def _np_tensor(arr):
    arr = np.asarray(arr, dtype=np.float32)
    t = MagicMock()
    t.detach.return_value = t
    t.to.return_value = t
    t.float.return_value = t
    t.numpy.return_value = arr
    t.shape = arr.shape
    return t


def _patched_setup(*, embedding_arr=None, with_cuda=False, sample_rate=24000):
    """Install fake transformers + torch + librosa on the live module.

    Returns (mod, fake_processor, fake_model, fake_extractor_class,
    fake_model_class, fake_torch, fake_librosa).
    """
    if embedding_arr is None:
        embedding_arr = np.array([[0.1] * 768], dtype=np.float32)

    mod = _mert_script()

    # Fake feature extractor: __call__(audio_list, sampling_rate=...,
    # return_tensors='pt') -> inputs
    fake_inputs = MagicMock()
    fake_inputs.to.return_value = fake_inputs
    fake_processor = MagicMock(return_value=fake_inputs)
    fake_extractor_class = MagicMock()
    fake_extractor_class.from_pretrained.return_value = fake_processor

    # Fake model: forward returns outputs; mean(dim=1) on
    # last_hidden_state returns the wrapped tensor.
    pooled_tensor = _np_tensor(embedding_arr)
    seq = MagicMock()
    seq.mean = MagicMock(return_value=pooled_tensor)

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

    fake_librosa = MagicMock()
    fake_librosa.load = MagicMock(
        return_value=(np.zeros(sample_rate, dtype=np.float32), sample_rate),
    )

    mod.AutoModel = fake_model_class
    mod.AutoFeatureExtractor = fake_extractor_class
    mod.torch = fake_torch
    mod.librosa = fake_librosa
    return (
        mod, fake_processor, fake_model,
        fake_extractor_class, fake_model_class,
        fake_torch, fake_librosa,
    )


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    yield
    mod = _mert_script()
    mod.torch = None
    mod.AutoModel = None
    mod.AutoFeatureExtractor = None
    mod.librosa = None


def test_manifest_required_fields():
    m = _manifest()
    assert m["model_id"] == "mert-v1-95m"
    assert m["modality"] == "audio/embedding"
    assert m["hf_repo"] == "m-a-p/MERT-v1-95M"
    assert "pip_extras" in m
    assert "torch>=2.6.0" in m["pip_extras"]
    assert any("transformers" in x for x in m["pip_extras"])
    assert any("librosa" in x for x in m["pip_extras"])


def test_manifest_capabilities_shape():
    caps = _manifest()["capabilities"]
    assert caps["device"] == "auto"
    assert caps["dimensions"] == 768
    assert caps["sample_rate"] == 24000
    assert caps["max_duration_seconds"] == 60.0
    assert caps["supports_text_embeddings_too"] is False
    assert caps["trust_remote_code"] is True
    assert "memory_gb" in caps


def test_manifest_allow_patterns_includes_required_files():
    """Weights, processor config, and code are all required locally."""
    patterns = _manifest()["allow_patterns"]
    assert any("preprocessor_config" in p for p in patterns)
    assert "pytorch_model.bin" in patterns
    assert any(".py" in p for p in patterns)


def test_manifest_pins_reviewed_remote_code_revision():
    revision = _manifest()["revision"]
    assert len(revision) == 40
    assert all(c in "0123456789abcdef" for c in revision)


def test_manifest_license_is_noncommercial_cc_by_nc_4_0():
    assert _manifest()["license"] == "CC-BY-NC-4.0"


def test_model_class_exists():
    mod = _mert_script()
    assert hasattr(mod, "Model")
    assert mod.Model.model_id == "mert-v1-95m"
    assert mod.Model.dimensions == 768


def test_model_construction_lazy_imports():
    """Model.__init__ should call _ensure_deps and read transformers
    via the module's sentinels so tests can patch them."""
    mod, _, _, fake_ext_class, fake_model_class, _, _ = _patched_setup()
    m = mod.Model(hf_repo="m-a-p/MERT-v1-95M", local_dir=None, device="cpu")
    assert m._device == "cpu"
    fake_ext_class.from_pretrained.assert_called_once()
    fake_model_class.from_pretrained.assert_called_once()


def test_model_prefers_local_dir():
    mod, _, _, fake_ext_class, fake_model_class, _, _ = _patched_setup()
    mod.Model(
        hf_repo="m-a-p/MERT-v1-95M",
        local_dir="/tmp/mert",
        device="cpu",
    )
    args, _ = fake_ext_class.from_pretrained.call_args
    assert args[0] == "/tmp/mert"
    args2, _ = fake_model_class.from_pretrained.call_args
    assert args2[0] == "/tmp/mert"


def test_model_falls_back_to_hf_repo_when_no_local_dir():
    mod, _, _, fake_ext_class, _, _, _ = _patched_setup()
    mod.Model(hf_repo="m-a-p/MERT-v1-95M", local_dir=None, device="cpu")
    args, _ = fake_ext_class.from_pretrained.call_args
    assert args[0] == "m-a-p/MERT-v1-95M"


def test_model_threads_trust_remote_code_to_processor_and_model():
    mod, _, _, fake_ext_class, fake_model_class, _, _ = _patched_setup()
    mod.Model(hf_repo="m-a-p/MERT-v1-95M", device="cpu", trust_remote_code=True)
    _, kwargs = fake_ext_class.from_pretrained.call_args
    assert kwargs.get("trust_remote_code") is True
    _, mkwargs = fake_model_class.from_pretrained.call_args
    assert mkwargs.get("trust_remote_code") is True


def test_remote_fallback_threads_reviewed_revision():
    mod, _, _, fake_ext_class, fake_model_class, _, _ = _patched_setup()
    mod.Model(hf_repo="m-a-p/MERT-v1-95M", local_dir=None, device="cpu")
    revision = mod.MANIFEST["revision"]
    assert fake_ext_class.from_pretrained.call_args.kwargs["revision"] == revision
    assert fake_model_class.from_pretrained.call_args.kwargs["revision"] == revision


def test_embed_returns_audio_embedding_result():
    mod, _, _, _, _, _, _ = _patched_setup()
    m = mod.Model(hf_repo="m-a-p/MERT-v1-95M", device="cpu")
    out = m.embed([b"FAKEWAV"])
    assert isinstance(out, AudioEmbeddingResult)
    assert out.model_id == "mert-v1-95m"
    assert out.n_audio_clips == 1


def test_embed_returns_768_dim_by_default():
    arr = np.tile(np.arange(768, dtype=np.float32) / 768, (1, 1))
    mod, _, _, _, _, _, _ = _patched_setup(embedding_arr=arr)
    m = mod.Model(hf_repo="m-a-p/MERT-v1-95M", device="cpu")
    out = m.embed([b"FAKEWAV"])
    assert out.dimensions == 768
    assert len(out.embeddings[0]) == 768


def test_embed_passes_return_tensors_pt():
    mod, fake_processor, _, _, _, _, _ = _patched_setup()
    m = mod.Model(hf_repo="m-a-p/MERT-v1-95M", device="cpu")
    m.embed([b"FAKEWAV"])
    _, kwargs = fake_processor.call_args
    assert kwargs.get("return_tensors") == "pt"


def test_embed_passes_sampling_rate_to_processor():
    mod, fake_processor, _, _, _, _, _ = _patched_setup()
    m = mod.Model(hf_repo="m-a-p/MERT-v1-95M", device="cpu", sample_rate=24000)
    m.embed([b"FAKEWAV"])
    _, kwargs = fake_processor.call_args
    assert kwargs.get("sampling_rate") == 24000


def test_embed_calls_librosa_load_per_clip():
    mod, _, _, _, _, _, fake_librosa = _patched_setup(
        embedding_arr=np.array([[0.1] * 768, [0.2] * 768], dtype=np.float32),
    )
    m = mod.Model(hf_repo="m-a-p/MERT-v1-95M", device="cpu")
    m.embed([b"a", b"b"])
    assert fake_librosa.load.call_count == 2


def test_embed_calls_librosa_load_with_target_sample_rate():
    mod, _, _, _, _, _, fake_librosa = _patched_setup()
    m = mod.Model(hf_repo="m-a-p/MERT-v1-95M", device="cpu", sample_rate=24000)
    m.embed([b"FAKEWAV"])
    _, kwargs = fake_librosa.load.call_args
    assert kwargs["sr"] == 24000
    assert kwargs["mono"] is True


def test_embed_wraps_single_input_into_list():
    mod, fake_processor, _, _, _, _, _ = _patched_setup()
    m = mod.Model(hf_repo="m-a-p/MERT-v1-95M", device="cpu")
    m.embed(b"single-not-in-list")
    args, _ = fake_processor.call_args
    # processor was called with a list of decoded audio
    assert isinstance(args[0], list)
    assert len(args[0]) == 1


def test_embed_n_audio_clips_matches_input_length():
    arr = np.array([[0.1] * 768, [0.2] * 768, [0.3] * 768], dtype=np.float32)
    mod, _, _, _, _, _, _ = _patched_setup(embedding_arr=arr)
    m = mod.Model(hf_repo="m-a-p/MERT-v1-95M", device="cpu")
    out = m.embed([b"a", b"b", b"c"])
    assert out.n_audio_clips == 3


def test_embed_metadata_marks_source_as_mert():
    mod, _, _, _, _, _, _ = _patched_setup()
    m = mod.Model(hf_repo="m-a-p/MERT-v1-95M", device="cpu")
    out = m.embed([b"FAKEWAV"])
    assert out.metadata.get("source") == "mert"


def test_embed_metadata_records_sample_rate_used():
    mod, _, _, _, _, _, _ = _patched_setup()
    m = mod.Model(hf_repo="m-a-p/MERT-v1-95M", device="cpu", sample_rate=24000)
    out = m.embed([b"FAKEWAV"])
    assert out.metadata.get("sample_rate_used") == 24000


def test_model_raises_when_transformers_missing():
    """Stub _ensure_deps so it leaves AutoModel as None
    (simulates transformers not being installed in the venv)."""
    mod = _mert_script()
    fake_torch = MagicMock()
    fake_librosa = MagicMock()
    with patch.object(mod, "AutoModel", None), \
            patch.object(mod, "AutoFeatureExtractor", None), \
            patch.object(mod, "_ensure_deps", lambda: None), \
            patch.object(mod, "torch", fake_torch), \
            patch.object(mod, "librosa", fake_librosa):
        with pytest.raises(RuntimeError, match="transformers"):
            mod.Model(hf_repo="m-a-p/MERT-v1-95M", local_dir=None)


def test_model_raises_when_librosa_missing():
    """Stub _ensure_deps so librosa stays None."""
    mod = _mert_script()
    fake_torch = MagicMock()
    fake_model_class = MagicMock()
    fake_ext_class = MagicMock()
    with patch.object(mod, "AutoModel", fake_model_class), \
            patch.object(mod, "AutoFeatureExtractor", fake_ext_class), \
            patch.object(mod, "_ensure_deps", lambda: None), \
            patch.object(mod, "torch", fake_torch), \
            patch.object(mod, "librosa", None):
        with pytest.raises(RuntimeError, match="librosa"):
            mod.Model(hf_repo="m-a-p/MERT-v1-95M", local_dir=None)


def test_select_device_auto_falls_back_to_cpu_with_no_torch():
    mod = _mert_script()
    with patch.object(mod, "torch", None):
        assert mod._select_device("auto") == "cpu"


def test_select_device_auto_picks_cuda():
    mod = _mert_script()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.object(mod, "torch", fake_torch):
        assert mod._select_device("auto") == "cuda"


def test_select_device_auto_picks_mps_when_cuda_absent():
    mod = _mert_script()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = True
    with patch.object(mod, "torch", fake_torch):
        assert mod._select_device("auto") == "mps"


def test_select_device_explicit_passes_through():
    mod = _mert_script()
    with patch.object(mod, "torch", MagicMock()):
        assert mod._select_device("cuda:1") == "cuda:1"


def test_set_inference_mode_invokes_method_when_present():
    mod = _mert_script()
    m = MagicMock()
    mod._set_inference_mode(m)
    assert m.eval.call_count == 1


def test_set_inference_mode_safe_when_method_missing():
    mod = _mert_script()

    class NoEvalMethod:
        pass

    mod._set_inference_mode(NoEvalMethod())  # no exception


def test_mean_pool_time_calls_mean_dim_1():
    mod = _mert_script()
    expected = MagicMock()
    seq = MagicMock()
    seq.mean = MagicMock(return_value=expected)
    outputs = MagicMock()
    outputs.last_hidden_state = seq
    out = mod._mean_pool_time(outputs)
    assert out is expected
    seq.mean.assert_called_once_with(dim=1)


def test_mert_capability_supports_text_embeddings_too_is_false():
    """MERT v1 is purely audio-side; no text tower."""
    caps = _manifest()["capabilities"]
    assert caps["supports_text_embeddings_too"] is False


def test_model_class_dimensions_class_attr_is_768():
    """Class-level dimensions attribute matches manifest value."""
    mod = _mert_script()
    assert mod.Model.dimensions == 768


def test_decode_audio_truncates_to_max_seconds():
    mod = _mert_script()
    long = np.arange(48000, dtype=np.float32)  # 2 seconds at 24kHz
    fake_librosa = MagicMock()
    fake_librosa.load = MagicMock(return_value=(long, 24000))
    with patch.object(mod, "librosa", fake_librosa):
        out = mod._decode_audio(b"raw", sample_rate=24000, max_seconds=1.0)
    assert len(out) == 24000
