from muse.models.starvector_1b_im2svg import MANIFEST, Model
from muse.modalities.image_vectorization.runtimes import StarVectorRuntime


def test_manifest_shape_and_exact_revision():
    assert MANIFEST["model_id"] == "starvector-1b-im2svg"
    assert MANIFEST["modality"] == "image/vectorization"
    assert MANIFEST["hf_repo"] == "starvector/starvector-1b-im2svg"
    assert MANIFEST["license"] == "Apache 2.0"
    capabilities = MANIFEST["capabilities"]
    assert len(capabilities["revision"]) == 40
    assert capabilities["supports_image_to_svg"] is True
    assert capabilities["static_svg_only"] is True
    assert capabilities["memory_gb"] <= 12
    assert "*.safetensors" in capabilities["allow_patterns"]
    assert "*.py" not in capabilities["allow_patterns"]


def test_model_uses_audited_runtime():
    assert issubclass(Model, StarVectorRuntime)


def test_manifest_dependencies_are_inference_only():
    extras = " ".join(MANIFEST["pip_extras"])
    for required in ("torch", "transformers==4.49.0", "Pillow", "numpy"):
        assert required in extras
    for excluded in ("flash_attn", "gradio", "deepspeed", "starvector @"):
        assert excluded not in extras
