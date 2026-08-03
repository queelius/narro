"""Bundled vitpose-base-simple discovery + manifest tests."""
from muse.models.vitpose_base_simple import MANIFEST, Model


def test_manifest_required_keys():
    assert MANIFEST["model_id"] == "vitpose-base-simple"
    assert MANIFEST["modality"] == "image/cv"
    assert MANIFEST["hf_repo"] == "usyd-community/vitpose-base-simple"


def test_manifest_capabilities():
    caps = MANIFEST["capabilities"]
    assert caps["supports_depth"] is False
    assert caps["supports_keypoints"] is True
    assert caps["supports_detection"] is False
    assert caps["device"] == "auto"


def test_model_inherits_runtime():
    from muse.modalities.image_cv.runtimes import HFKeypointRuntime
    assert issubclass(Model, HFKeypointRuntime)


def test_pip_extras_require_vitpose_transformers_and_scipy():
    """Fresh pulls include the explicit ViTPose API and processor deps."""
    extras = MANIFEST["pip_extras"]
    transformers_pins = [e for e in extras if e.startswith("transformers")]
    assert transformers_pins == ["transformers>=4.48.0"]
    assert "scipy" in extras


def test_pip_extras_include_transformers_5_image_processor_dependency():
    assert "torch>=2.1.0" in MANIFEST["pip_extras"]
    assert any(e.startswith("torchvision") for e in MANIFEST["pip_extras"])


def test_license():
    assert MANIFEST["license"] == "Apache 2.0"
