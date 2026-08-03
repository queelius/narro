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


def test_pip_extras_requires_transformers_4_46():
    """AutoModelForKeypointDetection arrived in 4.46; the bundled
    pip_extras must pin that minimum so a fresh-venv pull doesn't
    silently install an older transformers and crash on load."""
    extras = MANIFEST["pip_extras"]
    transformers_pins = [e for e in extras if e.startswith("transformers")]
    assert transformers_pins
    # Must be >= 4.46.x.
    pin = transformers_pins[0]
    assert "4.46" in pin or ">4.46" in pin or ">=4.46" in pin


def test_pip_extras_include_transformers_5_image_processor_dependency():
    assert "torch>=2.1.0" in MANIFEST["pip_extras"]
    assert any(e.startswith("torchvision") for e in MANIFEST["pip_extras"])


def test_license():
    assert MANIFEST["license"] == "Apache 2.0"
