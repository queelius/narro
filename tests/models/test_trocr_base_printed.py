"""Bundled trocr-base-printed discovery + manifest tests."""
from muse.models.trocr_base_printed import MANIFEST, Model


def test_manifest_required_keys():
    assert MANIFEST["model_id"] == "trocr-base-printed"
    assert MANIFEST["modality"] == "image/ocr"
    assert MANIFEST["hf_repo"] == "microsoft/trocr-base-printed"


def test_manifest_capabilities():
    """Default printed-text OCR; no handwritten or math support."""
    caps = MANIFEST["capabilities"]
    assert caps["device"] == "auto"
    assert caps["memory_gb"] == 0.7
    assert caps["max_new_tokens"] == 256
    assert caps["supports_handwritten"] is False
    assert caps["supports_math"] is False


def test_model_inherits_runtime():
    from muse.modalities.image_ocr.runtimes import HFVision2SeqRuntime
    assert issubclass(Model, HFVision2SeqRuntime)


def test_pip_extras_includes_pillow():
    """OCR runtime requires PIL; if pip_extras drops Pillow the smoke
    test (fresh-venv-smoke.yml) catches a load-time ImportError."""
    extras = MANIFEST["pip_extras"]
    assert any("Pillow" in e for e in extras)
    assert "torch>=2.1.0" in extras
    assert any(e.startswith("torchvision") for e in extras)
    assert any(e.startswith("transformers") for e in extras)


def test_license_is_mit():
    assert MANIFEST["license"] == "MIT"
