"""Tests for the image/embedding HF resolver plugin."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from muse.modalities.image_embedding.hf import HF_PLUGIN


def _fake_info(repo_id, tags=(), siblings=(), card=None):
    return SimpleNamespace(
        id=repo_id,
        tags=list(tags),
        siblings=[SimpleNamespace(rfilename=s) for s in siblings],
        card_data=card,
    )


def test_plugin_keys_present():
    keys = {"modality", "runtime_path", "pip_extras", "system_packages",
            "priority", "sniff", "resolve", "search"}
    assert keys.issubset(HF_PLUGIN.keys())


def test_plugin_modality_and_priority():
    assert HF_PLUGIN["modality"] == "image/embedding"
    # 105: between embedding/text (110) and image-generation file-pattern (100).
    assert HF_PLUGIN["priority"] == 105


def test_plugin_runtime_path():
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.image_embedding.runtimes.transformers_image"
        ":ImageEmbeddingRuntime"
    )


def test_plugin_pip_extras_include_image_processor_runtime_dependencies():
    extras = HF_PLUGIN["pip_extras"]
    assert "torch>=2.1.0" in extras
    assert any(e.startswith("torchvision") for e in extras)
    assert any("transformers" in e for e in extras)
    assert any("Pillow" in e for e in extras)


def test_sniff_true_for_image_feature_extraction_tag_with_processor_config():
    info = _fake_info(
        "facebook/dinov2-small",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json", "model.safetensors"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_feature_extraction_tag_with_processor_config():
    info = _fake_info(
        "openai/clip-vit-base-patch32",
        tags=["feature-extraction", "vision"],
        siblings=["preprocessor_config.json", "config.json"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_image_classification_tag_with_processor_config():
    """Image-classification repos that ship preprocessor_config also count
    (you can use a classifier as an embedder by extracting hidden states)."""
    info = _fake_info(
        "google/vit-base-patch16-224",
        tags=["image-classification"],
        siblings=["preprocessor_config.json"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_when_processor_config_missing():
    """Tag is present but no preprocessor_config.json -> not an image embedder."""
    info = _fake_info(
        "acme/text-only-feature-extractor",
        tags=["feature-extraction"],
        siblings=["model.safetensors", "config.json"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_when_no_relevant_tag():
    """preprocessor_config.json is present but no relevant tag -> skip."""
    info = _fake_info(
        "acme/random-image-thing",
        tags=["text-classification"],  # wrong family entirely
        siblings=["preprocessor_config.json"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_chat_completion_repo():
    info = _fake_info("Qwen/Qwen3-8B-GGUF",
                     tags=["chat", "text-generation"],
                     siblings=["model.gguf"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_summarization_repo():
    info = _fake_info("facebook/bart-large-cnn",
                     tags=["summarization"],
                     siblings=["pytorch_model.bin"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_repo_without_tags():
    info = _fake_info("acme/empty",
                     tags=[],
                     siblings=["preprocessor_config.json"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_no_tags_attribute():
    """Defensive: tags attribute missing entirely."""
    info = SimpleNamespace(
        id="x/y",
        siblings=[SimpleNamespace(rfilename="preprocessor_config.json")],
        card_data=None,
    )
    # Missing tags -> sniff returns False.
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_synthesizes_manifest_for_dinov2_small():
    info = _fake_info(
        "facebook/dinov2-small",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json", "model.safetensors"],
        card=SimpleNamespace(license="apache-2.0"),
    )
    resolved = HF_PLUGIN["resolve"]("facebook/dinov2-small", None, info)
    m = resolved.manifest
    assert m["model_id"] == "dinov2-small"
    assert m["modality"] == "image/embedding"
    assert m["hf_repo"] == "facebook/dinov2-small"
    assert m["license"] == "apache-2.0"
    caps = m["capabilities"]
    assert caps["device"] == "auto"
    assert caps["dimensions"] == 384  # DINOv2 small heuristic
    assert caps["supports_text_embeddings_too"] is False
    assert caps["image_size"] == 224


def test_resolve_clip_capability_defaults_base():
    info = _fake_info(
        "openai/clip-vit-base-patch32",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"](
        "openai/clip-vit-base-patch32", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 512
    assert caps["supports_text_embeddings_too"] is True


def test_resolve_clip_capability_defaults_large():
    info = _fake_info(
        "openai/clip-vit-large-patch14",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"](
        "openai/clip-vit-large-patch14", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 768
    assert caps["supports_text_embeddings_too"] is True


def test_resolve_siglip_capability_defaults_base():
    info = _fake_info(
        "google/siglip2-base-patch16-256",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"](
        "google/siglip2-base-patch16-256", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 768
    assert caps["supports_text_embeddings_too"] is True
    assert caps["image_size"] == 256


def test_resolve_siglip_capability_defaults_large():
    info = _fake_info(
        "google/siglip-large-patch16-384",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"](
        "google/siglip-large-patch16-384", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 1024
    assert caps["supports_text_embeddings_too"] is True


def test_resolve_dinov2_small_dim_384():
    info = _fake_info(
        "facebook/dinov2-small",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("facebook/dinov2-small", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 384
    assert caps["supports_text_embeddings_too"] is False


def test_resolve_dinov2_base_dim_768():
    info = _fake_info(
        "facebook/dinov2-base",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("facebook/dinov2-base", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 768


def test_resolve_dinov2_large_dim_1024():
    info = _fake_info(
        "facebook/dinov2-large",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("facebook/dinov2-large", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 1024


def test_resolve_vit_capability_default_dim_768():
    info = _fake_info(
        "google/vit-base-patch16-224",
        tags=["image-classification"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"](
        "google/vit-base-patch16-224", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 768
    assert caps["supports_text_embeddings_too"] is False


def test_resolve_vit_large_dim_1024():
    info = _fake_info(
        "google/vit-large-patch16-224",
        tags=["image-classification"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"](
        "google/vit-large-patch16-224", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 1024


def test_resolve_fallback_when_unknown_pattern():
    """Unknown architectures: dimensions absent (auto-detected at load),
    supports_text_embeddings_too=False."""
    info = _fake_info(
        "acme/mystery-vision",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("acme/mystery-vision", None, info)
    caps = resolved.manifest["capabilities"]
    assert "dimensions" not in caps  # left unset for auto-detection
    assert caps["supports_text_embeddings_too"] is False


def test_resolve_backend_path_points_to_runtime():
    info = _fake_info(
        "acme/x",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("acme/x", None, info)
    assert resolved.backend_path == HF_PLUGIN["runtime_path"]


def test_resolve_includes_pip_extras_in_manifest():
    info = _fake_info(
        "acme/x",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("acme/x", None, info)
    extras = resolved.manifest["pip_extras"]
    assert "torch>=2.1.0" in extras
    assert any(e.startswith("torchvision") for e in extras)
    assert any("transformers" in e for e in extras)
    assert any("Pillow" in e for e in extras)


def test_resolve_when_card_data_missing_license_is_none():
    info = _fake_info(
        "acme/x",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json"],
        card=None,
    )
    resolved = HF_PLUGIN["resolve"]("acme/x", None, info)
    assert resolved.manifest["license"] is None


def test_search_yields_results():
    api = MagicMock()
    repo1 = SimpleNamespace(id="facebook/dinov2-small", downloads=2_000_000)
    repo2 = SimpleNamespace(id="google/siglip2-base-patch16-256",
                            downloads=500_000)
    api.list_models.return_value = [repo1, repo2]
    out = list(HF_PLUGIN["search"](
        api, "vision", sort="downloads", limit=10,
    ))
    assert len(out) == 2
    assert out[0].uri == "hf://facebook/dinov2-small"
    assert out[0].modality == "image/embedding"
    assert out[0].downloads == 2_000_000
    assert out[1].model_id == "siglip2-base-patch16-256"


def test_search_calls_list_models_with_image_feature_extraction_filter():
    api = MagicMock()
    api.list_models.return_value = []
    list(HF_PLUGIN["search"](api, "x", sort="downloads", limit=5))
    _, kwargs = api.list_models.call_args
    assert kwargs["filter"] == "image-feature-extraction"
    assert kwargs["sort"] == "downloads"
    assert kwargs["limit"] == 5
    assert kwargs["search"] == "x"


def test_resolve_capabilities_include_memory_gb():
    """Every resolved manifest should declare memory_gb (used by
    `muse models list` for budgeting)."""
    info = _fake_info(
        "acme/x",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("acme/x", None, info)
    assert "memory_gb" in resolved.manifest["capabilities"]


def test_resolve_capabilities_include_device_auto():
    """device='auto' lets the runtime pick GPU when available."""
    info = _fake_info(
        "acme/x",
        tags=["image-feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("acme/x", None, info)
    assert resolved.manifest["capabilities"]["device"] == "auto"
