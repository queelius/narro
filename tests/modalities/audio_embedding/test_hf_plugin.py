"""Tests for the audio/embedding HF resolver plugin."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from muse.modalities.audio_embedding.hf import HF_PLUGIN


_FAKE_REVISION = "1" * 40


def _fake_info(repo_id, tags=(), siblings=(), card=None, sha=_FAKE_REVISION):
    return SimpleNamespace(
        id=repo_id,
        tags=list(tags),
        siblings=[SimpleNamespace(rfilename=s) for s in siblings],
        card_data=card,
        sha=sha,
    )


def test_plugin_keys_present():
    keys = {"modality", "runtime_path", "pip_extras", "system_packages",
            "priority", "sniff", "resolve", "search"}
    assert keys.issubset(HF_PLUGIN.keys())


def test_plugin_modality_and_priority():
    assert HF_PLUGIN["modality"] == "audio/embedding"
    # 105: between embedding/text (110) and image-generation file-pattern (100).
    assert HF_PLUGIN["priority"] == 105


def test_plugin_runtime_path():
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.audio_embedding.runtimes.transformers_audio"
        ":AudioEmbeddingRuntime"
    )


def test_plugin_pip_extras_includes_torch_transformers_librosa():
    extras = HF_PLUGIN["pip_extras"]
    assert any("torch" in e for e in extras)
    assert any("transformers" in e for e in extras)
    assert any("librosa" in e for e in extras)


def test_sniff_true_for_clap_with_feature_extraction_tag():
    info = _fake_info(
        "laion/clap-htsat-fused",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json", "model.safetensors"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_mert_with_feature_extraction_tag():
    info = _fake_info(
        "m-a-p/MERT-v1-95M",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json", "model.safetensors", "feature_extraction.py"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_wav2vec_with_feature_extraction_tag():
    info = _fake_info(
        "facebook/wav2vec2-base-960h",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_audio_encoder_pattern():
    info = _fake_info(
        "acme/some-audio-encoder",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_audio_embedding_pattern():
    info = _fake_info(
        "acme/audio-embedding-model",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_when_tag_missing():
    """No feature-extraction tag means we skip even with a matching name."""
    info = _fake_info(
        "laion/clap-htsat-fused",
        tags=["audio-classification"],
        siblings=["preprocessor_config.json"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_when_no_name_match():
    """Tag present but the repo name doesn't match an audio family."""
    info = _fake_info(
        "sentence-transformers/all-MiniLM-L6-v2",
        tags=["feature-extraction"],
        siblings=["model.safetensors"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_chat_completion_repo():
    info = _fake_info("Qwen/Qwen3-8B-GGUF",
                     tags=["chat", "text-generation"],
                     siblings=["model.gguf"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_no_tags_attribute():
    """Defensive: tags attribute missing entirely."""
    info = SimpleNamespace(
        id="laion/clap-htsat-fused",
        siblings=[SimpleNamespace(rfilename="preprocessor_config.json")],
        card_data=None,
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_synthesizes_manifest_for_clap_htsat_fused():
    info = _fake_info(
        "laion/clap-htsat-fused",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json", "model.safetensors"],
        card=SimpleNamespace(license="bsd-3-clause"),
    )
    resolved = HF_PLUGIN["resolve"]("laion/clap-htsat-fused", None, info)
    m = resolved.manifest
    assert m["model_id"] == "clap-htsat-fused"
    assert m["modality"] == "audio/embedding"
    assert m["hf_repo"] == "laion/clap-htsat-fused"
    assert m["license"] == "bsd-3-clause"
    caps = m["capabilities"]
    assert caps["device"] == "auto"
    assert caps["dimensions"] == 512
    assert caps["sample_rate"] == 48000
    assert caps["supports_text_embeddings_too"] is True
    assert caps["max_duration_seconds"] == 60.0


def test_resolve_clap_capability_defaults():
    info = _fake_info(
        "laion/clap-htsat-fused",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("laion/clap-htsat-fused", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 512
    assert caps["sample_rate"] == 48000
    assert caps["supports_text_embeddings_too"] is True


def test_resolve_mert_capability_defaults():
    info = _fake_info(
        "m-a-p/MERT-v1-95M",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("m-a-p/MERT-v1-95M", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 768
    assert caps["sample_rate"] == 24000
    assert caps["supports_text_embeddings_too"] is False


def test_resolve_wav2vec_base_capability_defaults():
    info = _fake_info(
        "facebook/wav2vec2-base-960h",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("facebook/wav2vec2-base-960h", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 768
    assert caps["sample_rate"] == 16000
    assert caps["supports_text_embeddings_too"] is False


def test_resolve_wav2vec_large_capability_defaults():
    info = _fake_info(
        "facebook/wav2vec2-large-960h",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("facebook/wav2vec2-large-960h", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["dimensions"] == 1024


def test_resolve_fallback_when_unknown_pattern():
    """Generic audio-encoder: dimensions absent (auto-detected),
    sample_rate=16000, supports_text_embeddings_too=False."""
    info = _fake_info(
        "acme/some-audio-encoder",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("acme/some-audio-encoder", None, info)
    caps = resolved.manifest["capabilities"]
    assert "dimensions" not in caps
    assert caps["sample_rate"] == 16000
    assert caps["supports_text_embeddings_too"] is False


def test_resolve_mert_repo_sets_trust_remote_code_true():
    """Finding 3 (Low): _download deliberately pulls *.py 'so
    trust_remote_code paths can read the custom feature extractor
    (e.g. MERT)', but _resolve never set the capability, so the
    runtime's trust_remote_code=False default made a pulled
    hf://...mert-* repo raise transformers' trust_remote_code error at
    load. MERT-family repos must carry trust_remote_code=True."""
    info = _fake_info(
        "m-a-p/MERT-v1-95M",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("m-a-p/MERT-v1-95M", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["trust_remote_code"] is True
    assert resolved.manifest["revision"] == _FAKE_REVISION


def test_resolve_mert_refuses_nonhex_remote_code_revision():
    import pytest

    from muse.core.resolvers import ResolverError

    info = _fake_info(
        "m-a-p/MERT-v1-95M",
        tags=["feature-extraction"],
        sha="z" * 40,
    )
    with pytest.raises(ResolverError, match="immutable Hugging Face commit"):
        HF_PLUGIN["resolve"]("m-a-p/MERT-v1-95M", None, info)


def test_mert_download_uses_resolved_commit(tmp_path, monkeypatch):
    import muse.modalities.audio_embedding.hf as plugin_module

    calls = {}
    monkeypatch.setattr(
        plugin_module,
        "snapshot_download",
        lambda **kwargs: calls.update(kwargs) or str(tmp_path / "snapshot"),
    )
    resolved = HF_PLUGIN["resolve"](
        "m-a-p/MERT-v1-95M",
        None,
        _fake_info("m-a-p/MERT-v1-95M", tags=["feature-extraction"]),
    )
    resolved.download(tmp_path)
    assert calls["revision"] == _FAKE_REVISION


def test_resolve_wav2vec_repo_does_not_set_trust_remote_code():
    """Security-conservative: only the MERT family gets
    trust_remote_code=True. Every other family (wav2vec here) must NOT
    carry the flag at all (absent, not even False)."""
    info = _fake_info(
        "facebook/wav2vec2-base-960h",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("facebook/wav2vec2-base-960h", None, info)
    caps = resolved.manifest["capabilities"]
    assert "trust_remote_code" not in caps


def test_resolve_backend_path_points_to_runtime():
    info = _fake_info(
        "laion/clap-htsat-fused",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("laion/clap-htsat-fused", None, info)
    assert resolved.backend_path == HF_PLUGIN["runtime_path"]


def test_resolve_includes_pip_extras_in_manifest():
    info = _fake_info(
        "laion/clap-htsat-fused",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("laion/clap-htsat-fused", None, info)
    extras = resolved.manifest["pip_extras"]
    assert any("torch" in e for e in extras)
    assert any("transformers" in e for e in extras)
    assert any("librosa" in e for e in extras)


def test_resolve_when_card_data_missing_license_is_none():
    info = _fake_info(
        "laion/clap-htsat-fused",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
        card=None,
    )
    resolved = HF_PLUGIN["resolve"]("laion/clap-htsat-fused", None, info)
    assert resolved.manifest["license"] is None


def test_search_yields_results_for_audio_pattern():
    api = MagicMock()
    repo1 = SimpleNamespace(id="laion/clap-htsat-fused", downloads=2_000_000)
    repo2 = SimpleNamespace(id="m-a-p/MERT-v1-95M", downloads=500_000)
    api.list_models.return_value = [repo1, repo2]
    out = list(HF_PLUGIN["search"](
        api, "audio", sort="downloads", limit=10,
    ))
    assert len(out) == 2
    assert out[0].uri == "hf://laion/clap-htsat-fused"
    assert out[0].modality == "audio/embedding"
    assert out[0].downloads == 2_000_000
    assert out[1].model_id == "mert-v1-95m"


def test_search_filters_out_non_audio_repos():
    """Repos tagged feature-extraction but without audio-name match are dropped."""
    api = MagicMock()
    audio_repo = SimpleNamespace(id="laion/clap-htsat-fused", downloads=1000)
    text_repo = SimpleNamespace(id="sentence-transformers/all-MiniLM-L6-v2", downloads=2000)
    api.list_models.return_value = [audio_repo, text_repo]
    out = list(HF_PLUGIN["search"](
        api, "embedding", sort="downloads", limit=10,
    ))
    assert len(out) == 1
    assert out[0].model_id == "clap-htsat-fused"


def test_search_calls_list_models_with_feature_extraction_filter():
    api = MagicMock()
    api.list_models.return_value = []
    list(HF_PLUGIN["search"](api, "x", sort="downloads", limit=5))
    _, kwargs = api.list_models.call_args
    assert kwargs["filter"] == "feature-extraction"
    assert kwargs["sort"] == "downloads"
    assert kwargs["limit"] == 5
    assert kwargs["search"] == "x"


def test_resolve_capabilities_include_memory_gb():
    """Every resolved manifest should declare memory_gb (used by
    `muse models list` for budgeting)."""
    info = _fake_info(
        "laion/clap-htsat-fused",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("laion/clap-htsat-fused", None, info)
    assert "memory_gb" in resolved.manifest["capabilities"]


def test_resolve_capabilities_include_device_auto():
    info = _fake_info(
        "laion/clap-htsat-fused",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("laion/clap-htsat-fused", None, info)
    assert resolved.manifest["capabilities"]["device"] == "auto"


def test_resolve_capabilities_include_max_duration_seconds():
    info = _fake_info(
        "laion/clap-htsat-fused",
        tags=["feature-extraction"],
        siblings=["preprocessor_config.json"],
    )
    resolved = HF_PLUGIN["resolve"]("laion/clap-htsat-fused", None, info)
    assert "max_duration_seconds" in resolved.manifest["capabilities"]
