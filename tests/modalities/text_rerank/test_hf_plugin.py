"""Tests for the text/rerank HF resolver plugin."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from muse.modalities.text_rerank.hf import HF_PLUGIN


_FAKE_REVISION = "1" * 40


def _fake_info(repo_id, tags=(), siblings=(), card=None, sha=_FAKE_REVISION):
    return SimpleNamespace(
        id=repo_id,
        tags=list(tags),
        siblings=[SimpleNamespace(rfilename=s) for s in siblings],
        card_data=card,
        sha=sha,
    )


def test_download_uses_resolved_commit_and_includes_dynamic_modules(
    tmp_path, monkeypatch,
):
    import muse.modalities.text_rerank.hf as plugin_module

    calls = {}
    monkeypatch.setattr(
        plugin_module,
        "snapshot_download",
        lambda **kwargs: calls.update(kwargs) or str(tmp_path / "snapshot"),
    )
    result = HF_PLUGIN["resolve"](
        "jinaai/jina-reranker-v2-base-multilingual",
        None,
        _fake_info(
            "jinaai/jina-reranker-v2-base-multilingual",
            tags=["cross-encoder"],
        ),
    )
    result.download(tmp_path)

    assert result.manifest["revision"] == _FAKE_REVISION
    assert calls["revision"] == _FAKE_REVISION
    assert "*.py" in calls["allow_patterns"]


def test_nonhex_hub_sha_is_not_persisted_as_revision():
    result = HF_PLUGIN["resolve"](
        "org/reranker",
        None,
        _fake_info("org/reranker", tags=["cross-encoder"], sha="z" * 40),
    )
    assert "revision" not in result.manifest


def test_plugin_keys_present():
    keys = {"modality", "runtime_path", "pip_extras", "system_packages",
            "priority", "sniff", "resolve", "search"}
    assert keys.issubset(HF_PLUGIN.keys())


def test_plugin_modality_and_priority():
    assert HF_PLUGIN["modality"] == "text/rerank"
    # 115: between embedding/text (110) and text/classification (200).
    # Wins over text-classification's catch-all because reranker repos
    # commonly carry the text-classification tag too.
    assert HF_PLUGIN["priority"] == 115


def test_plugin_runtime_path():
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.text_rerank.runtimes.cross_encoder"
        ":CrossEncoderRuntime"
    )


def test_sniff_true_for_cross_encoder_tag():
    info = _fake_info("any/repo", tags=["cross-encoder"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_text_classification_with_rerank_in_name():
    info = _fake_info("BAAI/bge-reranker-v2-m3",
                     tags=["text-classification"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_for_text_classification_without_rerank_name():
    info = _fake_info("KoalaAI/Text-Moderation",
                     tags=["text-classification"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_unrelated_repo():
    info = _fake_info("Qwen/Qwen3-8B-GGUF",
                     tags=["chat", "text-generation"])
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_synthesizes_manifest():
    info = _fake_info(
        "BAAI/bge-reranker-v2-m3",
        tags=["cross-encoder", "text-classification"],
        card=SimpleNamespace(license="apache-2.0"),
    )
    resolved = HF_PLUGIN["resolve"]("BAAI/bge-reranker-v2-m3", None, info)
    m = resolved.manifest
    assert m["model_id"] == "bge-reranker-v2-m3"
    assert m["modality"] == "text/rerank"
    assert m["hf_repo"] == "BAAI/bge-reranker-v2-m3"
    assert m["license"] == "apache-2.0"
    # max_length heuristic: m3 lineage ships 8K context.
    assert m["capabilities"]["max_length"] == 8192


def test_resolve_max_length_heuristic_default_for_unknown_repo():
    info = _fake_info("acme/some-cross-encoder", tags=["cross-encoder"])
    resolved = HF_PLUGIN["resolve"]("acme/some-cross-encoder", None, info)
    assert resolved.manifest["capabilities"]["max_length"] == 512


def test_resolve_max_length_heuristic_jina_v2():
    info = _fake_info("jinaai/jina-reranker-v2-base-multilingual",
                     tags=["cross-encoder"])
    resolved = HF_PLUGIN["resolve"](
        "jinaai/jina-reranker-v2-base-multilingual", None, info,
    )
    assert resolved.manifest["capabilities"]["max_length"] == 1024


def test_resolve_backend_path_points_to_cross_encoder_runtime():
    info = _fake_info("acme/x", tags=["cross-encoder"])
    resolved = HF_PLUGIN["resolve"]("acme/x", None, info)
    assert resolved.backend_path == HF_PLUGIN["runtime_path"]


def test_search_yields_results():
    api = MagicMock()
    repo1 = SimpleNamespace(id="BAAI/bge-reranker-v2-m3", downloads=1000)
    repo2 = SimpleNamespace(id="cross-encoder/ms-marco-MiniLM-L-6-v2",
                            downloads=500)
    api.list_models.return_value = [repo1, repo2]
    out = list(HF_PLUGIN["search"](api, "rerank", sort="downloads", limit=10))
    assert len(out) == 2
    assert out[0].uri == "hf://BAAI/bge-reranker-v2-m3"
    assert out[0].modality == "text/rerank"
    assert out[0].downloads == 1000
