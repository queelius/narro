"""Tests for HFResolver (huggingface_hub mocked; no network)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.core.resolvers import ResolverError, _reset_registry_for_tests


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


def _fake_repo_info(siblings=(), tags=()):
    """Build a MagicMock that looks like HfApi().repo_info() output."""
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f, size=1_000_000) for f in siblings]
    info.tags = list(tags)
    info.card_data = MagicMock(license="apache-2.0")
    info.downloads = 123
    return info


def test_sniff_recognizes_gguf_repo():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = _fake_repo_info(
        siblings=["qwen3-8b-q4_k_m.gguf", "README.md", "config.json"],
        tags=["text-generation"],
    )
    assert _sniff_repo_shape(info) == "gguf"


def test_sniff_recognizes_sentence_transformers_via_tag():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = _fake_repo_info(
        siblings=["config.json", "tokenizer.json"],
        tags=["sentence-transformers"],
    )
    assert _sniff_repo_shape(info) == "sentence-transformers"


def test_sniff_recognizes_sentence_transformers_via_config_file():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = _fake_repo_info(
        siblings=["config.json", "sentence_transformers_config.json"],
        tags=[],
    )
    assert _sniff_repo_shape(info) == "sentence-transformers"


def test_sniff_returns_unknown_for_unrecognized_repo():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = _fake_repo_info(
        siblings=["model.safetensors", "config.json"],
        tags=["text-classification"],
    )
    assert _sniff_repo_shape(info) == "unknown"


def test_resolve_gguf_requires_variant():
    """GGUF repos MUST specify @variant; no magic default."""
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["a-q4_k_m.gguf", "a-q5_k_m.gguf", "a-q8_0.gguf"],
            tags=["text-generation"],
        )
        r = HFResolver()
        with pytest.raises(ResolverError, match="variant.*required.*available"):
            r.resolve("hf://org/repo-gguf")


def test_resolve_gguf_variant_not_found_lists_available():
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["a-q4_k_m.gguf", "a-q5_k_m.gguf"],
            tags=["text-generation"],
        )
        r = HFResolver()
        with pytest.raises(ResolverError, match="variant.*q8_0.*not found"):
            r.resolve("hf://org/repo-gguf@q8_0")


def test_resolve_gguf_exact_variant_match():
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi, \
         patch("muse.core.resolvers_hf._try_sniff_tools_from_repo", return_value=None), \
         patch("muse.core.resolvers_hf._try_sniff_context_length_from_repo", return_value=None):
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["qwen3-8b-q4_k_m.gguf"],
            tags=["text-generation"],
        )
        r = HFResolver()
        rm = r.resolve("hf://Qwen/Qwen3-8B-GGUF@q4_k_m")
        assert rm.manifest["modality"] == "chat/completion"
        assert rm.manifest["model_id"] == "qwen3-8b-gguf-q4-k-m"
        assert rm.manifest["hf_repo"] == "Qwen/Qwen3-8B-GGUF"
        assert rm.manifest["capabilities"]["gguf_file"] == "qwen3-8b-q4_k_m.gguf"
        assert any("llama-cpp-python" in p for p in rm.manifest["pip_extras"])
        assert rm.backend_path.endswith(":LlamaCppModel")


def test_resolve_sentence_transformer_repo():
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["config.json", "sentence_transformers_config.json"],
            tags=["sentence-transformers"],
        )
        r = HFResolver()
        rm = r.resolve("hf://sentence-transformers/all-MiniLM-L6-v2")
        assert rm.manifest["modality"] == "embedding/text"
        assert rm.manifest["hf_repo"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert "sentence-transformers" in " ".join(rm.manifest["pip_extras"])
        assert rm.backend_path.endswith(":SentenceTransformerModel")


def test_resolve_rejects_non_hf_scheme():
    from muse.core.resolvers_hf import HFResolver
    r = HFResolver()
    with pytest.raises(ResolverError):
        r.resolve("civitai://something")


def test_resolve_rejects_non_uri():
    from muse.core.resolvers_hf import HFResolver
    r = HFResolver()
    with pytest.raises(ResolverError):
        r.resolve("not-a-uri")


def test_resolve_unrecognized_repo_shape_raises():
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["model.safetensors"],
            tags=["text-classification"],
        )
        r = HFResolver()
        with pytest.raises(ResolverError, match="cannot infer.*text-classification"):
            r.resolve("hf://org/weird-repo")


def test_tool_template_detection():
    """When tokenizer_config.json's chat_template mentions tools,
    capabilities.supports_tools = True."""
    from muse.core.resolvers_hf import _sniff_supports_tools
    tmpl = '{% if tools %}{{ tools | tojson }}{% endif %}{% for m in messages %}{{ m }}{% endfor %}'
    assert _sniff_supports_tools(tmpl) is True
    tmpl2 = '{% for m in messages %}{{ m }}{% endfor %}'
    assert _sniff_supports_tools(tmpl2) is False
    assert _sniff_supports_tools(None) is False


def test_gguf_variant_tag_normalizes_filename():
    """Variant `q4_k_m` should match filename `model-q4_k_m.gguf`."""
    from muse.core.resolvers_hf import _match_gguf_variant
    files = ["qwen3-8b-q4_k_m.gguf", "qwen3-8b-q5_k_m.gguf", "qwen3-8b-q8_0.gguf"]
    assert _match_gguf_variant(files, "q4_k_m") == "qwen3-8b-q4_k_m.gguf"
    assert _match_gguf_variant(files, "q5_k_m") == "qwen3-8b-q5_k_m.gguf"
    assert _match_gguf_variant(files, "q8_0") == "qwen3-8b-q8_0.gguf"
    assert _match_gguf_variant(files, "Q4_K_M") == "qwen3-8b-q4_k_m.gguf"
    assert _match_gguf_variant(files, "q2_k") is None


def test_search_gguf_returns_variant_rows():
    """Each GGUF file in a matched repo becomes a separate SearchResult."""
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        api = MockApi.return_value
        fake_repo = MagicMock(
            id="org/Qwen3-8B-GGUF",
            downloads=1000,
            tags=["text-generation"],
            siblings=[
                MagicMock(rfilename="x-q4_k_m.gguf", size=4_500_000_000),
                MagicMock(rfilename="x-q8_0.gguf", size=8_500_000_000),
                MagicMock(rfilename="README.md", size=10_000),
            ],
        )
        api.list_models.return_value = [fake_repo]
        r = HFResolver()
        results = list(r.search("qwen3", modality="chat/completion"))
        assert len(results) == 2
        uris = {res.uri for res in results}
        assert "hf://org/Qwen3-8B-GGUF@q4_k_m" in uris
        assert "hf://org/Qwen3-8B-GGUF@q8_0" in uris


def test_search_embeddings_returns_repo_rows():
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        api = MockApi.return_value
        api.list_models.return_value = [
            MagicMock(
                id="sentence-transformers/all-MiniLM-L6-v2",
                downloads=50_000_000,
                tags=["sentence-transformers", "feature-extraction"],
                siblings=[MagicMock(rfilename="config.json", size=1000)],
            ),
        ]
        r = HFResolver()
        results = list(r.search("minilm", modality="embedding/text"))
        assert len(results) == 1
        assert results[0].uri == "hf://sentence-transformers/all-MiniLM-L6-v2"
        assert results[0].modality == "embedding/text"


def test_hf_resolver_registers_on_import():
    """Importing muse.core.resolvers_hf should register an HFResolver."""
    import importlib
    from muse.core import resolvers_hf  # noqa: F401
    importlib.reload(resolvers_hf)  # _clean_registry fixture cleared the prior registration
    from muse.core.resolvers import get_resolver
    r = get_resolver("hf://anything/anywhere")
    assert r.scheme == "hf"
