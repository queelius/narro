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


def test_search_gguf_dedupes_variants_per_repo():
    """Sharded GGUFs (model-q4_k_m-00001-of-00003.gguf) and repos that
    publish the same quant in multiple files emit ONE row per (repo, variant)
    with sizes summed across files. Without this dedup, search output is
    flooded with duplicates (the bug v0.10.2 fixes)."""
    from muse.core.resolvers_hf import HFResolver
    # list_models returns a repo without sibling info; resolver falls back
    # to repo_info(files_metadata=True) to fetch siblings + sizes.
    list_repo = MagicMock(id="unsloth/Qwen3-122B-GGUF", downloads=500_000, tags=[])
    list_repo.siblings = []  # force the repo_info fallback
    info = MagicMock()
    info.siblings = [
        # Three shards of one bf16 quant
        MagicMock(rfilename="m-bf16-00001-of-00003.gguf", size=80_000_000_000),
        MagicMock(rfilename="m-bf16-00002-of-00003.gguf", size=80_000_000_000),
        MagicMock(rfilename="m-bf16-00003-of-00003.gguf", size=80_000_000_000),
        # Also a single-file q4_k_m
        MagicMock(rfilename="m-q4_k_m.gguf", size=12_000_000_000),
    ]
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        api = MockApi.return_value
        api.list_models.return_value = [list_repo]
        api.repo_info.return_value = info
        r = HFResolver()
        results = list(r.search("qwen3", modality="chat/completion"))

    uris = [res.uri for res in results]
    # Exactly one row per variant
    assert uris.count("hf://unsloth/Qwen3-122B-GGUF@bf16") == 1
    assert uris.count("hf://unsloth/Qwen3-122B-GGUF@q4_k_m") == 1
    assert len(results) == 2
    # Sharded bf16's size is the sum of all three shards (240 GB)
    bf16 = next(r for r in results if r.uri.endswith("@bf16"))
    assert abs(bf16.size_gb - 240.0) < 0.001
    # Single-file q4_k_m is 12 GB
    q4 = next(r for r in results if r.uri.endswith("@q4_k_m"))
    assert abs(q4.size_gb - 12.0) < 0.001


def test_search_gguf_passes_files_metadata_when_repo_info_called():
    """Without files_metadata=True, RepoSibling.size is None and our
    --max-size-gb filter is meaningless. v0.10.2 fix: always request it."""
    from muse.core.resolvers_hf import HFResolver
    list_repo = MagicMock(id="org/repo-gguf", downloads=1, tags=[])
    list_repo.siblings = []  # force fallback
    info = MagicMock()
    info.siblings = [MagicMock(rfilename="model-q4_k_m.gguf", size=4_000_000_000)]
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        api = MockApi.return_value
        api.list_models.return_value = [list_repo]
        api.repo_info.return_value = info
        list(HFResolver().search("anything", modality="chat/completion"))
        # The fallback repo_info call must include files_metadata=True
        api.repo_info.assert_called_once()
        kwargs = api.repo_info.call_args.kwargs
        assert kwargs.get("files_metadata") is True


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


def test_resolve_gguf_applies_chat_format_hints_from_yaml():
    """When chat_formats.yaml has a pattern matching the repo, the resolver
    populates capabilities.chat_format and capabilities.supports_tools."""
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi, \
         patch("muse.core.resolvers_hf._try_sniff_tools_from_repo", return_value=None), \
         patch("muse.core.resolvers_hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.core.chat_formats.lookup_chat_format", return_value={
             "chat_format": "chatml-function-calling",
             "supports_tools": True,
         }):
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["model-q4_k_m.gguf"],
            tags=["text-generation"],
        )
        rm = HFResolver().resolve("hf://unsloth/Qwen3.5-4B-GGUF@q4_k_m")
    caps = rm.manifest["capabilities"]
    assert caps["chat_format"] == "chatml-function-calling"
    assert caps["supports_tools"] is True


def test_resolve_gguf_yaml_supports_tools_overrides_sniff_result():
    """If sniff returned None but YAML says True, YAML wins (it's curated)."""
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi, \
         patch("muse.core.resolvers_hf._try_sniff_tools_from_repo", return_value=None), \
         patch("muse.core.resolvers_hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.core.chat_formats.lookup_chat_format", return_value={
             "supports_tools": True,
         }):
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["model-q4_k_m.gguf"],
            tags=[],
        )
        rm = HFResolver().resolve("hf://acme/some-model-GGUF@q4_k_m")
    assert rm.manifest["capabilities"]["supports_tools"] is True


def test_resolve_gguf_no_yaml_match_leaves_supports_tools_as_sniff_value():
    """Unknown repo: capabilities.chat_format absent, supports_tools = sniff."""
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi, \
         patch("muse.core.resolvers_hf._try_sniff_tools_from_repo", return_value=None), \
         patch("muse.core.resolvers_hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.core.chat_formats.lookup_chat_format", return_value=None):
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["model-q4_k_m.gguf"],
            tags=[],
        )
        rm = HFResolver().resolve("hf://nobody/random-GGUF@q4_k_m")
    caps = rm.manifest["capabilities"]
    assert "chat_format" not in caps
    assert caps["supports_tools"] is None  # sniff returned None and no YAML hint
