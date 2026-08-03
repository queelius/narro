"""Tests for HFResolver (huggingface_hub mocked; no network)."""
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.core.resolvers import (
    ResolvedModel,
    ResolverError,
    _reset_registry_for_tests,
)


_FAKE_REVISION = "1" * 40


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
    info.sha = _FAKE_REVISION
    return info


def _fake_resolved(repo_id="org/repo", revision=_FAKE_REVISION, **manifest_values):
    manifest = {
        "model_id": repo_id.split("/", 1)[-1],
        "modality": "embedding/text",
        "hf_repo": repo_id,
        "revision": revision,
        **manifest_values,
    }
    return ResolvedModel(
        manifest=manifest,
        backend_path="muse.fake:Model",
        download=lambda cache: Path(cache) / "model",
    )


def test_resolve_threads_reviewed_revision_to_hub_metadata():
    from muse.core.resolvers_hf import HFResolver

    plugin = {
        "sniff": lambda info: True,
        "resolve": lambda repo_id, variant, info: _fake_resolved(
            repo_id, info.sha,
        ),
    }
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info()
        resolver = HFResolver(plugins=[plugin])
        resolver.resolve("hf://org/repo", revision=_FAKE_REVISION)

    MockApi.return_value.repo_info.assert_called_once_with(
        "org/repo", revision=_FAKE_REVISION,
    )


def test_resolve_rejects_plugin_that_drops_metadata_commit():
    from muse.core.resolvers_hf import HFResolver

    plugin = {
        "sniff": lambda info: True,
        "resolve": lambda repo_id, variant, info: _fake_resolved(
            repo_id, revision=None,
        ),
    }
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info()
        resolver = HFResolver(plugins=[plugin])

        with pytest.raises(ResolverError, match="did not preserve.*commit"):
            resolver.resolve("hf://org/repo")


def test_resolve_adds_primary_immutable_artifact_receipt():
    from muse.core.resolvers_hf import HFResolver

    plugin = {
        "sniff": lambda info: True,
        "resolve": lambda repo_id, variant, info: _fake_resolved(
            repo_id, info.sha,
        ),
    }
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info()
        resolved = HFResolver(plugins=[plugin]).resolve("hf://org/repo")

    assert resolved.artifact_provenance == ({
        "repo_id": "org/repo",
        "revision": _FAKE_REVISION,
        "subdir": ".",
    },)


def test_resolve_preserves_required_artifact_patterns():
    from muse.core.resolvers_hf import HFResolver

    receipt = ({
        "repo_id": "org/repo",
        "revision": _FAKE_REVISION,
        "subdir": "adapter",
        "allow_patterns": ("*.json", "*.safetensors"),
        "required_patterns": ("model_index.json", "*.safetensors"),
    },)
    resolved = ResolvedModel(
        manifest={
            "model_id": "repo",
            "modality": "image/animation",
            "hf_repo": "org/repo",
            "revision": _FAKE_REVISION,
        },
        backend_path="muse.fake:Model",
        download=lambda cache: Path(cache) / "model",
        artifact_provenance=receipt,
    )
    plugin = {
        "sniff": lambda info: True,
        "resolve": lambda repo_id, variant, info: resolved,
    }
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info()
        result = HFResolver(plugins=[plugin]).resolve("hf://org/repo")

    assert result.artifact_provenance[0]["allow_patterns"] == [
        "*.json",
        "*.safetensors",
    ]
    assert result.artifact_provenance[0]["required_patterns"] == [
        "model_index.json",
        "*.safetensors",
    ]


def test_resolve_rejects_hub_revision_mismatch():
    from muse.core.resolvers_hf import HFResolver

    plugin = {"sniff": lambda info: True, "resolve": MagicMock()}
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        info = _fake_repo_info()
        info.sha = "2" * 40
        MockApi.return_value.repo_info.return_value = info
        resolver = HFResolver(plugins=[plugin])
        with pytest.raises(ResolverError, match="unexpected commit"):
            resolver.resolve("hf://org/repo", revision=_FAKE_REVISION)


@pytest.mark.parametrize("via_modality", [False, True])
def test_resolve_rejects_missing_immutable_hub_commit(via_modality):
    from muse.core.resolvers_hf import HFResolver

    plugin = {
        "modality": "embedding/text",
        "sniff": lambda info: True,
        "resolve": MagicMock(),
    }
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        info = _fake_repo_info()
        info.sha = None
        MockApi.return_value.repo_info.return_value = info
        resolver = HFResolver(plugins=[plugin])
        with pytest.raises(ResolverError, match="immutable 40-character commit"):
            if via_modality:
                resolver.resolve_via_modality(
                    "hf://org/repo",
                    "embedding/text",
                )
            else:
                resolver.resolve("hf://org/repo")

    plugin["resolve"].assert_not_called()


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
            tags=["some-unsupported-tag"],
        )
        r = HFResolver()
        with pytest.raises(ResolverError, match="no HF plugin matched"):
            r.resolve("hf://org/weird-repo")


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


# --- faster-whisper branch ---

def _fake_ct2_whisper_siblings():
    return [
        SimpleNamespace(rfilename="model.bin"),
        SimpleNamespace(rfilename="config.json"),
        SimpleNamespace(rfilename="vocabulary.txt"),
        SimpleNamespace(rfilename="README.md"),
    ]


def test_resolve_faster_whisper_synthesizes_manifest():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    info = SimpleNamespace(
        siblings=_fake_ct2_whisper_siblings(),
        tags=["automatic-speech-recognition"],
        card_data=SimpleNamespace(license="mit"),
        sha=_FAKE_REVISION,
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        resolved = resolver.resolve("hf://Systran/faster-whisper-tiny")
    assert resolved.manifest["modality"] == "audio/transcription"
    assert resolved.manifest["hf_repo"] == "Systran/faster-whisper-tiny"
    assert resolved.manifest["model_id"] == "faster-whisper-tiny"
    assert "faster-whisper>=1.0.0" in resolved.manifest["pip_extras"]
    assert "ffmpeg" in resolved.manifest["system_packages"]
    assert resolved.backend_path == (
        "muse.modalities.audio_transcription.runtimes.faster_whisper"
        ":FasterWhisperModel"
    )


def test_search_faster_whisper_yields_results():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    fake_repos = [
        SimpleNamespace(id="Systran/faster-whisper-tiny", downloads=12345, siblings=[]),
        SimpleNamespace(id="Systran/faster-whisper-base", downloads=8000, siblings=[]),
    ]
    with patch.object(resolver._api, "list_models", return_value=fake_repos):
        results = list(resolver.search("whisper", modality="audio/transcription"))
    assert len(results) == 2
    assert all(r.modality == "audio/transcription" for r in results)
    assert all(r.uri.startswith("hf://Systran/faster-whisper-") for r in results)
    assert results[0].model_id == "faster-whisper-tiny"
    assert results[1].model_id == "faster-whisper-base"


# --- text-classification branch ---

def test_resolve_text_classification_synthesizes_manifest():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="config.json"),
            SimpleNamespace(rfilename="model.safetensors"),
            SimpleNamespace(rfilename="tokenizer.json"),
        ],
        tags=["text-classification"],
        card_data=SimpleNamespace(license="apache-2.0"),
        sha=_FAKE_REVISION,
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        resolved = resolver.resolve("hf://KoalaAI/Text-Moderation")
    assert resolved.manifest["modality"] == "text/classification"
    assert resolved.manifest["hf_repo"] == "KoalaAI/Text-Moderation"
    assert resolved.manifest["model_id"] == "text-moderation"
    assert "transformers>=4.36.0" in resolved.manifest["pip_extras"]
    assert "torch>=2.1.0" in resolved.manifest["pip_extras"]
    assert resolved.backend_path == (
        "muse.modalities.text_classification.runtimes.hf_text_classifier"
        ":HFTextClassifier"
    )


def test_search_text_classification_yields_results():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    fake_repos = [
        SimpleNamespace(id="KoalaAI/Text-Moderation", downloads=5000, siblings=[]),
        SimpleNamespace(id="unitary/toxic-bert", downloads=12000, siblings=[]),
    ]
    with patch.object(resolver._api, "list_models", return_value=fake_repos):
        results = list(resolver.search("toxic", modality="text/classification"))
    assert len(results) == 2
    assert all(r.modality == "text/classification" for r in results)
    assert results[0].model_id == "text-moderation"


def test_resolve_unknown_error_message_includes_repo_diagnostics():
    """When no plugin matches, the error surfaces the repo id plus the seen
    tags and siblings so users can debug."""
    from muse.core.resolvers_hf import HFResolver, ResolverError
    resolver = HFResolver()
    info = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename="random.bin")],
        tags=["something-unknown"],
        sha=_FAKE_REVISION,
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        try:
            resolver.resolve("hf://x/y")
        except ResolverError as e:
            msg = str(e)
            assert "no HF plugin matched" in msg
            assert "x/y" in msg
            assert "something-unknown" in msg
            assert "random.bin" in msg
        else:
            raise AssertionError("expected ResolverError")


def test_resolve_via_modality_picks_named_plugin_not_priority_winner():
    """Regression for v0.41.1: a curated alias declaring `modality:
    text/rerank` for a sentence-transformers reranker repo
    (BAAI/bge-reranker-base) must route through the text/rerank plugin,
    not the embedding/text plugin (which would otherwise win on sniff).
    """
    from muse.core.resolvers_hf import HFResolver

    rerank_plugin_resolved = _fake_resolved(
        "BAAI/bge-reranker-base",
        modality="text/rerank",
    )
    embedding_plugin = {
        "modality": "embedding/text",
        "sniff": MagicMock(return_value=True),  # would win on sniff
        "resolve": MagicMock(name="embedding-resolve"),
    }
    rerank_plugin = {
        "modality": "text/rerank",
        "sniff": MagicMock(return_value=False),  # wouldn't win on sniff
        "resolve": MagicMock(return_value=rerank_plugin_resolved),
    }
    resolver = HFResolver(plugins=[embedding_plugin, rerank_plugin])

    info = SimpleNamespace(
        siblings=[], tags=[], id="BAAI/bge-reranker-base", sha=_FAKE_REVISION,
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        out = resolver.resolve_via_modality(
            "hf://BAAI/bge-reranker-base", "text/rerank",
        )

    assert out.manifest["modality"] == "text/rerank"
    rerank_plugin["resolve"].assert_called_once()
    # The embedding plugin must NOT have been consulted, even though
    # its sniff would have returned True under priority dispatch.
    embedding_plugin["sniff"].assert_not_called()
    embedding_plugin["resolve"].assert_not_called()


def test_resolve_forwards_base_override_when_plugin_resolve_accepts_it():
    """I2: HFResolver.resolve forwards base_override to a plugin's
    resolve callable ONLY when that callable's signature accepts it
    (inspect-guarded), so the other 15+ modality plugins with a plain
    3-arg resolve(repo_id, variant, info) keep working untouched."""
    from muse.core.resolvers_hf import HFResolver

    def _lora_resolve(repo_id, variant, info, base_override=None):
        return _fake_resolved(repo_id, info.sha, base_override=base_override)

    lora_plugin = {
        "modality": "image/generation",
        "sniff": MagicMock(return_value=True),
        "resolve": _lora_resolve,
    }
    resolver = HFResolver(plugins=[lora_plugin])
    info = SimpleNamespace(
        siblings=[], tags=[], id="nerijs/pixel-art-xl", sha=_FAKE_REVISION,
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        out = resolver.resolve(
            "hf://nerijs/pixel-art-xl", base_override="sdxl-turbo",
        )
    assert out.manifest["base_override"] == "sdxl-turbo"


def test_resolve_does_not_forward_base_override_to_plain_plugin_resolve():
    """A plugin with a plain resolve(repo_id, variant, info) signature
    must not receive base_override (that would raise TypeError). Uses a
    real function (not a bare MagicMock) so inspect.signature sees a
    genuine 3-parameter signature, matching the real modality plugins'
    resolve callables.
    """
    from muse.core.resolvers_hf import HFResolver

    calls = []

    def plain_resolve(repo_id, variant, info):
        calls.append((repo_id, variant, info))
        return _fake_resolved(repo_id, info.sha)

    plain_plugin = {
        "modality": "audio/speech",
        "sniff": MagicMock(return_value=True),
        "resolve": plain_resolve,
    }
    resolver = HFResolver(plugins=[plain_plugin])
    info = SimpleNamespace(
        siblings=[], tags=[], id="org/repo", sha=_FAKE_REVISION,
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        out = resolver.resolve("hf://org/repo", base_override="sdxl-turbo")
    assert out.manifest["hf_repo"] == "org/repo"
    assert calls == [("org/repo", None, info)]


def test_resolve_via_modality_forwards_base_override_when_accepted():
    from muse.core.resolvers_hf import HFResolver

    def _lora_resolve(repo_id, variant, info, base_override=None):
        return _fake_resolved(repo_id, info.sha, base_override=base_override)

    lora_plugin = {
        "modality": "image/generation",
        "sniff": MagicMock(return_value=False),
        "resolve": _lora_resolve,
    }
    resolver = HFResolver(plugins=[lora_plugin])
    info = SimpleNamespace(
        siblings=[], tags=[], id="nerijs/pixel-art-xl", sha=_FAKE_REVISION,
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        out = resolver.resolve_via_modality(
            "hf://nerijs/pixel-art-xl", "image/generation",
            base_override="flux-schnell",
        )
    assert out.manifest["base_override"] == "flux-schnell"


def test_resolve_via_modality_raises_when_no_plugin_for_modality():
    from muse.core.resolvers_hf import HFResolver

    plugin = {"modality": "audio/speech", "sniff": MagicMock(), "resolve": MagicMock()}
    resolver = HFResolver(plugins=[plugin])
    try:
        resolver.resolve_via_modality("hf://x/y", "image/segmentation")
    except ResolverError as e:
        msg = str(e)
        assert "no HF plugin for modality" in msg
        assert "image/segmentation" in msg
    else:
        raise AssertionError("expected ResolverError")


# --- transient repo_info resilience (resolve does not crash on a flaky Hub) ---

def test_repo_info_retries_transient_error_then_succeeds():
    """A transient repo_info failure (e.g. the TypeError huggingface_hub
    raises on a partial/rate-limited response) is retried, not propagated."""
    from muse.core.resolvers_hf import HFResolver
    good = _fake_repo_info(siblings=["model.safetensors"], tags=["x"])
    with patch("muse.core.resolvers_hf.HfApi") as MockApi, \
            patch("muse.core.resolvers_hf.time.sleep"):
        api = MockApi.return_value
        api.repo_info.side_effect = [
            TypeError("unsupported format string passed to NoneType.__format__"),
            TypeError("unsupported format string passed to NoneType.__format__"),
            good,
        ]
        r = HFResolver(plugins=[])
        assert r._repo_info("org/repo") is good
        assert api.repo_info.call_count == 3


def test_repo_info_repository_not_found_surfaces_without_retry():
    """A missing/gated repo is deterministic: surface it immediately, do not
    retry or mask it behind a generic 'rate-limited' message."""
    from huggingface_hub.utils import RepositoryNotFoundError
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi, \
            patch("muse.core.resolvers_hf.time.sleep"):
        api = MockApi.return_value
        # RepositoryNotFoundError inherits HfHubHTTPError, whose __init__ takes a
        # required keyword-only `response` (huggingface_hub >= 1.x). Construct it
        # accordingly so the test exercises the no-retry path instead of crashing
        # at setup (regression: positional-only construction raised TypeError on
        # hf_hub 1.20.1, silently voiding this test's coverage).
        api.repo_info.side_effect = RepositoryNotFoundError(
            "404 not found", response=MagicMock())
        r = HFResolver(plugins=[])
        with pytest.raises(RepositoryNotFoundError):
            r._repo_info("org/missing")
        assert api.repo_info.call_count == 1  # no retry on a meaningful error


def test_repo_info_persistent_transient_wrapped_as_resolver_error():
    """When transient failures exhaust the retry budget, the raw exception
    is wrapped in a clear, retryable ResolverError (not leaked as TypeError)."""
    from muse.core.resolvers_hf import HFResolver, _REPO_INFO_MAX_ATTEMPTS
    with patch("muse.core.resolvers_hf.HfApi") as MockApi, \
            patch("muse.core.resolvers_hf.time.sleep"):
        api = MockApi.return_value
        api.repo_info.side_effect = TypeError("boom")
        r = HFResolver(plugins=[])
        with pytest.raises(ResolverError, match="rate-limit|retry|metadata"):
            r._repo_info("org/flaky")
        assert api.repo_info.call_count == _REPO_INFO_MAX_ATTEMPTS
