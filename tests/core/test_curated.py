"""Tests for muse.core.curated: YAML loader + helpers."""
from unittest.mock import patch

import pytest

from muse.core.curated import (
    CuratedEntry,
    expand_curated_pull,
    find_curated,
    find_curated_by_uri,
    load_curated,
    _reset_curated_cache_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    _reset_curated_cache_for_tests()
    yield
    _reset_curated_cache_for_tests()


def _patch_yaml(yaml_text: str):
    """Replace the YAML resource with a string for one test."""
    from unittest.mock import MagicMock
    fake_resource = MagicMock()
    fake_resource.read_text.return_value = yaml_text
    return patch("muse.core.curated._curated_yaml_path", return_value=fake_resource)


def test_load_curated_returns_real_bundled_yaml():
    """The bundled curated.yaml is valid and parses cleanly."""
    entries = load_curated()
    assert len(entries) > 0
    for e in entries:
        assert isinstance(e, CuratedEntry)
        assert e.id
        assert e.bundled or e.uri  # one of the two must be set


def test_load_curated_resolver_entry_required_fields():
    """A resolver entry without `uri` and without `bundled` should be skipped."""
    yaml_text = """
- id: legit
  uri: hf://x/y
  modality: chat/completion
  description: ok
- id: bogus
  modality: chat/completion
"""
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert len(entries) == 1
    assert entries[0].id == "legit"


def test_load_curated_bundled_and_uri_mutually_exclusive():
    yaml_text = """
- id: conflict
  uri: hf://x/y
  bundled: true
"""
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert entries == []


def test_load_curated_handles_missing_id():
    yaml_text = """
- uri: hf://x/y
  modality: chat/completion
- id: ok
  bundled: true
"""
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert len(entries) == 1
    assert entries[0].id == "ok"


def test_load_curated_handles_top_level_non_list():
    """If the YAML root is a dict, log a warning and return []."""
    yaml_text = """
some_key: some_value
"""
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert entries == []


def test_load_curated_handles_malformed_yaml():
    """Bad YAML should produce a warning and an empty list, not raise."""
    yaml_text = "not: valid: yaml: at: all: : :"
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert entries == []


def test_load_curated_handles_missing_yaml_file():
    """Missing curated.yaml is non-fatal."""
    from unittest.mock import MagicMock
    fake_resource = MagicMock()
    fake_resource.read_text.side_effect = FileNotFoundError()
    with patch("muse.core.curated._curated_yaml_path", return_value=fake_resource):
        entries = load_curated()
    assert entries == []


def test_load_curated_caches_after_first_call():
    """Second call returns the same list object (cached)."""
    first = load_curated()
    second = load_curated()
    assert first is second


def test_find_curated_returns_entry():
    yaml_text = """
- id: alpha
  uri: hf://a/b
  modality: chat/completion
  description: alpha desc
- id: beta
  bundled: true
"""
    with _patch_yaml(yaml_text):
        a = find_curated("alpha")
        b = find_curated("beta")
        none = find_curated("nonexistent")

    assert a is not None
    assert a.id == "alpha"
    assert a.uri == "hf://a/b"
    assert b is not None
    assert b.bundled is True
    assert none is None


def test_expand_curated_pull_returns_uri_for_resolver_entry():
    yaml_text = """
- id: friendly-id
  uri: hf://Qwen/Qwen3-8B-GGUF@q4_k_m
  modality: chat/completion
  description: friendly
"""
    with _patch_yaml(yaml_text):
        target = expand_curated_pull("friendly-id")
    assert target == "hf://Qwen/Qwen3-8B-GGUF@q4_k_m"


def test_expand_curated_pull_returns_id_for_bundled_entry():
    yaml_text = """
- id: kokoro-82m
  bundled: true
"""
    with _patch_yaml(yaml_text):
        target = expand_curated_pull("kokoro-82m")
    assert target == "kokoro-82m"


def test_expand_curated_pull_returns_none_for_unknown_id():
    yaml_text = """
- id: known
  bundled: true
"""
    with _patch_yaml(yaml_text):
        target = expand_curated_pull("unknown")
    assert target is None


def test_load_curated_optional_fields_default_correctly():
    """When uri-shape entry omits modality/size/description, those stay None."""
    yaml_text = """
- id: minimal-resolver
  uri: hf://x/y
"""
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert len(entries) == 1
    e = entries[0]
    assert e.modality is None
    assert e.size_gb is None
    assert e.description is None
    assert e.tags == ()


def test_load_curated_parses_capabilities_overlay():
    yaml_text = """
- id: q3e
  uri: hf://Qwen/Qwen3-Embedding-0.6B
  revision: "1111111111111111111111111111111111111111"
  modality: embedding/text
  size_gb: 0.6
  description: Qwen3-Embedding 0.6B
  capabilities:
    trust_remote_code: true
    matryoshka: true
"""
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert len(entries) == 1
    e = entries[0]
    assert e.id == "q3e"
    assert e.capabilities == {"trust_remote_code": True, "matryoshka": True}
    assert e.revision == "1" * 40


def test_load_curated_rejects_remote_code_without_concrete_revision():
    yaml_text = """
- id: unsafe
  uri: hf://org/unsafe
  modality: embedding/text
  capabilities:
    trust_remote_code: true
"""
    with _patch_yaml(yaml_text):
        assert load_curated() == []


def test_load_curated_rejects_non_commit_code_revision():
    yaml_text = """
- id: unsafe-code
  uri: hf://org/unsafe
  revision: "1111111111111111111111111111111111111111"
  code_revision: main
  modality: embedding/text
  capabilities:
    trust_remote_code: true
"""
    with _patch_yaml(yaml_text):
        assert load_curated() == []


def test_load_curated_capabilities_defaults_to_empty_dict():
    yaml_text = """
- id: minimal
  uri: hf://x/y
  modality: chat/completion
"""
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert entries[0].capabilities == {}


def test_load_curated_capabilities_non_dict_is_rejected():
    yaml_text = """
- id: bad
  uri: hf://x/y
  modality: chat/completion
  capabilities: "not a dict"
"""
    with _patch_yaml(yaml_text):
        entries = load_curated()
    assert entries == []


def test_load_curated_includes_whisper_entries():
    """ASR curated shortcuts: whisper-tiny, whisper-base, whisper-large-v3."""
    entries = load_curated()
    asr_ids = {e.id for e in entries if e.modality == "audio/transcription"}
    assert {"whisper-tiny", "whisper-base", "whisper-large-v3"}.issubset(asr_ids)
    # Each points at a faster-whisper CT2 repo. The bundled trio are
    # Systran builds; community CT2 conversions (e.g. deepdml's turbo) are
    # equally valid, so assert the faster-whisper lineage, not the publisher.
    for e in entries:
        if e.modality == "audio/transcription":
            assert e.uri is not None
            assert e.uri.startswith("hf://")
            assert "faster-whisper" in e.uri


def test_load_curated_includes_audio_quality_entries():
    entries = load_curated()
    quality = {e.id: e for e in entries if e.modality == "audio/quality"}
    assert set(quality) >= {"utmos", "audiobox-aesthetics"}
    assert quality["utmos"].uri == "hf://Blinorot/UTMOS-PyTorch"
    assert quality["audiobox-aesthetics"].uri == (
        "hf://facebook/audiobox-aesthetics"
    )
    assert all(entry.size_gb and entry.size_gb < 0.5 for entry in quality.values())


def test_load_curated_includes_qwen3_forced_aligner():
    entries = load_curated()
    aligners = {e.id: e for e in entries if e.modality == "audio/alignment"}
    entry = aligners["qwen3-forced-aligner-0.6b"]
    assert entry.uri == "hf://Qwen/Qwen3-ForcedAligner-0.6B-hf"
    assert entry.size_gb == 1.84
    assert entry.capabilities["max_duration_seconds"] == 300.0
    assert entry.capabilities["max_input_tokens"] == 8192
    assert entry.capabilities["max_reference_words"] == 2048
    assert len(entry.capabilities["supported_languages"]) == 11


def test_load_curated_includes_starvector():
    entries = {entry.id: entry for entry in load_curated()}
    entry = entries["starvector-1b-im2svg"]
    assert entry.bundled is True
    assert entry.uri is None


def test_load_curated_includes_v046_model_refresh():
    """v0.46.0 model-refresh additions across existing modalities. Each is a
    pure resolver-pull entry (no new runtime), verified to resolve to the
    modality named here."""
    by_id = {e.id: e for e in load_curated()}
    expected = {
        "qwen3-0.6b-q4": "chat/completion",
        "qwen3-4b-q4": "chat/completion",
        "qwen3-8b-q4": "chat/completion",
        "qwen2.5-vl-7b-instruct": "chat/completion",
        "bge-m3": "embedding/text",
        "nomic-embed-text-v1.5": "embedding/text",
        "whisper-large-v3-turbo": "audio/transcription",
        "distilbart-cnn-12-6": "text/summarization",
        "dinov2-base": "image/embedding",
        "dinov2-large": "image/embedding",
        "rtdetr-v2-r50vd": "image/cv",
        "yolos-small": "image/cv",
        "depth-anything-v2-large": "image/cv",
        "sam2.1-hiera-large": "image/segmentation",
    }
    for cid, modality in expected.items():
        assert cid in by_id, f"missing curated entry {cid!r}"
        e = by_id[cid]
        assert e.modality == modality, f"{cid}: {e.modality!r} != {modality!r}"
        assert e.uri and e.uri.startswith("hf://")


def test_load_curated_includes_sdxl_turbo_and_flux_schnell():
    """v0.16.0 adds two image/generation curated aliases via the new HF plugin."""
    entries = load_curated()
    by_id = {e.id: e for e in entries}

    assert "sdxl-turbo" in by_id
    e = by_id["sdxl-turbo"]
    assert e.modality == "image/generation"
    assert e.uri == "hf://stabilityai/sdxl-turbo"

    assert "flux-schnell" in by_id
    e = by_id["flux-schnell"]
    assert e.modality == "image/generation"
    assert e.uri == "hf://black-forest-labs/FLUX.1-schnell"


def test_load_curated_includes_bge_reranker_v2_m3():
    """v0.19.0 adds bge-reranker-v2-m3 as a bundled alias under text/rerank."""
    entries = load_curated()
    by_id = {e.id: e for e in entries}

    assert "bge-reranker-v2-m3" in by_id
    e = by_id["bge-reranker-v2-m3"]
    assert e.bundled is True
    assert e.uri is None  # bundled entries don't have URIs


def test_load_curated_includes_animation_entries():
    """v0.18.0 adds image/animation curated aliases."""
    entries = load_curated()
    by_id = {e.id: e for e in entries}

    assert "animatediff-motion-v3" in by_id
    e = by_id["animatediff-motion-v3"]
    assert e.bundled is True
    assert e.uri is None  # bundled entries don't have URIs

    assert "animatelcm" in by_id
    e = by_id["animatelcm"]
    assert e.modality == "image/animation"
    assert e.uri == "hf://wangfuyun/AnimateLCM"


def test_load_curated_includes_stable_audio_open_1_0():
    """v0.20.0 adds stable-audio-open-1.0 as a bundled alias under audio/generation."""
    entries = load_curated()
    by_id = {e.id: e for e in entries}

    assert "stable-audio-open-1.0" in by_id
    e = by_id["stable-audio-open-1.0"]
    assert e.bundled is True
    assert e.uri is None  # bundled entries don't have URIs


def test_load_curated_includes_summarization_entries():
    """v0.22.0 adds bart-large-cnn (bundled) and bart-cnn-samsum (URI) under
    text/summarization."""
    entries = load_curated()
    by_id = {e.id: e for e in entries}

    assert "bart-large-cnn" in by_id
    bart = by_id["bart-large-cnn"]
    assert bart.bundled is True
    assert bart.uri is None  # bundled entries don't have URIs

    assert "bart-cnn-samsum" in by_id
    samsum = by_id["bart-cnn-samsum"]
    assert samsum.modality == "text/summarization"
    assert samsum.uri == "hf://philschmid/bart-large-cnn-samsum"
    # Capabilities overlay carries the dialog flag and CPU annotation.
    assert samsum.capabilities.get("supports_dialog_summarization") is True
    assert samsum.capabilities.get("device") == "cpu"
    assert samsum.capabilities.get("memory_gb") == 1.5


def test_load_curated_includes_text_moderation_entry():
    """Curated text-moderation alias exists and points at KoalaAI.

    Also asserts the `safe_labels: ["OK"]` capability overlay is present.
    KoalaAI/Text-Moderation includes "OK" as one of nine single-label
    classes; without this overlay, benign inputs (where the model
    correctly assigns >0.99 to OK) erroneously get `flagged=True`. The
    overlay is the fix; the test guards against accidental removal.
    """
    entries = load_curated()
    by_id = {e.id: e for e in entries}
    assert "text-moderation" in by_id
    e = by_id["text-moderation"]
    assert e.modality == "text/classification"
    assert e.uri == "hf://KoalaAI/Text-Moderation"
    assert e.capabilities.get("safe_labels") == ["OK"], (
        "text-moderation must declare safe_labels=['OK'] or benign inputs "
        "get flagged when the model is highly confident they're safe"
    )


def test_load_curated_includes_memory_annotations():
    """v0.18.2: curated entries declare memory_gb under capabilities.

    We spot-check a representative sample (chat, embedding, image,
    audio). At least 5 entries must carry the annotation.
    """
    entries = load_curated()
    by_id = {e.id: e for e in entries}
    annotated = [e for e in entries if e.capabilities.get("memory_gb") is not None]
    assert len(annotated) >= 5, (
        "expected memory_gb on most curated entries, got "
        f"{len(annotated)}/{len(entries)}"
    )
    # Spot checks
    assert by_id["qwen3.5-4b-q4"].capabilities["memory_gb"] == 3.5
    assert by_id["all-minilm-l6-v2"].capabilities["memory_gb"] == 0.1
    assert by_id["whisper-tiny"].capabilities["memory_gb"] == 0.1
    assert by_id["text-moderation"].capabilities["memory_gb"] == 0.5


def test_find_curated_by_uri_round_trips():
    """find_curated_by_uri(e.uri) == e for every URI-shaped curated entry."""
    for e in load_curated():
        if not e.uri:
            continue
        match = find_curated_by_uri(e.uri)
        assert match is e, f"URI lookup for {e.id!r} failed: {match}"


def test_find_curated_by_uri_unknown_returns_none():
    assert find_curated_by_uri("hf://nobody/nothing") is None


def test_find_curated_by_uri_lets_uri_pull_inherit_capabilities():
    """The curated text-moderation overlay is reachable from raw URI."""
    match = find_curated_by_uri("hf://KoalaAI/Text-Moderation")
    assert match is not None
    assert match.id == "text-moderation"
    assert match.capabilities.get("safe_labels") == ["OK"]


def test_load_curated_includes_supertonic_3():
    """v0.47.0: Supertonic-3 bundled TTS alias."""
    by_id = {e.id: e for e in load_curated()}
    assert "supertonic-3" in by_id
    assert by_id["supertonic-3"].bundled is True
