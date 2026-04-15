"""Tests for muse.core.curated: YAML loader + helpers."""
from unittest.mock import patch

import pytest

from muse.core.curated import (
    CuratedEntry,
    expand_curated_pull,
    find_curated,
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
