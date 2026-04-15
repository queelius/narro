"""Tests for muse.core.chat_formats: YAML lookup."""
from unittest.mock import MagicMock, patch

import pytest

from muse.core.chat_formats import (
    lookup_chat_format,
    _reset_cache_for_tests,
)


@pytest.fixture(autouse=True)
def _reset():
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


def _patch_yaml(yaml_text: str):
    fake = MagicMock()
    fake.read_text.return_value = yaml_text
    return patch("muse.core.chat_formats._yaml_path", return_value=fake)


def test_lookup_returns_hint_for_substring_match():
    yaml_text = """
- pattern: qwen3
  chat_format: chatml-function-calling
  supports_tools: true
"""
    with _patch_yaml(yaml_text):
        hint = lookup_chat_format("unsloth/Qwen3.5-4B-GGUF")
    assert hint == {"chat_format": "chatml-function-calling", "supports_tools": True}


def test_lookup_case_insensitive():
    yaml_text = """
- pattern: QWEN3
  chat_format: chatml-function-calling
"""
    with _patch_yaml(yaml_text):
        hint = lookup_chat_format("user/qwen3-something")
    assert hint is not None
    assert hint["chat_format"] == "chatml-function-calling"


def test_lookup_first_match_wins():
    """More-specific patterns must come first in the YAML to shadow more-general."""
    yaml_text = """
- pattern: qwen2.5-coder
  chat_format: special
  supports_tools: true
- pattern: qwen2.5
  chat_format: general
  supports_tools: false
"""
    with _patch_yaml(yaml_text):
        coder = lookup_chat_format("Qwen/Qwen2.5-Coder-7B-Instruct")
        general = lookup_chat_format("Qwen/Qwen2.5-7B-Instruct")
    assert coder["chat_format"] == "special"
    assert general["chat_format"] == "general"


def test_lookup_returns_none_for_unmatched():
    yaml_text = """
- pattern: qwen3
  chat_format: chatml-function-calling
"""
    with _patch_yaml(yaml_text):
        hint = lookup_chat_format("some/random-model")
    assert hint is None


def test_lookup_with_real_bundled_yaml_finds_qwen3():
    """Sanity: the real shipped chat_formats.yaml maps Qwen3 correctly."""
    hint = lookup_chat_format("unsloth/Qwen3.5-4B-GGUF")
    assert hint is not None
    assert hint["chat_format"] == "chatml-function-calling"
    assert hint["supports_tools"] is True


def test_lookup_with_real_bundled_yaml_finds_llama_3_2():
    hint = lookup_chat_format("bartowski/Llama-3.2-3B-Instruct-GGUF")
    assert hint is not None
    assert hint["chat_format"] == "llama-3"
    assert hint["supports_tools"] is True


def test_lookup_omits_pattern_key_from_returned_dict():
    """The pattern is a lookup key, not part of the hint payload."""
    yaml_text = """
- pattern: foo
  chat_format: bar
  supports_tools: true
"""
    with _patch_yaml(yaml_text):
        hint = lookup_chat_format("test/foo-model")
    assert "pattern" not in hint


def test_lookup_handles_missing_yaml_file():
    fake = MagicMock()
    fake.read_text.side_effect = FileNotFoundError()
    with patch("muse.core.chat_formats._yaml_path", return_value=fake):
        assert lookup_chat_format("anything") is None


def test_lookup_handles_malformed_yaml():
    yaml_text = "not: valid: yaml: at: : :"
    with _patch_yaml(yaml_text):
        assert lookup_chat_format("anything") is None


def test_lookup_skips_rows_without_pattern():
    yaml_text = """
- chat_format: orphan
- pattern: ok
  chat_format: keeper
"""
    with _patch_yaml(yaml_text):
        hint = lookup_chat_format("test/ok")
    assert hint == {"chat_format": "keeper"}
