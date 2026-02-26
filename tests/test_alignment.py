"""Tests for narro/alignment.py — word-level alignment from attention weights."""

import json
import os
import tempfile

import numpy as np
import pytest

from narro.alignment import extract_alignment, save_alignment, extract_alignment_from_encoded
from narro.encoded import SentenceEncoding, EncodedSpeech


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sentence_with_attention(text, seq_len, input_len,
                                   text_index=0, sentence_index=0):
    """Create a SentenceEncoding with random attention weights."""
    attention = np.random.rand(seq_len, input_len).astype(np.float32)
    return SentenceEncoding(
        hidden_states=np.random.randn(seq_len, 512).astype(np.float32),
        token_ids=np.random.randint(0, 8192, size=seq_len, dtype=np.int32),
        token_entropy=np.random.rand(seq_len).astype(np.float32),
        finish_reason='stop',
        text=text,
        text_index=text_index,
        sentence_index=sentence_index,
        attention_weights=attention,
    )


# ---------------------------------------------------------------------------
# extract_alignment tests
# ---------------------------------------------------------------------------

class TestExtractAlignment:

    def test_basic_alignment_two_words(self):
        """Two words should produce two alignment entries in order."""
        T = 10  # generated tokens
        input_len = 4  # e.g. [STOP][TEXT] word1 word2

        # Create attention where first half attends to token 2 (word1),
        # second half attends to token 3 (word2)
        attention = np.zeros((T, input_len), dtype=np.float32)
        attention[:5, 2] = 1.0   # first 5 generated tokens attend to word1
        attention[5:, 3] = 1.0   # last 5 generated tokens attend to word2

        token_to_word = {2: "hello", 3: "world"}
        token_duration = 0.064

        result = extract_alignment(attention, token_to_word, token_duration)

        assert len(result) == 2
        assert result[0]['word'] == 'hello'
        assert result[1]['word'] == 'world'
        # hello should start before world
        assert result[0]['start'] < result[1]['start']

    def test_empty_attention(self):
        """Zero-size attention should return empty list."""
        attention = np.zeros((0, 5), dtype=np.float32)
        token_to_word = {}
        result = extract_alignment(attention, token_to_word, 0.064)
        assert result == []

    def test_single_word(self):
        """Single word should span a reasonable portion of the total duration."""
        T = 10
        input_len = 3  # [STOP][TEXT] word

        attention = np.zeros((T, input_len), dtype=np.float32)
        attention[:, 2] = 1.0  # all generated tokens attend to the single word

        token_to_word = {2: "hello"}
        token_duration = 0.064

        result = extract_alignment(attention, token_to_word, token_duration)

        assert len(result) == 1
        assert result[0]['word'] == 'hello'
        # With uniform attention, center-of-mass is at the middle and spread
        # covers a symmetric region. Start should be >= 0, end <= total.
        total_duration = T * token_duration
        assert result[0]['start'] >= 0.0
        assert result[0]['end'] <= total_duration + 0.001
        # The span should be non-trivial (word covers some time range)
        assert result[0]['end'] > result[0]['start']

    def test_timestamps_are_rounded(self):
        """All timestamps should be rounded to 3 decimal places."""
        T = 7
        input_len = 4
        attention = np.random.rand(T, input_len).astype(np.float32)
        token_to_word = {2: "hello", 3: "world"}
        token_duration = 0.064

        result = extract_alignment(attention, token_to_word, token_duration)

        for entry in result:
            # Check 3 decimal places: multiply by 1000 and verify it's an integer
            assert entry['start'] == round(entry['start'], 3)
            assert entry['end'] == round(entry['end'], 3)

    def test_word_ordering_by_start_time(self):
        """Words should be ordered by start time, not by input token position."""
        T = 10
        input_len = 4

        # Reverse the attention: word2 (token 3) is attended to first
        attention = np.zeros((T, input_len), dtype=np.float32)
        attention[:5, 3] = 1.0   # first 5 tokens attend to word2
        attention[5:, 2] = 1.0   # last 5 tokens attend to word1

        token_to_word = {2: "world", 3: "hello"}
        token_duration = 0.064

        result = extract_alignment(attention, token_to_word, token_duration)

        assert len(result) == 2
        # Should be ordered by start time
        assert result[0]['start'] <= result[1]['start']

    def test_no_word_tokens_returns_empty(self):
        """If token_to_word is empty, return empty list."""
        T = 10
        input_len = 4
        attention = np.random.rand(T, input_len).astype(np.float32)
        token_to_word = {}
        token_duration = 0.064

        result = extract_alignment(attention, token_to_word, token_duration)
        assert result == []

    def test_duplicate_words_produce_separate_entries(self):
        """Repeated words (e.g., 'the ... the') should each get their own entry."""
        T = 12
        input_len = 6  # e.g. [STOP][TEXT] the cat the mat

        attention = np.zeros((T, input_len), dtype=np.float32)
        attention[:3, 2] = 1.0   # first "the"
        attention[3:6, 3] = 1.0  # "cat"
        attention[6:9, 4] = 1.0  # second "the"
        attention[9:, 5] = 1.0   # "mat"

        token_to_word = {2: "the", 3: "cat", 4: "the", 5: "mat"}
        token_duration = 0.064

        result = extract_alignment(attention, token_to_word, token_duration)

        assert len(result) == 4
        words = [r['word'] for r in result]
        assert words.count('the') == 2
        # The two "the" entries should have different start times
        the_entries = [r for r in result if r['word'] == 'the']
        assert the_entries[0]['start'] != the_entries[1]['start']

    def test_zero_spread_has_minimum_width(self):
        """When all attention is on a single token, the entry should still have non-zero width."""
        T = 10
        input_len = 3

        attention = np.zeros((T, input_len), dtype=np.float32)
        # All attention on a single generated token
        attention[5, 2] = 1.0

        token_to_word = {2: "hello"}
        token_duration = 0.064

        result = extract_alignment(attention, token_to_word, token_duration)

        assert len(result) == 1
        assert result[0]['end'] > result[0]['start']

    def test_zero_attention_word(self):
        """A word with zero attention should appear with zero duration."""
        T = 5
        input_len = 4

        attention = np.zeros((T, input_len), dtype=np.float32)
        attention[:, 2] = 1.0  # only "hello" gets attention

        token_to_word = {2: "hello", 3: "ghost"}
        token_duration = 0.064

        result = extract_alignment(attention, token_to_word, token_duration)

        assert len(result) == 2
        ghost = [r for r in result if r['word'] == 'ghost']
        assert len(ghost) == 1
        assert ghost[0]['start'] == 0.0
        assert ghost[0]['end'] == 0.0


# ---------------------------------------------------------------------------
# save_alignment tests
# ---------------------------------------------------------------------------

class TestSaveAlignment:

    def test_save_alignment_writes_json(self):
        """save_alignment should write valid JSON with indent=2."""
        alignment = [
            {"word": "hello", "start": 0.0, "end": 0.32},
            {"word": "world", "start": 0.32, "end": 0.64},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            save_alignment(alignment, path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == alignment
            # Check indent=2 formatting
            with open(path) as f:
                raw = f.read()
            assert '  ' in raw  # indented
        finally:
            os.unlink(path)

    def test_save_alignment_empty_list(self):
        """save_alignment should handle empty alignment list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            save_alignment([], path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# extract_alignment_from_encoded tests
# ---------------------------------------------------------------------------

class TestExtractAlignmentFromEncoded:
    """Tests for proportional word timing within sentence boundaries."""

    def _make_sentence(self, text, T, **kwargs):
        """Helper to create a SentenceEncoding with given text and token count."""
        defaults = dict(
            hidden_states=np.random.randn(T, 512).astype(np.float32),
            token_ids=np.random.randint(0, 8192, size=T, dtype=np.int32),
            token_entropy=np.random.rand(T).astype(np.float32),
            finish_reason='stop',
            text=text,
            text_index=0,
            sentence_index=0,
        )
        defaults.update(kwargs)
        return SentenceEncoding(**defaults)

    def test_single_sentence_two_words(self):
        """Two equal-length words should each get half the sentence duration."""
        sentence = self._make_sentence("hello world", T=10)
        encoded = EncodedSpeech(sentences=[sentence], model_id='test/model')
        result = extract_alignment_from_encoded(encoded)

        assert len(result) == 2
        assert result[0]['word'] == 'hello'
        assert result[1]['word'] == 'world'
        # Non-overlapping, sequential
        assert result[0]['start'] == 0.0
        assert result[0]['end'] == result[1]['start']
        # Both have positive duration
        assert result[0]['end'] > result[0]['start']
        assert result[1]['end'] > result[1]['start']

    def test_proportional_by_character_length(self):
        """Longer words should get proportionally more time."""
        sentence = self._make_sentence("I unbelievable", T=10)
        encoded = EncodedSpeech(sentences=[sentence], model_id='test/model')
        result = extract_alignment_from_encoded(encoded)

        assert len(result) == 2
        # "unbelievable" (12 chars) should get 12/13 of the duration
        # "I" (1 char) should get 1/13
        duration_i = result[0]['end'] - result[0]['start']
        duration_unbelievable = result[1]['end'] - result[1]['start']
        assert duration_unbelievable > duration_i * 5  # much longer

    def test_cumulative_timestamps_across_sentences(self):
        """Timestamps should be cumulative across sentences."""
        sent1 = self._make_sentence("hello world", T=10, sentence_index=0)
        sent2 = self._make_sentence("foo bar", T=10, sentence_index=1)
        encoded = EncodedSpeech(sentences=[sent1, sent2], model_id='test/model')
        result = extract_alignment_from_encoded(encoded)

        assert len(result) == 4
        assert result[0]['word'] == 'hello'
        assert result[1]['word'] == 'world'
        assert result[2]['word'] == 'foo'
        assert result[3]['word'] == 'bar'
        # Second sentence starts after first ends
        sentence1_duration = 10 * 2048 / 32000
        assert result[2]['start'] >= sentence1_duration - 0.001

    def test_no_overlap_between_words(self):
        """Word timestamps should never overlap."""
        sentence = self._make_sentence("the quick brown fox jumps", T=20)
        encoded = EncodedSpeech(sentences=[sentence], model_id='test/model')
        result = extract_alignment_from_encoded(encoded)

        assert len(result) == 5
        for i in range(len(result) - 1):
            assert result[i]['end'] <= result[i + 1]['start'] + 0.001

    def test_total_duration_matches_sentence(self):
        """Word timestamps should cover the full sentence duration."""
        T = 15
        sentence = self._make_sentence("hello beautiful world", T=T)
        encoded = EncodedSpeech(sentences=[sentence], model_id='test/model')
        result = extract_alignment_from_encoded(encoded)

        expected_duration = T * 2048 / 32000
        assert result[0]['start'] == 0.0
        assert abs(result[-1]['end'] - expected_duration) < 0.002

    def test_empty_text_produces_no_words(self):
        """Sentence with empty text should produce no alignment entries."""
        sentence = self._make_sentence("", T=10)
        encoded = EncodedSpeech(sentences=[sentence], model_id='test/model')
        result = extract_alignment_from_encoded(encoded)
        assert result == []

    def test_empty_sentences_list(self):
        """Empty sentence list should return empty alignment."""
        encoded = EncodedSpeech(sentences=[], model_id='test/model')
        result = extract_alignment_from_encoded(encoded)
        assert result == []

    def test_timestamps_are_rounded(self):
        """All timestamps should be rounded to 3 decimal places."""
        sentence = self._make_sentence("hello world foo bar", T=17)
        encoded = EncodedSpeech(sentences=[sentence], model_id='test/model')
        result = extract_alignment_from_encoded(encoded)

        for entry in result:
            assert entry['start'] == round(entry['start'], 3)
            assert entry['end'] == round(entry['end'], 3)


# ---------------------------------------------------------------------------
# _build_token_word_map tests
# ---------------------------------------------------------------------------

class TestBuildTokenWordMap:

    def test_basic_mapping(self):
        """Should map token positions to words using offset_mapping."""
        from unittest.mock import MagicMock
        from narro.backends.base import BaseModel

        model = BaseModel()
        # Mock tokenizer that returns offset_mapping
        model.tokenizer = MagicMock()
        # Prompt: [STOP][TEXT]hello world[START]
        #          0123456789012345678901234567
        #          [STOP] = 0-6, [TEXT] = 6-12, hello = 12-17, " " = 17-18, world = 18-23, [START] = 23-30
        prompt = "[STOP][TEXT]hello world[START]"
        model.tokenizer.return_value = {
            'offset_mapping': [
                (0, 1),    # [
                (1, 5),    # STOP
                (5, 6),    # ]
                (6, 7),    # [
                (7, 11),   # TEXT
                (11, 12),  # ]
                (12, 17),  # hello
                (17, 18),  # (space - between words)
                (18, 23),  # world
                (23, 24),  # [
                (24, 29),  # START
                (29, 30),  # ]
            ]
        }

        result = model._build_token_word_map(prompt, "hello world")

        assert result is not None
        # Token 6 (chars 12-17) -> "hello"
        assert result[6] == "hello"
        # Token 8 (chars 18-23) -> "world"
        assert result[8] == "world"
        # Prefix/suffix tokens should not be mapped
        assert 0 not in result
        assert 1 not in result
        assert 10 not in result
        assert 11 not in result

    def test_returns_none_without_offset_mapping(self):
        """Should return None if tokenizer doesn't support offset_mapping."""
        from unittest.mock import MagicMock
        from narro.backends.base import BaseModel

        model = BaseModel()
        model.tokenizer = MagicMock()
        model.tokenizer.side_effect = TypeError("no offset_mapping")

        result = model._build_token_word_map("[STOP][TEXT]hello[START]", "hello")
        assert result is None

    def test_multi_token_words(self):
        """Words split into multiple subword tokens should all map to the same word."""
        from unittest.mock import MagicMock
        from narro.backends.base import BaseModel

        model = BaseModel()
        model.tokenizer = MagicMock()
        # "unbelievable" split into "un" + "believ" + "able"
        prompt = "[STOP][TEXT]unbelievable[START]"
        #          0          1          2
        #          0123456789012345678901234567890
        model.tokenizer.return_value = {
            'offset_mapping': [
                (0, 6),    # [STOP]
                (6, 12),   # [TEXT]
                (12, 14),  # un
                (14, 20),  # believ
                (20, 24),  # able
                (24, 31),  # [START]
            ]
        }

        result = model._build_token_word_map(prompt, "unbelievable")

        assert result is not None
        assert result[2] == "unbelievable"
        assert result[3] == "unbelievable"
        assert result[4] == "unbelievable"


# ---------------------------------------------------------------------------
# CLI --align flag tests
# ---------------------------------------------------------------------------

class TestCLIAlignFlag:

    def test_cli_accepts_align_flag(self):
        """The speak subcommand should accept --align."""
        from narro.cli import main
        from unittest.mock import patch, MagicMock

        with patch('sys.argv', ['narro', 'speak', 'Hello world', '--align', 'output.json']), \
             patch('narro.cli.argparse.ArgumentParser.parse_args') as mock_parse:
            import argparse
            mock_parse.return_value = argparse.Namespace(
                command='speak',
                func=MagicMock(),
                text='Hello world',
                output='output.wav',
                align='output.json',
            )
            main()
            mock_parse.return_value.func.assert_called_once()

    def test_cli_align_flag_short(self):
        """The speak subcommand should accept -a shorthand."""
        import narro.cli as cli_mod

        parser = cli_mod.argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        speak_parser = subparsers.add_parser('speak')
        cli_mod._add_speak_args(speak_parser)
        cli_mod._add_common_args(speak_parser)

        args = parser.parse_args(['speak', 'Hello', '-a', 'out.json'])
        assert args.align == 'out.json'

    def test_cli_align_flag_defaults_none(self):
        """--align should default to None when not provided."""
        import narro.cli as cli_mod

        parser = cli_mod.argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        speak_parser = subparsers.add_parser('speak')
        cli_mod._add_speak_args(speak_parser)
        cli_mod._add_common_args(speak_parser)

        args = parser.parse_args(['speak', 'Hello'])
        assert args.align is None
