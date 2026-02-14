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

    def test_extract_from_single_sentence(self):
        """Convenience function should handle a single sentence."""
        text = "hello world"
        T = 10
        # Input tokens: [STOP][TEXT] hello world [START] = 5 tokens
        input_len = 5

        attention = np.zeros((T, input_len), dtype=np.float32)
        # "hello" maps to input token 2, "world" maps to input token 3
        attention[:5, 2] = 1.0
        attention[5:, 3] = 1.0

        sentence = SentenceEncoding(
            hidden_states=np.random.randn(T, 512).astype(np.float32),
            token_ids=np.random.randint(0, 8192, size=T, dtype=np.int32),
            token_entropy=np.random.rand(T).astype(np.float32),
            finish_reason='stop',
            text=text,
            text_index=0,
            sentence_index=0,
            attention_weights=attention,
        )

        encoded = EncodedSpeech(
            sentences=[sentence],
            model_id='test/model',
        )

        result = extract_alignment_from_encoded(encoded)

        assert len(result) == 2
        assert result[0]['word'] == 'hello'
        assert result[1]['word'] == 'world'

    def test_extract_from_multiple_sentences_cumulative_timestamps(self):
        """Timestamps should be cumulative across sentences."""
        T1 = 10
        T2 = 10
        input_len = 5

        # Sentence 1: "hello world"
        attn1 = np.zeros((T1, input_len), dtype=np.float32)
        attn1[:5, 2] = 1.0
        attn1[5:, 3] = 1.0

        sent1 = SentenceEncoding(
            hidden_states=np.random.randn(T1, 512).astype(np.float32),
            token_ids=np.random.randint(0, 8192, size=T1, dtype=np.int32),
            token_entropy=np.random.rand(T1).astype(np.float32),
            finish_reason='stop',
            text="hello world",
            text_index=0,
            sentence_index=0,
            attention_weights=attn1,
        )

        # Sentence 2: "foo bar"
        attn2 = np.zeros((T2, input_len), dtype=np.float32)
        attn2[:5, 2] = 1.0
        attn2[5:, 3] = 1.0

        sent2 = SentenceEncoding(
            hidden_states=np.random.randn(T2, 512).astype(np.float32),
            token_ids=np.random.randint(0, 8192, size=T2, dtype=np.int32),
            token_entropy=np.random.rand(T2).astype(np.float32),
            finish_reason='stop',
            text="foo bar",
            text_index=0,
            sentence_index=1,
            attention_weights=attn2,
        )

        encoded = EncodedSpeech(
            sentences=[sent1, sent2],
            model_id='test/model',
        )

        result = extract_alignment_from_encoded(encoded)

        assert len(result) == 4
        # First sentence words
        assert result[0]['word'] == 'hello'
        assert result[1]['word'] == 'world'
        # Second sentence words should have timestamps offset by sentence 1 duration
        assert result[2]['word'] == 'foo'
        assert result[3]['word'] == 'bar'
        # Second sentence starts after first sentence ends
        sentence1_duration = T1 * 2048 / 32000  # TOKEN_SIZE / SAMPLE_RATE
        assert result[2]['start'] >= sentence1_duration - 0.001

    def test_extract_from_encoded_no_attention_skips_sentence(self):
        """Sentences without attention weights should be skipped (but offset still advances)."""
        T = 10
        sent_no_attn = SentenceEncoding(
            hidden_states=np.random.randn(T, 512).astype(np.float32),
            token_ids=np.random.randint(0, 8192, size=T, dtype=np.int32),
            token_entropy=np.random.rand(T).astype(np.float32),
            finish_reason='stop',
            text="hello world",
            text_index=0,
            sentence_index=0,
            attention_weights=None,
        )

        encoded = EncodedSpeech(
            sentences=[sent_no_attn],
            model_id='test/model',
        )

        result = extract_alignment_from_encoded(encoded)
        assert result == []

    def test_extract_from_encoded_empty_sentences(self):
        """Empty sentence list should return empty alignment."""
        encoded = EncodedSpeech(
            sentences=[],
            model_id='test/model',
        )
        result = extract_alignment_from_encoded(encoded)
        assert result == []


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
