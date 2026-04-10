"""Tests for narro/alignment.py — sentence-level alignment from hidden-state counts."""

import json
import os
import tempfile

import numpy as np
import pytest

from narro.alignment import (
    save_alignment,
    extract_alignment_from_encoded,
    extract_paragraph_alignment,
)
from narro.encoded import SentenceEncoding, EncodedSpeech


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sentence(text, seq_len, text_index=0, sentence_index=0):
    """Create a SentenceEncoding with random hidden states (no attention)."""
    return SentenceEncoding(
        hidden_states=np.random.randn(seq_len, 512).astype(np.float32),
        token_ids=np.random.randint(0, 8192, size=seq_len, dtype=np.int32),
        token_entropy=np.random.rand(seq_len).astype(np.float32),
        finish_reason='stop',
        text=text,
        text_index=text_index,
        sentence_index=sentence_index,
        attention_weights=None,
    )


# ---------------------------------------------------------------------------
# save_alignment tests
# ---------------------------------------------------------------------------

class TestSaveAlignment:

    def test_save_alignment_writes_json(self):
        """save_alignment should write valid JSON with indent=2."""
        alignment = [
            {"text": "Hello world.", "start": 0.0, "end": 0.64},
            {"text": "Goodbye.", "start": 0.64, "end": 1.28},
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

TOKEN_DURATION = 2048 / 32000  # 0.064 seconds


class TestExtractAlignmentFromEncoded:

    def test_single_sentence(self):
        """Single sentence should produce one entry with exact timestamps."""
        T = 10
        sentence = _make_sentence("Hello world.", T)
        encoded = EncodedSpeech(sentences=[sentence], model_id='test/model')

        result = extract_alignment_from_encoded(encoded)

        assert len(result) == 1
        assert result[0]['text'] == 'Hello world.'
        assert result[0]['start'] == 0.0
        assert result[0]['end'] == round(T * TOKEN_DURATION, 3)

    def test_multiple_sentences_cumulative_timestamps(self):
        """Timestamps should be cumulative across sentences."""
        T1, T2 = 10, 15
        sent1 = _make_sentence("Hello world.", T1, sentence_index=0)
        sent2 = _make_sentence("Goodbye moon.", T2, sentence_index=1)
        encoded = EncodedSpeech(sentences=[sent1, sent2], model_id='test/model')

        result = extract_alignment_from_encoded(encoded)

        assert len(result) == 2
        assert result[0]['text'] == 'Hello world.'
        assert result[1]['text'] == 'Goodbye moon.'

        # First sentence: [0, T1 * token_duration]
        assert result[0]['start'] == 0.0
        s1_end = round(T1 * TOKEN_DURATION, 3)
        assert result[0]['end'] == s1_end

        # Second sentence starts where first ends
        assert result[1]['start'] == s1_end
        assert result[1]['end'] == round(s1_end + T2 * TOKEN_DURATION, 3)

    def test_empty_text_skipped_but_offset_advances(self):
        """Sentences with empty text should be skipped, but offset still advances."""
        T1, T2, T3 = 10, 5, 10
        sent1 = _make_sentence("Hello.", T1, sentence_index=0)
        sent_empty = _make_sentence("", T2, sentence_index=1)
        sent3 = _make_sentence("World.", T3, sentence_index=2)
        encoded = EncodedSpeech(
            sentences=[sent1, sent_empty, sent3], model_id='test/model',
        )

        result = extract_alignment_from_encoded(encoded)

        assert len(result) == 2
        assert result[0]['text'] == 'Hello.'
        assert result[1]['text'] == 'World.'

        # World should start after both Hello and the empty sentence
        expected_start = round((T1 + T2) * TOKEN_DURATION, 3)
        assert result[1]['start'] == expected_start

    def test_whitespace_only_text_skipped(self):
        """Sentences with only whitespace should be treated as empty."""
        sentence = _make_sentence("   \n\t  ", 10)
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
        T = 7  # 7 * 0.064 = 0.448 — clean, but tests rounding path
        sentence = _make_sentence("Test sentence.", T)
        encoded = EncodedSpeech(sentences=[sentence], model_id='test/model')

        result = extract_alignment_from_encoded(encoded)

        for entry in result:
            assert entry['start'] == round(entry['start'], 3)
            assert entry['end'] == round(entry['end'], 3)

    def test_text_is_stripped(self):
        """Leading/trailing whitespace in sentence text should be stripped."""
        sentence = _make_sentence("  Hello world.  ", 10)
        encoded = EncodedSpeech(sentences=[sentence], model_id='test/model')

        result = extract_alignment_from_encoded(encoded)

        assert len(result) == 1
        assert result[0]['text'] == 'Hello world.'

    def test_duration_is_exact(self):
        """Sentence duration should be exactly T * token_duration."""
        T = 42
        sentence = _make_sentence("A fairly long test sentence for exactness.", T)
        encoded = EncodedSpeech(sentences=[sentence], model_id='test/model')

        result = extract_alignment_from_encoded(encoded)

        duration = result[0]['end'] - result[0]['start']
        expected = round(T * TOKEN_DURATION, 3)
        assert abs(duration - expected) < 1e-9

    def test_no_attention_weights_still_works(self):
        """Sentence-level alignment does not require attention weights."""
        sentence = _make_sentence("No attention needed.", 10)
        assert sentence.attention_weights is None  # sanity check

        encoded = EncodedSpeech(sentences=[sentence], model_id='test/model')
        result = extract_alignment_from_encoded(encoded)

        assert len(result) == 1
        assert result[0]['text'] == 'No attention needed.'


# ---------------------------------------------------------------------------
# extract_paragraph_alignment tests
# ---------------------------------------------------------------------------

class TestExtractParagraphAlignment:

    def test_single_paragraph_single_sentence(self):
        """One paragraph with one sentence produces one entry."""
        sent = _make_sentence("Hello world.", 10, text_index=0, sentence_index=0)
        encoded = EncodedSpeech(sentences=[sent], model_id='test/model')

        result = extract_paragraph_alignment(encoded)

        assert len(result) == 1
        assert result[0]['paragraph'] == 0
        assert result[0]['start'] == 0.0
        assert result[0]['end'] == round(10 * TOKEN_DURATION, 3)

    def test_single_paragraph_multiple_sentences(self):
        """Multiple sentences in one paragraph produce one entry spanning all."""
        T1, T2 = 10, 15
        sent1 = _make_sentence("Hello.", T1, text_index=0, sentence_index=0)
        sent2 = _make_sentence("World.", T2, text_index=0, sentence_index=1)
        encoded = EncodedSpeech(sentences=[sent1, sent2], model_id='test/model')

        result = extract_paragraph_alignment(encoded)

        assert len(result) == 1
        assert result[0]['paragraph'] == 0
        assert result[0]['start'] == 0.0
        assert result[0]['end'] == round((T1 + T2) * TOKEN_DURATION, 3)

    def test_multiple_paragraphs(self):
        """Each text_index becomes one paragraph entry with correct timing."""
        T1, T2, T3 = 10, 15, 20
        sent1 = _make_sentence("Para one.", T1, text_index=0, sentence_index=0)
        sent2 = _make_sentence("Para two A.", T2, text_index=1, sentence_index=0)
        sent3 = _make_sentence("Para two B.", T3, text_index=1, sentence_index=1)
        encoded = EncodedSpeech(sentences=[sent1, sent2, sent3], model_id='test/model')

        result = extract_paragraph_alignment(encoded)

        assert len(result) == 2
        assert result[0]['paragraph'] == 0
        assert result[1]['paragraph'] == 1

        # Paragraph 0: just sent1
        assert result[0]['start'] == 0.0
        p0_end = round(T1 * TOKEN_DURATION, 3)
        assert result[0]['end'] == p0_end

        # Paragraph 1: sent2 + sent3
        assert result[1]['start'] == p0_end
        assert result[1]['end'] == round(p0_end + (T2 + T3) * TOKEN_DURATION, 3)

    def test_empty_sentences_list(self):
        """No sentences produces empty alignment."""
        encoded = EncodedSpeech(sentences=[], model_id='test/model')
        result = extract_paragraph_alignment(encoded)
        assert result == []

    def test_empty_text_advances_offset(self):
        """Sentences with empty text still advance the cumulative offset."""
        T1, T2, T3 = 10, 5, 10
        sent1 = _make_sentence("Para one.", T1, text_index=0, sentence_index=0)
        sent_empty = _make_sentence("", T2, text_index=0, sentence_index=1)
        sent3 = _make_sentence("Para two.", T3, text_index=1, sentence_index=0)
        encoded = EncodedSpeech(
            sentences=[sent1, sent_empty, sent3], model_id='test/model',
        )

        result = extract_paragraph_alignment(encoded)

        # Paragraph 0 includes the empty sentence's duration
        assert result[0]['end'] == round((T1 + T2) * TOKEN_DURATION, 3)
        # Paragraph 1 starts after both
        assert result[1]['start'] == round((T1 + T2) * TOKEN_DURATION, 3)

    def test_timestamps_are_rounded(self):
        """All timestamps should be rounded to 3 decimal places."""
        T = 7
        sent = _make_sentence("Test.", T, text_index=0)
        encoded = EncodedSpeech(sentences=[sent], model_id='test/model')

        result = extract_paragraph_alignment(encoded)

        for entry in result:
            assert entry['start'] == round(entry['start'], 3)
            assert entry['end'] == round(entry['end'], 3)


# ---------------------------------------------------------------------------
# CLI --align flag tests
# ---------------------------------------------------------------------------

# CLI --align flag was removed along with the speak subcommand.
# Alignment is now accessed via the HTTP API (align: true in the request)
# and via NarroClient.generate_with_alignment().
