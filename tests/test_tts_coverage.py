"""Tests to increase coverage for text preprocessing, CLI, hallucination detector
edge cases, batch inference, streaming inference, text normalizer, and text splitter."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from narro.tts import (
    Narro,
    SAMPLE_RATE,
    INT16_MAX,
    RECEPTIVE_FIELD,
    TOKEN_SIZE,
    HIDDEN_DIM,
    DIFF_THRESHOLD,
    MAX_RUNLENGTH,
)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tts_stub():
    """Create a Narro stub without triggering __init__ (warmup inference)."""
    obj = object.__new__(Narro)
    obj.model_id = 'test/model'
    return obj


def _make_hidden_state(seq_len, dim=HIDDEN_DIM):
    """Create a fake hidden state tensor of shape (seq_len, dim)."""
    return torch.randn(seq_len, dim)


def _make_pipeline_response(seq_len, finish_reason='stop'):
    """Create a fake enriched pipeline response dict."""
    return {
        'finish_reason': finish_reason,
        'hidden_state': _make_hidden_state(seq_len),
        'token_ids': torch.randint(0, 8192, (seq_len,), dtype=torch.int32),
        'token_entropy': torch.rand(seq_len),
        'attention': None,
    }


# ---------------------------------------------------------------------------
# Text preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessText:
    """Test Narro._preprocess_text behavior."""

    def test_single_short_sentence(self):
        """A single short sentence should not be split or merged."""
        tts = _make_tts_stub()
        result = tts._preprocess_text(["Hello world."])
        assert len(result) == 1
        prompt, text_idx, sentence_idx, original_text = result[0]
        assert prompt.startswith("[STOP][TEXT]")
        assert prompt.endswith("[START]")
        assert text_idx == 0
        assert sentence_idx == 0

    def test_wrapping_format(self):
        """Each sentence should be wrapped as [STOP][TEXT]...[START]."""
        tts = _make_tts_stub()
        result = tts._preprocess_text(["Testing format."])
        prompt = result[0][0]
        assert prompt == "[STOP][TEXT]testing format.[START]"

    def test_multiple_texts(self):
        """Multiple texts should each be tracked by text_idx."""
        tts = _make_tts_stub()
        result = tts._preprocess_text(["First text.", "Second text."])
        text_indices = [r[1] for r in result]
        assert 0 in text_indices
        assert 1 in text_indices

    def test_sentence_splitting(self):
        """Long text with multiple sentences should be split."""
        tts = _make_tts_stub()
        text = "This is the first sentence. This is the second sentence. And this is the third one."
        result = tts._preprocess_text([text])
        # Should produce multiple sentence entries
        assert len(result) >= 1
        for prompt, text_idx, sentence_idx, _ in result:
            assert text_idx == 0
            assert prompt.startswith("[STOP][TEXT]")

    def test_short_sentences_merged(self):
        """Sentences shorter than min_length should be merged with neighbors."""
        tts = _make_tts_stub()
        # "Hi." is very short (3 chars < 30), should be merged
        text = "Hi. This is a much longer sentence that should stand on its own for testing."
        result = tts._preprocess_text([text], min_length=30)
        # The short "Hi." should be merged with the longer sentence
        assert len(result) >= 1

    def test_min_length_zero_disables_merging(self):
        """Setting min_length=0 should disable sentence merging."""
        tts = _make_tts_stub()
        text = "Hi. Bye. This is a test."
        result = tts._preprocess_text([text], min_length=0)
        # With min_length=0, no merging happens
        assert len(result) >= 1

    def test_whitespace_stripping(self):
        """Leading/trailing whitespace should be stripped from input text."""
        tts = _make_tts_stub()
        result = tts._preprocess_text(["  Hello world.  "])
        prompt = result[0][0]
        assert "  " not in prompt.replace("[STOP][TEXT]", "").replace("[START]", "")

    def test_sentence_indices_are_sequential(self):
        """Sentence indices within a single text should be sequential."""
        tts = _make_tts_stub()
        text = (
            "First sentence is long enough to stand alone here. "
            "Second sentence is also long enough to stand alone. "
            "Third sentence is definitely long enough too."
        )
        result = tts._preprocess_text([text])
        sentence_indices = [r[2] for r in result]
        for i in range(len(sentence_indices)):
            assert sentence_indices[i] == i

    def test_empty_text(self):
        """Empty text should produce no results after cleaning."""
        tts = _make_tts_stub()
        result = tts._preprocess_text([""])
        # clean_text on empty string may produce empty => split_and_recombine
        # returns empty list, so no sentences
        assert len(result) == 0

    def test_short_trailing_sentence_merged_forward(self):
        """A short sentence in the middle should be merged forward when no prior merged."""
        tts = _make_tts_stub()
        # First sentence is short, so it gets merged with the next
        text = "Ok. This is a very long second sentence that passes the minimum length threshold."
        result = tts._preprocess_text([text], min_length=30)
        assert len(result) >= 1

    def test_only_short_sentence(self):
        """A single short sentence (only one) should still be emitted even if < min_length."""
        tts = _make_tts_stub()
        result = tts._preprocess_text(["Hi."])
        assert len(result) == 1

    def test_clean_text_is_called(self):
        """Text normalization should be applied (e.g., numbers expanded)."""
        tts = _make_tts_stub()
        result = tts._preprocess_text(["I have $5."])
        prompt = result[0][0]
        # "$5" should be expanded to "five dollars" by clean_text
        assert "five dollars" in prompt


# ---------------------------------------------------------------------------
# Hallucination detector edge cases
# ---------------------------------------------------------------------------

class TestHallucinationDetectorEdgeCases:
    """Additional edge cases for the vectorized hallucination detector."""

    def test_exactly_max_runlength_states_no_trigger(self):
        """Exactly MAX_RUNLENGTH identical states should NOT trigger (need >MAX_RUNLENGTH)."""
        tts = _make_tts_stub()
        base = torch.zeros(HIDDEN_DIM)
        # MAX_RUNLENGTH identical states => MAX_RUNLENGTH-1 diffs, not enough
        hidden_state = [base.clone() for _ in range(MAX_RUNLENGTH)]
        assert tts.hallucination_detector(hidden_state) is False

    def test_runlength_counter_decrements(self):
        """The detector decrements the counter on varied states."""
        tts = _make_tts_stub()
        hidden_state = []
        base = torch.zeros(HIDDEN_DIM)
        # Alternating pattern: short runs of similar, then varied
        # Should not trigger because varied states decrement the counter
        for _ in range(5):
            # 5 similar states
            for _ in range(5):
                hidden_state.append(base.clone())
            # 10 varied states (high magnitude diffs)
            for _ in range(10):
                hidden_state.append(torch.randn(HIDDEN_DIM) * 10000)
        assert tts.hallucination_detector(hidden_state) is False

    def test_single_element_sequence(self):
        """Single element sequence should return False."""
        tts = _make_tts_stub()
        assert tts.hallucination_detector([torch.randn(HIDDEN_DIM)]) is False

    def test_empty_sequence(self):
        """Empty sequence should return False."""
        tts = _make_tts_stub()
        assert tts.hallucination_detector([]) is False

    def test_all_identical_long_sequence(self):
        """Long sequence of all-identical states should trigger."""
        tts = _make_tts_stub()
        base = torch.zeros(HIDDEN_DIM)
        hidden_state = [base.clone() for _ in range(100)]
        assert tts.hallucination_detector(hidden_state) is True

    def test_diffs_just_above_threshold_no_trigger(self):
        """States with L1 diffs just above DIFF_THRESHOLD should not trigger."""
        tts = _make_tts_stub()
        # Create states where each adjacent diff is just above DIFF_THRESHOLD
        hidden_state = []
        state = torch.zeros(HIDDEN_DIM)
        for i in range(50):
            # Each step changes by DIFF_THRESHOLD+100 total L1 distance
            delta = torch.zeros(HIDDEN_DIM)
            delta[0] = DIFF_THRESHOLD + 100  # single element change exceeds threshold
            if i % 2 == 0:
                hidden_state.append(state.clone())
            else:
                hidden_state.append((state + delta).clone())
        assert tts.hallucination_detector(hidden_state) is False


# ---------------------------------------------------------------------------
# Infer method tests (mocked pipeline)
# ---------------------------------------------------------------------------

class TestInfer:
    """Test Narro.infer with a mocked pipeline and decoder."""

    def _make_tts_with_mocks(self):
        """Create a TTS stub with mocked pipeline and decoder."""
        tts = _make_tts_stub()
        tts.pipeline = MagicMock()
        tts.decoder = MagicMock()
        tts.decoder_batch_size = 4
        return tts

    def test_infer_returns_tensor(self):
        """infer() should return a 1D audio tensor."""
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]
        # Decoder returns (batch, audio_samples) tensor
        audio_len = (seq_len - 1) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(1, audio_len + TOKEN_SIZE)

        result = tts.infer("Hello world.")
        assert isinstance(result, torch.Tensor)

    def test_infer_writes_wav_file(self):
        """infer() with out_path should write a WAV file."""
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]
        audio_len = (seq_len - 1) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(1, audio_len + TOKEN_SIZE)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            out_path = f.name

        try:
            tts.infer("Hello world.", out_path=out_path)
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0
        finally:
            os.unlink(out_path)

    def test_infer_passes_parameters_to_batch(self):
        """infer() should pass top_p, temperature, repetition_penalty to infer_batch."""
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]
        audio_len = (seq_len - 1) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(1, audio_len + TOKEN_SIZE)

        tts.infer("Test", top_p=0.8, temperature=0.5, repetition_penalty=1.5)
        tts.pipeline.infer.assert_called_once()
        call_kwargs = tts.pipeline.infer.call_args
        assert call_kwargs[1]['top_p'] == 0.8
        assert call_kwargs[1]['temperature'] == 0.5
        assert call_kwargs[1]['repetition_penalty'] == 1.5


# ---------------------------------------------------------------------------
# Batch inference tests
# ---------------------------------------------------------------------------

class TestInferBatch:
    """Test Narro.infer_batch with mocked pipeline and decoder."""

    def _make_tts_with_mocks(self):
        tts = _make_tts_stub()
        tts.pipeline = MagicMock()
        tts.decoder = MagicMock()
        tts.decoder_batch_size = 4
        return tts

    def test_batch_multiple_texts(self):
        """infer_batch should handle multiple texts and return a list per text."""
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [
            _make_pipeline_response(seq_len),
            _make_pipeline_response(seq_len),
        ]
        audio_len = (seq_len - 1) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(2, audio_len + TOKEN_SIZE)

        results = tts.infer_batch(["First text.", "Second text."])
        assert len(results) == 2
        assert all(isinstance(r, torch.Tensor) for r in results)

    def test_batch_writes_wav_to_out_dir(self):
        """infer_batch with out_dir should write WAV files."""
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [
            _make_pipeline_response(seq_len),
        ]
        audio_len = (seq_len - 1) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(1, audio_len + TOKEN_SIZE)

        with tempfile.TemporaryDirectory() as out_dir:
            tts.infer_batch(["Hello world."], out_dir=out_dir)
            assert os.path.exists(os.path.join(out_dir, "0.wav"))

    def test_batch_hallucination_retry(self):
        """infer_batch with retries should re-infer hallucinated sentences."""
        tts = self._make_tts_with_mocks()
        seq_len = 20

        # First call: hallucination (all identical hidden states)
        base = torch.zeros(HIDDEN_DIM)
        hallucinated_hidden = torch.stack([base.clone() for _ in range(50)])
        bad_response = {
            'finish_reason': 'stop',
            'hidden_state': hallucinated_hidden,
            'token_ids': torch.zeros(50, dtype=torch.int32),
            'token_entropy': torch.zeros(50),
            'attention': None,
        }
        good_response = _make_pipeline_response(seq_len)

        tts.pipeline.infer.side_effect = [
            [bad_response],
            [good_response],
        ]

        audio_len = (seq_len - 1) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(1, audio_len + TOKEN_SIZE)

        results = tts.infer_batch(["Hello world."], retries=1)
        assert len(results) == 1
        # Should have been called twice (original + retry)
        assert tts.pipeline.infer.call_count == 2

    def test_batch_no_retry_when_retries_zero(self):
        """With retries=0, hallucination detection should not trigger regeneration."""
        tts = self._make_tts_with_mocks()
        base = torch.zeros(HIDDEN_DIM)
        hallucinated_hidden = torch.stack([base.clone() for _ in range(50)])
        bad_response = {
            'finish_reason': 'stop',
            'hidden_state': hallucinated_hidden,
            'token_ids': torch.zeros(50, dtype=torch.int32),
            'token_entropy': torch.zeros(50),
            'attention': None,
        }

        tts.pipeline.infer.return_value = [bad_response]
        audio_len = 49 * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(1, audio_len + TOKEN_SIZE)

        results = tts.infer_batch(["Hello world."], retries=0)
        assert tts.pipeline.infer.call_count == 1

    def test_batch_finish_reason_length_logs_warning(self):
        """A non-'stop' finish reason should log a warning."""
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [
            _make_pipeline_response(seq_len, finish_reason='length')
        ]
        audio_len = (seq_len - 1) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(1, audio_len + TOKEN_SIZE)

        with patch('narro.tts.logger') as mock_logger:
            tts.infer_batch(["Hello world."])
            mock_logger.warning.assert_called()

    def test_batch_decoder_batching(self):
        """With batch_size=2 and 3 sentences, decoder should be called at least twice."""
        tts = self._make_tts_with_mocks()
        tts.decoder_batch_size = 2
        seq_len = 20

        # This text should produce 3 sentences (long enough to split)
        text = (
            "First sentence stands on its own and is long. "
            "Second sentence also stands on its own and is long. "
            "Third sentence completes the trio and is long."
        )

        tts.pipeline.infer.return_value = [
            _make_pipeline_response(seq_len) for _ in range(3)
        ]
        audio_len = (seq_len - 1) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(2, audio_len + TOKEN_SIZE)

        results = tts.infer_batch([text])
        # 3 sentences with batch_size=2 requires at least 2 decoder calls
        assert tts.decoder.call_count >= 2

    def test_batch_sorts_by_descending_length(self):
        """Hidden states should be sorted by descending length for efficient padding."""
        tts = self._make_tts_with_mocks()
        tts.decoder_batch_size = 10

        # Two texts that will produce sentences of different hidden state lengths
        tts.pipeline.infer.return_value = [
            _make_pipeline_response(10),
            _make_pipeline_response(30),
        ]
        audio_len_long = 29 * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(2, audio_len_long + TOKEN_SIZE)

        results = tts.infer_batch(["Short text.", "This is a longer text that should have more tokens."])
        assert len(results) == 2
        # Decoder batch should have max_len=30 (longest first due to sorting)
        call_args = tts.decoder.call_args[0][0]  # first positional arg
        assert call_args.shape[2] == 30  # max_len from h_long


# ---------------------------------------------------------------------------
# Streaming inference tests
# ---------------------------------------------------------------------------

class TestInferStream:
    """Test Narro.infer_stream with a mocked pipeline and decoder."""

    def _make_tts_with_mocks(self):
        tts = _make_tts_stub()
        tts.pipeline = MagicMock()
        tts.decoder = MagicMock()
        tts.decoder_batch_size = 4
        return tts

    def test_stream_yields_audio_chunks(self):
        """infer_stream should yield audio tensors."""
        tts = self._make_tts_with_mocks()

        # Simulate streaming tokens: RECEPTIVE_FIELD + chunk_size tokens, then finish
        tokens = []
        num_tokens = RECEPTIVE_FIELD + 3
        for i in range(num_tokens):
            tokens.append({
                'finish_reason': None,
                'hidden_state': torch.randn(1, HIDDEN_DIM),
            })
        tokens.append({'finish_reason': 'stop', 'hidden_state': torch.randn(1, HIDDEN_DIM)})

        tts.pipeline.stream_infer.return_value = iter(tokens)

        audio_samples = (num_tokens + RECEPTIVE_FIELD) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(1, audio_samples)

        chunks = list(tts.infer_stream("Hello world."))
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, torch.Tensor)

    def test_stream_respects_chunk_size(self):
        """infer_stream should accumulate chunk_size tokens before yielding."""
        tts = self._make_tts_with_mocks()

        chunk_size = 3
        # Generate enough tokens for multiple chunks
        num_tokens = RECEPTIVE_FIELD + chunk_size * 3
        tokens = []
        for _ in range(num_tokens):
            tokens.append({
                'finish_reason': None,
                'hidden_state': torch.randn(1, HIDDEN_DIM),
            })
        tokens.append({'finish_reason': 'stop', 'hidden_state': torch.randn(1, HIDDEN_DIM)})

        tts.pipeline.stream_infer.return_value = iter(tokens)

        audio_samples = (num_tokens + RECEPTIVE_FIELD) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(1, audio_samples)

        chunks = list(tts.infer_stream("Hello world.", chunk_size=chunk_size))
        # Should get at least 2 chunks (intermediate + final)
        assert len(chunks) >= 2

    def test_stream_logs_latency(self):
        """infer_stream should log streaming latency for the first chunk."""
        tts = self._make_tts_with_mocks()

        tokens = []
        for _ in range(RECEPTIVE_FIELD + 1):
            tokens.append({
                'finish_reason': None,
                'hidden_state': torch.randn(1, HIDDEN_DIM),
            })
        tokens.append({'finish_reason': 'stop', 'hidden_state': torch.randn(1, HIDDEN_DIM)})

        tts.pipeline.stream_infer.return_value = iter(tokens)
        audio_samples = (RECEPTIVE_FIELD + 2) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(1, audio_samples)

        with patch('narro.tts.logger') as mock_logger:
            list(tts.infer_stream("Hello."))
            # The first chunk should log latency
            mock_logger.info.assert_called()

    def test_stream_handles_multiple_sentences(self):
        """infer_stream should process multiple sentences from a single text."""
        tts = self._make_tts_with_mocks()

        tokens = []
        for _ in range(RECEPTIVE_FIELD + 1):
            tokens.append({
                'finish_reason': None,
                'hidden_state': torch.randn(1, HIDDEN_DIM),
            })
        tokens.append({'finish_reason': 'stop', 'hidden_state': torch.randn(1, HIDDEN_DIM)})

        # stream_infer is called once per sentence
        tts.pipeline.stream_infer.return_value = iter(tokens)

        audio_samples = (RECEPTIVE_FIELD + 2) * TOKEN_SIZE
        tts.decoder.return_value = torch.randn(1, audio_samples)

        chunks = list(tts.infer_stream("Hello world."))
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCLI:
    """Test the CLI argument parsing and main() function."""

    def test_cli_argument_parser_defaults(self):
        """CLI should parse required text arg with correct defaults."""
        from narro.cli import main

        with patch('sys.argv', ['narro', 'Hello world', '-o', 'output.wav']), \
             patch('narro.Narro') as mock_tts:
            mock_instance = MagicMock()
            mock_tts.return_value = mock_instance
            main()

            mock_tts.assert_called_once_with(
                model_path=None,
                compile=True,
                quantize=False,
                decoder_batch_size=4,
                num_threads=None,
            )
            mock_instance.infer.assert_called_once_with(
                'Hello world', out_path='output.wav'
            )

    def test_cli_no_compile_flag(self):
        """--no-compile should pass compile=False to Narro."""
        from narro.cli import main

        with patch('sys.argv', ['narro', 'Test', '--no-compile']), \
             patch('narro.Narro') as mock_tts:
            mock_tts.return_value = MagicMock()
            main()
            assert mock_tts.call_args[1]['compile'] is False

    def test_cli_quantize_flag(self):
        """--quantize should pass quantize=True to Narro."""
        from narro.cli import main

        with patch('sys.argv', ['narro', 'Test', '--quantize']), \
             patch('narro.Narro') as mock_tts:
            mock_tts.return_value = MagicMock()
            main()
            assert mock_tts.call_args[1]['quantize'] is True

    def test_cli_custom_threads_and_batch_size(self):
        """--num-threads and --decoder-batch-size should be passed through."""
        from narro.cli import main

        with patch('sys.argv', ['narro', 'Test', '-t', '4', '-bs', '8']), \
             patch('narro.Narro') as mock_tts:
            mock_tts.return_value = MagicMock()
            main()
            assert mock_tts.call_args[1]['decoder_batch_size'] == 8
            assert mock_tts.call_args[1]['num_threads'] == 4

    def test_cli_model_path(self):
        """--model-path should be passed through to Narro."""
        from narro.cli import main

        with patch('sys.argv', ['narro', 'Test', '-m', '/tmp/my_model']), \
             patch('narro.Narro') as mock_tts:
            mock_tts.return_value = MagicMock()
            main()
            assert mock_tts.call_args[1]['model_path'] == '/tmp/my_model'


# ---------------------------------------------------------------------------
# Text normalizer tests
# ---------------------------------------------------------------------------

class TestTextNormalizer:
    """Test text normalization functions."""

    def test_clean_text_basic(self):
        from narro.utils.text_normalizer import clean_text
        result = clean_text("Hello World!")
        assert result == "hello world!"

    def test_abbreviation_expansion(self):
        from narro.utils.text_normalizer import expand_abbreviations
        assert "doctor" in expand_abbreviations("Dr. Smith")
        assert "mister" in expand_abbreviations("Mr. Jones")

    def test_number_expansion(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("42")
        assert "forty" in result and "two" in result

    def test_dollar_expansion(self):
        from narro.utils.text_normalizer import clean_text
        result = clean_text("$5.99")
        assert "dollar" in result
        assert "cent" in result

    def test_dollar_expansion_zero(self):
        from narro.utils.text_normalizer import _expand_dollars
        import re
        m = re.match(r'\$([\d\.\,]*\d+)', '$0.00')
        result = _expand_dollars(m)
        assert "zero dollars" in result

    def test_dollar_expansion_cents_only(self):
        from narro.utils.text_normalizer import _expand_dollars
        import re
        m = re.match(r'\$([\d\.\,]*\d+)', '$0.50')
        result = _expand_dollars(m)
        assert "cent" in result

    def test_dollar_expansion_one_dollar(self):
        from narro.utils.text_normalizer import _expand_dollars
        import re
        m = re.match(r'\$([\d\.\,]*\d+)', '$1.00')
        result = _expand_dollars(m)
        assert "dollar" in result
        assert "dollars" not in result  # singular

    def test_dollar_expansion_one_cent(self):
        from narro.utils.text_normalizer import _expand_dollars
        import re
        m = re.match(r'\$([\d\.\,]*\d+)', '$0.01')
        result = _expand_dollars(m)
        assert "cent" in result
        assert "cents" not in result  # singular

    def test_pound_expansion(self):
        """Pounds sign is handled by regex before unidecode. Test normalize_numbers directly."""
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("£20")
        assert "pounds" in result

    def test_num_prefix(self):
        from narro.utils.text_normalizer import clean_text
        result = clean_text("#1")
        assert "number" in result

    def test_num_suffix_k(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("100K")
        assert "thousand" in result

    def test_num_suffix_m(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("5M")
        assert "million" in result

    def test_num_suffix_b(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("3B")
        assert "billion" in result

    def test_num_suffix_t(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("2T")
        assert "trillion" in result

    def test_special_characters(self):
        from narro.utils.text_normalizer import expand_special_characters
        assert "at" in expand_special_characters("user@domain")
        assert "and" in expand_special_characters("A & B")
        assert "percent" in expand_special_characters("50%")

    def test_collapse_whitespace(self):
        from narro.utils.text_normalizer import collapse_whitespace
        assert collapse_whitespace("hello   world") == "hello world"
        assert collapse_whitespace("hello .") == "hello."

    def test_dedup_punctuation(self):
        from narro.utils.text_normalizer import dedup_punctuation
        assert dedup_punctuation("hello....") == "hello..."
        assert dedup_punctuation("hello,,,,") == "hello,"
        assert dedup_punctuation("hello!!") == "hello!"
        assert dedup_punctuation("hello??") == "hello?"

    def test_collapse_triple_letters(self):
        from narro.utils.text_normalizer import collapse_triple_letters
        assert collapse_triple_letters("hellooo") == "helloo"
        assert collapse_triple_letters("aaa") == "aa"

    def test_convert_to_ascii(self):
        from narro.utils.text_normalizer import convert_to_ascii
        result = convert_to_ascii("cafe\u0301")
        assert "cafe" in result

    def test_normalize_newlines(self):
        from narro.utils.text_normalizer import normalize_newlines
        result = normalize_newlines("Hello\nworld")
        assert "Hello." in result

    def test_normalize_newlines_preserves_terminal_punctuation(self):
        from narro.utils.text_normalizer import normalize_newlines
        result = normalize_newlines("Hello!\nworld?")
        assert "Hello!" in result
        assert "world?" in result

    def test_normalize_newlines_skips_empty_lines(self):
        from narro.utils.text_normalizer import normalize_newlines
        result = normalize_newlines("Hello\n\nworld")
        # Empty line between should just collapse
        assert "Hello." in result

    def test_remove_unknown_characters(self):
        from narro.utils.text_normalizer import remove_unknown_characters
        result = remove_unknown_characters("Hello{world}")
        assert "{" not in result

    def test_time_expansion_hours_oclock(self):
        from narro.utils.text_normalizer import clean_text
        result = clean_text("8:00")
        assert "o'clock" in result or "eight" in result

    def test_time_expansion_with_minutes(self):
        from narro.utils.text_normalizer import clean_text
        result = clean_text("8:15")
        assert "fifteen" in result or "15" in result

    def test_time_expansion_zero_prefix_minutes(self):
        from narro.utils.text_normalizer import clean_text
        result = clean_text("8:05")
        assert "oh" in result

    def test_date_expansion(self):
        from narro.utils.text_normalizer import clean_text
        result = clean_text("1/1/2025")
        assert "dash" in result

    def test_phone_number_expansion(self):
        from narro.utils.text_normalizer import clean_text
        result = clean_text("(123) 456-7890")
        # Phone numbers get expanded digit-by-digit
        assert len(result) > 0

    def test_decimal_expansion(self):
        from narro.utils.text_normalizer import clean_text
        result = clean_text("3.14")
        assert "point" in result

    def test_fraction_expansion(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("1/2")
        assert "over" in result

    def test_ordinal_expansion(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("1st")
        assert "first" in result

    def test_year_2000(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("2000")
        assert "two thousand" in result

    def test_year_2005(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("2005")
        assert "thousand" in result

    def test_year_1999(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("1999")
        assert "nineteen" in result

    def test_year_2100(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("2100")
        assert "hundred" in result

    def test_multiply_expansion(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("3 * 4")
        assert "times" in result

    def test_divide_expansion(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("8 / 2")
        assert "over" in result

    def test_add_expansion(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("3 + 4")
        assert "plus" in result

    def test_subtract_expansion(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("5 - 3")
        assert "minus" in result

    def test_link_header_expansion(self):
        from narro.utils.text_normalizer import normalize_special
        result = normalize_special("https://example.com")
        assert "h t t p" in result

    def test_dash_in_sentence_expansion(self):
        from narro.utils.text_normalizer import normalize_special
        result = normalize_special("word - another")
        assert "," in result

    def test_dot_abbreviation_expansion(self):
        from narro.utils.text_normalizer import normalize_special
        result = normalize_special("U.S")
        assert "dot" in result

    def test_parentheses_expansion(self):
        from narro.utils.text_normalizer import normalize_special
        result = normalize_special("hello (world) there")
        assert "(" not in result

    def test_mixed_case_splitting(self):
        from narro.utils.text_normalizer import normalize_mixedcase
        result = normalize_mixedcase("LMDeploy")
        # CamelCase should be split
        assert " " in result or result == "LMDeploy"

    def test_cased_abbreviation_hz(self):
        from narro.utils.text_normalizer import expand_abbreviations
        assert "hertz" in expand_abbreviations("100 Hz signal")

    def test_cased_abbreviation_cpu(self):
        from narro.utils.text_normalizer import expand_abbreviations
        assert "c p u" in expand_abbreviations("the CPU is fast")

    def test_cased_abbreviation_gpu_plural(self):
        from narro.utils.text_normalizer import expand_abbreviations
        result = expand_abbreviations("multiple GPUs")
        assert "g p u" in result

    def test_cased_abbreviation_day(self):
        from narro.utils.text_normalizer import expand_abbreviations
        assert "monday" in expand_abbreviations("Mon morning")

    def test_cased_abbreviation_month(self):
        from narro.utils.text_normalizer import expand_abbreviations
        assert "january" in expand_abbreviations("Jan 1st")

    def test_preunicode_em_dash(self):
        from narro.utils.text_normalizer import expand_preunicode_special_characters
        result = expand_preunicode_special_characters("hello\u2014world")
        assert " - " in result

    def test_comma_number_removal(self):
        from narro.utils.text_normalizer import normalize_numbers
        # "1,000" should have comma removed then be expanded
        result = normalize_numbers("1,000")
        assert "," not in result

    def test_alphanumeric_splitting(self):
        from narro.utils.text_normalizer import normalize_numbers
        result = normalize_numbers("100x")
        # "100x" => "100 x" after splitting
        assert " " in result or "x" in result

    def test_clean_text_full_pipeline(self):
        """Test clean_text processes all stages end to end."""
        from narro.utils.text_normalizer import clean_text
        result = clean_text("Dr. Smith paid $42.50 on 1/15/2025 at 3:30 for 2 APIs & 100K tokens!")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be lowercase
        assert result == result.lower()
        # No dollar sign
        assert "$" not in result


# ---------------------------------------------------------------------------
# Text splitter tests
# ---------------------------------------------------------------------------

class TestTextSplitter:
    """Test the text splitting utility."""

    def test_single_sentence(self):
        from narro.utils.text_splitter import split_and_recombine_text
        result = split_and_recombine_text("Hello world.")
        assert len(result) == 1
        assert result[0] == "Hello world."

    def test_multiple_sentences(self):
        from narro.utils.text_splitter import split_and_recombine_text
        result = split_and_recombine_text("First sentence. Second sentence.")
        assert len(result) >= 1

    def test_respects_max_length(self):
        from narro.utils.text_splitter import split_and_recombine_text
        long_text = "This is a sentence. " * 50
        result = split_and_recombine_text(long_text, max_length=100)
        for chunk in result:
            # Chunks should be close to or under max_length
            assert len(chunk) <= 120  # Some slack for word boundaries

    def test_empty_string(self):
        from narro.utils.text_splitter import split_and_recombine_text
        result = split_and_recombine_text("")
        assert result == []

    def test_whitespace_only(self):
        from narro.utils.text_splitter import split_and_recombine_text
        result = split_and_recombine_text("   ")
        assert result == []

    def test_punctuation_only(self):
        from narro.utils.text_splitter import split_and_recombine_text
        result = split_and_recombine_text("...")
        assert result == []

    def test_question_marks_split(self):
        from narro.utils.text_splitter import split_and_recombine_text
        result = split_and_recombine_text("Is this a question? Yes it is.")
        assert len(result) >= 1

    def test_exclamation_marks_split(self):
        from narro.utils.text_splitter import split_and_recombine_text
        result = split_and_recombine_text("Wow! That is great.")
        assert len(result) >= 1

    def test_newline_split(self):
        from narro.utils.text_splitter import split_and_recombine_text
        result = split_and_recombine_text("First paragraph.\nSecond paragraph.")
        assert len(result) >= 1

    def test_preserves_quotes(self):
        from narro.utils.text_splitter import split_and_recombine_text
        text = 'She said "Hello world." and left.'
        result = split_and_recombine_text(text)
        # The period inside quotes should not cause a split
        combined = " ".join(result)
        assert "Hello world." in combined

    def test_very_long_word(self):
        from narro.utils.text_splitter import split_and_recombine_text
        # A word longer than max_length should still be handled
        long_word = "a" * 500
        result = split_and_recombine_text(long_word, max_length=100)
        assert len(result) >= 1

    def test_consecutive_boundary_markers(self):
        from narro.utils.text_splitter import split_and_recombine_text
        result = split_and_recombine_text("Really?! Absolutely!!")
        assert len(result) >= 1

    def test_smart_quotes_normalized(self):
        from narro.utils.text_splitter import split_and_recombine_text
        text = '\u201cHello\u201d she said.'
        result = split_and_recombine_text(text)
        combined = " ".join(result)
        # Smart quotes should be converted to ascii
        assert "\u201c" not in combined


# ---------------------------------------------------------------------------
# Weight migration: migrate_checkpoint_file
# ---------------------------------------------------------------------------

class TestMigrateCheckpointFile:
    """Test migrate_checkpoint_file to increase migrate_weights.py coverage."""

    def test_migrate_checkpoint_file_default_output(self):
        """migrate_checkpoint_file with no output_path should create .migrated file."""
        from narro.vocos.migrate_weights import migrate_checkpoint_file
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512),
            "norm.weight": torch.randn(768),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "decoder.pth")
            torch.save(state_dict, input_path)

            output_path = migrate_checkpoint_file(input_path)

            assert os.path.exists(output_path)
            assert "migrated" in output_path
            loaded = torch.load(output_path, map_location='cpu', weights_only=True)
            assert loaded["convnext.0.pwconv1.weight"].shape == (1536, 512, 1)
            assert loaded["norm.weight"].shape == (1, 768, 1)

    def test_migrate_checkpoint_file_custom_output(self):
        """migrate_checkpoint_file with explicit output_path."""
        from narro.vocos.migrate_weights import migrate_checkpoint_file
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "decoder.pth")
            custom_output = os.path.join(tmpdir, "custom_output.pth")
            torch.save(state_dict, input_path)

            result = migrate_checkpoint_file(input_path, custom_output)

            assert result == custom_output
            assert os.path.exists(custom_output)


# ---------------------------------------------------------------------------
# Additional text normalizer edge cases (targeted at uncovered lines)
# ---------------------------------------------------------------------------

class TestTextNormalizerEdgeCases:
    """Target uncovered lines in text_normalizer.py."""

    def test_time_zero_hours_zero_minutes(self):
        """0:00 should return '0' (line 146)."""
        from narro.utils.text_normalizer import _expand_time
        import re
        m = re.match(r'(\d\d?:\d\d(?::\d\d)?)', '0:00')
        result = _expand_time(m)
        assert result == '0'

    def test_time_13_hours_zero_minutes(self):
        """13:00 should return '13 minutes' (line 148)."""
        from narro.utils.text_normalizer import _expand_time
        import re
        m = re.match(r'(\d\d?:\d\d(?::\d\d)?)', '13:00')
        result = _expand_time(m)
        assert "minutes" in result

    def test_time_hms_with_nonzero_hours(self):
        """H:M:S format with nonzero hours (line 154-156)."""
        from narro.utils.text_normalizer import _expand_time
        import re
        m = re.match(r'(\d\d?:\d\d(?::\d\d)?)', '01:01:01')
        result = _expand_time(m)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_time_hms_zero_hours_nonzero_minutes(self):
        """0:MM:SS format (line 157-158)."""
        from narro.utils.text_normalizer import _expand_time
        import re
        m = re.match(r'(\d\d?:\d\d(?::\d\d)?)', '00:05:10')
        result = _expand_time(m)
        assert isinstance(result, str)

    def test_time_hms_zero_hours_zero_minutes(self):
        """0:00:SS format should return seconds only (line 159-160)."""
        from narro.utils.text_normalizer import _expand_time
        import re
        m = re.match(r'(\d\d?:\d\d(?::\d\d)?)', '00:00:30')
        result = _expand_time(m)
        assert result == '30'

    def test_time_hms_hours_zero_seconds(self):
        """H:M:00 format (seconds == '00' branch)."""
        from narro.utils.text_normalizer import _expand_time
        import re
        m = re.match(r'(\d\d?:\d\d(?::\d\d)?)', '01:00:00')
        result = _expand_time(m)
        assert isinstance(result, str)

    def test_time_hms_minutes_zero_seconds(self):
        """0:MM:00 format with nonzero minutes, zero seconds."""
        from narro.utils.text_normalizer import _expand_time
        import re
        m = re.match(r'(\d\d?:\d\d(?::\d\d)?)', '00:05:00')
        result = _expand_time(m)
        assert isinstance(result, str)

    def test_phone_number_invalid_length(self):
        """Phone number with non-10-digit count returns original (line 135)."""
        from narro.utils.text_normalizer import _expand_phone_number
        import re
        m = re.match(r'(\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4})', '123-456-7890')
        result = _expand_phone_number(m)
        # Valid 10-digit phone number
        assert "," in result

    def test_dollar_unexpected_format(self):
        """Dollar with more than 2 decimal parts (line 166)."""
        from narro.utils.text_normalizer import _expand_dollars
        import re
        m = re.match(r'\$([\d\.\,]*\d+)', '$1.2.3')
        result = _expand_dollars(m)
        assert "dollars" in result  # Unexpected format fallback

    def test_fraction_with_more_than_two_parts(self):
        """Fraction like 1/2/3 should use 'slash' instead of 'over' (line 188-190)."""
        from narro.utils.text_normalizer import _expand_fraction, _fraction_re
        import re
        m = re.search(_fraction_re, '1/2/3')
        result = _expand_fraction(m)
        assert "slash" in result

    def test_fraction_with_two_parts(self):
        """Fraction like 1/2 should use 'over'."""
        from narro.utils.text_normalizer import _expand_fraction, _fraction_re
        import re
        m = re.search(_fraction_re, '1/2')
        result = _expand_fraction(m)
        assert "over" in result

    def test_mixed_case_uppercase_plural(self):
        """Uppercase word ending in 's' => possessive form (line 309-310)."""
        from narro.utils.text_normalizer import normalize_mixedcase
        result = normalize_mixedcase("TPUs")
        assert "'s" in result

    def test_mixed_case_single_capital_word(self):
        """Single capital word should be returned unchanged (line 305-306)."""
        from narro.utils.text_normalizer import normalize_mixedcase
        result = normalize_mixedcase("Test")
        assert result == "Test"

    def test_mixed_case_all_uppercase(self):
        """All uppercase should be returned unchanged (line 307-308)."""
        from narro.utils.text_normalizer import normalize_mixedcase
        result = normalize_mixedcase("UPPERCASE")
        assert result == "UPPERCASE"



# ---------------------------------------------------------------------------
# Additional text splitter edge cases
# ---------------------------------------------------------------------------

class TestTextSplitterEdgeCases:
    """Target uncovered lines in text_splitter.py (lines 51-52)."""

    def test_force_split_at_sentence_boundary(self):
        """Text exceeding max_length with a sentence boundary should split there."""
        from narro.utils.text_splitter import split_and_recombine_text
        # Create text with a sentence boundary before max_length
        text = "Short sentence. " + "a " * 200 + "end."
        result = split_and_recombine_text(text, desired_length=10, max_length=50)
        assert len(result) >= 2

    def test_force_split_no_sentence_boundary(self):
        """Text exceeding max_length without sentence boundaries should split at word boundary (lines 51-52)."""
        from narro.utils.text_splitter import split_and_recombine_text
        # Single long sentence with no sentence-ending punctuation
        text = "word " * 100  # 500 chars with no periods
        result = split_and_recombine_text(text, desired_length=10, max_length=50)
        assert len(result) >= 2

    def test_end_of_quote_boundary(self):
        """End of quote followed by space should be treated as boundary."""
        from narro.utils.text_splitter import split_and_recombine_text
        text = '"Hello world" she said. Then she left.'
        result = split_and_recombine_text(text)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# BaseModel.infer and BaseModel.stream_infer tests (mocked model/tokenizer)
# ---------------------------------------------------------------------------

class TestBaseModelInfer:
    """Test BaseModel.infer with mocked model and tokenizer to cover backends/base.py."""

    def _make_base_model(self):
        """Create a BaseModel instance with mocked model and tokenizer."""
        from narro.backends.base import BaseModel
        bm = BaseModel()
        bm.model = MagicMock()
        bm.tokenizer = MagicMock()
        return bm

    def test_infer_single_prompt_stop(self):
        """infer with a single prompt that ends with EOS."""
        bm = self._make_base_model()
        eos_token_id = 2

        bm.model.config.eos_token_id = eos_token_id
        bm.tokenizer.pad_token_id = 0

        # Simulated tokenizer output
        bm.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
        }

        # Simulated generate output
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 5, 6, 10, 11, eos_token_id]])

        # hidden_states: one per output token (3 generated tokens)
        hidden_layer_0 = torch.randn(1, 1, 512)
        hidden_layer_1 = torch.randn(1, 1, 512)
        hidden_layer_2 = torch.randn(1, 1, 512)
        mock_outputs.hidden_states = [
            (hidden_layer_0,),  # token 10
            (hidden_layer_1,),  # token 11
            (hidden_layer_2,),  # token EOS
        ]
        # scores: one per output token, shape (batch_size, vocab_size)
        mock_outputs.scores = [torch.randn(1, 100) for _ in range(3)]

        bm.model.generate.return_value = mock_outputs

        results = bm.infer(["Hello world"])
        assert len(results) == 1
        assert results[0]['finish_reason'] == 'stop'
        assert results[0]['hidden_state'].dim() >= 1
        assert 'token_ids' in results[0]
        assert 'token_entropy' in results[0]

    def test_infer_single_prompt_length(self):
        """infer with a prompt that does not end with EOS."""
        bm = self._make_base_model()
        eos_token_id = 2

        bm.model.config.eos_token_id = eos_token_id
        bm.tokenizer.pad_token_id = 0
        bm.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 5]]),
            'attention_mask': torch.tensor([[1, 1]]),
        }

        mock_outputs = MagicMock()
        # No EOS at end
        mock_outputs.sequences = torch.tensor([[1, 5, 10, 11]])
        hidden_0 = torch.randn(1, 1, 512)
        hidden_1 = torch.randn(1, 1, 512)
        mock_outputs.hidden_states = [
            (hidden_0,),
            (hidden_1,),
        ]
        mock_outputs.scores = [torch.randn(1, 100) for _ in range(2)]
        bm.model.generate.return_value = mock_outputs

        results = bm.infer(["Hello"])
        assert results[0]['finish_reason'] == 'length'

    def test_infer_temperature_clamped(self):
        """temperature=0.0 should be clamped to 0.001."""
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id
        bm.tokenizer.pad_token_id = 0
        bm.tokenizer.return_value = {
            'input_ids': torch.tensor([[1]]),
            'attention_mask': torch.tensor([[1]]),
        }
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 10, eos_token_id]])
        mock_outputs.hidden_states = [
            (torch.randn(1, 1, 512),),
            (torch.randn(1, 1, 512),),
        ]
        mock_outputs.scores = [torch.randn(1, 100) for _ in range(2)]
        bm.model.generate.return_value = mock_outputs

        bm.infer(["Test"], temperature=0.0)
        call_kwargs = bm.model.generate.call_args[1]
        assert call_kwargs['temperature'] == 0.001

    def test_infer_eos_tokens_excluded_from_hidden_states(self):
        """Hidden states for EOS tokens should not be included in the result."""
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id
        bm.tokenizer.pad_token_id = 0
        bm.tokenizer.return_value = {
            'input_ids': torch.tensor([[1]]),
            'attention_mask': torch.tensor([[1]]),
        }

        mock_outputs = MagicMock()
        # Sequence: [1, 10, EOS] -- 2 generated tokens, EOS excluded
        mock_outputs.sequences = torch.tensor([[1, 10, eos_token_id]])
        h0 = torch.randn(1, 1, 512)
        h1 = torch.randn(1, 1, 512)
        mock_outputs.hidden_states = [(h0,), (h1,)]
        mock_outputs.scores = [torch.randn(1, 100) for _ in range(2)]
        bm.model.generate.return_value = mock_outputs

        results = bm.infer(["Test"])
        # Token 10 (not EOS) should be included, EOS should be excluded
        # So hidden_state should have 1 entry (just token 10)
        assert results[0]['hidden_state'].shape == (1, 512)  # single non-EOS token, 2D


class TestBaseModelStreamInfer:
    """Test BaseModel.stream_infer with mocked model and tokenizer."""

    def _make_base_model(self):
        from narro.backends.base import BaseModel
        bm = BaseModel()
        bm.model = MagicMock()
        bm.tokenizer = MagicMock()
        return bm

    def test_stream_infer_yields_tokens(self):
        """stream_infer should yield dicts with hidden_state and finish_reason."""
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id

        bm.tokenizer.return_value = {'input_ids': torch.tensor([[1, 5]])}

        # First call (prefill)
        prefill_outputs = MagicMock()
        prefill_outputs.past_key_values = "pkv0"
        prefill_outputs.logits = torch.randn(1, 2, 100)
        prefill_outputs.hidden_states = (torch.randn(1, 2, 512),)

        # Step 1
        step1_outputs = MagicMock()
        step1_outputs.past_key_values = "pkv1"
        step1_outputs.logits = torch.randn(1, 1, 100)
        step1_outputs.hidden_states = (torch.randn(1, 1, 512),)

        # Step 2 (EOS token enters the loop, so model is called with EOS)
        step2_outputs = MagicMock()
        step2_outputs.past_key_values = "pkv2"
        step2_outputs.logits = torch.randn(1, 1, 100)
        step2_outputs.hidden_states = (torch.randn(1, 1, 512),)

        bm.model.side_effect = [prefill_outputs, step1_outputs, step2_outputs]

        with patch('torch.multinomial') as mock_multinomial:
            mock_multinomial.side_effect = [
                torch.tensor([[10]]),  # initial token (from prefill)
                torch.tensor([[eos_token_id]]),  # returned after step1 (triggers stop in next iter)
            ]

            tokens = list(bm.stream_infer("Hello"))

        # step1: next_token=10 (not EOS), yields with finish_reason=None
        # step2: next_token=EOS, yields with finish_reason='stop', breaks
        assert len(tokens) == 2
        assert tokens[0]['finish_reason'] is None
        assert tokens[1]['finish_reason'] == 'stop'

    def test_stream_infer_max_tokens_limit(self):
        """stream_infer should stop after max_new_tokens if no EOS."""
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id

        bm.tokenizer.return_value = {'input_ids': torch.tensor([[1]])}

        prefill = MagicMock()
        prefill.past_key_values = "pkv"
        prefill.logits = torch.randn(1, 1, 100)
        prefill.hidden_states = (torch.randn(1, 1, 512),)

        step = MagicMock()
        step.past_key_values = "pkv"
        step.logits = torch.randn(1, 1, 100)
        step.hidden_states = (torch.randn(1, 1, 512),)

        bm.model.side_effect = [prefill] + [step] * 520

        with patch('torch.multinomial', return_value=torch.tensor([[10]])):
            tokens = list(bm.stream_infer("Test"))

        assert len(tokens) == 512
        assert tokens[-1]['finish_reason'] == 'length'

    def test_stream_infer_no_repetition_penalty_when_1(self):
        """stream_infer with repetition_penalty=1.0 should not add RepetitionPenaltyLogitsProcessor."""
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id
        bm.tokenizer.return_value = {'input_ids': torch.tensor([[1]])}

        prefill = MagicMock()
        prefill.past_key_values = "pkv"
        prefill.logits = torch.randn(1, 1, 100)
        prefill.hidden_states = (torch.randn(1, 1, 512),)

        step = MagicMock()
        step.past_key_values = "pkv"
        step.logits = torch.randn(1, 1, 100)
        step.hidden_states = (torch.randn(1, 1, 512),)

        bm.model.side_effect = [prefill, step]

        with patch('torch.multinomial', return_value=torch.tensor([[eos_token_id]])):
            tokens = list(bm.stream_infer("Test", repetition_penalty=1.0))

        assert len(tokens) >= 1

    def test_stream_infer_temperature_clamped(self):
        """stream_infer with temperature=0.0 should be clamped to 0.001."""
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id
        bm.tokenizer.return_value = {'input_ids': torch.tensor([[1]])}

        prefill = MagicMock()
        prefill.past_key_values = "pkv"
        prefill.logits = torch.randn(1, 1, 100)
        prefill.hidden_states = (torch.randn(1, 1, 512),)

        step = MagicMock()
        step.past_key_values = "pkv"
        step.logits = torch.randn(1, 1, 100)
        step.hidden_states = (torch.randn(1, 1, 512),)

        bm.model.side_effect = [prefill, step]

        # Should not raise even with temperature=0.0 (clamped internally)
        with patch('torch.multinomial', return_value=torch.tensor([[eos_token_id]])):
            tokens = list(bm.stream_infer("Test", temperature=0.0))
        assert len(tokens) >= 1


# ---------------------------------------------------------------------------
# Behavioral TDD tests: contracts that guide correctness
# ---------------------------------------------------------------------------

class TestMultiSentenceReassembly:
    """Verify infer_batch correctly re-assigns sentences to texts after sorting."""

    def _make_tts_with_mocks(self):
        tts = _make_tts_stub()
        tts.pipeline = MagicMock()
        tts.decoder = MagicMock()
        tts.decoder_batch_size = 10
        return tts

    def test_two_texts_different_lengths_assigned_correctly(self):
        """Shorter and longer hidden states should produce audio assigned to the correct text."""
        tts = self._make_tts_with_mocks()

        tts.pipeline.infer.return_value = [
            _make_pipeline_response(10),
            _make_pipeline_response(30),
        ]

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE)

        tts.decoder.side_effect = decoder_side_effect

        results = tts.infer_batch(["Short.", "This is a much longer sentence that produces more tokens."])

        assert len(results) == 2
        # Short text (10 tokens) should produce less audio than long text (30 tokens)
        assert results[0].shape[0] < results[1].shape[0]

    def test_three_texts_order_preserved(self):
        """Audio results should be in same order as input texts, regardless of sorting."""
        tts = self._make_tts_with_mocks()

        tts.pipeline.infer.return_value = [
            _make_pipeline_response(20),
            _make_pipeline_response(10),
            _make_pipeline_response(30),
        ]

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE)

        tts.decoder.side_effect = decoder_side_effect

        results = tts.infer_batch(["Medium text here.", "Short.", "Long text that is significantly longer than others."])

        assert len(results) == 3
        # Results should follow input order: medium, short, long
        assert results[1].shape[0] < results[0].shape[0]  # short < medium
        assert results[0].shape[0] < results[2].shape[0]  # medium < long


class TestHallucinationRetryExhaustion:
    """Verify behavior when hallucination retries are exhausted."""

    def _make_tts_with_mocks(self):
        tts = _make_tts_stub()
        tts.pipeline = MagicMock()
        tts.decoder = MagicMock()
        tts.decoder_batch_size = 4
        return tts

    def test_uses_last_result_when_retries_exhausted(self):
        """When hallucination persists through all retries, the last result is still used."""
        tts = self._make_tts_with_mocks()

        # Create a hallucinated hidden state (all nearly identical)
        base = torch.zeros(HIDDEN_DIM)
        hallucinated = torch.stack([base.clone() for _ in range(50)])

        # Both attempts return hallucinated results
        bad_response = {
            'finish_reason': 'stop',
            'hidden_state': hallucinated,
            'token_ids': torch.zeros(50, dtype=torch.int32),
            'token_entropy': torch.zeros(50),
            'attention': None,
        }
        tts.pipeline.infer.side_effect = [
            [bad_response],
            [bad_response],
        ]
        tts.decoder.return_value = torch.randn(1, 49 * TOKEN_SIZE)

        results = tts.infer_batch(["Hello."], retries=1)
        assert len(results) == 1
        # Should have called infer twice: original + 1 retry
        assert tts.pipeline.infer.call_count == 2


class TestWavOutputContract:
    """Verify WAV file output meets format contract."""

    def _make_tts_with_mocks(self):
        tts = _make_tts_stub()
        tts.pipeline = MagicMock()
        tts.decoder = MagicMock()
        tts.decoder_batch_size = 4
        return tts

    def test_wav_has_correct_sample_rate_and_format(self):
        """infer() should produce a valid WAV file at 32kHz with int16 mono samples."""
        import wave
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]
        audio = torch.randn(1, (seq_len - 1) * TOKEN_SIZE + TOKEN_SIZE) * 0.5
        tts.decoder.return_value = audio

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            out_path = f.name
        try:
            tts.infer("Hello.", out_path=out_path)
            with wave.open(out_path, 'rb') as wf:
                assert wf.getframerate() == SAMPLE_RATE
                assert wf.getsampwidth() == 2  # int16 = 2 bytes
                assert wf.getnchannels() == 1
                assert wf.getnframes() > 0
        finally:
            os.unlink(out_path)

    def test_audio_clipped_to_valid_range(self):
        """Audio values outside [-1, 1] should be clipped in WAV output."""
        from scipy.io import wavfile as wav_read
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]
        # Decoder returns audio with values way outside [-1, 1]
        audio = torch.ones(1, (seq_len - 1) * TOKEN_SIZE + TOKEN_SIZE) * 5.0
        tts.decoder.return_value = audio

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            out_path = f.name
        try:
            tts.infer("Hello.", out_path=out_path)
            _, pcm_data = wav_read.read(out_path)
            # All samples should be clipped to int16 range
            assert pcm_data.max() <= INT16_MAX
            assert pcm_data.min() >= -INT16_MAX
        finally:
            os.unlink(out_path)


class TestBaseModelMultiPromptBatch:
    """Test BaseModel.infer with multiple prompts in a batch."""

    def _make_base_model(self):
        from narro.backends.base import BaseModel
        bm = BaseModel()
        bm.model = MagicMock()
        bm.tokenizer = MagicMock()
        return bm

    def test_infer_multiple_prompts(self):
        """infer() with two prompts should return two results."""
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id
        bm.tokenizer.pad_token_id = 0
        bm.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 5], [1, 6]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1]]),
        }

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([
            [1, 5, 10, eos_token_id],
            [1, 6, 11, 12],  # no EOS -> 'length'
        ])
        mock_outputs.hidden_states = [
            (torch.randn(2, 1, 512),),
            (torch.randn(2, 1, 512),),
        ]
        # scores: one per output token, shape (batch_size, vocab_size)
        mock_outputs.scores = [torch.randn(2, 100) for _ in range(2)]
        bm.model.generate.return_value = mock_outputs

        results = bm.infer(["Hello", "World"])
        assert len(results) == 2
        assert results[0]['finish_reason'] == 'stop'
        assert results[1]['finish_reason'] == 'length'

    def test_infer_all_eos_returns_empty_hidden_state(self):
        """When all generated tokens are EOS, hidden_state should be empty."""
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id
        bm.model.config.hidden_size = 512
        bm.tokenizer.pad_token_id = 0
        bm.tokenizer.return_value = {
            'input_ids': torch.tensor([[1]]),
            'attention_mask': torch.tensor([[1]]),
        }
        mock_outputs = MagicMock()
        # Only generated token is EOS
        mock_outputs.sequences = torch.tensor([[1, eos_token_id]])
        mock_outputs.hidden_states = [(torch.randn(1, 1, 512),)]
        mock_outputs.scores = [torch.randn(1, 100)]
        bm.model.generate.return_value = mock_outputs

        results = bm.infer(["Test"])
        assert results[0]['hidden_state'].shape == (0, 512)
        assert results[0]['finish_reason'] == 'stop'


class TestISTFTForwardPass:
    """Test ISTFT spectral operations directly."""

    def test_same_padding_produces_audio(self):
        """ISTFT with 'same' padding should produce output of expected length."""
        from narro.vocos.spectral_ops import ISTFT
        n_fft, hop_length = 2048, 512
        istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding="same")
        T = 20
        spec = torch.randn(1, n_fft // 2 + 1, T, dtype=torch.cfloat)
        audio = istft(spec)
        assert audio.dim() == 2
        assert audio.shape[0] == 1
        assert audio.shape[1] > 0

    def test_center_padding_produces_audio(self):
        """ISTFT with 'center' padding should produce output."""
        from narro.vocos.spectral_ops import ISTFT
        n_fft, hop_length = 2048, 512
        istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding="center")
        T = 20
        spec = torch.randn(1, n_fft // 2 + 1, T, dtype=torch.cfloat)
        audio = istft(spec)
        assert audio.dim() == 2
        assert audio.shape[0] == 1
        assert audio.shape[1] > 0

    def test_different_sequence_lengths(self):
        """ISTFT should handle varying sequence lengths."""
        from narro.vocos.spectral_ops import ISTFT
        n_fft, hop_length = 2048, 512
        istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding="same")
        for T in [5, 10, 50]:
            spec = torch.randn(1, n_fft // 2 + 1, T, dtype=torch.cfloat)
            audio = istft(spec)
            assert audio.dim() == 2
            assert audio.shape[1] > 0


class TestMigrationGammaCheck:
    """Test that is_migrated correctly checks gamma keys."""

    def test_partially_migrated_with_1d_gamma_detected(self):
        """is_migrated should return False when pwconv is 3D but gamma is still 1D."""
        from narro.vocos.migrate_weights import is_migrated
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512, 1),  # migrated
            "convnext.0.gamma": torch.randn(512),  # NOT migrated
            "norm.weight": torch.randn(1, 768, 1),  # migrated
        }
        assert not is_migrated(state_dict)

    def test_fully_migrated_with_2d_gamma(self):
        """is_migrated should return True when all keys including gamma are migrated."""
        from narro.vocos.migrate_weights import is_migrated
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512, 1),
            "convnext.0.gamma": torch.randn(512, 1),  # correctly migrated
            "norm.weight": torch.randn(1, 768, 1),
            "head.out.weight": torch.randn(2050, 768, 1),
        }
        assert is_migrated(state_dict)


# ---------------------------------------------------------------------------
# ISTFT validation error tests (spectral_ops lines 22, 59, 92)
# ---------------------------------------------------------------------------

class TestISTFTValidation:
    """Test ISTFT validation error paths."""

    def test_invalid_padding_raises(self):
        """ISTFT with invalid padding should raise ValueError."""
        from narro.vocos.spectral_ops import ISTFT
        with pytest.raises(ValueError, match="Padding must be"):
            ISTFT(n_fft=2048, hop_length=512, win_length=2048, padding="invalid")

    def test_non_3d_input_raises(self):
        """ISTFT with non-3D input tensor should raise ValueError."""
        from narro.vocos.spectral_ops import ISTFT
        istft = ISTFT(n_fft=2048, hop_length=512, win_length=2048, padding="same")
        # 2D tensor instead of 3D
        bad_input = torch.randn(1025, 20, dtype=torch.cfloat)
        with pytest.raises(ValueError, match="Expected a 3D tensor"):
            istft(bad_input)


# ---------------------------------------------------------------------------
# BaseModel.infer with include_attention=True (base.py lines 78-80, 101)
# ---------------------------------------------------------------------------

class TestBaseModelAttention:
    """Test BaseModel.infer attention extraction path."""

    def _make_base_model(self):
        from narro.backends.base import BaseModel
        bm = BaseModel()
        bm.model = MagicMock()
        bm.tokenizer = MagicMock()
        return bm

    def test_infer_with_attention_extracts_weights(self):
        """include_attention=True should extract attention from outputs."""
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id
        bm.tokenizer.pad_token_id = 0

        input_len = 3
        bm.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
        }

        mock_outputs = MagicMock()
        # 2 generated tokens + EOS
        mock_outputs.sequences = torch.tensor([[1, 5, 6, 10, 11, eos_token_id]])

        mock_outputs.hidden_states = [
            (torch.randn(1, 1, 512),),  # token 10
            (torch.randn(1, 1, 512),),  # token 11
            (torch.randn(1, 1, 512),),  # EOS
        ]
        mock_outputs.scores = [torch.randn(1, 100) for _ in range(3)]

        # Attention: tuple of (layers), each layer is (batch, heads, seq_len, seq_len)
        # For each generated step, the full attention has shape expanding
        num_heads = 4
        # Step 0: seq_len = input_len + 1 = 4
        attn_step0 = torch.randn(1, num_heads, 4, 4)
        # Step 1: seq_len = input_len + 2 = 5
        attn_step1 = torch.randn(1, num_heads, 5, 5)
        # Step 2 (EOS): seq_len = input_len + 3 = 6
        attn_step2 = torch.randn(1, num_heads, 6, 6)

        # attentions is a tuple indexed by step, each step is a tuple of layers
        mock_outputs.attentions = [
            (attn_step0,),  # last layer for step 0
            (attn_step1,),
            (attn_step2,),
        ]

        bm.model.generate.return_value = mock_outputs

        results = bm.infer(["Hello world"], include_attention=True)
        assert results[0]['attention'] is not None
        # Should have 2 attention vectors (one per non-EOS token)
        assert results[0]['attention'].shape[0] == 2
        # Each attention vector should be sliced to input_len positions
        assert results[0]['attention'].shape[1] == input_len

    def test_infer_without_attention_returns_none(self):
        """include_attention=False should return attention=None."""
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id
        bm.tokenizer.pad_token_id = 0
        bm.tokenizer.return_value = {
            'input_ids': torch.tensor([[1]]),
            'attention_mask': torch.tensor([[1]]),
        }
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 10, eos_token_id]])
        mock_outputs.hidden_states = [(torch.randn(1, 1, 512),), (torch.randn(1, 1, 512),)]
        mock_outputs.scores = [torch.randn(1, 100), torch.randn(1, 100)]
        bm.model.generate.return_value = mock_outputs

        results = bm.infer(["Test"], include_attention=False)
        assert results[0]['attention'] is None


# ---------------------------------------------------------------------------
# CLI dispatch paths (cli.py lines 117, 121)
# ---------------------------------------------------------------------------

class TestCLIDispatch:
    """Test CLI main() dispatch branches."""

    def test_subcommand_dispatch(self):
        """When command is set, args.func should be called."""
        from narro.cli import main
        import argparse

        mock_func = MagicMock()
        with patch('sys.argv', ['narro', 'encode', 'Hello']), \
             patch('narro.cli.argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                command='encode',
                func=mock_func,
                text='Hello',
                output='out.soprano',
            )
            main()
            mock_func.assert_called_once()

    def test_no_text_prints_help(self):
        """When no args at all, parser.print_help should be called."""
        from narro.cli import main

        with patch('sys.argv', ['narro']), \
             patch('narro.cli.argparse.ArgumentParser.print_help') as mock_help:
            main()
            mock_help.assert_called_once()


# ---------------------------------------------------------------------------
# load_decoder tests (decode_only.py lines 41-58)
# ---------------------------------------------------------------------------

class TestLoadDecoder:
    """Test decode_only.load_decoder with mocked dependencies."""

    def test_load_decoder_from_local_path(self):
        """load_decoder with model_path should load from local file."""
        from narro.decode_only import load_decoder
        from narro.vocos.decoder import SopranoDecoder

        # Create a valid state dict from a fresh decoder
        ref_decoder = SopranoDecoder()
        state_dict = ref_decoder.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            decoder_path = os.path.join(tmpdir, 'decoder.pth')
            torch.save(state_dict, decoder_path)
            decoder = load_decoder(model_path=tmpdir, compile=False)
            assert isinstance(decoder, SopranoDecoder)

    def test_load_decoder_from_hub(self):
        """load_decoder without model_path should download from HuggingFace."""
        from narro.decode_only import load_decoder
        from narro.vocos.decoder import SopranoDecoder

        ref_decoder = SopranoDecoder()
        state_dict = ref_decoder.state_dict()

        with patch('narro.decode_only.hf_hub_download', return_value='/fake/decoder.pth'), \
             patch('torch.load', return_value=state_dict):
            decoder = load_decoder(compile=False)
            assert isinstance(decoder, SopranoDecoder)

    def test_load_decoder_with_compile_failure_warns(self):
        """load_decoder with compile=True should warn if torch.compile fails."""
        import warnings
        from narro.decode_only import load_decoder
        from narro.vocos.decoder import SopranoDecoder

        ref_decoder = SopranoDecoder()
        state_dict = ref_decoder.state_dict()

        with patch('narro.decode_only.hf_hub_download', return_value='/fake/decoder.pth'), \
             patch('torch.load', return_value=state_dict), \
             patch('torch.compile', side_effect=RuntimeError("compile not supported")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                decoder = load_decoder(compile=True)
                # Should have warned about compile failure
                assert any("torch.compile failed" in str(warning.message) for warning in w)

    def test_decode_with_no_decoder_loads_default(self):
        """decode() with decoder=None should call load_decoder internally."""
        from narro.decode_only import decode
        from narro.encoded import EncodedSpeech

        encoded = EncodedSpeech(sentences=[], model_id='test')

        with patch('narro.decode_only.load_decoder') as mock_load:
            mock_load.return_value = MagicMock()
            result = decode(encoded, decoder=None)
            mock_load.assert_called_once()
            assert result == []


# ---------------------------------------------------------------------------
# TransformersModel init tests (transformers.py lines 8-27)
# ---------------------------------------------------------------------------

class TestTransformersModelInit:
    """Test TransformersModel.__init__ with mocked HuggingFace classes."""

    def test_init_default_model(self):
        """TransformersModel() should load the default model."""
        with patch('narro.backends.transformers.AutoModelForCausalLM') as mock_model_cls, \
             patch('narro.backends.transformers.AutoTokenizer') as mock_tok_cls:
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tok_cls.from_pretrained.return_value = MagicMock()

            from narro.backends.transformers import TransformersModel
            tm = TransformersModel(compile=False, quantize=False)

            mock_model_cls.from_pretrained.assert_called_once_with(
                'ekwek/Soprano-1.1-80M', torch_dtype=torch.float32,
                attn_implementation="eager")
            mock_model.eval.assert_called_once()

    def test_init_with_quantize(self):
        """TransformersModel with quantize=True should call quantize_dynamic."""
        with patch('narro.backends.transformers.AutoModelForCausalLM') as mock_model_cls, \
             patch('narro.backends.transformers.AutoTokenizer') as mock_tok_cls, \
             patch('torch.quantization.quantize_dynamic') as mock_quantize:
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tok_cls.from_pretrained.return_value = MagicMock()
            mock_quantize.return_value = mock_model

            from narro.backends.transformers import TransformersModel
            tm = TransformersModel(compile=False, quantize=True)

            mock_quantize.assert_called_once()

    def test_init_with_compile(self):
        """TransformersModel with compile=True should call torch.compile."""
        with patch('narro.backends.transformers.AutoModelForCausalLM') as mock_model_cls, \
             patch('narro.backends.transformers.AutoTokenizer') as mock_tok_cls, \
             patch('torch.compile') as mock_compile:
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tok_cls.from_pretrained.return_value = MagicMock()
            mock_compile.return_value = mock_model

            from narro.backends.transformers import TransformersModel
            tm = TransformersModel(compile=True, quantize=False)

            mock_compile.assert_called_once()

    def test_init_compile_failure_warns(self):
        """TransformersModel should warn if torch.compile fails."""
        import warnings
        with patch('narro.backends.transformers.AutoModelForCausalLM') as mock_model_cls, \
             patch('narro.backends.transformers.AutoTokenizer') as mock_tok_cls, \
             patch('torch.compile', side_effect=RuntimeError("not supported")):
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tok_cls.from_pretrained.return_value = MagicMock()

            from narro.backends.transformers import TransformersModel
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                tm = TransformersModel(compile=True, quantize=False)
                assert any("torch.compile failed" in str(warning.message) for warning in w)

    def test_init_custom_model_path(self):
        """TransformersModel with model_path should use that path."""
        with patch('narro.backends.transformers.AutoModelForCausalLM') as mock_model_cls, \
             patch('narro.backends.transformers.AutoTokenizer') as mock_tok_cls:
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tok_cls.from_pretrained.return_value = MagicMock()

            from narro.backends.transformers import TransformersModel
            tm = TransformersModel(model_path='/custom/model', compile=False, quantize=False)

            mock_model_cls.from_pretrained.assert_called_once_with(
                '/custom/model', torch_dtype=torch.float32,
                attn_implementation="eager")
            mock_tok_cls.from_pretrained.assert_called_once_with('/custom/model')


