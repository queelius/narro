"""Tests for the encode/decode API in narro/tts.py and narro/decode_only.py."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from narro.tts import (
    Narro,
    SAMPLE_RATE,
    INT16_MAX,
    TOKEN_SIZE,
    HIDDEN_DIM,
)
from narro.encoded import EncodedSpeech, SentenceEncoding, save, load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tts_stub():
    """Create a Narro stub without triggering __init__."""
    obj = object.__new__(Narro)
    obj.model_id = 'test/model'
    return obj


def _make_hidden_state(seq_len, dim=HIDDEN_DIM):
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


def _make_encoded_speech(token_lengths, num_texts=None):
    """Create an EncodedSpeech with given per-sentence token lengths."""
    if num_texts is None:
        num_texts = len(token_lengths)
    sentences = []
    for i, length in enumerate(token_lengths):
        text_idx = min(i, num_texts - 1)
        sentences.append(SentenceEncoding(
            hidden_states=np.random.randn(length, HIDDEN_DIM).astype(np.float32),
            token_ids=np.random.randint(0, 8192, size=length, dtype=np.int32),
            token_entropy=np.random.rand(length).astype(np.float32),
            finish_reason='stop',
            text=f"sentence {i}",
            text_index=text_idx,
            sentence_index=0,
        ))
    return EncodedSpeech(sentences=sentences, model_id='test/model')


# ---------------------------------------------------------------------------
# Encode tests
# ---------------------------------------------------------------------------

class TestEncode:

    def _make_tts_with_mocks(self):
        tts = _make_tts_stub()
        tts.pipeline = MagicMock()
        tts.decoder = MagicMock()
        tts.decoder_batch_size = 4
        return tts

    def test_encode_returns_encoded_speech(self):
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]

        encoded = tts.encode("Hello world.")
        assert isinstance(encoded, EncodedSpeech)
        assert len(encoded.sentences) >= 1

    def test_encode_includes_entropy(self):
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]

        encoded = tts.encode("Hello world.")
        s = encoded.sentences[0]
        assert s.token_entropy is not None
        assert len(s.token_entropy) == seq_len

    def test_encode_includes_token_ids(self):
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]

        encoded = tts.encode("Hello world.")
        s = encoded.sentences[0]
        assert s.token_ids is not None
        assert len(s.token_ids) == seq_len

    def test_encode_preserves_text(self):
        tts = self._make_tts_with_mocks()
        tts.pipeline.infer.return_value = [_make_pipeline_response(20)]

        encoded = tts.encode("Hello world.")
        s = encoded.sentences[0]
        # Text is the cleaned/split version, not the wrapped prompt
        assert "[STOP]" not in s.text
        assert "[START]" not in s.text

    def test_encode_stores_model_id(self):
        tts = self._make_tts_with_mocks()
        tts.model_id = 'ekwek/Soprano-1.1-80M'
        tts.pipeline.infer.return_value = [_make_pipeline_response(20)]

        encoded = tts.encode("Hello world.")
        assert encoded.model_id == 'ekwek/Soprano-1.1-80M'

    def test_encode_stores_generation_params(self):
        tts = self._make_tts_with_mocks()
        tts.pipeline.infer.return_value = [_make_pipeline_response(20)]

        encoded = tts.encode("Hello world.", top_p=0.8, temperature=0.5,
                             repetition_penalty=1.5)
        assert encoded.top_p == 0.8
        assert encoded.temperature == 0.5
        assert encoded.repetition_penalty == 1.5

    def test_encode_batch_multiple_texts(self):
        tts = self._make_tts_with_mocks()
        tts.pipeline.infer.return_value = [
            _make_pipeline_response(20),
            _make_pipeline_response(30),
        ]

        encoded = tts.encode_batch(["First text.", "Second text."])
        text_indices = set(s.text_index for s in encoded.sentences)
        assert 0 in text_indices
        assert 1 in text_indices

    def test_encode_hallucination_retry(self):
        tts = self._make_tts_with_mocks()
        seq_len = 20

        # First call: hallucination
        base = torch.zeros(HIDDEN_DIM)
        hallucinated = torch.stack([base.clone() for _ in range(50)])
        bad_response = {
            'finish_reason': 'stop',
            'hidden_state': hallucinated,
            'token_ids': torch.zeros(50, dtype=torch.int32),
            'token_entropy': torch.zeros(50),
            'attention': None,
        }
        good_response = _make_pipeline_response(seq_len)

        tts.pipeline.infer.side_effect = [[bad_response], [good_response]]

        encoded = tts.encode("Hello world.", retries=1)
        assert tts.pipeline.infer.call_count == 2

    def test_encode_attention_opt_in(self):
        tts = self._make_tts_with_mocks()
        seq_len = 20
        response = _make_pipeline_response(seq_len)
        response['attention'] = torch.randn(seq_len, 10)  # 10 input tokens
        tts.pipeline.infer.return_value = [response]

        encoded = tts.encode("Hello world.", include_attention=True)
        s = encoded.sentences[0]
        assert s.attention_weights is not None
        assert s.attention_weights.shape == (seq_len, 10)

    def test_encode_no_attention_by_default(self):
        tts = self._make_tts_with_mocks()
        tts.pipeline.infer.return_value = [_make_pipeline_response(20)]

        encoded = tts.encode("Hello world.")
        s = encoded.sentences[0]
        assert s.attention_weights is None


# ---------------------------------------------------------------------------
# Decode tests
# ---------------------------------------------------------------------------

class TestDecode:

    def _make_tts_with_mocks(self):
        tts = _make_tts_stub()
        tts.pipeline = MagicMock()
        tts.decoder = MagicMock()
        tts.decoder_batch_size = 4
        return tts

    def test_decode_returns_audio_tensors(self):
        tts = self._make_tts_with_mocks()
        encoded = _make_encoded_speech([20], num_texts=1)

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE)

        tts.decoder.side_effect = decoder_side_effect

        audio = tts.decode(encoded)
        assert len(audio) == 1
        assert isinstance(audio[0], torch.Tensor)

    def test_decode_audio_length_matches_tokens(self):
        tts = self._make_tts_with_mocks()
        seq_len = 20
        encoded = _make_encoded_speech([seq_len], num_texts=1)

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE)

        tts.decoder.side_effect = decoder_side_effect

        audio = tts.decode(encoded)
        expected_samples = (seq_len - 1) * TOKEN_SIZE
        assert audio[0].shape[0] == expected_samples

    def test_decode_batch_ordering_preserved(self):
        tts = self._make_tts_with_mocks()
        tts.decoder_batch_size = 10

        # Short and long sentences for different texts
        sentences = [
            SentenceEncoding(
                hidden_states=np.random.randn(10, HIDDEN_DIM).astype(np.float32),
                token_ids=np.zeros(10, dtype=np.int32),
                token_entropy=np.zeros(10, dtype=np.float32),
                finish_reason='stop', text="short", text_index=0, sentence_index=0,
            ),
            SentenceEncoding(
                hidden_states=np.random.randn(30, HIDDEN_DIM).astype(np.float32),
                token_ids=np.zeros(30, dtype=np.int32),
                token_entropy=np.zeros(30, dtype=np.float32),
                finish_reason='stop', text="long", text_index=1, sentence_index=0,
            ),
        ]
        encoded = EncodedSpeech(sentences=sentences, model_id='test/model')

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE)

        tts.decoder.side_effect = decoder_side_effect

        audio = tts.decode(encoded)
        assert len(audio) == 2
        # Short (10 tokens) should produce less audio than long (30 tokens)
        assert audio[0].shape[0] < audio[1].shape[0]

    def test_decode_multi_sentence_per_text(self):
        tts = self._make_tts_with_mocks()
        tts.decoder_batch_size = 10

        sentences = [
            SentenceEncoding(
                hidden_states=np.random.randn(15, HIDDEN_DIM).astype(np.float32),
                token_ids=np.zeros(15, dtype=np.int32),
                token_entropy=np.zeros(15, dtype=np.float32),
                finish_reason='stop', text="first", text_index=0, sentence_index=0,
            ),
            SentenceEncoding(
                hidden_states=np.random.randn(10, HIDDEN_DIM).astype(np.float32),
                token_ids=np.zeros(10, dtype=np.int32),
                token_entropy=np.zeros(10, dtype=np.float32),
                finish_reason='stop', text="second", text_index=0, sentence_index=1,
            ),
        ]
        encoded = EncodedSpeech(sentences=sentences, model_id='test/model')

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE)

        tts.decoder.side_effect = decoder_side_effect

        audio = tts.decode(encoded)
        assert len(audio) == 1  # Both sentences belong to text_index=0
        # Total audio = (15-1)*TOKEN_SIZE + (10-1)*TOKEN_SIZE
        expected = (15 - 1) * TOKEN_SIZE + (10 - 1) * TOKEN_SIZE
        assert audio[0].shape[0] == expected

    def test_decode_zero_token_sentence_skipped(self):
        """A 0-token sentence should be skipped — never sent to the decoder."""
        tts = self._make_tts_with_mocks()
        tts.decoder_batch_size = 10

        sentences = [
            SentenceEncoding(
                hidden_states=np.zeros((0, HIDDEN_DIM), dtype=np.float32),
                token_ids=np.zeros(0, dtype=np.int32),
                token_entropy=np.zeros(0, dtype=np.float32),
                finish_reason='stop', text="empty", text_index=0, sentence_index=0,
            ),
            SentenceEncoding(
                hidden_states=np.random.randn(20, HIDDEN_DIM).astype(np.float32),
                token_ids=np.zeros(20, dtype=np.int32),
                token_entropy=np.zeros(20, dtype=np.float32),
                finish_reason='stop', text="real", text_index=0, sentence_index=1,
            ),
        ]
        encoded = EncodedSpeech(sentences=sentences, model_id='test/model')

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            # Real decoder would crash if T=0 (interpolate to size -3)
            assert T > 0, "Decoder should never receive 0-length hidden states"
            return torch.randn(B, T * TOKEN_SIZE)

        tts.decoder.side_effect = decoder_side_effect

        audio = tts.decode(encoded)
        assert len(audio) == 1
        # Only the 20-token sentence contributes audio
        assert audio[0].shape[0] == (20 - 1) * TOKEN_SIZE

    def test_decode_all_zero_token_sentences(self):
        """All 0-token sentences should produce empty audio without calling decoder."""
        tts = self._make_tts_with_mocks()
        tts.decoder_batch_size = 10

        sentences = [
            SentenceEncoding(
                hidden_states=np.zeros((0, HIDDEN_DIM), dtype=np.float32),
                token_ids=np.zeros(0, dtype=np.int32),
                token_entropy=np.zeros(0, dtype=np.float32),
                finish_reason='stop', text="empty", text_index=0, sentence_index=0,
            ),
        ]
        encoded = EncodedSpeech(sentences=sentences, model_id='test/model')

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            assert T > 0, "Decoder should never receive 0-length hidden states"
            return torch.randn(B, T * TOKEN_SIZE)

        tts.decoder.side_effect = decoder_side_effect

        audio = tts.decode(encoded)
        assert len(audio) == 1
        assert audio[0].shape[0] == 0  # empty audio
        tts.decoder.assert_not_called()

    def test_decode_single_token_sentence(self):
        """A 1-token sentence produces 0 audio samples (trim = 1*TOKEN_SIZE - TOKEN_SIZE = 0)."""
        tts = self._make_tts_with_mocks()
        tts.decoder_batch_size = 10

        sentences = [
            SentenceEncoding(
                hidden_states=np.random.randn(1, HIDDEN_DIM).astype(np.float32),
                token_ids=np.zeros(1, dtype=np.int32),
                token_entropy=np.zeros(1, dtype=np.float32),
                finish_reason='stop', text="one token", text_index=0, sentence_index=0,
            ),
            SentenceEncoding(
                hidden_states=np.random.randn(10, HIDDEN_DIM).astype(np.float32),
                token_ids=np.zeros(10, dtype=np.int32),
                token_entropy=np.zeros(10, dtype=np.float32),
                finish_reason='stop', text="real", text_index=0, sentence_index=1,
            ),
        ]
        encoded = EncodedSpeech(sentences=sentences, model_id='test/model')

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE)

        tts.decoder.side_effect = decoder_side_effect

        audio = tts.decode(encoded)
        assert len(audio) == 1
        # 1-token -> trim=0 -> 0 samples; 10-token -> 9*TOKEN_SIZE
        assert audio[0].shape[0] == (10 - 1) * TOKEN_SIZE

    def test_decode_empty_encoded_speech(self):
        """Decoding an EncodedSpeech with no sentences should return empty list."""
        tts = self._make_tts_with_mocks()
        encoded = EncodedSpeech(sentences=[], model_id='test/model')
        audio = tts.decode(encoded)
        assert audio == []


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------

class TestRoundtrip:

    def _make_tts_with_mocks(self):
        tts = _make_tts_stub()
        tts.pipeline = MagicMock()
        tts.decoder = MagicMock()
        tts.decoder_batch_size = 4
        return tts

    def test_encode_decode_produces_audio(self):
        """encode + decode should produce audio tensors."""
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE)

        tts.decoder.side_effect = decoder_side_effect

        encoded = tts.encode("Hello world.")
        audio = tts.decode(encoded)
        assert len(audio) >= 1
        assert all(isinstance(a, torch.Tensor) for a in audio)

    def test_infer_uses_encode_decode(self):
        """infer() should internally use encode + decode."""
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE)

        tts.decoder.side_effect = decoder_side_effect

        result = tts.infer("Hello world.")
        assert isinstance(result, torch.Tensor)

    def test_encode_save_load_decode(self):
        """Full pipeline: encode -> save -> load -> decode."""
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]

        encoded = tts.encode("Hello world.")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.narro")
            save(encoded, path)
            loaded = load(path)

            def decoder_side_effect(batch):
                B, C, T = batch.shape
                return torch.randn(B, T * TOKEN_SIZE)

            tts.decoder.side_effect = decoder_side_effect

            audio = tts.decode(loaded)
            assert len(audio) >= 1
            assert all(isinstance(a, torch.Tensor) for a in audio)

    def test_decode_to_wav(self):
        """decode_to_wav should write a valid WAV file."""
        tts = self._make_tts_with_mocks()
        seq_len = 20
        tts.pipeline.infer.return_value = [_make_pipeline_response(seq_len)]

        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE) * 0.5

        tts.decoder.side_effect = decoder_side_effect

        encoded = tts.encode("Hello world.")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            out_path = f.name

        try:
            tts.decode_to_wav(encoded, out_path)
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0
        finally:
            os.unlink(out_path)


# ---------------------------------------------------------------------------
# Decode-only module tests
# ---------------------------------------------------------------------------

class TestDecodeOnly:

    def test_decode_function_with_mock_decoder(self):
        """decode_only.decode() should work with a pre-loaded decoder."""
        from narro.decode_only import decode

        encoded = _make_encoded_speech([20], num_texts=1)

        mock_decoder = MagicMock()
        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE)
        mock_decoder.side_effect = decoder_side_effect

        audio = decode(encoded, decoder=mock_decoder)
        assert len(audio) == 1
        assert isinstance(audio[0], torch.Tensor)

    def test_decode_to_wav_with_mock_decoder(self):
        """decode_only.decode_to_wav() should write a WAV file."""
        from narro.decode_only import decode_to_wav

        encoded = _make_encoded_speech([20], num_texts=1)

        mock_decoder = MagicMock()
        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE) * 0.5
        mock_decoder.side_effect = decoder_side_effect

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            out_path = f.name

        try:
            decode_to_wav(encoded, out_path, decoder=mock_decoder)
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0
        finally:
            os.unlink(out_path)

    def test_decode_multiple_texts(self):
        """decode_only.decode() should handle multiple texts."""
        from narro.decode_only import decode

        sentences = [
            SentenceEncoding(
                hidden_states=np.random.randn(15, HIDDEN_DIM).astype(np.float32),
                token_ids=np.zeros(15, dtype=np.int32),
                token_entropy=np.zeros(15, dtype=np.float32),
                finish_reason='stop', text="first", text_index=0, sentence_index=0,
            ),
            SentenceEncoding(
                hidden_states=np.random.randn(20, HIDDEN_DIM).astype(np.float32),
                token_ids=np.zeros(20, dtype=np.int32),
                token_entropy=np.zeros(20, dtype=np.float32),
                finish_reason='stop', text="second", text_index=1, sentence_index=0,
            ),
        ]
        encoded = EncodedSpeech(sentences=sentences, model_id='test/model')

        mock_decoder = MagicMock()
        def decoder_side_effect(batch):
            B, C, T = batch.shape
            return torch.randn(B, T * TOKEN_SIZE)
        mock_decoder.side_effect = decoder_side_effect

        audio = decode(encoded, decoder=mock_decoder)
        assert len(audio) == 2


# ---------------------------------------------------------------------------
# BaseModel enriched output tests
# ---------------------------------------------------------------------------

class TestBaseModelEnrichedOutput:

    def _make_base_model(self):
        from narro.backends.base import BaseModel
        bm = BaseModel()
        bm.model = MagicMock()
        bm.tokenizer = MagicMock()
        return bm

    def test_infer_returns_token_ids(self):
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id
        bm.model.config.hidden_size = 512
        bm.tokenizer.pad_token_id = 0
        bm.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
        }

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 5, 6, 10, 11, eos_token_id]])
        mock_outputs.hidden_states = [
            (torch.randn(1, 1, 512),),
            (torch.randn(1, 1, 512),),
            (torch.randn(1, 1, 512),),
        ]
        # Scores: one per output token
        mock_outputs.scores = [
            torch.randn(1, 100),
            torch.randn(1, 100),
            torch.randn(1, 100),
        ]
        bm.model.generate.return_value = mock_outputs

        results = bm.infer(["Hello world"])
        assert 'token_ids' in results[0]
        assert 'token_entropy' in results[0]
        # 2 non-EOS tokens: 10, 11
        assert results[0]['token_ids'].shape == (2,)
        assert results[0]['token_entropy'].shape == (2,)

    def test_infer_entropy_is_nonnegative(self):
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
        mock_outputs.sequences = torch.tensor([[1, 10, eos_token_id]])
        mock_outputs.hidden_states = [
            (torch.randn(1, 1, 512),),
            (torch.randn(1, 1, 512),),
        ]
        mock_outputs.scores = [
            torch.randn(1, 100),
            torch.randn(1, 100),
        ]
        bm.model.generate.return_value = mock_outputs

        results = bm.infer(["Test"])
        # Entropy should be non-negative
        assert (results[0]['token_entropy'] >= 0).all()

    def test_stream_infer_returns_enriched_tokens(self):
        bm = self._make_base_model()
        eos_token_id = 2
        bm.model.config.eos_token_id = eos_token_id

        bm.tokenizer.return_value = {'input_ids': torch.tensor([[1, 5]])}

        prefill = MagicMock()
        prefill.past_key_values = "pkv0"
        prefill.logits = torch.randn(1, 2, 100)
        prefill.hidden_states = (torch.randn(1, 2, 512),)

        step1 = MagicMock()
        step1.past_key_values = "pkv1"
        step1.logits = torch.randn(1, 1, 100)
        step1.hidden_states = (torch.randn(1, 1, 512),)

        step2 = MagicMock()
        step2.past_key_values = "pkv2"
        step2.logits = torch.randn(1, 1, 100)
        step2.hidden_states = (torch.randn(1, 1, 512),)

        bm.model.side_effect = [prefill, step1, step2]

        with patch('torch.multinomial') as mock_multinomial:
            mock_multinomial.side_effect = [
                torch.tensor([[10]]),
                torch.tensor([[eos_token_id]]),
            ]
            tokens = list(bm.stream_infer("Hello"))

        assert len(tokens) == 2
        assert 'token_id' in tokens[0]
        assert 'token_entropy' in tokens[0]


# ---------------------------------------------------------------------------
# CLI encode/decode tests
# ---------------------------------------------------------------------------

class TestCLISubcommands:

    def test_encode_subcommand(self):
        """encode subcommand should call tts.encode and save."""
        from narro.cli import cmd_encode
        import argparse

        args = argparse.Namespace(
            text='Hello world',
            output='/tmp/test.narro',
            model_path=None,
            no_compile=True,
            quantize=False,
            num_threads=None,
            include_attention=False,
        )

        with patch('narro.Narro') as mock_tts_cls, \
             patch('narro.encoded.save') as mock_save:
            mock_tts = MagicMock()
            mock_tts_cls.return_value = mock_tts
            mock_encoded = MagicMock()
            mock_encoded.total_tokens = 50
            mock_encoded.estimated_duration = 3.2
            mock_tts.encode.return_value = mock_encoded

            cmd_encode(args)

            mock_tts.encode.assert_called_once_with('Hello world', include_attention=False)
            mock_save.assert_called_once_with(mock_encoded, '/tmp/test.narro')

    def test_decode_subcommand(self):
        """decode subcommand should call load and decode_to_wav."""
        from narro.cli import cmd_decode
        import argparse

        args = argparse.Namespace(
            input='/tmp/test.narro',
            output='/tmp/test.wav',
            model_path=None,
            no_compile=True,
            quantize=False,
            num_threads=None,
            decoder_batch_size=4,
        )

        with patch('narro.encoded.load') as mock_load, \
             patch('narro.decode_only.load_decoder') as mock_load_decoder, \
             patch('narro.decode_only.decode_to_wav') as mock_decode_wav:
            mock_encoded = MagicMock()
            mock_load.return_value = mock_encoded
            mock_decoder = MagicMock()
            mock_load_decoder.return_value = mock_decoder

            cmd_decode(args)

            mock_load.assert_called_once_with('/tmp/test.narro')
            mock_decode_wav.assert_called_once()
