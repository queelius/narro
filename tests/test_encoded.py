"""Tests for soprano/encoded.py — IR dataclasses and serialization."""

import os
import tempfile

import numpy as np
import pytest

from soprano.encoded import (
    SentenceEncoding,
    EncodedSpeech,
    FORMAT_VERSION,
    save,
    load,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sentence(seq_len=20, text_index=0, sentence_index=0, text="hello world",
                   finish_reason='stop', attention=None, hidden_dim=512):
    """Create a SentenceEncoding with random data."""
    return SentenceEncoding(
        hidden_states=np.random.randn(seq_len, hidden_dim).astype(np.float32),
        token_ids=np.random.randint(0, 8192, size=seq_len, dtype=np.int32),
        token_entropy=np.random.rand(seq_len).astype(np.float32),
        finish_reason=finish_reason,
        text=text,
        text_index=text_index,
        sentence_index=sentence_index,
        attention_weights=attention,
    )


def _make_encoded(sentences=None, model_id='test/model'):
    """Create an EncodedSpeech with default params."""
    if sentences is None:
        sentences = [_make_sentence()]
    return EncodedSpeech(
        sentences=sentences,
        model_id=model_id,
    )


# ---------------------------------------------------------------------------
# SentenceEncoding tests
# ---------------------------------------------------------------------------

class TestSentenceEncoding:

    def test_construction(self):
        s = _make_sentence(seq_len=10, text="test sentence")
        assert s.hidden_states.shape == (10, 512)
        assert s.token_ids.shape == (10,)
        assert s.token_entropy.shape == (10,)
        assert s.finish_reason == 'stop'
        assert s.text == "test sentence"
        assert s.text_index == 0
        assert s.sentence_index == 0

    def test_optional_fields_default_none(self):
        s = _make_sentence()
        assert s.attention_weights is None
        assert s.input_token_offsets is None

    def test_with_attention(self):
        attn = np.random.randn(20, 15).astype(np.float32)
        s = _make_sentence(seq_len=20, attention=attn)
        assert s.attention_weights is not None
        assert s.attention_weights.shape == (20, 15)

    def test_finish_reason_length(self):
        s = _make_sentence(finish_reason='length')
        assert s.finish_reason == 'length'


# ---------------------------------------------------------------------------
# EncodedSpeech tests
# ---------------------------------------------------------------------------

class TestEncodedSpeech:

    def test_total_tokens(self):
        s1 = _make_sentence(seq_len=10)
        s2 = _make_sentence(seq_len=20)
        enc = _make_encoded(sentences=[s1, s2])
        assert enc.total_tokens == 30

    def test_mean_entropy(self):
        # Use known entropy values
        s = _make_sentence(seq_len=5)
        s.token_entropy = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        enc = _make_encoded(sentences=[s])
        assert abs(enc.mean_entropy - 3.0) < 1e-6

    def test_mean_entropy_empty(self):
        s = _make_sentence(seq_len=0)
        s.hidden_states = np.zeros((0, 512), dtype=np.float32)
        s.token_ids = np.zeros(0, dtype=np.int32)
        s.token_entropy = np.zeros(0, dtype=np.float32)
        enc = _make_encoded(sentences=[s])
        assert enc.mean_entropy == 0.0

    def test_estimated_duration(self):
        # 100 tokens * 2048 samples/token / 32000 Hz = 6.4 seconds
        s = _make_sentence(seq_len=100)
        enc = _make_encoded(sentences=[s])
        assert abs(enc.estimated_duration - 6.4) < 0.01

    def test_num_texts_single(self):
        s = _make_sentence(text_index=0)
        enc = _make_encoded(sentences=[s])
        assert enc.num_texts == 1

    def test_num_texts_multiple(self):
        s1 = _make_sentence(text_index=0)
        s2 = _make_sentence(text_index=1)
        s3 = _make_sentence(text_index=2)
        enc = _make_encoded(sentences=[s1, s2, s3])
        assert enc.num_texts == 3

    def test_num_texts_empty(self):
        enc = _make_encoded(sentences=[])
        assert enc.num_texts == 0

    def test_defaults(self):
        enc = _make_encoded()
        assert enc.format_version == FORMAT_VERSION
        assert enc.sample_rate == 32000
        assert enc.token_audio_samples == 2048
        assert enc.hidden_dim == 512

    def test_generation_params(self):
        enc = EncodedSpeech(
            sentences=[],
            model_id='test',
            top_p=0.8,
            temperature=0.5,
            repetition_penalty=1.5,
        )
        assert enc.top_p == 0.8
        assert enc.temperature == 0.5
        assert enc.repetition_penalty == 1.5


# ---------------------------------------------------------------------------
# Save/Load roundtrip tests
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_roundtrip_basic(self):
        """Save and load should preserve all data."""
        enc = _make_encoded(sentences=[
            _make_sentence(seq_len=15, text_index=0, sentence_index=0, text="hello"),
            _make_sentence(seq_len=25, text_index=0, sentence_index=1, text="world"),
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soprano")
            save(enc, path)
            loaded = load(path)  # np.savez appends .npz

            assert loaded.model_id == enc.model_id
            assert loaded.format_version == enc.format_version
            assert loaded.sample_rate == enc.sample_rate
            assert loaded.top_p == enc.top_p
            assert loaded.temperature == enc.temperature
            assert loaded.repetition_penalty == enc.repetition_penalty
            assert len(loaded.sentences) == 2
            assert loaded.sentences[0].text == "hello"
            assert loaded.sentences[1].text == "world"
            assert loaded.sentences[0].finish_reason == 'stop'

    def test_roundtrip_float16_precision(self):
        """Float16 wire format should preserve data within expected precision."""
        s = _make_sentence(seq_len=10)
        enc = _make_encoded(sentences=[s])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soprano")
            save(enc, path)
            loaded = load(path)

            # Float16 has ~3 decimal digits of precision
            np.testing.assert_allclose(
                loaded.sentences[0].hidden_states,
                s.hidden_states,
                rtol=1e-2, atol=1e-3,
            )

    def test_roundtrip_with_attention(self):
        """Attention weights should survive save/load."""
        attn = np.random.randn(10, 8).astype(np.float32)
        s = _make_sentence(seq_len=10, attention=attn)
        enc = _make_encoded(sentences=[s])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soprano")
            save(enc, path)
            loaded = load(path)

            assert loaded.sentences[0].attention_weights is not None
            np.testing.assert_allclose(
                loaded.sentences[0].attention_weights,
                attn,
                rtol=1e-2, atol=1e-3,
            )

    def test_roundtrip_without_attention(self):
        """When no attention is saved, load should return None."""
        s = _make_sentence(seq_len=10)
        enc = _make_encoded(sentences=[s])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soprano")
            save(enc, path)
            loaded = load(path)

            assert loaded.sentences[0].attention_weights is None

    def test_roundtrip_metadata_preserved(self):
        """All metadata fields should survive serialization."""
        enc = EncodedSpeech(
            sentences=[_make_sentence(text_index=0, sentence_index=0, text="test",
                                      finish_reason='length')],
            model_id='custom/model-v2',
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "meta_test.soprano")
            save(enc, path)
            loaded = load(path)

            assert loaded.model_id == 'custom/model-v2'
            assert loaded.top_p == 0.9
            assert loaded.temperature == 0.7
            assert loaded.repetition_penalty == 1.3
            assert loaded.sentences[0].finish_reason == 'length'
            assert loaded.sentences[0].text == 'test'

    def test_roundtrip_uncompressed(self):
        """Save with compress=False should also work."""
        enc = _make_encoded()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_uncompressed.soprano")
            save(enc, path, compress=False)
            loaded = load(path)
            assert loaded.model_id == enc.model_id

    def test_roundtrip_token_ids_dtype(self):
        """Token IDs should be int32 after load (stored as uint16 on wire)."""
        s = _make_sentence(seq_len=10)
        enc = _make_encoded(sentences=[s])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soprano")
            save(enc, path)
            loaded = load(path)

            assert loaded.sentences[0].token_ids.dtype == np.int32

    def test_roundtrip_multiple_sentences(self):
        """Multiple sentences should all survive roundtrip."""
        sentences = [
            _make_sentence(seq_len=10, text_index=0, sentence_index=0, text="first"),
            _make_sentence(seq_len=20, text_index=0, sentence_index=1, text="second"),
            _make_sentence(seq_len=15, text_index=1, sentence_index=0, text="third"),
        ]
        enc = _make_encoded(sentences=sentences)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "multi.soprano")
            save(enc, path)
            loaded = load(path)

            assert len(loaded.sentences) == 3
            assert loaded.sentences[0].text == "first"
            assert loaded.sentences[1].text == "second"
            assert loaded.sentences[2].text == "third"
            assert loaded.sentences[0].hidden_states.shape == (10, 512)
            assert loaded.sentences[1].hidden_states.shape == (20, 512)
            assert loaded.sentences[2].hidden_states.shape == (15, 512)

    def test_roundtrip_symmetric_paths(self):
        """save(path) and load(path) should use the same path — no manual .npz."""
        enc = _make_encoded()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soprano")
            save(enc, path)
            loaded = load(path)  # NOT path + ".npz"
            assert loaded.model_id == enc.model_id
            assert len(loaded.sentences) == 1

    def test_load_with_explicit_npz_extension(self):
        """load() should also work when given a path ending in .npz."""
        enc = _make_encoded()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soprano")
            save(enc, path)
            loaded = load(path)  # explicit .npz still works
            assert loaded.model_id == enc.model_id

    def test_save_with_npz_extension_no_double_suffix(self):
        """save() with a .npz path should not create file.npz.npz."""
        enc = _make_encoded()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            save(enc, path)
            assert os.path.exists(path)
            loaded = load(path)
            assert loaded.model_id == enc.model_id

    def test_load_future_format_version_raises(self):
        """load() should raise ValueError for unsupported future format versions."""
        from soprano.encoded import _tamper_format_version_for_test
        enc = _make_encoded()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "future.soprano")
            save(enc, path)
            _tamper_format_version_for_test(path, 999)
            with pytest.raises(ValueError, match="format version 999"):
                load(path)

    def test_properties_after_load(self):
        """Properties should work correctly on loaded EncodedSpeech."""
        sentences = [
            _make_sentence(seq_len=10, text_index=0),
            _make_sentence(seq_len=20, text_index=1),
        ]
        enc = _make_encoded(sentences=sentences)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "props.soprano")
            save(enc, path)
            loaded = load(path)

            assert loaded.total_tokens == 30
            assert loaded.num_texts == 2
            assert loaded.estimated_duration > 0
