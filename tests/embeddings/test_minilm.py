"""Tests for MiniLMBackend (fully mocked; sentence-transformers not loaded)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from muse.embeddings.protocol import EmbeddingResult


def _mock_model(encode_return=None, dim=384):
    """Build a fake SentenceTransformer instance with a predictable shape."""
    m = MagicMock()
    if encode_return is None:
        m.encode.return_value = np.zeros((1, dim), dtype=np.float32)
    else:
        m.encode.return_value = np.asarray(encode_return, dtype=np.float32)
    m.get_sentence_embedding_dimension.return_value = dim
    # tokenize returns a dict with attention_mask we can sum
    def _tok(texts):
        n = len(texts)
        # pretend each text is 5 tokens
        return {
            "input_ids": np.ones((n, 5), dtype=np.int64),
            "attention_mask": np.ones((n, 5), dtype=np.int64),
        }
    m.tokenize.side_effect = _tok
    return m


def test_minilm_model_id_and_dimensions():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="sentence-transformers/all-MiniLM-L6-v2",
                          local_dir="/fake")
        assert m.model_id == "all-minilm-l6-v2"
        assert m.dimensions == 384


def test_minilm_embed_single_string():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        fake = _mock_model(encode_return=[[0.1] * 384])
        mock_cls.return_value = fake
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="sentence-transformers/all-MiniLM-L6-v2",
                          local_dir="/fake")
        result = m.embed("hello")
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 384
        assert result.dimensions == 384
        assert result.model_id == "all-minilm-l6-v2"
        # Tokenize was called with ["hello"]
        call = fake.tokenize.call_args.args[0]
        assert call == ["hello"]


def test_minilm_embed_list_of_strings():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        fake = _mock_model(encode_return=np.zeros((3, 384), dtype=np.float32))
        mock_cls.return_value = fake
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="sentence-transformers/all-MiniLM-L6-v2",
                          local_dir="/fake")
        result = m.embed(["a", "b", "c"])
        assert len(result.embeddings) == 3
        call = fake.tokenize.call_args.args[0]
        assert call == ["a", "b", "c"]


def test_minilm_counts_tokens_from_attention_mask():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        fake = _mock_model(encode_return=np.zeros((2, 384), dtype=np.float32))
        mock_cls.return_value = fake
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="...", local_dir="/fake")
        result = m.embed(["a", "b"])
        # _mock_model returns attention_mask of shape (2, 5) of all 1s
        assert result.prompt_tokens == 10


def test_minilm_dimensions_truncation_l2_normalizes():
    """Truncating + renormalizing produces unit vectors at the smaller dim."""
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        # Encode returns a known vector so we can check the math
        raw = np.array([[3.0, 4.0, 0.0, 0.0] + [0.0] * 380], dtype=np.float32)
        fake = _mock_model(encode_return=raw)
        mock_cls.return_value = fake
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="...", local_dir="/fake")
        result = m.embed("hi", dimensions=2)
        assert result.dimensions == 2
        truncated = result.embeddings[0]
        assert len(truncated) == 2
        # L2 norm of (3, 4) / sqrt(25) = (0.6, 0.8)
        assert abs(truncated[0] - 0.6) < 1e-5
        assert abs(truncated[1] - 0.8) < 1e-5


def test_minilm_full_dimensions_skips_truncation():
    """If dimensions >= model dim, no truncation applied."""
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        raw = np.ones((1, 384), dtype=np.float32)
        fake = _mock_model(encode_return=raw)
        mock_cls.return_value = fake
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="...", local_dir="/fake")
        result = m.embed("hi", dimensions=384)
        assert result.dimensions == 384


def test_minilm_prefers_local_dir_over_hf_repo():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.minilm import MiniLMBackend
        MiniLMBackend(hf_repo="sentence-transformers/all-MiniLM-L6-v2",
                      local_dir="/real/local/path")
        # First positional arg should be local_dir
        args = mock_cls.call_args.args
        assert args[0] == "/real/local/path"


def test_minilm_falls_back_to_hf_repo_when_local_dir_none():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.minilm import MiniLMBackend
        MiniLMBackend(hf_repo="sentence-transformers/all-MiniLM-L6-v2",
                      local_dir=None)
        args = mock_cls.call_args.args
        assert args[0] == "sentence-transformers/all-MiniLM-L6-v2"


def test_minilm_accepts_unknown_kwargs():
    """Future catalog kwargs (**_) should be absorbed."""
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.minilm import MiniLMBackend
        # Should not TypeError
        MiniLMBackend(
            hf_repo="x", local_dir="/fake",
            device="cpu", future_param="ignored",
        )


def test_minilm_module_imports_without_sentence_transformers_installed():
    """Module-level try/except leaves SentenceTransformer=None when missing."""
    import muse.embeddings.backends.minilm as mod
    assert hasattr(mod, "SentenceTransformer")
