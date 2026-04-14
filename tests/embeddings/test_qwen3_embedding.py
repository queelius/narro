"""Tests for Qwen3Embedding06BBackend (fully mocked; no weights loaded)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from muse.embeddings.protocol import EmbeddingResult


def _mock_model(encode_return=None, dim=1024):
    m = MagicMock()
    if encode_return is None:
        m.encode.return_value = np.zeros((1, dim), dtype=np.float32)
    else:
        m.encode.return_value = np.asarray(encode_return, dtype=np.float32)
    def _tok(texts):
        n = len(texts)
        return {
            "input_ids": np.ones((n, 7), dtype=np.int64),
            "attention_mask": np.ones((n, 7), dtype=np.int64),
        }
    m.tokenize.side_effect = _tok
    return m


def test_qwen3_model_id_and_dimensions():
    with patch("muse.embeddings.backends.qwen3_embedding.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.qwen3_embedding import Qwen3Embedding06BBackend
        m = Qwen3Embedding06BBackend(
            hf_repo="Qwen/Qwen3-Embedding-0.6B", local_dir="/fake",
        )
        assert m.model_id == "qwen3-embedding-0.6b"
        assert m.dimensions == 1024


def test_qwen3_passes_trust_remote_code_true():
    """Qwen3-Embedding publishes custom architecture in its HF repo."""
    with patch("muse.embeddings.backends.qwen3_embedding.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.qwen3_embedding import Qwen3Embedding06BBackend
        Qwen3Embedding06BBackend(hf_repo="fake", local_dir="/fake")
        kwargs = mock_cls.call_args.kwargs
        assert kwargs.get("trust_remote_code") is True


def test_qwen3_embed_single_string():
    with patch("muse.embeddings.backends.qwen3_embedding.SentenceTransformer") as mock_cls:
        fake = _mock_model(encode_return=[[0.1] * 1024])
        mock_cls.return_value = fake
        from muse.embeddings.backends.qwen3_embedding import Qwen3Embedding06BBackend
        m = Qwen3Embedding06BBackend(hf_repo="fake", local_dir="/fake")
        result = m.embed("hello")
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert result.dimensions == 1024
        assert result.model_id == "qwen3-embedding-0.6b"


def test_qwen3_embed_list_of_strings():
    with patch("muse.embeddings.backends.qwen3_embedding.SentenceTransformer") as mock_cls:
        fake = _mock_model(encode_return=np.zeros((3, 1024), dtype=np.float32))
        mock_cls.return_value = fake
        from muse.embeddings.backends.qwen3_embedding import Qwen3Embedding06BBackend
        m = Qwen3Embedding06BBackend(hf_repo="fake", local_dir="/fake")
        result = m.embed(["a", "b", "c"])
        assert len(result.embeddings) == 3


def test_qwen3_dimensions_truncation_renormalizes():
    """Matryoshka truncation should produce unit-norm vectors at the smaller dim."""
    with patch("muse.embeddings.backends.qwen3_embedding.SentenceTransformer") as mock_cls:
        raw = np.array([[3.0, 4.0] + [0.0] * 1022], dtype=np.float32)
        fake = _mock_model(encode_return=raw)
        mock_cls.return_value = fake
        from muse.embeddings.backends.qwen3_embedding import Qwen3Embedding06BBackend
        m = Qwen3Embedding06BBackend(hf_repo="fake", local_dir="/fake")
        result = m.embed("hi", dimensions=2)
        assert result.dimensions == 2
        truncated = result.embeddings[0]
        assert len(truncated) == 2
        # L2 norm of (3, 4) -> (0.6, 0.8)
        assert abs(truncated[0] - 0.6) < 1e-5
        assert abs(truncated[1] - 0.8) < 1e-5


def test_qwen3_prefers_local_dir_over_hf_repo():
    with patch("muse.embeddings.backends.qwen3_embedding.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.qwen3_embedding import Qwen3Embedding06BBackend
        Qwen3Embedding06BBackend(
            hf_repo="Qwen/Qwen3-Embedding-0.6B", local_dir="/real/path",
        )
        assert mock_cls.call_args.args[0] == "/real/path"


def test_qwen3_falls_back_to_hf_repo_when_local_dir_none():
    with patch("muse.embeddings.backends.qwen3_embedding.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.qwen3_embedding import Qwen3Embedding06BBackend
        Qwen3Embedding06BBackend(
            hf_repo="Qwen/Qwen3-Embedding-0.6B", local_dir=None,
        )
        assert mock_cls.call_args.args[0] == "Qwen/Qwen3-Embedding-0.6B"


def test_qwen3_counts_tokens_from_attention_mask():
    with patch("muse.embeddings.backends.qwen3_embedding.SentenceTransformer") as mock_cls:
        fake = _mock_model(encode_return=np.zeros((2, 1024), dtype=np.float32))
        mock_cls.return_value = fake
        from muse.embeddings.backends.qwen3_embedding import Qwen3Embedding06BBackend
        m = Qwen3Embedding06BBackend(hf_repo="fake", local_dir="/fake")
        result = m.embed(["a", "b"])
        # _mock_model returns attention_mask shape (2, 7) of all 1s
        assert result.prompt_tokens == 14
