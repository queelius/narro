"""Tests for SentenceTransformerModel (generic ST runtime; library mocked)."""
from unittest.mock import MagicMock, patch

import numpy as np

from muse.modalities.embedding_text.protocol import EmbeddingResult


def _mock_st_model(dim=384, encode_return=None):
    m = MagicMock()
    if encode_return is None:
        m.encode.return_value = np.zeros((1, dim), dtype=np.float32)
    else:
        m.encode.return_value = np.asarray(encode_return, dtype=np.float32)
    m.get_sentence_embedding_dimension.return_value = dim
    m.max_seq_length = 512

    def _tok(texts):
        n = len(texts)
        return {
            "input_ids": np.ones((n, 7), dtype=np.int64),
            "attention_mask": np.ones((n, 7), dtype=np.int64),
        }

    m.tokenize.side_effect = _tok
    return m


def test_auto_detects_dimensions_from_model():
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer"
    ) as mock_cls:
        mock_cls.return_value = _mock_st_model(dim=768)
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        m = SentenceTransformerModel(
            model_id="some-model",
            hf_repo="some/model",
            local_dir="/fake",
        )
        assert m.dimensions == 768


def test_embed_returns_embedding_result():
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer"
    ) as mock_cls:
        fake = _mock_st_model(dim=384, encode_return=np.zeros((1, 384), dtype=np.float32))
        mock_cls.return_value = fake
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        m = SentenceTransformerModel(
            model_id="all-minilm-l6-v2",
            hf_repo="sentence-transformers/all-MiniLM-L6-v2",
            local_dir="/fake",
        )
        r = m.embed("hello")
        assert isinstance(r, EmbeddingResult)
        assert r.dimensions == 384
        assert r.model_id == "all-minilm-l6-v2"
        assert len(r.embeddings) == 1


def test_trust_remote_code_forwarded():
    """For reviewed repos that require it (Nomic and some custom models)."""
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer"
    ) as mock_cls:
        mock_cls.return_value = _mock_st_model()
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        SentenceTransformerModel(
            model_id="x", hf_repo="x", local_dir="/fake",
            trust_remote_code=True,
        )
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["trust_remote_code"] is True


def test_external_code_revision_forwarded_to_config_and_model():
    revision = "2" * 40
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer"
    ) as mock_cls:
        mock_cls.return_value = _mock_st_model()
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        SentenceTransformerModel(
            model_id="nomic",
            hf_repo="nomic-ai/nomic-embed-text-v1.5",
            local_dir="/fake",
            trust_remote_code=True,
            code_revision=revision,
        )

    kwargs = mock_cls.call_args.kwargs
    assert kwargs["model_kwargs"]["code_revision"] == revision
    assert kwargs["config_kwargs"]["code_revision"] == revision
    assert kwargs["model_kwargs"].pop("code_revision") == revision
    assert kwargs["model_kwargs"]["code_revision"] == revision


def test_defaults_trust_remote_code_false():
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer"
    ) as mock_cls:
        mock_cls.return_value = _mock_st_model()
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        SentenceTransformerModel(model_id="x", hf_repo="x", local_dir="/fake")
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["trust_remote_code"] is False


def test_prefers_local_dir_over_hf_repo():
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer"
    ) as mock_cls:
        mock_cls.return_value = _mock_st_model()
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        SentenceTransformerModel(
            model_id="x", hf_repo="remote/x", local_dir="/real/local/path",
        )
        assert mock_cls.call_args.args[0] == "/real/local/path"


def test_falls_back_to_hf_repo_when_no_local_dir():
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer"
    ) as mock_cls:
        mock_cls.return_value = _mock_st_model()
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        SentenceTransformerModel(model_id="x", hf_repo="remote/x", local_dir=None)
        assert mock_cls.call_args.args[0] == "remote/x"


def test_matryoshka_truncation_renormalizes():
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer"
    ) as mock_cls:
        raw = np.array([[3.0, 4.0] + [0.0] * 1022], dtype=np.float32)
        mock_cls.return_value = _mock_st_model(dim=1024, encode_return=raw)
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        m = SentenceTransformerModel(model_id="x", hf_repo="x", local_dir="/fake")
        r = m.embed("hi", dimensions=2)
        assert r.dimensions == 2
        assert abs(r.embeddings[0][0] - 0.6) < 1e-5
        assert abs(r.embeddings[0][1] - 0.8) < 1e-5


def test_counts_tokens_from_attention_mask():
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer"
    ) as mock_cls:
        mock_cls.return_value = _mock_st_model(
            dim=384, encode_return=np.zeros((2, 384), dtype=np.float32),
        )
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        m = SentenceTransformerModel(model_id="x", hf_repo="x", local_dir="/fake")
        r = m.embed(["a", "b"])
        assert r.prompt_tokens == 14  # 2 texts * 7 mask positions


def test_accepts_unknown_kwargs():
    """Future MANIFEST kwargs absorbed by **_."""
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer"
    ) as mock_cls:
        mock_cls.return_value = _mock_st_model()
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        SentenceTransformerModel(
            model_id="x", hf_repo="x", local_dir="/fake",
            future_kwarg="absorbed", device="cpu",
        )


def test_raises_clear_error_when_deps_missing():
    """Simulate 'sentence-transformers not importable' on the host.

    `sentence_transformers` IS installed in the test env, so we have to
    neutralize both the module-level sentinel AND `_ensure_deps` (which
    would otherwise re-import the real SentenceTransformer and clobber
    the None patch).
    """
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer",
        new=None,
    ), patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers._ensure_deps",
    ):
        import pytest
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            SentenceTransformerModel(model_id="x", hf_repo="x", local_dir="/fake")


def test_string_input_wraps_to_single_element_list():
    """embed('hi') should behave like embed(['hi'])."""
    with patch(
        "muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer"
    ) as mock_cls:
        fake = _mock_st_model(dim=384, encode_return=np.zeros((1, 384), dtype=np.float32))
        mock_cls.return_value = fake
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        m = SentenceTransformerModel(model_id="x", hf_repo="x", local_dir="/fake")
        m.embed("hello")
        assert fake.encode.call_args.args[0] == ["hello"]
