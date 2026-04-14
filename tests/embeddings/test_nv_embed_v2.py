"""Tests for NVEmbedV2Backend (fully mocked; 14GB model not loaded)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _mock_model(encode_return=None, dim=4096):
    """Fake transformers AutoModel with NV-Embed-style encode()."""
    import torch
    m = MagicMock()
    if encode_return is None:
        tensor = torch.zeros((1, dim))
    else:
        tensor = torch.tensor(np.asarray(encode_return, dtype=np.float32))
    m.encode.return_value = tensor
    # .to(device) returns self
    m.to.return_value = m
    # parameters() for the requires_grad_ loop
    m.parameters.return_value = []
    # tokenizer for _count_tokens
    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {"input_ids": [[1, 2, 3, 4]]}
    m.tokenizer = fake_tokenizer
    return m


def test_nv_embed_model_id_and_dimensions():
    with patch("muse.embeddings.backends.nv_embed_v2.AutoModel") as mock_cls:
        mock_cls.from_pretrained.return_value = _mock_model()
        from muse.embeddings.backends.nv_embed_v2 import NVEmbedV2Backend
        m = NVEmbedV2Backend(hf_repo="nvidia/NV-Embed-v2", local_dir="/fake")
        assert m.model_id == "nv-embed-v2"
        assert m.dimensions == 4096


def test_nv_embed_passes_trust_remote_code():
    """NV-Embed ships a custom Mistral-based architecture in its HF repo."""
    with patch("muse.embeddings.backends.nv_embed_v2.AutoModel") as mock_cls:
        mock_cls.from_pretrained.return_value = _mock_model()
        from muse.embeddings.backends.nv_embed_v2 import NVEmbedV2Backend
        NVEmbedV2Backend(hf_repo="nvidia/NV-Embed-v2", local_dir="/fake")
        kwargs = mock_cls.from_pretrained.call_args.kwargs
        assert kwargs.get("trust_remote_code") is True


def test_nv_embed_embed_single_string():
    from muse.embeddings.protocol import EmbeddingResult
    with patch("muse.embeddings.backends.nv_embed_v2.AutoModel") as mock_cls:
        fake = _mock_model(encode_return=[[0.1] * 4096])
        mock_cls.from_pretrained.return_value = fake
        from muse.embeddings.backends.nv_embed_v2 import NVEmbedV2Backend
        m = NVEmbedV2Backend(hf_repo="nvidia/NV-Embed-v2", local_dir="/fake")
        result = m.embed("hello")
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 4096
        assert result.dimensions == 4096
        assert result.model_id == "nv-embed-v2"
        assert result.metadata["license"] == "CC-BY-NC-4.0"


def test_nv_embed_default_instruction_is_empty_passage_side():
    """Default instruction is "" (passage-side: embed documents for retrieval)."""
    with patch("muse.embeddings.backends.nv_embed_v2.AutoModel") as mock_cls:
        fake = _mock_model(encode_return=[[0.1] * 4096])
        mock_cls.from_pretrained.return_value = fake
        from muse.embeddings.backends.nv_embed_v2 import NVEmbedV2Backend
        m = NVEmbedV2Backend(hf_repo="fake", local_dir="/fake")
        m.embed("doc to retrieve")
        call_kwargs = fake.encode.call_args.kwargs
        assert call_kwargs.get("instruction") == ""


def test_nv_embed_custom_instruction_overrides_default_per_call():
    """Callers can pass instruction= to embed() for query-side embeddings."""
    with patch("muse.embeddings.backends.nv_embed_v2.AutoModel") as mock_cls:
        fake = _mock_model(encode_return=[[0.1] * 4096])
        mock_cls.from_pretrained.return_value = fake
        from muse.embeddings.backends.nv_embed_v2 import NVEmbedV2Backend
        m = NVEmbedV2Backend(hf_repo="fake", local_dir="/fake")
        query_instr = "Instruct: Given a question, retrieve passages.\nQuery: "
        m.embed("what is photosynthesis", instruction=query_instr)
        call_kwargs = fake.encode.call_args.kwargs
        assert call_kwargs.get("instruction") == query_instr


def test_nv_embed_constructor_instruction_used_when_no_override():
    with patch("muse.embeddings.backends.nv_embed_v2.AutoModel") as mock_cls:
        fake = _mock_model(encode_return=[[0.1] * 4096])
        mock_cls.from_pretrained.return_value = fake
        from muse.embeddings.backends.nv_embed_v2 import NVEmbedV2Backend
        m = NVEmbedV2Backend(
            hf_repo="fake", local_dir="/fake",
            instruction="classification: ",
        )
        m.embed("something")
        call_kwargs = fake.encode.call_args.kwargs
        assert call_kwargs.get("instruction") == "classification: "


def test_nv_embed_passes_max_length():
    """max_length flows through to encode()."""
    with patch("muse.embeddings.backends.nv_embed_v2.AutoModel") as mock_cls:
        fake = _mock_model(encode_return=[[0.1] * 4096])
        mock_cls.from_pretrained.return_value = fake
        from muse.embeddings.backends.nv_embed_v2 import NVEmbedV2Backend
        m = NVEmbedV2Backend(hf_repo="fake", local_dir="/fake", max_length=16384)
        m.embed("hi")
        call_kwargs = fake.encode.call_args.kwargs
        assert call_kwargs.get("max_length") == 16384


def test_nv_embed_normalizes_output_to_unit_length():
    """NV-Embed doesn't L2-normalize internally; backend must."""
    import torch
    with patch("muse.embeddings.backends.nv_embed_v2.AutoModel") as mock_cls:
        # Non-unit-length tensor so we can verify normalization happened
        raw = [[3.0, 4.0] + [0.0] * 4094]
        fake = _mock_model(encode_return=raw)
        mock_cls.from_pretrained.return_value = fake
        from muse.embeddings.backends.nv_embed_v2 import NVEmbedV2Backend
        m = NVEmbedV2Backend(hf_repo="fake", local_dir="/fake")
        result = m.embed("x")
        vec = np.asarray(result.embeddings[0])
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"expected unit-norm, got {norm}"


def test_nv_embed_dimensions_truncation_renormalizes():
    with patch("muse.embeddings.backends.nv_embed_v2.AutoModel") as mock_cls:
        raw = [[3.0, 4.0] + [0.0] * 4094]
        fake = _mock_model(encode_return=raw)
        mock_cls.from_pretrained.return_value = fake
        from muse.embeddings.backends.nv_embed_v2 import NVEmbedV2Backend
        m = NVEmbedV2Backend(hf_repo="fake", local_dir="/fake")
        result = m.embed("x", dimensions=2)
        assert result.dimensions == 2
        assert len(result.embeddings[0]) == 2
        # After normalize of (3,4) to unit, truncate to (0.6, 0.8),
        # renormalize again -> still (0.6, 0.8)
        assert abs(result.embeddings[0][0] - 0.6) < 1e-5
        assert abs(result.embeddings[0][1] - 0.8) < 1e-5


def test_nv_embed_prefers_local_dir_over_hf_repo():
    with patch("muse.embeddings.backends.nv_embed_v2.AutoModel") as mock_cls:
        mock_cls.from_pretrained.return_value = _mock_model()
        from muse.embeddings.backends.nv_embed_v2 import NVEmbedV2Backend
        NVEmbedV2Backend(hf_repo="nvidia/NV-Embed-v2", local_dir="/real/path")
        # from_pretrained's first positional arg is the source path
        assert mock_cls.from_pretrained.call_args.args[0] == "/real/path"


def test_nv_embed_accepts_unknown_kwargs():
    """Future catalog kwargs should be absorbed by **_."""
    with patch("muse.embeddings.backends.nv_embed_v2.AutoModel") as mock_cls:
        mock_cls.from_pretrained.return_value = _mock_model()
        from muse.embeddings.backends.nv_embed_v2 import NVEmbedV2Backend
        NVEmbedV2Backend(
            hf_repo="fake", local_dir="/fake",
            device="cpu", future_kwarg="ignored",
        )
