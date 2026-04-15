"""Tests for EmbeddingsModel protocol."""
import pytest

from muse.modalities.embedding_text.protocol import EmbeddingsModel, EmbeddingResult


def test_embedding_result_stores_all_fields():
    r = EmbeddingResult(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        dimensions=2,
        model_id="fake-embed",
        prompt_tokens=10,
        metadata={"prompt": "hi"},
    )
    assert r.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert r.dimensions == 2
    assert r.model_id == "fake-embed"
    assert r.prompt_tokens == 10
    assert r.metadata["prompt"] == "hi"


def test_embedding_result_metadata_defaults_empty():
    r = EmbeddingResult(
        embeddings=[[0.1]], dimensions=1, model_id="x", prompt_tokens=0,
    )
    assert r.metadata == {}


def test_embedding_result_metadata_is_independent_per_instance():
    """Regression: shared mutable default would leak state across instances."""
    a = EmbeddingResult(embeddings=[[0.1]], dimensions=1, model_id="x", prompt_tokens=0)
    b = EmbeddingResult(embeddings=[[0.2]], dimensions=1, model_id="y", prompt_tokens=0)
    a.metadata["k"] = 1
    assert "k" not in b.metadata


def test_embeddings_model_protocol_accepts_structural_impl():
    class MyModel:
        model_id = "fake-embed"
        dimensions = 384
        def embed(self, input, **kwargs):
            ...

    assert isinstance(MyModel(), EmbeddingsModel)


def test_embeddings_model_protocol_rejects_incomplete():
    class Missing:
        pass

    assert not isinstance(Missing(), EmbeddingsModel)
