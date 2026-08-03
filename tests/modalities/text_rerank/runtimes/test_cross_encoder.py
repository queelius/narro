"""Tests for CrossEncoderRuntime (sentence-transformers CrossEncoder wrapper)."""
from unittest.mock import MagicMock, patch

import pytest

import muse.modalities.text_rerank.runtimes.cross_encoder as ce_mod
from muse.modalities.text_rerank import RerankResult
from muse.modalities.text_rerank.runtimes.cross_encoder import (
    CrossEncoderRuntime,
)


def _patched_runtime(predict_return):
    """Return a CrossEncoderRuntime with sentence_transformers stubbed."""
    fake_ce = MagicMock()
    fake_ce.predict.return_value = predict_return
    fake_ce_class = MagicMock(return_value=fake_ce)
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(ce_mod, "CrossEncoder", fake_ce_class), \
            patch.object(ce_mod, "torch", fake_torch):
        rt = CrossEncoderRuntime(
            model_id="test", hf_repo="org/repo",
            local_dir=None, device="cpu", max_length=512,
        )
    return rt, fake_ce, fake_ce_class


def test_runtime_constructs_with_local_dir_preference():
    """Runtime prefers local_dir over hf_repo as the source path."""
    fake_ce = MagicMock()
    fake_ce_class = MagicMock(return_value=fake_ce)
    with patch.object(ce_mod, "CrossEncoder", fake_ce_class), \
            patch.object(ce_mod, "torch", MagicMock()):
        CrossEncoderRuntime(
            model_id="m", hf_repo="org/repo",
            local_dir="/tmp/cache/abc", device="cpu",
        )
    args, kwargs = fake_ce_class.call_args
    assert args[0] == "/tmp/cache/abc"


def test_runtime_falls_back_to_hf_repo_when_no_local_dir():
    fake_ce = MagicMock()
    fake_ce_class = MagicMock(return_value=fake_ce)
    with patch.object(ce_mod, "CrossEncoder", fake_ce_class), \
            patch.object(ce_mod, "torch", MagicMock()):
        CrossEncoderRuntime(
            model_id="m", hf_repo="org/repo",
            local_dir=None, device="cpu",
        )
    args, _ = fake_ce_class.call_args
    assert args[0] == "org/repo"


def test_runtime_forwards_remote_code_security_arguments():
    fake_ce_class = MagicMock(return_value=MagicMock())
    code_revision = "2" * 40
    with patch.object(ce_mod, "CrossEncoder", fake_ce_class), \
            patch.object(ce_mod, "torch", MagicMock()):
        CrossEncoderRuntime(
            model_id="jina",
            hf_repo="jinaai/jina-reranker-v2-base-multilingual",
            local_dir="/fake",
            device="cpu",
            trust_remote_code=True,
            code_revision=code_revision,
        )

    kwargs = fake_ce_class.call_args.kwargs
    assert kwargs["trust_remote_code"] is True
    assert kwargs["model_kwargs"]["code_revision"] == code_revision
    assert kwargs["config_kwargs"]["code_revision"] == code_revision


def test_rerank_returns_descending_score_order():
    rt, fake_ce, _ = _patched_runtime([0.1, 0.9, 0.4])
    out = rt.rerank("q", ["a", "b", "c"])
    assert isinstance(out, list)
    assert all(isinstance(r, RerankResult) for r in out)
    indices = [r.index for r in out]
    assert indices == [1, 2, 0]
    scores = [r.relevance_score for r in out]
    assert scores == sorted(scores, reverse=True)


def test_rerank_passes_pairs_to_predict():
    rt, fake_ce, _ = _patched_runtime([0.5, 0.5])
    rt.rerank("hello", ["doc1", "doc2"])
    args, _ = fake_ce.predict.call_args
    pairs = args[0]
    assert pairs == [("hello", "doc1"), ("hello", "doc2")]


def test_rerank_top_n_truncates():
    rt, fake_ce, _ = _patched_runtime([0.1, 0.9, 0.5, 0.7])
    out = rt.rerank("q", ["a", "b", "c", "d"], top_n=2)
    assert len(out) == 2
    assert [r.index for r in out] == [1, 3]


def test_rerank_top_n_none_returns_all():
    rt, fake_ce, _ = _patched_runtime([0.1, 0.9, 0.5])
    out = rt.rerank("q", ["a", "b", "c"], top_n=None)
    assert len(out) == 3


def test_rerank_empty_documents_returns_empty():
    rt, fake_ce, _ = _patched_runtime([])
    out = rt.rerank("q", [])
    assert out == []
    fake_ce.predict.assert_not_called()


def test_rerank_preserves_document_text():
    rt, fake_ce, _ = _patched_runtime([0.2, 0.8])
    out = rt.rerank("q", ["alpha", "beta"])
    text_by_index = {r.index: r.document_text for r in out}
    assert text_by_index == {0: "alpha", 1: "beta"}


def test_runtime_raises_when_sentence_transformers_missing():
    """Simulate missing sentence_transformers by stubbing _ensure_deps so
    it leaves the module-level CrossEncoder sentinel as None."""
    with patch.object(ce_mod, "CrossEncoder", None), \
            patch.object(ce_mod, "_ensure_deps", lambda: None), \
            patch.object(ce_mod, "torch", MagicMock()):
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            CrossEncoderRuntime(
                model_id="m", hf_repo="org/repo",
                local_dir=None, device="cpu",
            )


def test_select_device_auto_falls_back_to_cpu_with_no_torch():
    with patch.object(ce_mod, "torch", None):
        from muse.modalities.text_rerank.runtimes.cross_encoder import _select_device
        assert _select_device("auto") == "cpu"


def test_select_device_auto_picks_cuda():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.object(ce_mod, "torch", fake_torch):
        from muse.modalities.text_rerank.runtimes.cross_encoder import _select_device
        assert _select_device("auto") == "cuda"
