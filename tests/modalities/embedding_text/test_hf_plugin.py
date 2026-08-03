"""Tests for the embedding_text HF plugin (sentence-transformers)."""
from unittest.mock import MagicMock

from muse.modalities.embedding_text.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


_FAKE_REVISION = "1" * 40


def _fake_info(siblings=None, tags=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    info.sha = _FAKE_REVISION
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "embedding/text"
    assert HF_PLUGIN["runtime_path"].endswith(":SentenceTransformerModel")
    assert HF_PLUGIN["priority"] == 110


def test_sniff_true_on_sentence_transformers_tag():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["sentence-transformers"])) is True


def test_sniff_true_on_st_config_file():
    info = _fake_info(siblings=["sentence_transformers_config.json"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_on_random_repo():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["text-generation"])) is False


def test_resolve_returns_resolved_model():
    info = _fake_info(tags=["sentence-transformers"])
    result = HF_PLUGIN["resolve"]("sentence-transformers/all-MiniLM-L6-v2", None, info)
    assert isinstance(result, ResolvedModel)
    assert result.manifest["modality"] == "embedding/text"
    assert result.manifest["model_id"] == "all-minilm-l6-v2"
    assert result.manifest["revision"] == _FAKE_REVISION


def test_nonhex_hub_sha_is_not_persisted_as_revision():
    info = _fake_info(tags=["sentence-transformers"])
    info.sha = "z" * 40
    result = HF_PLUGIN["resolve"]("org/model", None, info)
    assert "revision" not in result.manifest


def test_download_uses_resolved_commit_and_includes_dynamic_modules(
    tmp_path, monkeypatch,
):
    import muse.modalities.embedding_text.hf as plugin_module

    calls = {}
    monkeypatch.setattr(
        plugin_module,
        "snapshot_download",
        lambda **kwargs: calls.update(kwargs) or str(tmp_path / "snapshot"),
    )
    result = HF_PLUGIN["resolve"](
        "nomic-ai/nomic-embed-text-v1.5",
        None,
        _fake_info(tags=["sentence-transformers"]),
    )
    result.download(tmp_path)

    assert calls["revision"] == _FAKE_REVISION
    assert "*.py" in calls["allow_patterns"]


def test_search_yields_results():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/repo", downloads=50)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "minilm", sort="downloads", limit=20))
    assert len(rows) == 1
    assert rows[0].modality == "embedding/text"
