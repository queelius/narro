"""Tests for HFResolver dispatch over per-modality plugins."""
from unittest.mock import MagicMock, patch

import pytest

from muse.core.resolvers import ResolvedModel, ResolverError, SearchResult
from muse.core.resolvers_hf import HFResolver


def _make_plugin(modality, *, priority=100, sniff_returns=False, resolve_returns=None, search_returns=()):
    """Build an HF_PLUGIN dict with controllable callbacks for tests."""
    return {
        "modality": modality,
        "runtime_path": f"muse.modalities.{modality.replace('/', '_')}.runtimes.fake:Fake",
        "pip_extras": (),
        "system_packages": (),
        "priority": priority,
        "sniff": MagicMock(return_value=sniff_returns),
        "resolve": MagicMock(return_value=resolve_returns),
        "search": MagicMock(return_value=iter(search_returns)),
    }


def test_resolve_first_matching_plugin_wins():
    p_low = _make_plugin("a/first", priority=100, sniff_returns=False)
    p_high = _make_plugin(
        "b/second", priority=200, sniff_returns=True,
        resolve_returns=ResolvedModel(
            manifest={
                "model_id": "x",
                "hf_repo": "org/repo",
                "revision": "a" * 40,
            },
            backend_path="a:B",
            download=lambda root: root,
        ),
    )
    resolver = HFResolver(plugins=[p_low, p_high])
    fake_info = MagicMock()
    fake_info.sha = "a" * 40
    with patch.object(resolver._api, "repo_info", return_value=fake_info):
        result = resolver.resolve("hf://org/repo")
    assert result.manifest["model_id"] == "x"
    p_low["sniff"].assert_called_once_with(fake_info)
    p_high["sniff"].assert_called_once_with(fake_info)
    p_high["resolve"].assert_called_once()


def test_resolve_no_plugin_matches_raises_clean_error():
    p1 = _make_plugin("x/y", sniff_returns=False)
    resolver = HFResolver(plugins=[p1])
    fake_info = MagicMock(siblings=[], tags=["random"], sha="a" * 40)
    with patch.object(resolver._api, "repo_info", return_value=fake_info):
        with pytest.raises(ResolverError, match="no HF plugin matched"):
            resolver.resolve("hf://org/repo")


def test_resolve_short_circuits_on_first_match():
    """Once a plugin's sniff returns True, later plugins are not consulted."""
    p_first = _make_plugin(
        "a/x", priority=100, sniff_returns=True,
        resolve_returns=ResolvedModel(
            manifest={
                "model_id": "a",
                "hf_repo": "org/repo",
                "revision": "a" * 40,
            },
            backend_path="a:B",
            download=lambda root: root,
        ),
    )
    p_second = _make_plugin("b/y", priority=200, sniff_returns=True)
    resolver = HFResolver(plugins=[p_first, p_second])
    with patch.object(
        resolver._api,
        "repo_info",
        return_value=MagicMock(sha="a" * 40),
    ):
        resolver.resolve("hf://org/repo")
    p_first["sniff"].assert_called_once()
    p_second["sniff"].assert_not_called()


def test_search_with_modality_filter_consults_only_matching():
    p_a = _make_plugin("a/x", search_returns=[SearchResult(
        uri="hf://a/1", model_id="m1", modality="a/x",
        size_gb=None, downloads=None, license=None, description=None,
    )])
    p_b = _make_plugin("b/y", search_returns=[SearchResult(
        uri="hf://b/2", model_id="m2", modality="b/y",
        size_gb=None, downloads=None, license=None, description=None,
    )])
    resolver = HFResolver(plugins=[p_a, p_b])
    rows = list(resolver.search("foo", modality="a/x"))
    assert [r.modality for r in rows] == ["a/x"]
    p_a["search"].assert_called_once()
    p_b["search"].assert_not_called()


def test_search_with_no_modality_filter_consults_all_plugins():
    p_a = _make_plugin("a/x", search_returns=[SearchResult(
        uri="hf://a/1", model_id="m1", modality="a/x",
        size_gb=None, downloads=None, license=None, description=None,
    )])
    p_b = _make_plugin("b/y", search_returns=[SearchResult(
        uri="hf://b/2", model_id="m2", modality="b/y",
        size_gb=None, downloads=None, license=None, description=None,
    )])
    resolver = HFResolver(plugins=[p_a, p_b])
    rows = list(resolver.search("foo"))
    assert sorted(r.modality for r in rows) == ["a/x", "b/y"]
    p_a["search"].assert_called_once()
    p_b["search"].assert_called_once()


def test_search_with_unknown_modality_raises_clean_error():
    p_a = _make_plugin("a/x")
    resolver = HFResolver(plugins=[p_a])
    with pytest.raises(ResolverError, match="does not support modality"):
        list(resolver.search("foo", modality="never/heard-of-it"))


def test_default_constructor_loads_from_disk():
    """No plugins= arg falls back to discover_hf_plugins(default_dirs)."""
    with patch("muse.core.resolvers_hf.discover_hf_plugins", return_value=[]) as mock_discover:
        HFResolver()
    mock_discover.assert_called_once()
