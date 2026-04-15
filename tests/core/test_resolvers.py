"""Tests for the muse.core.resolvers abstraction.

The resolver module only defines the ABCs, dataclasses, and registry
dispatch. Concrete resolvers (hf://, etc.) live in separate modules
and register themselves at import time.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from muse.core.resolvers import (
    Resolver,
    ResolvedModel,
    SearchResult,
    ResolverError,
    register_resolver,
    get_resolver,
    resolve,
    search,
    _reset_registry_for_tests,
    parse_uri,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


class _FakeResolver(Resolver):
    scheme = "fake"

    def resolve(self, uri):
        return ResolvedModel(
            manifest={"model_id": "fake-model", "modality": "fake/type", "hf_repo": "x/y"},
            backend_path="muse.fake:FakeModel",
            download=lambda cache: cache / "fake",
        )

    def search(self, query, **filters):
        return [SearchResult(
            uri="fake://a/b", model_id="a-b", modality="fake/type",
            size_gb=0.5, downloads=10, license=None, description=None,
        )]


def test_register_and_get_resolver():
    r = _FakeResolver()
    register_resolver(r)
    assert get_resolver("fake://anything") is r


def test_get_resolver_raises_for_unknown_scheme():
    with pytest.raises(ResolverError, match="no resolver for scheme"):
        get_resolver("unknown://whatever")


def test_get_resolver_raises_for_non_uri():
    with pytest.raises(ResolverError, match="not a resolver URI"):
        get_resolver("just-a-bare-id")


def test_resolve_dispatches_by_scheme():
    register_resolver(_FakeResolver())
    rm = resolve("fake://a/b")
    assert isinstance(rm, ResolvedModel)
    assert rm.manifest["model_id"] == "fake-model"


def test_search_dispatches_to_named_backend():
    register_resolver(_FakeResolver())
    results = list(search("anything", backend="fake"))
    assert len(results) == 1
    assert results[0].uri == "fake://a/b"


def test_search_defaults_to_only_registered_backend():
    """If exactly one resolver is registered, search defaults to it."""
    register_resolver(_FakeResolver())
    results = list(search("q"))  # no backend kwarg
    assert len(results) == 1


def test_search_raises_when_multiple_and_no_backend():
    class _Other(Resolver):
        scheme = "other"
        def resolve(self, uri): raise NotImplementedError
        def search(self, q, **k): return []
    register_resolver(_FakeResolver())
    register_resolver(_Other())
    with pytest.raises(ResolverError, match="multiple resolvers"):
        list(search("q"))


def test_parse_uri_splits_scheme_ref_and_variant():
    scheme, ref, variant = parse_uri("hf://Qwen/Qwen3-8B-GGUF@q4_k_m")
    assert scheme == "hf"
    assert ref == "Qwen/Qwen3-8B-GGUF"
    assert variant == "q4_k_m"


def test_parse_uri_handles_missing_variant():
    scheme, ref, variant = parse_uri("hf://org/repo")
    assert scheme == "hf"
    assert ref == "org/repo"
    assert variant is None


def test_parse_uri_rejects_bad_scheme():
    with pytest.raises(ResolverError, match="not a resolver URI"):
        parse_uri("bare-id")
