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


def test_search_raises_helpful_message_when_no_resolvers_registered():
    """An empty registry (zero resolvers registered) must not claim
    "multiple resolvers registered []" -- that message is misleading
    when there are actually zero. Reserve the disambiguation message
    for len > 1."""
    # _clean_registry autouse fixture guarantees an empty registry here.
    with pytest.raises(ResolverError, match="no resolvers registered"):
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


class _BaseAwareResolver(Resolver):
    """A resolver whose resolve() accepts base_override (I2 opt-in shape)."""

    scheme = "based"

    def resolve(self, uri, *, base_override=None):
        return ResolvedModel(
            manifest={
                "model_id": "based-model",
                "modality": "fake/type",
                "hf_repo": "x/y",
                "capabilities": {"base_model": base_override},
            },
            backend_path="muse.fake:FakeModel",
            download=lambda cache: cache / "based",
        )

    def search(self, query, **filters):
        return []


def test_resolve_forwards_base_override_to_signature_accepting_resolver():
    """I2: resolve() inspects the target resolver's signature and only
    forwards base_override when the resolver actually declares the
    kwarg, so the 15+ modality plugins with a plain resolve(uri) keep
    working untouched."""
    register_resolver(_BaseAwareResolver())
    rm = resolve("based://a/b", base_override="sdxl-turbo")
    assert rm.manifest["capabilities"]["base_model"] == "sdxl-turbo"


def test_resolve_does_not_forward_base_override_to_signature_lacking_resolver():
    """A resolver without a base_override parameter must not be called
    with it (that would raise TypeError); resolve() must guard via
    inspect.signature rather than blindly passing the kwarg through."""
    register_resolver(_FakeResolver())
    # No TypeError: base_override is silently dropped for resolvers
    # that don't accept it.
    rm = resolve("fake://a/b", base_override="sdxl-turbo")
    assert rm.manifest["model_id"] == "fake-model"


def test_resolve_base_override_none_is_harmless_for_plain_resolver():
    register_resolver(_FakeResolver())
    rm = resolve("fake://a/b")
    assert rm.manifest["model_id"] == "fake-model"


def test_resolve_rejects_reviewed_revision_when_resolver_cannot_accept_it():
    register_resolver(_FakeResolver())

    with pytest.raises(ResolverError, match="does not accept.*revision"):
        resolve("fake://a/b", revision="1" * 40)


def test_resolve_forwards_reviewed_revision_to_opted_in_resolver():
    class _RevisionAwareResolver(_FakeResolver):
        scheme = "revisioned"

        def resolve(self, uri, *, revision=None):
            result = super().resolve(uri)
            result.manifest["revision"] = revision
            return result

    register_resolver(_RevisionAwareResolver())

    result = resolve("revisioned://a/b", revision="1" * 40)

    assert result.manifest["revision"] == "1" * 40
