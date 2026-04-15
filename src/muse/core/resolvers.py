"""Resolver abstraction: URIs in, synthesized model records out.

A Resolver translates a URI like `hf://Qwen/Qwen3-8B-GGUF@q4_k_m` into a
ResolvedModel (synthesized manifest + backend class path + downloader
function). Resolvers also expose `search(query, **filters)` for model
discovery across their backing source (e.g. HuggingFace Hub).

Design goals:
 - Pluggable: register_resolver(instance) at import time from submodules.
 - Dispatching: `resolve(uri)` / `search(query, backend=...)` find the
   right resolver and forward.
 - Stateless: resolvers hold configuration but no per-call state.

The resolver output feeds directly into the existing pull path:
 - manifest -> catalog.json persisted alongside normal pull state
 - backend_path -> load_backend() imports and instantiates
 - download(cache_dir) -> fetches weights to a local directory
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


logger = logging.getLogger(__name__)


class ResolverError(Exception):
    """Raised when resolution or dispatch fails."""


@dataclass
class ResolvedModel:
    """Output of Resolver.resolve().

    Fields:
      - manifest: dict with at minimum `model_id`, `modality`, `hf_repo`
        keys, shaped like a MANIFEST in a model script. Flows into
        catalog.json and registry manifest passthrough.
      - backend_path: "module.path:ClassName" for load_backend(). The
        class must accept (hf_repo, local_dir, **kwargs) in its
        constructor, same protocol as scripted models.
      - download: callable that takes a cache directory and returns the
        path to the downloaded weights. Called during `pull`. Allows
        each resolver to control download semantics (snapshot_download,
        single-file download, etc.).
    """
    manifest: dict
    backend_path: str
    download: Callable[[Path], Path]


@dataclass
class SearchResult:
    """One candidate model returned from `Resolver.search`.

    Fields mirror what a user sees in a table listing. All optional
    fields may be None when the backend doesn't surface the data.
    """
    uri: str
    model_id: str
    modality: str
    size_gb: float | None = None
    downloads: int | None = None
    license: str | None = None
    description: str | None = None
    metadata: dict = field(default_factory=dict)


class Resolver(ABC):
    """Abstract resolver for a URI scheme."""

    scheme: str  # subclasses MUST set, e.g. "hf"

    @abstractmethod
    def resolve(self, uri: str) -> ResolvedModel:
        """Translate a URI into a ResolvedModel."""

    @abstractmethod
    def search(self, query: str, **filters: Any) -> Iterable[SearchResult]:
        """Search the backend for candidate models."""


_RESOLVERS: dict[str, Resolver] = {}


def register_resolver(resolver: Resolver) -> None:
    """Register a resolver instance under its scheme.

    Re-registration of the same scheme is allowed (overwrites); this
    simplifies test fixtures and future escape-hatch env-var overrides.
    """
    _RESOLVERS[resolver.scheme] = resolver


def _reset_registry_for_tests() -> None:
    """Test hook: clear all registered resolvers."""
    _RESOLVERS.clear()


def parse_uri(uri: str) -> tuple[str, str, str | None]:
    """Split `scheme://ref[@variant]` into (scheme, ref, variant | None).

    Raises ResolverError if the input has no `://` separator.
    """
    if "://" not in uri:
        raise ResolverError(f"not a resolver URI: {uri!r}")
    scheme, rest = uri.split("://", 1)
    if "@" in rest:
        ref, variant = rest.rsplit("@", 1)
    else:
        ref, variant = rest, None
    return scheme, ref, variant


def get_resolver(uri: str) -> Resolver:
    """Return the resolver registered for `uri`'s scheme."""
    scheme, _, _ = parse_uri(uri)
    try:
        return _RESOLVERS[scheme]
    except KeyError:
        raise ResolverError(
            f"no resolver for scheme {scheme!r}; "
            f"registered: {sorted(_RESOLVERS)}"
        )


def resolve(uri: str) -> ResolvedModel:
    """Resolve a URI through the matching resolver."""
    return get_resolver(uri).resolve(uri)


def search(query: str, *, backend: str | None = None, **filters: Any) -> Iterable[SearchResult]:
    """Search one backend (or the only-registered backend) for candidates.

    `backend` is the resolver scheme (e.g. "hf"). When omitted and
    exactly one resolver is registered, that one is used. When omitted
    and multiple are registered, raises ResolverError asking the caller
    to pick.
    """
    if backend is None:
        if len(_RESOLVERS) == 1:
            backend = next(iter(_RESOLVERS))
        else:
            raise ResolverError(
                f"multiple resolvers registered {sorted(_RESOLVERS)!r}; "
                f"pass backend= to disambiguate"
            )
    if backend not in _RESOLVERS:
        raise ResolverError(
            f"no resolver registered for backend {backend!r}; "
            f"registered: {sorted(_RESOLVERS)}"
        )
    return _RESOLVERS[backend].search(query, **filters)
