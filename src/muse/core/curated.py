"""Curated recommendations: opinionated newbie list surfaced in `muse models list`.

Reads `src/muse/curated.yaml` (a hand-edited file shipped with the
package) and exposes helpers for the CLI:

  load_curated() -> list[CuratedEntry]
  find_curated(id) -> CuratedEntry | None
  expand_curated_pull(id) -> str | None  # returns URI or bundled-id

Two entry shapes:
  - Resolver entry: has `uri` field (e.g. "hf://Qwen/Qwen3-8B-GGUF@q4_k_m").
    All metadata (modality, size_gb, description) comes from the YAML.
  - Bundled entry: has `bundled: true`. The `id` must match an existing
    bundled script's model_id; metadata is read from that script's
    MANIFEST at display time.

Optional `capabilities:` mapping on either shape: runtime-specific
overrides (e.g. `trust_remote_code: true`, `chat_format: "..."`,
`context_length`) that merge into the resolver-synthesized manifest at
pull time. See `catalog._pull_via_resolver` for merge semantics (overlay
wins on key collision). Only applied to resolver entries; ignored for
bundled entries since those carry their own MANIFEST.

Resolver entries that enable `trust_remote_code` must also declare a full
immutable `revision`; `code_revision` pins an external repository referenced
by Transformers `auto_map`.

The list is loaded once at import and cached. Restart muse to pick up
edits to the YAML (matches the rest of muse's "static at startup"
discovery model).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import yaml


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CuratedEntry:
    """One row in the curated recommendations YAML."""
    id: str
    bundled: bool
    uri: str | None
    modality: str | None
    size_gb: float | None
    description: str | None
    tags: tuple[str, ...]
    # Runtime-specific overrides that merge into the resolver-synthesized
    # manifest at pull time (shallow merge; overlay wins on collision).
    # See catalog._pull_via_resolver.
    capabilities: dict = field(default_factory=dict)
    # Immutable Hugging Face commits selected during curation.  `revision`
    # pins model artifacts; `code_revision` separately pins an external
    # repository named by a Transformers auto_map (for example Nomic BERT).
    revision: str | None = None
    code_revision: str | None = None


_CURATED_CACHE: list[CuratedEntry] | None = None
_CONCRETE_HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _curated_yaml_path():
    """Locate src/muse/curated.yaml as a package resource."""
    return files("muse").joinpath("curated.yaml")


def load_curated() -> list[CuratedEntry]:
    """Return the curated list. Cached after first call.

    Returns [] if the YAML is missing or malformed (logs a warning;
    discovery should never refuse to start because of a bad curated file).
    """
    global _CURATED_CACHE
    if _CURATED_CACHE is not None:
        return _CURATED_CACHE
    try:
        text = _curated_yaml_path().read_text()
        raw = yaml.safe_load(text) or []
    except FileNotFoundError:
        logger.debug("no curated.yaml; skipping recommendations")
        _CURATED_CACHE = []
        return _CURATED_CACHE
    except Exception as e:  # noqa: BLE001
        logger.warning("curated.yaml could not be loaded: %s", e)
        _CURATED_CACHE = []
        return _CURATED_CACHE

    if not isinstance(raw, list):
        logger.warning("curated.yaml: top-level must be a list, got %s", type(raw).__name__)
        _CURATED_CACHE = []
        return _CURATED_CACHE

    entries: list[CuratedEntry] = []
    for i, raw_entry in enumerate(raw):
        if not isinstance(raw_entry, dict):
            logger.warning("curated.yaml entry %d is not a mapping; skipping", i)
            continue
        try:
            entry = _entry_from_dict(raw_entry)
        except ValueError as e:
            logger.warning("curated.yaml entry %d invalid: %s", i, e)
            continue
        entries.append(entry)
    _CURATED_CACHE = entries
    return _CURATED_CACHE


def _entry_from_dict(d: dict) -> CuratedEntry:
    """Validate + project a dict from YAML onto CuratedEntry."""
    if "id" not in d:
        raise ValueError("missing required key 'id'")
    bundled = bool(d.get("bundled", False))
    uri = d.get("uri")
    if not bundled and not uri:
        raise ValueError(
            f"entry {d['id']!r}: must set either 'uri' (resolver) "
            "or 'bundled: true' (script alias)"
        )
    if bundled and uri:
        raise ValueError(
            f"entry {d['id']!r}: cannot set both 'uri' and 'bundled: true'"
        )
    caps = d.get("capabilities", {})
    if not isinstance(caps, dict):
        raise ValueError(
            f"entry {d['id']!r}: 'capabilities' must be a mapping, got {type(caps).__name__}"
        )
    revision = d.get("revision")
    code_revision = d.get("code_revision")
    for field_name, value in (
        ("revision", revision),
        ("code_revision", code_revision),
    ):
        if value is not None and (
            not isinstance(value, str)
            or not _CONCRETE_HF_REVISION_RE.fullmatch(value)
        ):
            raise ValueError(
                f"entry {d['id']!r}: '{field_name}' must be a full "
                "40-character lowercase commit SHA"
            )
    if caps.get("trust_remote_code") and uri and revision is None:
        raise ValueError(
            f"entry {d['id']!r}: trust_remote_code requires a concrete "
            "reviewed 'revision'"
        )
    if code_revision is not None and not caps.get("trust_remote_code"):
        raise ValueError(
            f"entry {d['id']!r}: 'code_revision' requires "
            "capabilities.trust_remote_code: true"
        )
    return CuratedEntry(
        id=d["id"],
        bundled=bundled,
        uri=uri,
        modality=d.get("modality"),
        size_gb=d.get("size_gb"),
        description=d.get("description"),
        tags=tuple(d.get("tags", ())),
        capabilities=dict(caps),
        revision=revision,
        code_revision=code_revision,
    )


def all_curated() -> list[CuratedEntry]:
    """Return every curated entry. Thin alias for load_curated().

    Prefer this over load_curated() in code that reads entries for
    iteration: the name makes intent explicit and lets callers avoid
    importing load_curated directly.
    """
    return load_curated()


def find_curated(model_id: str) -> CuratedEntry | None:
    """Return the curated entry with this id, or None."""
    for e in load_curated():
        if e.id == model_id:
            return e
    return None


def find_curated_by_uri(uri: str) -> CuratedEntry | None:
    """Return the curated entry whose `uri` matches, or None.

    Lets the URI-only pull path (`muse pull hf://...`) inherit any
    curated `capabilities` overlay (e.g. `safe_labels`, `chat_format`)
    that was set for the same upstream repo. Without this, pulling by
    curated id and pulling by raw URI produce subtly different catalog
    manifests for the same model, a footgun where curated metadata
    silently disappears the moment a user copies the URI from `muse
    search` output.
    """
    for e in load_curated():
        if e.uri and e.uri == uri:
            return e
    return None


def expand_curated_pull(identifier: str) -> str | None:
    """Map a curated id to whatever `pull()` should actually receive.

    Returns:
      - the URI (e.g. "hf://...") for resolver entries
      - the bundled script's model_id for bundled entries (which equals
        the curated id by convention)
      - None if `identifier` is not a curated id

    `pull()` calls this first; if non-None, the original identifier was
    a curated alias and we substitute the underlying target.
    """
    entry = find_curated(identifier)
    if entry is None:
        return None
    return entry.uri if entry.uri else entry.id


def _reset_curated_cache_for_tests() -> None:
    """Test hook: clear the cache so reload picks up monkey-patched paths."""
    global _CURATED_CACHE
    _CURATED_CACHE = None
