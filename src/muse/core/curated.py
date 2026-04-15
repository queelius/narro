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

The list is loaded once at import and cached. Restart muse to pick up
edits to the YAML (matches the rest of muse's "static at startup"
discovery model).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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


_CURATED_CACHE: list[CuratedEntry] | None = None


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
    return CuratedEntry(
        id=d["id"],
        bundled=bundled,
        uri=uri,
        modality=d.get("modality"),
        size_gb=d.get("size_gb"),
        description=d.get("description"),
        tags=tuple(d.get("tags", ())),
    )


def find_curated(model_id: str) -> CuratedEntry | None:
    """Return the curated entry with this id, or None."""
    for e in load_curated():
        if e.id == model_id:
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
