"""GGUF chat-format hint lookup.

Reads `src/muse/chat_formats.yaml` (a hand-edited table shipped with the
package) and exposes one function:

  lookup_chat_format(hf_repo) -> dict | None

Used by the HF resolver to pre-populate `capabilities.chat_format` and
`capabilities.supports_tools` in synthesized GGUF manifests, so users
don't have to figure out llama-cpp-python's chat_format strings
themselves. Defensive: missing or malformed YAML returns `None`
universally; never refuses to start.
"""
from __future__ import annotations

import logging
from importlib.resources import files
from typing import Any

import yaml


logger = logging.getLogger(__name__)


_CACHE: list[dict[str, Any]] | None = None


def _yaml_path():
    return files("muse").joinpath("chat_formats.yaml")


def _load() -> list[dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        text = _yaml_path().read_text()
        raw = yaml.safe_load(text) or []
    except FileNotFoundError:
        logger.debug("chat_formats.yaml missing; no chat-format hints available")
        _CACHE = []
        return _CACHE
    except Exception as e:  # noqa: BLE001
        logger.warning("chat_formats.yaml unreadable: %s", e)
        _CACHE = []
        return _CACHE

    if not isinstance(raw, list):
        logger.warning(
            "chat_formats.yaml: top-level must be a list, got %s",
            type(raw).__name__,
        )
        _CACHE = []
        return _CACHE

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            logger.warning("chat_formats.yaml row %d is not a mapping; skipping", i)
            continue
        if "pattern" not in row:
            logger.warning("chat_formats.yaml row %d missing 'pattern'; skipping", i)
            continue
        rows.append(row)
    _CACHE = rows
    return _CACHE


def lookup_chat_format(hf_repo: str) -> dict | None:
    """Return chat-format hints for a HF repo, or None if no pattern matches.

    Pattern matching is case-insensitive substring against the lowercased
    `hf_repo`. First match wins (preserve order in YAML to make more-
    specific patterns shadow more-general ones).

    Returns a dict with whatever fields the matched row had (e.g.
    `{"chat_format": "chatml-function-calling", "supports_tools": true}`)
    minus the `pattern` key itself.
    """
    needle = hf_repo.lower()
    for row in _load():
        pattern = str(row["pattern"]).lower()
        if pattern in needle:
            out = {k: v for k, v in row.items() if k != "pattern"}
            return out
    return None


def _reset_cache_for_tests() -> None:
    global _CACHE
    _CACHE = None
