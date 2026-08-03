"""Node-membership model for the muse federation coordinator.

A "node" is a remote muse `serve` instance the coordinator forwards
requests to. This module is intentionally light: stdlib + `yaml` only
(no torch, no fastapi), so it can be imported early without pulling in
heavy ML deps.

Two node sources merge into one list:
  - CLI entries: plain URLs (`"http://host:8000"`) or named entries
    (`"name=http://host:8000"`).
  - A yaml file with shape `nodes: [{url, name?, token?}, ...]`.

Both sources are merged and deduped by normalized url; the first
occurrence wins (CLI entries take precedence over the yaml file).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml


def _normalize_url(url: object) -> str:
    """Validate and normalize one HTTP(S) Muse node base URL.

    Node configuration is later used as a forwarding destination, so fail
    closed on ambiguous strings instead of allowing httpx to reinterpret
    userinfo, fragments, or query parameters at request time.  A path prefix
    remains supported for reverse-proxy deployments.
    """
    if not isinstance(url, str) or not url or url != url.strip():
        raise ValueError("federation node url must be a non-empty trimmed string")
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError("federation node url contains control characters")
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            f"federation node url must be an absolute http(s) URL with a host: {url!r}"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("federation node url must not contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError("federation node url must not contain a query or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"federation node url has an invalid port: {url!r}") from exc
    if port == 0:
        raise ValueError(f"federation node url has an invalid port: {url!r}")
    return url.rstrip("/")


def _default_name(url: str) -> str:
    """Derive a default node name from the url's host, falling back to
    the url itself when it has no parseable hostname."""
    return urlparse(url).hostname or url


def _normalize_name(name: object | None, *, url: str) -> str:
    if name is None:
        return _default_name(url)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("federation node name must be a non-empty string")
    if name != name.strip() or any(
        ord(char) < 32 or ord(char) == 127 for char in name
    ):
        raise ValueError("federation node name contains unsafe whitespace")
    return name


def _normalize_token(token: object | None) -> str | None:
    if token is None or token == "":
        return None
    if not isinstance(token, str):
        raise ValueError("federation node token must be a string or null")
    if token != token.strip() or any(
        ord(char) < 32 or ord(char) == 127 for char in token
    ):
        raise ValueError("federation node token contains unsafe whitespace")
    return token


@dataclass(frozen=True)
class NodeSpec:
    url: str
    name: str
    token: str | None = None


def _node_from_cli_entry(entry: str) -> NodeSpec:
    """Parse one CLI node entry: 'http://h:8000' or 'name=http://h:8000'."""
    if not isinstance(entry, str):
        raise ValueError("federation CLI node entry must be a string")
    name, sep, rest = entry.partition("=")
    if sep and "://" in rest:
        url = _normalize_url(rest)
        return NodeSpec(
            url=url,
            name=_normalize_name(name, url=url),
            token=None,
        )
    url = _normalize_url(entry)
    return NodeSpec(url=url, name=_normalize_name(None, url=url), token=None)


def _nodes_from_yaml(config_path: str | Path) -> list[NodeSpec]:
    path = Path(config_path)
    try:
        text = path.read_text()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return []
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        return []
    entries = data.get("nodes", [])
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("federation config 'nodes' must be a list")
    nodes: list[NodeSpec] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "url" not in entry:
            raise ValueError(
                f"federation node entry {index} must be an object with a url"
            )
        url = _normalize_url(entry["url"])
        name = _normalize_name(entry.get("name"), url=url)
        token = _normalize_token(entry.get("token"))
        nodes.append(NodeSpec(url=url, name=name, token=token))
    return nodes


def load_nodes(
    cli_nodes: list[str] | None = None,
    config_path: str | Path | None = None,
) -> list[NodeSpec]:
    """Merge CLI-provided node entries with a yaml node-list file.

    Dedup by normalized url; the first occurrence wins, so CLI entries
    take precedence over entries from the yaml file with the same url.
    """
    nodes: list[NodeSpec] = []
    for entry in cli_nodes or []:
        nodes.append(_node_from_cli_entry(entry))
    if config_path is not None:
        nodes.extend(_nodes_from_yaml(config_path))

    seen: set[str] = set()
    deduped: list[NodeSpec] = []
    for node in nodes:
        if node.url in seen:
            continue
        seen.add(node.url)
        deduped.append(node)
    return deduped
