"""Binary I/O helpers for MCP tool args and responses.

Inputs accept three alternative fields per binary slot:
  - <name>_b64:  base64-encoded bytes
  - <name>_url:  data: or http(s) URL
  - <name>_path: local filesystem path

Exactly one must be present. ``resolve_binary_input`` returns raw bytes
or raises ``ValueError`` with a structured message the MCP server
forwards to the LLM client (so it can correct itself on the next turn).

Outputs use MCP content blocks (TextContent + ImageContent +
AudioContent). The pack helpers build plain dicts; the MCPServer
converts those to SDK type instances at the call boundary so this
module stays import-safe without the SDK installed.

Security hardening (v0.45.6):
  - URL inputs route through muse.core.net_fetch.fetch_url_bytes:
      * SSRF guard: host must resolve to a public IP (C1)
      * Size cap:   MUSE_IMAGE_INPUT_MAX_BYTES (default 10MB), streamed
                    so no full body is buffered before the cap fires (C3)
      * Open-redirect protection: each redirect hop re-validates the
        Location header's host (H7, shared with image routes)
  - Path inputs are gated by MUSE_MCP_ALLOWED_PATH_PREFIXES (C2):
      * Unset (default): all path inputs raise ValueError.
      * Set: the path is realpath()'d to defeat ../ traversal and
        symlink escapes, then checked against the realpath()'d prefixes.
  - Base64 and path inputs use the same byte cap as URL inputs; base64 is
    rejected by encoded length before decoding and files are read at most
    ``cap + 1`` bytes.
"""
from __future__ import annotations

import base64
import os
import stat
from typing import Any

from muse.core import config
from muse.core.net_fetch import fetch_url_bytes
from muse.modalities.image_generation.image_input import _default_max_bytes


def _enforce_size_cap(data: bytes, *, field_name: str, slot: str) -> bytes:
    """Reject a decoded blob larger than the shared image-input byte cap.

    The http(s) URL branch already enforces this cap via
    ``fetch_url_bytes`` (streamed, so oversized bodies never fully
    buffer). The b64 and data: URL branches decode eagerly with
    ``base64.b64decode`` first, so the cap has to be applied to the
    decoded length after the fact -- still a bounded amount of memory
    per request, but bounded is the point.
    """
    cap = _default_max_bytes()
    if len(data) > cap:
        raise ValueError(
            f"{field_name}_{slot} decodes to {len(data)} bytes, which "
            f"exceeds the max of {cap} bytes "
            f"(set MUSE_IMAGE_INPUT_MAX_BYTES to raise it)"
        )
    return data


def _decode_base64_bounded(
    encoded: str,
    *,
    field_name: str,
    slot: str,
) -> bytes:
    """Strictly decode base64 without allocating beyond the shared cap."""
    cap = _default_max_bytes()
    max_encoded = (cap * 4) // 3 + 4
    if len(encoded) > max_encoded:
        raise ValueError(
            f"{field_name}_{slot} exceeds the max encoded length "
            f"({len(encoded)} > {max_encoded})"
        )
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"malformed base64 in {field_name}_{slot}: {e}"
        ) from e
    return _enforce_size_cap(decoded, field_name=field_name, slot=slot)


def _normalize_input_slot(
    value: str | None,
    *,
    field_name: str,
    slot: str,
) -> str | None:
    """Validate one JSON-facing binary slot and normalize ``""`` away."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name}_{slot} must be a string, got "
            f"{type(value).__name__}"
        )
    return value or None


def resolve_binary_input(
    *,
    b64: str | None = None,
    url: str | None = None,
    path: str | None = None,
    field_name: str = "image",
) -> bytes:
    """Resolve a tri-modal binary input to raw bytes.

    Exactly one of ``b64``, ``url``, ``path`` must be provided. Anything
    else raises ``ValueError`` so the MCP server reports a structured
    error the LLM can correct from on retry.

    Security:
      - URL inputs are SSRF-protected and size-capped via net_fetch.
      - Path inputs require MUSE_MCP_ALLOWED_PATH_PREFIXES to be set.
    """
    # Normalize empty-string slots to None so the "exactly one" guard and
    # the dispatch below agree on what "absent" means. An LLM that leaves a
    # slot blank sends "" (falsy) rather than omitting it; without this an
    # empty b64="" would pass the truthiness guard yet be dispatched by the
    # `is not None` check, silently decoding b"" and dropping a real URL.
    b64 = _normalize_input_slot(
        b64, field_name=field_name, slot="b64",
    )
    url = _normalize_input_slot(
        url, field_name=field_name, slot="url",
    )
    path = _normalize_input_slot(
        path, field_name=field_name, slot="path",
    )
    provided = [k for k, v in (("b64", b64), ("url", url), ("path", path)) if v]
    if len(provided) == 0:
        raise ValueError(
            f"missing {field_name} input: provide exactly one of "
            f"{field_name}_b64 (base64), {field_name}_url (URL), or "
            f"{field_name}_path (local file path)"
        )
    if len(provided) > 1:
        raise ValueError(
            f"too many {field_name} inputs: provided {provided}; "
            f"provide exactly one"
        )
    if b64 is not None:
        # Strip a leading data: prefix if the LLM included it.
        if b64.startswith("data:"):
            comma = b64.find(",")
            if comma == -1:
                raise ValueError(f"malformed data URL in {field_name}_b64")
            b64 = b64[comma + 1:]
        return _decode_base64_bounded(
            b64, field_name=field_name, slot="b64",
        )
    if url is not None:
        if url.startswith("data:"):
            comma = url.find(",")
            if comma == -1:
                raise ValueError(f"malformed data URL in {field_name}_url")
            return _decode_base64_bounded(
                url[comma + 1:], field_name=field_name, slot="url",
            )
        if url.startswith(("http://", "https://")):
            # Route through the hardened primitive: SSRF guard, per-hop
            # redirect re-validation, and streaming size cap.
            cap = _default_max_bytes()
            return fetch_url_bytes(url, max_bytes=cap)
        raise ValueError(
            f"unsupported {field_name}_url scheme: {url[:32]}; "
            f"expected http(s):// or data:"
        )
    if path is not None:
        return _resolve_path(path, field_name=field_name)
    raise AssertionError("unreachable")


def _resolve_path(path: str, *, field_name: str) -> bytes:
    """Read a local file, gated by MUSE_MCP_ALLOWED_PATH_PREFIXES.

    Raises ``ValueError`` when:
      - The env var is unset or empty (default-deny).
      - The realpath of ``path`` is not under any allowed prefix (catches
        ../ traversal and symlink escapes by resolving both sides before
        comparing).
      - The file does not exist.
    """
    raw_prefixes = config.get("fetch.mcp_allowed_path_prefixes").strip()
    if not raw_prefixes:
        raise ValueError(
            f"{field_name}_path input is disabled: set "
            "MUSE_MCP_ALLOWED_PATH_PREFIXES to a "
            + os.pathsep
            + "-separated list of allowed directory prefixes to enable it"
        )
    # Realpath both the prefixes and the requested path so ../ traversal
    # and symlinks cannot escape the allowlist.
    allowed = [
        os.path.realpath(p)
        for p in raw_prefixes.split(os.pathsep)
        if p.strip()
    ]
    real_path = os.path.realpath(path)
    within = any(
        real_path == prefix or real_path.startswith(prefix + os.sep)
        for prefix in allowed
    )
    if not within:
        raise ValueError(
            f"{field_name}_path {path!r} is not within an allowed prefix; "
            f"set MUSE_MCP_ALLOWED_PATH_PREFIXES to include its directory"
        )
    cap = _default_max_bytes()
    fd = -1
    try:
        fd = _open_absolute_regular_nofollow(real_path)
        with os.fdopen(fd, "rb") as f:
            fd = -1
            data = f.read(cap + 1)
    except FileNotFoundError as e:
        raise ValueError(f"{field_name}_path not found: {path}") from e
    except OSError as e:
        raise ValueError(f"could not read {field_name}_path {path!r}: {e}") from e
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > cap:
        raise ValueError(
            f"{field_name}_path exceeds the max of {cap} bytes "
            f"(set MUSE_IMAGE_INPUT_MAX_BYTES to raise it)"
        )
    return data


def _open_absolute_regular_nofollow(path: str) -> int:
    """Open an absolute path without following a symlink in any component.

    ``realpath`` plus a later builtin ``open`` has a race: an allowed parent
    can be replaced by a symlink between the check and the read.  Walk from a
    descriptor for ``/`` with ``openat``-style calls instead.  Every directory
    and the final file is opened no-follow, and the returned descriptor is
    required to name a regular file.
    """
    if not os.path.isabs(path):
        raise OSError("resolved path is not absolute")

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    parts = [part for part in path.split(os.sep) if part]
    if not parts:
        raise OSError("path does not name a regular file")

    current_fd = os.open(os.sep, os.O_RDONLY | directory | cloexec)
    try:
        for component in parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | directory | nofollow | cloexec,
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        result_fd = os.open(
            parts[-1],
            os.O_RDONLY | nofollow | cloexec,
            dir_fd=current_fd,
        )
        try:
            if not stat.S_ISREG(os.fstat(result_fd).st_mode):
                raise OSError("path does not name a regular file")
        except BaseException:
            os.close(result_fd)
            raise
        return result_fd
    finally:
        os.close(current_fd)


def binary_input_schema(field_name: str = "image") -> dict[str, Any]:
    """Build the JSON-Schema fragment for the three input fields.

    Caller merges this dict into its tool's ``properties``. The fields
    are mutually exclusive but JSON Schema's ``oneOf`` would force the
    LLM into a stricter shape than necessary; we describe the
    exclusivity in prose and validate at call time.
    """
    return {
        f"{field_name}_b64": {
            "type": "string",
            "description": (
                f"Base64-encoded {field_name} bytes. Provide exactly "
                f"one of {field_name}_b64, {field_name}_url, "
                f"{field_name}_path."
            ),
        },
        f"{field_name}_url": {
            "type": "string",
            "description": (
                f"URL to the {field_name} (data: or http(s)://). "
                f"Mutually exclusive with the other two."
            ),
        },
        f"{field_name}_path": {
            "type": "string",
            "description": (
                f"Local filesystem path to the {field_name}. Only "
                f"valid when the MCP server has access to that path "
                f"and MUSE_MCP_ALLOWED_PATH_PREFIXES is configured."
            ),
        },
    }


def pack_image_output(
    image_bytes: bytes,
    *,
    mime: str = "image/png",
) -> dict[str, Any]:
    """Build an MCP ImageContent dict for raw image bytes."""
    return {
        "type": "image",
        "data": base64.b64encode(image_bytes).decode("ascii"),
        "mimeType": mime,
    }


def pack_audio_output(
    audio_bytes: bytes,
    *,
    mime: str = "audio/wav",
) -> dict[str, Any]:
    """Build an MCP AudioContent dict for raw audio bytes."""
    return {
        "type": "audio",
        "data": base64.b64encode(audio_bytes).decode("ascii"),
        "mimeType": mime,
    }


def pack_text_output(text: str) -> dict[str, Any]:
    """Build an MCP TextContent dict for a string payload."""
    return {"type": "text", "text": text}
