"""Decode user-supplied images for img2img on /v1/images/generations.

Accepted shapes:
  - data:image/{png,jpeg,webp};base64,XYZ
  - https?://... (fetched via httpx, content-type validated, size-capped,
    SSRF-protected: hostnames that resolve to private/loopback/link-local
    IPs are refused unless MUSE_ALLOW_PRIVATE_FETCH=1)

Returns PIL.Image. Failures raise ValueError so the route layer can
surface them as 400 responses with the OpenAI-shape error envelope.

PIL is already a pip_extra of the diffusers runtime. httpx is a
server-extras dep.
"""
from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any

from muse.core import config
from muse.core.net_fetch import (
    afetch_url_bytes,
    validate_public_host as _validate_public_host_impl,
)


logger = logging.getLogger(__name__)


_DATA_URL_RE = re.compile(
    r"^data:([a-zA-Z0-9.+/-]+);base64,(.*)$",
    re.DOTALL,
)
_ALLOWED_IMAGE_MIME = frozenset({
    "image/png", "image/jpeg", "image/jpg", "image/webp",
})
_HARD_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
_HTTP_TIMEOUT = 30.0


def _default_max_bytes() -> int:
    """Read the image-input byte cap via muse.core.config (env:
    MUSE_IMAGE_INPUT_MAX_BYTES), so operators can raise the cap without
    a server restart. config.get() already warns and falls back to the
    registry default on an unparseable value; a resolved value that is
    zero or negative (parseable but nonsensical as a byte cap) also
    falls back to the hardcoded default here, since the registry has no
    notion of "must be positive"."""
    n = config.get("limits.image_input_max_bytes")
    if n is None or n <= 0:
        return _HARD_DEFAULT_MAX_BYTES
    return n


async def decode_image_input(value: str, *, max_bytes: int | None = None) -> Any:
    """Parse a data URL or HTTP(S) URL into a PIL.Image.

    Async because the http path uses `httpx.AsyncClient` and must not
    block the event loop. The data-URL path is in-memory and stays sync
    internally; awaiting an already-resolved coroutine is cheap.

    `max_bytes` may be passed by the caller (e.g. an upscale route that
    wants a larger cap) or left None to read MUSE_IMAGE_INPUT_MAX_BYTES
    per request.
    """
    cap = max_bytes if max_bytes is not None else _default_max_bytes()
    if value.startswith("data:"):
        return _decode_data_url(value, max_bytes=cap)
    if value.startswith(("http://", "https://")):
        return await _fetch_http_url(value, max_bytes=cap)
    raise ValueError(
        f"image must be a data: URL or http(s):// URL; got {value[:30]!r}..."
    )


async def decode_image_file(
    file: Any,
    *,
    max_bytes: int | None = None,
    max_side: int | None = None,
) -> Any:
    """Decode a multipart UploadFile into a PIL.Image.

    Used by /v1/images/edits and /v1/images/variations, where the
    image (and mask) arrive as multipart/form-data file uploads rather
    than as data URLs in a JSON body. Mirrors the validation discipline
    of decode_image_input: empty file rejected, oversized rejected,
    undecodable bytes rejected. Failures raise ValueError so the route
    layer can surface them as 400 responses with the OpenAI-shape error
    envelope.

    The argument is typed Any so this helper doesn't bind us to FastAPI
    or Starlette at import time. Anything with an async .read() method
    works (UploadFile, plus any test double).

    Reads at most `max_bytes + 1` so a malicious giant upload doesn't
    fully buffer into worker memory before being rejected.

    `max_bytes` defaults to MUSE_IMAGE_INPUT_MAX_BYTES (or 10MB).
    When `max_side` is set, declared dimensions are checked before pixel
    decoding so an oversized image is rejected without allocating its
    full raster.
    """
    cap = max_bytes if max_bytes is not None else _default_max_bytes()
    raw = await file.read(cap + 1)
    if not raw:
        raise ValueError("empty image file")
    if len(raw) > cap:
        raise ValueError(
            f"image bytes exceeds max ({len(raw)} > {cap})"
        )
    return _bytes_to_pil(raw, max_side=max_side)


def _decode_data_url(value: str, *, max_bytes: int):
    m = _DATA_URL_RE.match(value)
    if not m:
        raise ValueError("malformed data URL")
    mime = m.group(1).lower()
    if mime not in _ALLOWED_IMAGE_MIME:
        raise ValueError(
            f"unsupported MIME {mime!r}; allowed: {sorted(_ALLOWED_IMAGE_MIME)}"
        )
    b64 = m.group(2)
    # Reject before decoding: base64 inflates ~4/3, so any b64 string
    # larger than (max_bytes * 4 / 3) cannot fit under the cap. Adding 4
    # for padding slack avoids false-positives on exact-fit boundaries.
    max_b64_len = (max_bytes * 4) // 3 + 4
    if len(b64) > max_b64_len:
        raise ValueError(
            f"image bytes exceeds max ({len(b64)} b64 chars > {max_b64_len} limit)"
        )
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"base64 decode failed: {e}") from e
    if len(raw) > max_bytes:
        raise ValueError(
            f"image bytes exceeds max ({len(raw)} > {max_bytes})"
        )
    return _bytes_to_pil(raw)


async def _fetch_http_url(value: str, *, max_bytes: int):
    try:
        raw = await afetch_url_bytes(value, max_bytes=max_bytes, timeout=_HTTP_TIMEOUT)
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"fetch failed: {e}") from e
    return _bytes_to_pil(raw)


def _validate_public_host(url: str) -> None:
    """Thin alias for muse.core.net_fetch.validate_public_host.

    Preserved for backward compatibility: existing tests and any external
    callers that reference
    ``muse.modalities.image_generation.image_input._validate_public_host``
    continue to work. The real implementation lives in
    ``muse.core.net_fetch.validate_public_host``.
    """
    _validate_public_host_impl(url)


def _bytes_to_pil(raw: bytes, *, max_side: int | None = None):
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(raw))
        width, height = img.size
        if max_side is not None and (
            width > max_side or height > max_side
        ):
            raise ValueError(
                f"image too large: {width}x{height} exceeds max input side "
                f"{max_side}"
            )
        img.load()  # force decode now so errors surface here, not later
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        # DecompressionBombError is a plain Exception (not OSError): a tiny
        # PNG declaring huge dimensions would otherwise escape as a 500. PIL
        # still blocks the memory DoS; we only reclassify it as a client
        # error (ValueError -> 400).
        raise ValueError(f"image decode failed: {e}") from e
    return img
