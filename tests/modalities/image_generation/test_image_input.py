"""Tests for image_input: parsing user-supplied images for img2img.

The helper accepts either:
  - a data URL: data:image/{png,jpeg,webp};base64,...
  - an http(s):// URL fetched via httpx (size-capped, content-type-checked,
    SSRF-protected: hostname must resolve to a public IP unless
    MUSE_ALLOW_PRIVATE_FETCH=1 is set).

Returns a PIL.Image. Decode failures raise ValueError so the route layer
can surface them as 400s.

`decode_image_input` is async (the http path uses httpx.AsyncClient to
avoid blocking the event loop on the calling worker), so all tests
that touch it are marked asyncio.
"""
import base64
import io

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from muse.modalities.image_generation.image_input import decode_image_input


def _png_bytes(width=64, height=64, color=(0, 128, 255)):
    """Build minimal PNG bytes via PIL."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_decode_data_url_png():
    raw = _png_bytes()
    data_url = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    img = await decode_image_input(data_url)
    assert img.size == (64, 64)
    assert img.mode in ("RGB", "RGBA")


@pytest.mark.asyncio
async def test_decode_data_url_jpeg():
    from PIL import Image
    rgb = Image.new("RGB", (32, 32), (255, 0, 0))
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG")
    data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    img = await decode_image_input(data_url)
    assert img.size == (32, 32)


def _patch_afetch(content: bytes):
    """Patch afetch_url_bytes as imported into image_input to return ``content``.

    image_input imports afetch_url_bytes via ``from muse.core.net_fetch import
    afetch_url_bytes``, so the patch target is the name in image_input's own
    namespace, not the origin module.
    """
    return patch(
        "muse.modalities.image_generation.image_input.afetch_url_bytes",
        new=AsyncMock(return_value=content),
    )


# Keep _patch_dns pointing at the real implementation location.
def _patch_dns(public_ip: str = "8.8.8.8"):
    """Patch socket.gethostbyname in muse.core.net_fetch."""
    return patch(
        "muse.core.net_fetch.socket.gethostbyname",
        return_value=public_ip,
    )


@pytest.mark.asyncio
async def test_decode_http_url_fetches_via_net_fetch():
    raw = _png_bytes()
    with _patch_afetch(raw):
        img = await decode_image_input("https://example.com/cat.png")
    assert img.size == (64, 64)


@pytest.mark.asyncio
async def test_decode_rejects_oversize_data_url():
    huge = b"\x00" * (11 * 1024 * 1024)  # 11MB
    data_url = f"data:image/png;base64,{base64.b64encode(huge).decode()}"
    with pytest.raises(ValueError, match="exceeds"):
        await decode_image_input(data_url, max_bytes=10 * 1024 * 1024)


@pytest.mark.asyncio
async def test_decode_data_url_rejects_oversize_before_b64_decode(monkeypatch):
    """Pre-b64-decode size guard: a 1GB b64 string must reject without
    inflating into RAM."""
    huge_b64 = "A" * (15 * 1024 * 1024)  # 15MB of b64 chars
    data_url = f"data:image/png;base64,{huge_b64}"
    # If b64decode were called we'd allocate ~11MB, but the pre-check
    # should reject first. Spy on b64decode to confirm.
    from muse.modalities.image_generation import image_input
    spy = MagicMock(side_effect=AssertionError("b64decode reached unexpectedly"))
    monkeypatch.setattr(image_input.base64, "b64decode", spy)
    with pytest.raises(ValueError, match="exceeds"):
        await decode_image_input(data_url, max_bytes=10 * 1024 * 1024)
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_decode_rejects_non_image_http_response():
    """HTTP path returns raw bytes; PIL decode rejects non-image content."""
    with _patch_afetch(b"<html>nope</html>"):
        with pytest.raises(ValueError, match="decode"):
            await decode_image_input("https://example.com/page.html")


@pytest.mark.asyncio
async def test_decode_rejects_unknown_data_url_mime():
    raw = b"some text"
    data_url = f"data:text/plain;base64,{base64.b64encode(raw).decode()}"
    with pytest.raises(ValueError, match="MIME"):
        await decode_image_input(data_url)


@pytest.mark.asyncio
async def test_decode_rejects_invalid_url_shape():
    with pytest.raises(ValueError, match="must be"):
        await decode_image_input("ftp://example.com/img.png")


@pytest.mark.asyncio
async def test_decode_rejects_corrupt_image_bytes():
    raw = b"not really a png"
    data_url = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    with pytest.raises(ValueError, match="decode"):
        await decode_image_input(data_url)


def test_bytes_to_pil_rejects_decompression_bomb(monkeypatch):
    """A decompression bomb must surface as ValueError (-> 400), not the raw
    Image.DecompressionBombError (which would escape to a 500). PIL still
    blocks the DoS; we only fix the error class."""
    from PIL import Image

    from muse.modalities.image_generation import image_input

    # A modest real PNG whose pixel count trips a lowered bomb threshold.
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (1, 2, 3)).save(buf, format="PNG")
    raw = buf.getvalue()

    # DecompressionBombError fires when pixels > 2 * MAX_IMAGE_PIXELS.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="decode"):
        image_input._bytes_to_pil(raw)


# ---------------- SSRF: blocked on private/loopback/link-local ----------------


@pytest.mark.asyncio
async def test_fetch_blocks_loopback(monkeypatch):
    """A URL whose host resolves to 127.0.0.1 must reject without fetching.

    SSRF guard now lives in muse.core.net_fetch; image_input delegates to it
    via afetch_url_bytes. Patch DNS at net_fetch and confirm fetch is rejected
    before any network I/O reaches afetch_url_bytes's inner httpx call.
    """
    monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
    with _patch_dns("127.0.0.1"):
        with pytest.raises(ValueError, match="non-public"):
            await decode_image_input("https://attacker.example/payload.png")


@pytest.mark.asyncio
async def test_fetch_blocks_link_local_aws_metadata(monkeypatch):
    """The classic SSRF target: 169.254.169.254 (AWS instance metadata)."""
    monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
    with _patch_dns("169.254.169.254"):
        with pytest.raises(ValueError, match="non-public"):
            await decode_image_input("http://metadata.example/iam/")


@pytest.mark.asyncio
async def test_fetch_blocks_private_lan(monkeypatch):
    """Private RFC1918 ranges (10.x, 172.16-31.x, 192.168.x) reject."""
    monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
    with _patch_dns("192.168.0.5"):
        with pytest.raises(ValueError, match="non-public"):
            await decode_image_input("https://router.lan/admin.png")


@pytest.mark.asyncio
async def test_fetch_blocks_internal_10_range(monkeypatch):
    monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
    with _patch_dns("10.0.0.1"):
        with pytest.raises(ValueError, match="non-public"):
            await decode_image_input("https://internal.corp/x.png")


@pytest.mark.asyncio
async def test_fetch_allows_override_via_env(monkeypatch):
    """Operators on a trusted network can opt out via MUSE_ALLOW_PRIVATE_FETCH=1."""
    monkeypatch.setenv("MUSE_ALLOW_PRIVATE_FETCH", "1")
    raw = _png_bytes()
    # With MUSE_ALLOW_PRIVATE_FETCH=1 the DNS check is bypassed entirely;
    # patch afetch to return valid PNG bytes so PIL decode succeeds.
    with _patch_afetch(raw):
        img = await decode_image_input("http://localhost:9001/internal.png")
    assert img.size == (64, 64)


@pytest.mark.asyncio
async def test_fetch_rejects_url_without_hostname(monkeypatch):
    """A schemed-but-empty-host URL has no IP to resolve."""
    monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
    with pytest.raises(ValueError, match="hostname"):
        await decode_image_input("http:///no-host.png")


# ---------------- decode_image_file (multipart UploadFile) ----------------


class _FakeUploadFile:
    """Minimal async-file shape compatible with decode_image_file.

    Honors the size argument to .read() so the bounded-read fix is
    actually exercised. Records the requested size for assertions.
    """

    def __init__(self, data: bytes):
        self._data = data
        self.read_size: int | None = None

    async def read(self, size: int | None = None) -> bytes:
        self.read_size = size
        if size is None:
            return self._data
        return self._data[:size]


@pytest.mark.asyncio
async def test_decode_image_file_reads_png_from_upload():
    from muse.modalities.image_generation.image_input import decode_image_file
    raw = _png_bytes(width=48, height=24)
    upload = _FakeUploadFile(raw)
    img = await decode_image_file(upload)
    assert img.size == (48, 24)


@pytest.mark.asyncio
async def test_decode_image_file_rejects_side_before_pixel_decode(
    monkeypatch,
):
    from muse.modalities.image_generation.image_input import decode_image_file
    from PIL import Image

    raw = _png_bytes(width=48, height=24)
    upload = _FakeUploadFile(raw)
    load = Image.Image.load

    def fail_if_loaded(self):
        raise AssertionError("oversized image pixels should not be decoded")

    monkeypatch.setattr(Image.Image, "load", fail_if_loaded)
    with pytest.raises(ValueError, match="max input side 32"):
        await decode_image_file(upload, max_side=32)
    monkeypatch.setattr(Image.Image, "load", load)


@pytest.mark.asyncio
async def test_decode_image_file_rejects_empty():
    from muse.modalities.image_generation.image_input import decode_image_file
    upload = _FakeUploadFile(b"")
    with pytest.raises(ValueError, match="empty"):
        await decode_image_file(upload)


@pytest.mark.asyncio
async def test_decode_image_file_rejects_oversized():
    from muse.modalities.image_generation.image_input import decode_image_file
    huge = b"\x00" * (11 * 1024 * 1024)
    upload = _FakeUploadFile(huge)
    with pytest.raises(ValueError, match="exceeds"):
        await decode_image_file(upload, max_bytes=10 * 1024 * 1024)


@pytest.mark.asyncio
async def test_decode_image_file_uses_bounded_read():
    """The fix: read() is called with `max_bytes + 1`, not unbounded.

    A malicious 1GB upload must not buffer fully into worker RAM
    before the size check fires.
    """
    from muse.modalities.image_generation.image_input import decode_image_file
    # 11MB of zeroes; cap at 10MB. Read should request only the first
    # 10MB+1 bytes; oversize check fires after.
    huge = b"\x00" * (11 * 1024 * 1024)
    upload = _FakeUploadFile(huge)
    cap = 10 * 1024 * 1024
    with pytest.raises(ValueError, match="exceeds"):
        await decode_image_file(upload, max_bytes=cap)
    assert upload.read_size == cap + 1, (
        f"expected bounded read of {cap + 1}, got {upload.read_size}"
    )


@pytest.mark.asyncio
async def test_decode_image_file_rejects_undecodable():
    from muse.modalities.image_generation.image_input import decode_image_file
    upload = _FakeUploadFile(b"not really an image at all")
    with pytest.raises(ValueError, match="decode"):
        await decode_image_file(upload)


# ---------------- v0.34.0 finding #11: MUSE_IMAGE_INPUT_MAX_BYTES ----------------


@pytest.mark.asyncio
async def test_decode_image_file_honors_env_cap(monkeypatch):
    """Operators can raise the cap via MUSE_IMAGE_INPUT_MAX_BYTES
    without a server restart. A 13000-byte upload past a 12345-byte
    cap is rejected; the cap is read per-call so changes take effect
    immediately on the next request."""
    from muse.modalities.image_generation.image_input import decode_image_file
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "12345")
    payload = b"\x00" * 13000
    upload = _FakeUploadFile(payload)
    with pytest.raises(ValueError, match="exceeds"):
        await decode_image_file(upload)
    # Read was bounded to env-derived cap + 1; not the full payload.
    assert upload.read_size == 12345 + 1


@pytest.mark.asyncio
async def test_decode_image_file_falls_back_on_unparseable_env(monkeypatch):
    """A garbled MUSE_IMAGE_INPUT_MAX_BYTES falls back to the 10MB
    hardcoded default. Operators see a warning in the worker log."""
    from muse.modalities.image_generation.image_input import decode_image_file
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "not-an-int")
    payload = b"\x00" * (11 * 1024 * 1024)
    upload = _FakeUploadFile(payload)
    with pytest.raises(ValueError, match="exceeds"):
        await decode_image_file(upload)


@pytest.mark.asyncio
async def test_decode_image_file_falls_back_on_zero_env(monkeypatch):
    """A non-positive value (0 or negative) falls back to the 10MB
    default. Zero would otherwise reject every upload."""
    from muse.modalities.image_generation.image_input import decode_image_file
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "0")
    payload = _png_bytes(width=8, height=8)
    upload = _FakeUploadFile(payload)
    img = await decode_image_file(upload)  # 10MB default lets this through
    assert img.size == (8, 8)


def test_default_max_bytes_helper_returns_default_when_unset(monkeypatch):
    from muse.modalities.image_generation.image_input import (
        _default_max_bytes, _HARD_DEFAULT_MAX_BYTES,
    )
    monkeypatch.delenv("MUSE_IMAGE_INPUT_MAX_BYTES", raising=False)
    assert _default_max_bytes() == _HARD_DEFAULT_MAX_BYTES


def test_default_max_bytes_helper_reads_env(monkeypatch):
    from muse.modalities.image_generation.image_input import _default_max_bytes
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "20971520")  # 20MB
    assert _default_max_bytes() == 20 * 1024 * 1024


# ---------------- config-hierarchy migration (limits.image_input_max_bytes) ----------------


def test_default_max_bytes_reads_config_env(monkeypatch):
    """The cap now resolves through muse.core.config, not raw os.environ."""
    from muse.core import config as cfg
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "12345")
    cfg.reset_config()
    from muse.modalities.image_generation.image_input import _default_max_bytes
    assert _default_max_bytes() == 12345
    cfg.reset_config()


def test_default_max_bytes_reads_config_file(tmp_path, monkeypatch):
    """A config.yaml value (no env var set) reaches the accessor too.

    This is the behavior that a bare os.environ.get() implementation
    cannot provide: proof the site is routed through the registry
    rather than reading the env var directly.
    """
    from muse.core import config as cfg
    monkeypatch.delenv("MUSE_IMAGE_INPUT_MAX_BYTES", raising=False)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("limits:\n  image_input_max_bytes: 54321\n")
    monkeypatch.setenv("MUSE_CONFIG", str(cfg_file))
    cfg.reset_config()
    from muse.modalities.image_generation.image_input import _default_max_bytes
    assert _default_max_bytes() == 54321
    cfg.reset_config()
