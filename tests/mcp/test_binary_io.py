"""Tests for muse.mcp.binary_io.

Cover the three input forms (b64 / url / path) plus error cases. Output
packing is tested for image and audio variants.
"""
from __future__ import annotations

import base64
import os
from unittest.mock import MagicMock

import pytest

from muse.mcp.binary_io import (
    binary_input_schema,
    pack_audio_output,
    pack_image_output,
    pack_text_output,
    resolve_binary_input,
)


SAMPLE_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10


class TestResolveBinaryInput:
    def test_b64(self):
        b64 = base64.b64encode(SAMPLE_BYTES).decode("ascii")
        out = resolve_binary_input(b64=b64, field_name="image")
        assert out == SAMPLE_BYTES

    def test_b64_with_data_prefix(self):
        b64 = "data:image/png;base64," + base64.b64encode(SAMPLE_BYTES).decode("ascii")
        out = resolve_binary_input(b64=b64, field_name="image")
        assert out == SAMPLE_BYTES

    def test_data_url(self):
        url = "data:image/png;base64," + base64.b64encode(SAMPLE_BYTES).decode("ascii")
        out = resolve_binary_input(url=url, field_name="image")
        assert out == SAMPLE_BYTES

    def test_path_roundtrip(self, monkeypatch, tmp_path):
        # Path inputs require MUSE_MCP_ALLOWED_PATH_PREFIXES (C2 fix).
        monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", str(tmp_path))
        f = tmp_path / "sample.png"
        f.write_bytes(SAMPLE_BYTES)
        out = resolve_binary_input(path=str(f), field_name="image")
        assert out == SAMPLE_BYTES

    def test_missing_input_raises(self):
        with pytest.raises(ValueError, match="missing audio input"):
            resolve_binary_input(field_name="audio")

    def test_too_many_inputs_raises(self):
        b64 = base64.b64encode(b"x").decode("ascii")
        with pytest.raises(ValueError, match="too many image inputs"):
            resolve_binary_input(b64=b64, url="data:," + b64, field_name="image")

    def test_unsupported_url_scheme(self):
        with pytest.raises(ValueError, match="unsupported"):
            resolve_binary_input(url="ftp://nope/path", field_name="image")

    def test_path_not_found(self, monkeypatch):
        # Path inputs need an allowlist; set /tmp as allowed so "not found"
        # fires rather than "disabled".
        monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", "/tmp")
        with pytest.raises(ValueError, match="not found"):
            resolve_binary_input(
                path="/tmp/__definitely_not_a_real_path__.bin",
                field_name="image",
            )

    def test_malformed_b64_raises(self):
        with pytest.raises(ValueError, match="malformed base64"):
            resolve_binary_input(b64="not!!!base64!!!", field_name="image")

    def test_oversized_b64_rejected_before_decode(self, monkeypatch):
        from muse.core import config
        from muse.mcp import binary_io

        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "3")
        config.reset_config()
        decode = MagicMock(side_effect=AssertionError("decode should not run"))
        monkeypatch.setattr(binary_io.base64, "b64decode", decode)
        try:
            with pytest.raises(ValueError, match="encoded length"):
                resolve_binary_input(b64="A" * 100, field_name="image")
            decode.assert_not_called()
        finally:
            config.reset_config()

    def test_malformed_data_url_base64_is_normalized(self):
        with pytest.raises(ValueError, match="malformed base64"):
            resolve_binary_input(
                url="data:image/png;base64,not!!!base64!!!",
                field_name="image",
            )

    def test_malformed_data_b64_no_comma(self):
        with pytest.raises(ValueError, match="malformed data URL"):
            resolve_binary_input(b64="data:image/png;base64", field_name="image")

    def test_malformed_data_url_no_comma(self):
        with pytest.raises(ValueError, match="malformed data URL"):
            resolve_binary_input(url="data:image/png;base64", field_name="image")

    def test_empty_string_slot_treated_as_absent(self, monkeypatch):
        """M7: the 'exactly one' guard counts slots by truthiness (`if v`),
        but dispatch used `is not None`. An LLM that leaves image_b64="" and
        fills image_url passed the guard yet fell into the empty-b64 branch,
        so b64decode("")==b"" silently dropped the URL. Guard and dispatch
        must agree: treat "" as absent."""
        from unittest.mock import patch as _patch

        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch(
            "muse.mcp.binary_io.fetch_url_bytes", return_value=SAMPLE_BYTES,
        ):
            out = resolve_binary_input(
                b64="", url="http://example.com/img.png", field_name="image",
            )
        assert out == SAMPLE_BYTES

    def test_all_empty_strings_raise_missing(self):
        """Empty strings in every slot must read as 'nothing provided',
        not 'ambiguous', so the LLM gets the actionable 'missing' hint."""
        with pytest.raises(ValueError, match="missing image input"):
            resolve_binary_input(b64="", url="", path="", field_name="image")

    @pytest.mark.parametrize("slot", ["b64", "url", "path"])
    @pytest.mark.parametrize(
        "value",
        [False, True, 0, 1, 1.5, [], {}, b"bytes"],
        ids=[
            "false", "true", "zero", "one", "float", "list", "dict", "bytes",
        ],
    )
    def test_non_string_slot_is_rejected(self, slot, value):
        """Malformed MCP JSON must fail as validation, not Python internals."""
        with pytest.raises(
            ValueError,
            match=rf"image_{slot} must be a string",
        ):
            resolve_binary_input(
                field_name="image",
                **{slot: value},
            )

    def test_http_url_routes_through_net_fetch(self, monkeypatch):
        """URL inputs now route through muse.core.net_fetch.fetch_url_bytes
        (SSRF-protected, size-capped). Patch at that boundary."""
        from unittest.mock import patch as _patch

        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        called = {}

        def fake_fetch(url, *, max_bytes, **kwargs):
            called["url"] = url
            called["max_bytes"] = max_bytes
            return SAMPLE_BYTES

        with _patch("muse.mcp.binary_io.fetch_url_bytes", side_effect=fake_fetch):
            out = resolve_binary_input(
                url="http://example.com/img.png", field_name="image",
            )
        assert out == SAMPLE_BYTES
        assert called["url"] == "http://example.com/img.png"
        assert isinstance(called["max_bytes"], int)

    def test_allowlisted_path_obeys_shared_size_cap(self, tmp_path, monkeypatch):
        from muse.core import config

        monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", str(tmp_path))
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "4")
        target = tmp_path / "too-large.bin"
        target.write_bytes(b"12345")
        config.reset_config()
        try:
            with pytest.raises(ValueError, match="exceeds the max of 4 bytes"):
                resolve_binary_input(path=str(target), field_name="image")
        finally:
            config.reset_config()

    def test_allowlisted_directory_is_clean_read_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", str(tmp_path))
        with pytest.raises(ValueError, match="could not read"):
            resolve_binary_input(path=str(tmp_path), field_name="image")

    def test_path_read_rejects_symlink_swapped_after_realpath(
        self, tmp_path, monkeypatch,
    ):
        """The descriptor walk must fail closed if the checked leaf changed."""
        from muse.mcp import binary_io

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"secret")
        stale_target = allowed / "target.bin"
        stale_target.symlink_to(outside)
        monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", str(allowed))

        realpath = binary_io.os.path.realpath

        def stale_realpath(value):
            if os.fspath(value) == os.fspath(stale_target):
                # Simulate a target that was regular during canonicalization
                # but became a symlink before descriptor acquisition.
                return os.fspath(stale_target)
            return realpath(value)

        monkeypatch.setattr(binary_io.os.path, "realpath", stale_realpath)
        with pytest.raises(ValueError, match="could not read"):
            resolve_binary_input(path=str(stale_target), field_name="image")


class TestPackOutputs:
    def test_pack_image(self):
        block = pack_image_output(SAMPLE_BYTES)
        assert block["type"] == "image"
        assert block["mimeType"] == "image/png"
        assert base64.b64decode(block["data"]) == SAMPLE_BYTES

    def test_pack_image_custom_mime(self):
        block = pack_image_output(SAMPLE_BYTES, mime="image/webp")
        assert block["mimeType"] == "image/webp"

    def test_pack_audio(self):
        block = pack_audio_output(b"RIFFxxxxxx")
        assert block["type"] == "audio"
        assert block["mimeType"] == "audio/wav"
        assert base64.b64decode(block["data"]) == b"RIFFxxxxxx"

    def test_pack_audio_custom_mime(self):
        block = pack_audio_output(b"\x00", mime="audio/opus")
        assert block["mimeType"] == "audio/opus"

    def test_pack_text(self):
        block = pack_text_output("hello")
        assert block == {"type": "text", "text": "hello"}


class TestSchema:
    def test_schema_has_three_fields(self):
        s = binary_input_schema("image")
        assert set(s.keys()) == {"image_b64", "image_url", "image_path"}
        for v in s.values():
            assert v["type"] == "string"
            assert "description" in v

    def test_schema_field_name_substituted(self):
        s = binary_input_schema("audio")
        assert "audio_b64" in s
        assert "audio_url" in s
        assert "audio_path" in s
