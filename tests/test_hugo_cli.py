"""Tests for narro.hugo.cli — Hugo integration CLI commands."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from narro.hugo.cli import (
    _check_ffmpeg,
    _validate_site,
    cmd_hugo_generate,
    cmd_hugo_install,
    cmd_hugo_status,
    find_tts_posts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hugo_site(tmp_path, posts=None):
    """Build a minimal Hugo site structure for testing."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "hugo.toml").write_text('baseURL = "https://example.com/"\n')
    for post in (posts or []):
        post_dir = site / "content" / "post" / post["slug"]
        post_dir.mkdir(parents=True)
        fm = "---\n"
        for k, v in post.get("frontmatter", {}).items():
            if isinstance(v, bool):
                fm += f"{k}: {'true' if v else 'false'}\n"
            else:
                fm += f'{k}: "{v}"\n'
        fm += "---\n"
        (post_dir / "index.md").write_text(fm + post.get("body", ""))
    return site


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class TestValidateSite:
    def test_accepts_hugo_toml(self, tmp_path):
        site = tmp_path / "site"
        site.mkdir()
        (site / "hugo.toml").write_text("")
        # Should not raise
        _validate_site(str(site))

    def test_accepts_config_yaml(self, tmp_path):
        site = tmp_path / "site"
        site.mkdir()
        (site / "config.yaml").write_text("")
        _validate_site(str(site))

    def test_rejects_non_hugo_site(self, tmp_path):
        site = tmp_path / "empty"
        site.mkdir()
        with pytest.raises(SystemExit):
            _validate_site(str(site))


class TestCheckFfmpeg:
    def test_passes_when_available(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
        # Should not raise
        _check_ffmpeg()

    def test_fails_when_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(SystemExit):
            _check_ffmpeg()


# ---------------------------------------------------------------------------
# find_tts_posts
# ---------------------------------------------------------------------------

class TestFindTtsPosts:
    def test_finds_only_tts_true_posts(self, tmp_path):
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "hello", "frontmatter": {"title": "Hello", "tts": True},
             "body": "Hello world."},
            {"slug": "no-tts", "frontmatter": {"title": "No TTS", "tts": False},
             "body": "Not narrated."},
            {"slug": "missing", "frontmatter": {"title": "Missing"},
             "body": "No tts field."},
        ])
        posts = find_tts_posts(str(site))
        assert len(posts) == 1
        assert posts[0]["slug"] == "hello"
        assert posts[0]["title"] == "Hello"
        assert posts[0]["has_audio"] is False
        assert "Hello world." in posts[0]["body"]

    def test_detects_existing_audio(self, tmp_path):
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "narrated", "frontmatter": {"title": "Done", "tts": True},
             "body": "Already narrated."},
        ])
        # Place narration.opus in the post dir
        opus_path = site / "content" / "post" / "narrated" / "narration.opus"
        opus_path.write_bytes(b"\x00" * 10)

        posts = find_tts_posts(str(site))
        assert len(posts) == 1
        assert posts[0]["has_audio"] is True


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

class TestHugoInstall:
    def test_install_copies_assets(self, tmp_path):
        site = make_hugo_site(tmp_path)
        cmd_hugo_install(str(site))

        assert (site / "layouts" / "partials" / "tts-player.html").exists()
        assert (site / "static" / "js" / "tts-player.js").exists()
        assert (site / "static" / "css" / "tts-player.css").exists()

        # Verify content is non-empty
        html = (site / "layouts" / "partials" / "tts-player.html").read_text()
        assert "tts-player" in html

    def test_install_rejects_non_hugo_site(self, tmp_path):
        bad_dir = tmp_path / "nope"
        bad_dir.mkdir()
        with pytest.raises(SystemExit):
            cmd_hugo_install(str(bad_dir))


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TestHugoStatus:
    def test_status_finds_tts_posts(self, tmp_path, capsys):
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "a", "frontmatter": {"title": "Post A", "tts": True},
             "body": "AAA"},
            {"slug": "b", "frontmatter": {"title": "Post B", "tts": False},
             "body": "BBB"},
            {"slug": "c", "frontmatter": {"title": "Post C"},
             "body": "CCC"},
        ])
        cmd_hugo_status(str(site))
        out = capsys.readouterr().out
        assert "Post A" in out
        assert "Post B" not in out

    def test_status_detects_existing_audio(self, tmp_path, capsys):
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "done", "frontmatter": {"title": "Done Post", "tts": True},
             "body": "Done."},
        ])
        (site / "content" / "post" / "done" / "narration.opus").write_bytes(b"\x00")
        cmd_hugo_status(str(site))
        out = capsys.readouterr().out
        # Should indicate audio exists
        assert "Done Post" in out
        assert "yes" in out.lower() or "\u2713" in out or "ready" in out.lower()


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

class TestHugoGenerate:
    def test_generate_skips_posts_with_audio(self, tmp_path, monkeypatch):
        """Posts with existing narration.opus should be skipped."""
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "done", "frontmatter": {"title": "Done", "tts": True},
             "body": "Already done."},
        ])
        (site / "content" / "post" / "done" / "narration.opus").write_bytes(b"\x00")
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")

        result = cmd_hugo_generate(str(site))
        assert result["generated"] == 0
        assert result["skipped"] == 1

    def test_generate_dry_run(self, tmp_path, monkeypatch):
        """Dry run should not load model, should report pending count."""
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "pending1", "frontmatter": {"title": "P1", "tts": True},
             "body": "Pending one."},
            {"slug": "pending2", "frontmatter": {"title": "P2", "tts": True},
             "body": "Pending two."},
            {"slug": "done", "frontmatter": {"title": "Done", "tts": True},
             "body": "Already done."},
        ])
        (site / "content" / "post" / "done" / "narration.opus").write_bytes(b"\x00")
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")

        result = cmd_hugo_generate(str(site), dry_run=True)
        assert result["generated"] == 0
        assert result["pending"] == 2
        assert result["skipped"] == 1

    def test_generate_extracts_text_and_calls_tts(self, tmp_path, monkeypatch):
        """Mock Narro + ffmpeg, verify prose extracted correctly (no code)."""
        body = (
            "This is speakable text.\n\n"
            "```python\nprint('hidden')\n```\n\n"
            "More speakable text."
        )
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "test-post", "frontmatter": {"title": "Test", "tts": True},
             "body": body},
        ])
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")

        # Track what encode was called with
        encode_calls = []
        fake_encoded = MagicMock()

        class FakeNarro:
            def __init__(self):
                pass

            def encode(self, text, **kwargs):
                encode_calls.append(text)
                return fake_encoded

            def decode_to_wav(self, encoded, out_path):
                # Write a dummy wav so the pipeline continues
                Path(out_path).write_bytes(b"RIFF" + b"\x00" * 100)

        monkeypatch.setattr("narro.hugo.cli.Narro", FakeNarro)

        # Mock ffmpeg to create the opus file
        post_dir = site / "content" / "post" / "test-post"

        def fake_subprocess_run(cmd, **kwargs):
            # Find the output path (after -y)
            if "ffmpeg" in cmd[0]:
                opus_out = cmd[-1]
                Path(opus_out).write_bytes(b"\x00" * 10)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("narro.hugo.cli.subprocess.run", fake_subprocess_run)

        fake_alignment = [{"word": "hello", "start": 0.0, "end": 0.5}]
        monkeypatch.setattr(
            "narro.hugo.cli.extract_alignment_from_encoded",
            lambda enc: fake_alignment,
        )
        monkeypatch.setattr(
            "narro.hugo.cli.save_alignment",
            lambda alignment, path: None,
        )

        result = cmd_hugo_generate(str(site))
        assert result["generated"] == 1
        assert result["errors"] == 0

        # Verify encode was called with clean text (no code blocks)
        assert len(encode_calls) == 1
        assert "print" not in encode_calls[0]
        assert "speakable" in encode_calls[0]

        # Verify opus file was created
        assert (post_dir / "narration.opus").exists()

    def test_generate_single_post(self, tmp_path, monkeypatch):
        """With post_slug, only generate for matching slug."""
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "target", "frontmatter": {"title": "Target", "tts": True},
             "body": "Generate me."},
            {"slug": "other", "frontmatter": {"title": "Other", "tts": True},
             "body": "Skip me."},
        ])
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")

        encode_calls = []
        fake_encoded = MagicMock()

        class FakeNarro:
            def __init__(self):
                pass

            def encode(self, text, **kwargs):
                encode_calls.append(text)
                return fake_encoded

            def decode_to_wav(self, encoded, out_path):
                Path(out_path).write_bytes(b"RIFF" + b"\x00" * 100)

        monkeypatch.setattr("narro.hugo.cli.Narro", FakeNarro)

        def fake_subprocess_run(cmd, **kwargs):
            if "ffmpeg" in cmd[0]:
                Path(cmd[-1]).write_bytes(b"\x00" * 10)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("narro.hugo.cli.subprocess.run", fake_subprocess_run)
        monkeypatch.setattr(
            "narro.hugo.cli.extract_alignment_from_encoded",
            lambda enc: [],
        )
        monkeypatch.setattr(
            "narro.hugo.cli.save_alignment",
            lambda alignment, path: None,
        )

        result = cmd_hugo_generate(str(site), post_slug="target")
        assert result["generated"] == 1
        # "other" should not have been generated
        assert not (site / "content" / "post" / "other" / "narration.opus").exists()

    def test_generate_force_regenerates(self, tmp_path, monkeypatch):
        """With force=True, regenerate even posts that have audio."""
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "done", "frontmatter": {"title": "Done", "tts": True},
             "body": "Regenerate me."},
        ])
        (site / "content" / "post" / "done" / "narration.opus").write_bytes(b"\x00")
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")

        fake_encoded = MagicMock()

        class FakeNarro:
            def __init__(self):
                pass

            def encode(self, text, **kwargs):
                return fake_encoded

            def decode_to_wav(self, encoded, out_path):
                Path(out_path).write_bytes(b"RIFF" + b"\x00" * 100)

        monkeypatch.setattr("narro.hugo.cli.Narro", FakeNarro)

        def fake_subprocess_run(cmd, **kwargs):
            if "ffmpeg" in cmd[0]:
                Path(cmd[-1]).write_bytes(b"\x00" * 10)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("narro.hugo.cli.subprocess.run", fake_subprocess_run)
        monkeypatch.setattr(
            "narro.hugo.cli.extract_alignment_from_encoded",
            lambda enc: [],
        )
        monkeypatch.setattr(
            "narro.hugo.cli.save_alignment",
            lambda alignment, path: None,
        )

        result = cmd_hugo_generate(str(site), force=True)
        assert result["generated"] == 1
        assert result["skipped"] == 0

    def test_generate_error_cleans_up(self, tmp_path, monkeypatch):
        """On error, partial files should be cleaned up and processing continues."""
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "aaa-fail", "frontmatter": {"title": "Fail", "tts": True},
             "body": "Will fail."},
            {"slug": "zzz-ok", "frontmatter": {"title": "OK", "tts": True},
             "body": "Will succeed."},
        ])
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")

        # Track which slugs were attempted
        encode_slugs = []
        fake_encoded = MagicMock()

        class FakeNarro:
            def __init__(self):
                pass

            def encode(self, text, **kwargs):
                # First call always fails (aaa-fail sorts first via os.walk)
                encode_slugs.append(text)
                if len(encode_slugs) == 1:
                    raise RuntimeError("TTS failed")
                return fake_encoded

            def decode_to_wav(self, encoded, out_path):
                Path(out_path).write_bytes(b"RIFF" + b"\x00" * 100)

        monkeypatch.setattr("narro.hugo.cli.Narro", FakeNarro)

        def fake_subprocess_run(cmd, **kwargs):
            if "ffmpeg" in cmd[0]:
                Path(cmd[-1]).write_bytes(b"\x00" * 10)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("narro.hugo.cli.subprocess.run", fake_subprocess_run)
        monkeypatch.setattr(
            "narro.hugo.cli.extract_alignment_from_encoded",
            lambda enc: [],
        )
        monkeypatch.setattr(
            "narro.hugo.cli.save_alignment",
            lambda alignment, path: None,
        )

        result = cmd_hugo_generate(str(site))
        assert result["errors"] == 1
        assert result["generated"] == 1
        # The failing post should NOT have an opus file
        assert not (site / "content" / "post" / "aaa-fail" / "narration.opus").exists()
        # The successful post should have one
        assert (site / "content" / "post" / "zzz-ok" / "narration.opus").exists()
