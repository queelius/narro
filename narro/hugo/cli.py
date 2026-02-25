"""Hugo integration CLI commands for narro.

Provides three commands:
- generate: Synthesize narration for Hugo posts with tts: true
- install:  Copy TTS player assets into a Hugo site
- status:   Show which posts have TTS enabled and audio status
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .extract import extract_prose, parse_frontmatter

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Hugo config filenames we recognize
_HUGO_CONFIG_FILES = (
    "hugo.toml", "hugo.yaml", "hugo.json",
    "config.toml", "config.yaml",
)

# Lazy-loaded references (populated by _lazy_import on first generate).
# Tests can monkeypatch these at narro.hugo.cli.Narro etc.
Narro = None
extract_alignment_from_encoded = None
save_alignment = None


def _lazy_import() -> None:
    """Import heavy dependencies into module globals on first use."""
    global Narro, extract_alignment_from_encoded, save_alignment
    if Narro is not None:
        return
    from narro.tts import Narro as _Narro
    from narro.alignment import (
        extract_alignment_from_encoded as _extract,
        save_alignment as _save,
    )
    Narro = _Narro
    extract_alignment_from_encoded = _extract
    save_alignment = _save


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_site(site_root: str) -> None:
    """Check that site_root looks like a Hugo site (has a config file).

    Calls sys.exit(1) with error message if no Hugo config is found.
    """
    for name in _HUGO_CONFIG_FILES:
        if os.path.isfile(os.path.join(site_root, name)):
            return
    print(
        f"Error: No Hugo config found in {site_root}\n"
        f"Expected one of: {', '.join(_HUGO_CONFIG_FILES)}",
        file=sys.stderr,
    )
    sys.exit(1)


def _check_ffmpeg() -> None:
    """Verify ffmpeg is available on PATH.

    Calls sys.exit(1) if ffmpeg is not found.
    """
    if not shutil.which("ffmpeg"):
        print(
            "Error: ffmpeg not found. Install ffmpeg to convert audio to opus.\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS: brew install ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Post discovery
# ---------------------------------------------------------------------------

def find_tts_posts(site_root: str) -> list[dict[str, Any]]:
    """Walk content/post/**/index.md, return posts with tts: true.

    Returns:
        List of dicts with keys: slug, dir, title, has_audio, body
    """
    content_dir = os.path.join(site_root, "content", "post")
    if not os.path.isdir(content_dir):
        return []

    posts: list[dict[str, Any]] = []
    for root, _dirs, files in os.walk(content_dir):
        if "index.md" not in files:
            continue
        index_path = os.path.join(root, "index.md")
        text = Path(index_path).read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)

        if not meta.get("tts", False):
            continue

        slug = os.path.basename(root)
        has_audio = os.path.isfile(os.path.join(root, "narration.opus"))
        posts.append({
            "slug": slug,
            "dir": root,
            "title": meta.get("title", slug),
            "has_audio": has_audio,
            "body": body,
        })

    posts.sort(key=lambda p: p["slug"])
    return posts


# ---------------------------------------------------------------------------
# install command
# ---------------------------------------------------------------------------

def cmd_hugo_install(site_root: str) -> None:
    """Copy TTS player assets into a Hugo site.

    Copies:
      - layouts/partials/tts-player.html
      - static/js/tts-player.js
      - static/css/tts-player.css

    Prints instructions for including the partial in a layout.
    """
    _validate_site(site_root)

    file_map = {
        "tts-player.html": os.path.join("layouts", "partials", "tts-player.html"),
        "tts-player.js": os.path.join("static", "js", "tts-player.js"),
        "tts-player.css": os.path.join("static", "css", "tts-player.css"),
    }

    for asset_name, dest_rel in file_map.items():
        src = os.path.join(ASSETS_DIR, asset_name)
        dest = os.path.join(site_root, dest_rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        print(f"  Copied {dest_rel}")

    print(
        "\nAdd the TTS player to your single post layout:\n"
        '  {{ partial "tts-player.html" . }}\n'
        "\n"
        "Place it inside your article template (e.g. layouts/post/single.html)\n"
        "after the content block."
    )


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

def cmd_hugo_status(site_root: str) -> None:
    """Print a status table of TTS-enabled posts."""
    _validate_site(site_root)
    posts = find_tts_posts(site_root)

    if not posts:
        print("No posts with tts: true found.")
        return

    # Column widths
    slug_w = max(len(p["slug"]) for p in posts)
    title_w = max(len(p["title"]) for p in posts)
    slug_w = max(slug_w, 4)  # min header width
    title_w = max(title_w, 5)

    header = f"{'Slug':<{slug_w}}  {'Title':<{title_w}}  Audio"
    print(header)
    print("-" * len(header))
    for p in posts:
        status = "yes" if p["has_audio"] else "pending"
        print(f"{p['slug']:<{slug_w}}  {p['title']:<{title_w}}  {status}")


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------

def cmd_hugo_generate(
    site_root: str,
    force: bool = False,
    dry_run: bool = False,
    post_slug: str | None = None,
) -> dict[str, int]:
    """Generate narration for Hugo posts with tts: true.

    Args:
        site_root: Path to the Hugo site root.
        force: Regenerate even if narration.opus already exists.
        dry_run: Print pending posts without generating.
        post_slug: Only generate for this specific post slug.

    Returns:
        Dict with keys: generated, skipped, errors (and pending for dry_run).
    """
    _validate_site(site_root)
    _check_ffmpeg()

    posts = find_tts_posts(site_root)

    # Filter by slug if requested
    if post_slug is not None:
        posts = [p for p in posts if p["slug"] == post_slug]

    # Partition into pending vs skipped
    if force:
        pending = posts
        skipped = []
    else:
        pending = [p for p in posts if not p["has_audio"]]
        skipped = [p for p in posts if p["has_audio"]]

    if dry_run:
        if pending:
            print(f"Pending ({len(pending)} posts):")
            for p in pending:
                print(f"  - {p['slug']}: {p['title']}")
        if skipped:
            print(f"Skipped ({len(skipped)} posts with existing audio)")
        return {
            "generated": 0,
            "skipped": len(skipped),
            "errors": 0,
            "pending": len(pending),
        }

    if not pending:
        print("Nothing to generate.")
        return {"generated": 0, "skipped": len(skipped), "errors": 0}

    # Lazy imports — only when actually generating.
    # Import into module globals so tests can monkeypatch
    # narro.hugo.cli.Narro, .extract_alignment_from_encoded, .save_alignment.
    _lazy_import()

    tts = Narro()
    generated = 0
    errors = 0

    for p in pending:
        post_dir = p["dir"]
        wav_path = os.path.join(post_dir, "narration.wav")
        opus_path = os.path.join(post_dir, "narration.opus")
        json_path = os.path.join(post_dir, "narration.json")

        print(f"Generating: {p['slug']} ({p['title']})")

        try:
            # Extract speakable prose
            prose = extract_prose(p["body"])

            # Encode with attention for alignment
            encoded = tts.encode(prose, include_attention=True)

            # Decode to WAV
            tts.decode_to_wav(encoded, wav_path)

            # Extract and save alignment
            alignment = extract_alignment_from_encoded(encoded)
            save_alignment(alignment, json_path)

            # Convert WAV -> Opus
            subprocess.run(
                ["ffmpeg", "-i", wav_path, "-c:a", "libopus", "-b:a", "32k", "-y", opus_path],
                check=True,
                capture_output=True,
            )

            # Clean up WAV
            if os.path.isfile(wav_path):
                os.remove(wav_path)

            generated += 1
            print(f"  Done: {opus_path}")

        except Exception as e:
            errors += 1
            print(f"  Error: {e}", file=sys.stderr)
            # Clean up partial files
            for partial in (wav_path, opus_path, json_path):
                if os.path.isfile(partial):
                    os.remove(partial)

    return {"generated": generated, "skipped": len(skipped), "errors": errors}
