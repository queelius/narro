# Hugo Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `narro hugo generate|install|status` CLI commands that generate TTS narration for Hugo blog posts and install the required player assets.

**Architecture:** A new `narro/hugo/` subpackage contains text extraction (`extract.py`), CLI commands (`cli.py`), and installable Hugo assets (`assets/`). The CLI uses argparse nested subparsers, matching the existing pattern. The generate command loads the Narro model once and iterates posts with `tts: true` in frontmatter.

**Tech Stack:** Python 3.10+, argparse, PyYAML (new dependency for frontmatter parsing), ffmpeg (system), existing Narro Python API.

---

### Task 1: Add PyYAML dependency

PyYAML is needed to parse Hugo post frontmatter reliably. The existing codebase has no YAML parser.

**Files:**
- Modify: `pyproject.toml:18-26` (dependencies list)

**Step 1: Add PyYAML to dependencies**

In `pyproject.toml`, add `"pyyaml"` to the `dependencies` list:

```toml
dependencies = [
  "huggingface_hub",
  "numpy",
  "pyyaml",
  "scipy",
  "torch>=2.1.0",
  "transformers>=4.51.0",
  "unidecode",
  "inflect"
]
```

**Step 2: Install updated dependencies**

Run: `pip install -e .`
Expected: Successfully installs with pyyaml available.

**Step 3: Verify**

Run: `python -c "import yaml; print(yaml.__version__)"`
Expected: Prints a version number.

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add pyyaml for hugo frontmatter parsing"
```

---

### Task 2: Create extract_prose() with tests (TDD)

The text extraction function strips markdown to speakable plain text. This is the most testable component — pure function, no IO.

**Files:**
- Create: `narro/hugo/__init__.py`
- Create: `narro/hugo/extract.py`
- Create: `tests/test_hugo_extract.py`

**Step 1: Write the failing tests**

Create `tests/test_hugo_extract.py`:

```python
"""Tests for narro.hugo.extract — markdown to speakable text."""

import pytest
from narro.hugo.extract import extract_prose, parse_frontmatter


class TestParseFrontmatter:
    """Test YAML frontmatter extraction."""

    def test_basic_frontmatter(self):
        md = '---\ntitle: "Hello"\ntts: true\n---\nBody text here.'
        meta, body = parse_frontmatter(md)
        assert meta['title'] == 'Hello'
        assert meta['tts'] is True
        assert body.strip() == 'Body text here.'

    def test_no_frontmatter(self):
        md = 'Just body text, no frontmatter.'
        meta, body = parse_frontmatter(md)
        assert meta == {}
        assert body.strip() == 'Just body text, no frontmatter.'

    def test_tts_false(self):
        md = '---\ntitle: "Post"\ntts: false\n---\nBody.'
        meta, body = parse_frontmatter(md)
        assert meta['tts'] is False

    def test_no_tts_field(self):
        md = '---\ntitle: "Post"\n---\nBody.'
        meta, body = parse_frontmatter(md)
        assert 'tts' not in meta


class TestExtractProse:
    """Test markdown stripping to speakable text."""

    def test_plain_text(self):
        assert extract_prose('Hello world.') == 'Hello world.'

    def test_strips_headings(self):
        md = '## Introduction\n\nSome text here.'
        result = extract_prose(md)
        assert result == 'Introduction\n\nSome text here.'

    def test_strips_fenced_code(self):
        md = 'Before code.\n\n```python\nprint("hello")\n```\n\nAfter code.'
        result = extract_prose(md)
        assert 'print' not in result
        assert 'Before code.' in result
        assert 'After code.' in result

    def test_strips_indented_code(self):
        md = 'Before.\n\n    code_line_1\n    code_line_2\n\nAfter.'
        result = extract_prose(md)
        assert 'code_line' not in result

    def test_strips_latex_block(self):
        md = 'Before.\n\n$$\nE = mc^2\n$$\n\nAfter.'
        result = extract_prose(md)
        assert 'mc^2' not in result
        assert 'Before.' in result
        assert 'After.' in result

    def test_strips_latex_inline(self):
        md = r'The value \(x = 5\) is important.'
        result = extract_prose(md)
        assert 'x = 5' not in result
        assert 'The value' in result
        assert 'is important.' in result

    def test_strips_latex_bracket_block(self):
        md = 'Before.\n\n\\[\nf(x) = x^2\n\\]\n\nAfter.'
        result = extract_prose(md)
        assert 'f(x)' not in result

    def test_strips_images(self):
        md = 'Text before.\n\n![Alt text](image.png)\n\nText after.'
        result = extract_prose(md)
        assert 'Alt text' not in result
        assert 'image.png' not in result

    def test_converts_links_to_text(self):
        md = 'See [this article](https://example.com) for details.'
        result = extract_prose(md)
        assert result == 'See this article for details.'

    def test_strips_hugo_shortcodes(self):
        md = 'Before.\n\n{{< tts src="audio.opus" >}}\n\nAfter.'
        result = extract_prose(md)
        assert 'tts' not in result
        assert 'Before.' in result

    def test_strips_self_closing_shortcodes(self):
        md = 'Before. {{< relurl "path" />}} After.'
        result = extract_prose(md)
        assert 'relurl' not in result

    def test_strips_shortcodes_with_body(self):
        md = 'Before.\n\n{{< cryptoid-encrypted >}}\nSecret stuff\n{{< /cryptoid-encrypted >}}\n\nAfter.'
        result = extract_prose(md)
        assert 'Secret' not in result
        assert 'cryptoid' not in result

    def test_strips_html_tags(self):
        md = 'Text with <strong>bold</strong> and <br/> breaks.'
        result = extract_prose(md)
        assert '<strong>' not in result
        assert '<br/>' not in result
        assert 'bold' in result

    def test_strips_bold_italic_markers(self):
        md = 'This is **bold** and *italic* text.'
        result = extract_prose(md)
        assert result == 'This is bold and italic text.'

    def test_strips_inline_code(self):
        md = 'Use the `extract_prose()` function.'
        result = extract_prose(md)
        assert '`' not in result
        assert 'extract_prose()' in result

    def test_collapses_whitespace(self):
        md = 'First.\n\n\n\n\n\nSecond.'
        result = extract_prose(md)
        # Multiple blank lines collapse to double newline (paragraph break)
        assert '\n\n\n' not in result

    def test_strips_horizontal_rules(self):
        md = 'Above.\n\n---\n\nBelow.'
        result = extract_prose(md)
        assert '---' not in result

    def test_strips_blockquotes(self):
        md = '> This is a quote.\n\nNormal text.'
        result = extract_prose(md)
        assert 'This is a quote.' in result
        assert '>' not in result

    def test_strips_list_markers(self):
        md = '- Item one\n- Item two\n\n1. First\n2. Second'
        result = extract_prose(md)
        assert '- ' not in result
        assert 'Item one' in result
        assert 'First' in result

    def test_realistic_post(self):
        md = """## Getting Started

Here's how to use [narro](https://github.com/queelius/narro) for TTS.

First, install it:

```bash
pip install narro
```

The model uses \\(80M\\) parameters. Given the equation:

$$
L = -\\sum_{i} \\log p(x_i)
$$

We can see that **loss decreases** as the model improves.

![Architecture diagram](arch.png)

{{< relurl "docs/api" />}}

> Note: CPU-only inference.

That's it!"""
        result = extract_prose(md)
        # Should contain readable text
        assert 'Getting Started' in result
        assert 'how to use narro for TTS' in result
        assert "That's it!" in result
        # Should not contain code, math, images, shortcodes
        assert 'pip install' not in result
        assert 'log p' not in result
        assert 'arch.png' not in result
        assert 'relurl' not in result
        assert '```' not in result
        assert '$$' not in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hugo_extract.py -v`
Expected: ImportError — `narro.hugo.extract` does not exist yet.

**Step 3: Create the package and implement**

Create `narro/hugo/__init__.py`:

```python
```

Create `narro/hugo/extract.py`:

```python
"""Extract speakable prose from Hugo markdown posts.

Strips frontmatter, code blocks, LaTeX math, images, shortcodes,
HTML tags, and markdown formatting to produce plain text suitable
for text-to-speech synthesis.
"""

import re

import yaml


def parse_frontmatter(markdown):
    """Parse YAML frontmatter from a markdown string.

    Args:
        markdown: Full markdown file content.

    Returns:
        Tuple of (metadata_dict, body_string). If no frontmatter
        found, metadata_dict is empty and body is the full input.
    """
    if not markdown.startswith('---'):
        return {}, markdown

    # Find closing --- (must be on its own line after the opening)
    end = markdown.find('\n---', 3)
    if end == -1:
        return {}, markdown

    front = markdown[3:end].strip()
    body = markdown[end + 4:]  # skip past \n---

    try:
        meta = yaml.safe_load(front) or {}
    except yaml.YAMLError:
        return {}, markdown

    return meta, body


def extract_prose(text):
    """Convert markdown text to speakable plain text.

    Processing order matters: code blocks and math are removed before
    HTML tags (code can contain angle brackets). Shortcodes before
    general tag stripping.

    Args:
        text: Markdown body text (frontmatter already stripped).

    Returns:
        Plain text suitable for TTS synthesis.
    """
    # 1. Fenced code blocks (``` or ~~~)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'~~~[\s\S]*?~~~', '', text)

    # 2. Indented code blocks (4 spaces or tab, preceded by blank line)
    text = re.sub(r'(?m)(?:^[ \t]*\n)((?:^(?:    |\t).+\n?)+)', '', text)

    # 3. LaTeX math
    # Block: $$...$$ (multiline)
    text = re.sub(r'\$\$[\s\S]*?\$\$', '', text)
    # Block: \[...\] (multiline)
    text = re.sub(r'\\\[[\s\S]*?\\\]', '', text)
    # Inline: \(...\)
    text = re.sub(r'\\\(.*?\\\)', '', text)

    # 4. Images: ![alt](url) or ![alt](url "title")
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # 5. Hugo shortcodes
    # Paired: {{< name >}}...{{< /name >}}
    text = re.sub(r'\{\{<\s*/?\s*\w[\w-]*(?:\s[^>]*)?>}}', '', text)
    # Self-closing: {{< name ... />}}
    text = re.sub(r'\{\{<\s*\w[\w-]*(?:\s[^>]*)?\s*/?>}}', '', text)

    # 6. HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 7. Markdown links: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 8. Heading markers
    text = re.sub(r'(?m)^#{1,6}\s+', '', text)

    # 9. Horizontal rules (---, ***, ___)
    text = re.sub(r'(?m)^[\s]*[-*_]{3,}\s*$', '', text)

    # 10. Blockquote markers
    text = re.sub(r'(?m)^>\s*', '', text)

    # 11. List markers (- , * , + , 1. , 2. , etc.)
    text = re.sub(r'(?m)^[\s]*[-*+]\s+', '', text)
    text = re.sub(r'(?m)^[\s]*\d+\.\s+', '', text)

    # 12. Bold/italic markers
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)

    # 13. Inline code (preserve text inside backticks)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # 14. Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hugo_extract.py -v`
Expected: All tests PASS.

**Step 5: Run coverage**

Run: `pytest tests/test_hugo_extract.py --cov=narro.hugo.extract --cov-report=term-missing`
Expected: 90%+ coverage on extract.py.

**Step 6: Commit**

```bash
git add narro/hugo/__init__.py narro/hugo/extract.py tests/test_hugo_extract.py
git commit -m "feat(hugo): add extract_prose() for markdown to speakable text"
```

---

### Task 3: Create Hugo assets (JS player, CSS, HTML partial)

Static files that `narro hugo install` copies into a Hugo site.

**Files:**
- Create: `narro/hugo/assets/tts-player.html`
- Create: `narro/hugo/assets/tts-player.js`
- Create: `narro/hugo/assets/tts-player.css`

**Step 1: Create the Hugo partial**

Create `narro/hugo/assets/tts-player.html`:

```html
{{/* TTS Player — installed by narro hugo install */}}
{{ if .Params.tts }}
  {{ $audio := .Resources.GetMatch "narration.opus" }}
  {{ $align := .Resources.GetMatch "narration.json" }}
  {{ if $audio }}
    <link rel="stylesheet" href="{{ "css/tts-player.css" | relURL }}">
    <div class="tts-player" {{ if $align }}data-align="{{ $align.RelPermalink }}"{{ end }}>
      <audio src="{{ $audio.RelPermalink }}" preload="metadata"></audio>
      <button class="tts-play" aria-label="Play narration">&#9654; Listen</button>
      <span class="tts-time"></span>
    </div>
    <script src="{{ "js/tts-player.js" | relURL }}" defer></script>
  {{ end }}
{{ end }}
```

**Step 2: Create the JS player**

Create `narro/hugo/assets/tts-player.js`:

```javascript
/**
 * TTS Player — word-level highlighting synchronized with audio playback.
 * Installed by: narro hugo install
 */
(function () {
  "use strict";

  const player = document.querySelector(".tts-player");
  if (!player) return;

  const audio = player.querySelector("audio");
  const btn = player.querySelector(".tts-play");
  const timeDisplay = player.querySelector(".tts-time");
  const alignUrl = player.dataset.align;

  let alignment = [];
  let wordSpans = [];
  let activeIdx = -1;

  // --- Initialization ---

  function init() {
    if (alignUrl) {
      fetch(alignUrl)
        .then((r) => r.json())
        .then((data) => {
          alignment = data;
          wrapWords();
        })
        .catch(() => {}); // Degrade gracefully: no highlighting
    }
  }

  /**
   * Walk text nodes in the article, wrap each word in a <span>.
   * Match words sequentially to alignment entries.
   */
  function wrapWords() {
    const article = document.querySelector("article") || document.querySelector(".content");
    if (!article || alignment.length === 0) return;

    let alignIdx = 0;
    const walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);

    for (const node of textNodes) {
      const text = node.textContent;
      if (!text.trim()) continue;

      const frag = document.createDocumentFragment();
      // Split on word boundaries, preserving whitespace
      const parts = text.split(/(\s+)/);

      for (const part of parts) {
        if (/^\s+$/.test(part)) {
          frag.appendChild(document.createTextNode(part));
          continue;
        }
        if (!part) continue;

        const span = document.createElement("span");
        span.className = "tts-word";

        if (alignIdx < alignment.length) {
          span.dataset.idx = alignIdx;
          wordSpans[alignIdx] = span;
          alignIdx++;
        }

        span.textContent = part;
        span.addEventListener("click", onWordClick);
        frag.appendChild(span);
      }

      node.parentNode.replaceChild(frag, node);
    }
  }

  // --- Playback ---

  function formatTime(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m + ":" + (sec < 10 ? "0" : "") + sec;
  }

  function updateTime() {
    if (!audio.duration) return;
    timeDisplay.textContent = formatTime(audio.currentTime) + " / " + formatTime(audio.duration);
  }

  /**
   * Binary search alignment array for the word active at currentTime.
   */
  function highlightWord() {
    if (alignment.length === 0) return;

    const t = audio.currentTime;
    let lo = 0, hi = alignment.length - 1, mid, idx = -1;

    while (lo <= hi) {
      mid = (lo + hi) >> 1;
      if (alignment[mid].start <= t) {
        idx = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }

    // Check if time is within the found word's range
    if (idx >= 0 && t > alignment[idx].end) idx = -1;

    if (idx !== activeIdx) {
      if (activeIdx >= 0 && wordSpans[activeIdx]) {
        wordSpans[activeIdx].classList.remove("tts-active");
      }
      if (idx >= 0 && wordSpans[idx]) {
        wordSpans[idx].classList.add("tts-active");
      }
      activeIdx = idx;
    }
  }

  function onWordClick(e) {
    const idx = parseInt(e.target.dataset.idx, 10);
    if (!isNaN(idx) && idx < alignment.length) {
      audio.currentTime = alignment[idx].start;
      if (audio.paused) audio.play();
    }
  }

  // --- Controls ---

  btn.addEventListener("click", function () {
    if (audio.paused) {
      audio.play();
      btn.textContent = "\u275A\u275A Pause";
    } else {
      audio.pause();
      btn.textContent = "\u25B6 Listen";
    }
  });

  audio.addEventListener("timeupdate", function () {
    updateTime();
    highlightWord();
  });

  audio.addEventListener("ended", function () {
    btn.textContent = "\u25B6 Listen";
    if (activeIdx >= 0 && wordSpans[activeIdx]) {
      wordSpans[activeIdx].classList.remove("tts-active");
    }
    activeIdx = -1;
  });

  document.addEventListener("keydown", function (e) {
    if (e.code === "Space" && e.target === document.body) {
      e.preventDefault();
      btn.click();
    }
  });

  init();
})();
```

**Step 3: Create the CSS**

Create `narro/hugo/assets/tts-player.css`:

```css
/* TTS Player — installed by narro hugo install */

.tts-player {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  margin: 1.5rem 0;
  border-radius: 8px;
  background: var(--surface-subtle, #f8f9fa);
  border: 1px solid var(--border-color, #e2e8f0);
}

.tts-play {
  cursor: pointer;
  border: none;
  background: none;
  font-size: 1rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  color: inherit;
}

.tts-play:hover {
  background: var(--border-color, #e2e8f0);
}

.tts-time {
  font-size: 0.8rem;
  color: var(--text-muted, #64748b);
  margin-left: auto;
  font-variant-numeric: tabular-nums;
}

.tts-word {
  cursor: pointer;
  border-radius: 2px;
  transition: background-color 0.15s ease;
}

.tts-word.tts-active {
  background-color: rgba(37, 99, 235, 0.12);
}

@media (prefers-reduced-motion: reduce) {
  .tts-word {
    transition: none;
  }
}
```

**Step 4: Commit**

```bash
git add narro/hugo/assets/
git commit -m "feat(hugo): add player assets (JS, CSS, HTML partial)"
```

---

### Task 4: Create Hugo CLI commands with tests (TDD)

The three CLI commands: `generate`, `install`, `status`. This task uses the existing Narro Python API, alignment module, and extract_prose from Task 2.

**Files:**
- Create: `narro/hugo/cli.py`
- Create: `tests/test_hugo_cli.py`
- Modify: `narro/cli.py:84-141` (register hugo subcommand)

**Step 1: Write the failing tests**

Create `tests/test_hugo_cli.py`:

```python
"""Tests for narro.hugo.cli — Hugo integration commands."""

import json
import os
import subprocess
import tempfile

import pytest


def make_hugo_site(tmp_path, posts=None):
    """Create a minimal Hugo site structure for testing.

    Args:
        tmp_path: Base directory.
        posts: List of dicts with keys: slug, frontmatter, body.

    Returns:
        Path to site root.
    """
    site = tmp_path / "site"
    site.mkdir()
    (site / "hugo.toml").write_text('baseURL = "https://example.com/"\n')
    content_dir = site / "content" / "post"
    content_dir.mkdir(parents=True)

    for post in (posts or []):
        post_dir = content_dir / post["slug"]
        post_dir.mkdir()
        fm = "---\n"
        for k, v in post.get("frontmatter", {}).items():
            if isinstance(v, bool):
                fm += f"{k}: {'true' if v else 'false'}\n"
            else:
                fm += f'{k}: "{v}"\n'
        fm += "---\n"
        (post_dir / "index.md").write_text(fm + post.get("body", ""))

    return site


class TestHugoInstall:
    """Test narro hugo install command."""

    def test_install_copies_assets(self, tmp_path):
        site = make_hugo_site(tmp_path)
        from narro.hugo.cli import cmd_hugo_install
        cmd_hugo_install(str(site))

        assert (site / "layouts" / "partials" / "tts-player.html").exists()
        assert (site / "static" / "js" / "tts-player.js").exists()
        assert (site / "static" / "css" / "tts-player.css").exists()

    def test_install_rejects_non_hugo_site(self, tmp_path):
        with pytest.raises(SystemExit):
            from narro.hugo.cli import cmd_hugo_install
            cmd_hugo_install(str(tmp_path))


class TestHugoStatus:
    """Test narro hugo status command."""

    def test_status_finds_tts_posts(self, tmp_path):
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "post-a", "frontmatter": {"title": "A", "tts": True}, "body": "Hello."},
            {"slug": "post-b", "frontmatter": {"title": "B", "tts": False}, "body": "World."},
            {"slug": "post-c", "frontmatter": {"title": "C"}, "body": "No tts field."},
        ])
        from narro.hugo.cli import find_tts_posts
        posts = find_tts_posts(str(site))

        assert len(posts) == 1
        assert posts[0]["slug"] == "post-a"
        assert posts[0]["has_audio"] is False

    def test_status_detects_existing_audio(self, tmp_path):
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "post-a", "frontmatter": {"title": "A", "tts": True}, "body": "Hello."},
        ])
        # Create fake audio file
        (site / "content" / "post" / "post-a" / "narration.opus").write_bytes(b"fake")

        from narro.hugo.cli import find_tts_posts
        posts = find_tts_posts(str(site))
        assert posts[0]["has_audio"] is True


class TestHugoGenerate:
    """Test narro hugo generate command."""

    def test_generate_skips_posts_with_audio(self, tmp_path):
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "post-a", "frontmatter": {"title": "A", "tts": True}, "body": "Hello."},
        ])
        (site / "content" / "post" / "post-a" / "narration.opus").write_bytes(b"fake")

        from narro.hugo.cli import find_tts_posts
        posts = find_tts_posts(str(site))
        pending = [p for p in posts if not p["has_audio"]]
        assert len(pending) == 0

    def test_generate_extracts_text_and_calls_tts(self, tmp_path, monkeypatch):
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "post-a", "frontmatter": {"title": "A", "tts": True},
             "body": "## Intro\n\nHello world.\n\n```python\nx = 1\n```\n"},
        ])

        calls = []

        class FakeNarro:
            def __init__(self, **kwargs):
                pass

            def encode(self, text, **kwargs):
                calls.append(("encode", text))
                return "fake_encoded"

            def decode_to_wav(self, encoded, path):
                calls.append(("decode_to_wav", path))
                # Create the wav file so ffmpeg step can be tested
                import wave
                import struct
                with wave.open(path, 'w') as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(32000)
                    f.writeframes(struct.pack('<h', 0) * 100)

        # Mock ffmpeg
        def fake_run(cmd, **kwargs):
            if 'ffmpeg' in cmd[0]:
                # Create the opus file
                opus_path = cmd[cmd.index('-y') + 1] if '-y' in cmd else cmd[-1]
                with open(opus_path, 'wb') as f:
                    f.write(b"fake_opus")
                return subprocess.CompletedProcess(cmd, 0)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("narro.hugo.cli.Narro", FakeNarro)
        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr(
            "narro.hugo.cli.extract_alignment_from_encoded",
            lambda e: [{"word": "Hello", "start": 0.0, "end": 0.5}],
        )
        monkeypatch.setattr(
            "narro.hugo.cli.save_alignment",
            lambda a, p: None,
        )

        from narro.hugo.cli import cmd_hugo_generate
        result = cmd_hugo_generate(str(site), force=False, dry_run=False, post_slug=None)

        assert result["generated"] == 1
        assert result["skipped"] == 0
        # Verify TTS was called with extracted prose (no code block)
        assert len(calls) > 0
        encode_text = calls[0][1]
        assert "Hello world." in encode_text
        assert "x = 1" not in encode_text

    def test_generate_dry_run(self, tmp_path, monkeypatch):
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "post-a", "frontmatter": {"title": "A", "tts": True}, "body": "Hello."},
        ])

        from narro.hugo.cli import cmd_hugo_generate
        # dry_run should not need Narro loaded
        result = cmd_hugo_generate(str(site), force=False, dry_run=True, post_slug=None)
        assert result["generated"] == 0
        assert result["pending"] == 1

    def test_generate_single_post(self, tmp_path, monkeypatch):
        site = make_hugo_site(tmp_path, posts=[
            {"slug": "post-a", "frontmatter": {"title": "A", "tts": True}, "body": "Hello."},
            {"slug": "post-b", "frontmatter": {"title": "B", "tts": True}, "body": "World."},
        ])

        class FakeNarro:
            def __init__(self, **kwargs): pass
            def encode(self, text, **kwargs): return "fake"
            def decode_to_wav(self, encoded, path):
                import wave, struct
                with wave.open(path, 'w') as f:
                    f.setnchannels(1); f.setsampwidth(2); f.setframerate(32000)
                    f.writeframes(struct.pack('<h', 0) * 100)

        def fake_run(cmd, **kwargs):
            opus_path = cmd[cmd.index('-y') + 1] if '-y' in cmd else cmd[-1]
            with open(opus_path, 'wb') as f: f.write(b"fake")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("narro.hugo.cli.Narro", FakeNarro)
        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("narro.hugo.cli.extract_alignment_from_encoded", lambda e: [])
        monkeypatch.setattr("narro.hugo.cli.save_alignment", lambda a, p: None)

        from narro.hugo.cli import cmd_hugo_generate
        result = cmd_hugo_generate(str(site), force=False, dry_run=False, post_slug="post-a")

        assert result["generated"] == 1
        assert result["skipped"] == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hugo_cli.py -v`
Expected: ImportError — `narro.hugo.cli` does not exist yet.

**Step 3: Implement the CLI module**

Create `narro/hugo/cli.py`:

```python
"""Hugo integration CLI commands for Narro TTS.

Provides generate, install, and status commands for adding TTS
narration to Hugo blog posts.
"""

import logging
import os
import shutil
import subprocess
import sys

from .extract import extract_prose, parse_frontmatter

logger = logging.getLogger(__name__)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def _validate_site(site_root):
    """Check that site_root looks like a Hugo site."""
    config = os.path.join(site_root, "hugo.toml")
    if not os.path.isfile(config):
        # Also check hugo.yaml and hugo.json
        for alt in ("hugo.yaml", "hugo.json", "config.toml", "config.yaml"):
            if os.path.isfile(os.path.join(site_root, alt)):
                return
        print(f"Error: {site_root} does not look like a Hugo site (no hugo.toml found).", file=sys.stderr)
        sys.exit(1)


def _check_ffmpeg():
    """Verify ffmpeg is available."""
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Install it: sudo apt install ffmpeg", file=sys.stderr)
        sys.exit(1)


def find_tts_posts(site_root):
    """Find all posts with tts: true in frontmatter.

    Returns:
        List of dicts with keys: slug, dir, title, has_audio, body.
    """
    content_dir = os.path.join(site_root, "content", "post")
    if not os.path.isdir(content_dir):
        return []

    posts = []
    for entry in sorted(os.listdir(content_dir)):
        index_path = os.path.join(content_dir, entry, "index.md")
        if not os.path.isfile(index_path):
            continue

        with open(index_path, "r") as f:
            content = f.read()

        meta, body = parse_frontmatter(content)
        if not meta.get("tts"):
            continue

        has_audio = os.path.isfile(os.path.join(content_dir, entry, "narration.opus"))
        posts.append({
            "slug": entry,
            "dir": os.path.join(content_dir, entry),
            "title": meta.get("title", entry),
            "has_audio": has_audio,
            "body": body,
        })

    return posts


def cmd_hugo_install(site_root):
    """Install TTS player assets into a Hugo site."""
    _validate_site(site_root)

    copies = [
        ("tts-player.html", os.path.join("layouts", "partials", "tts-player.html")),
        ("tts-player.js", os.path.join("static", "js", "tts-player.js")),
        ("tts-player.css", os.path.join("static", "css", "tts-player.css")),
    ]

    for asset_name, dest_rel in copies:
        src = os.path.join(ASSETS_DIR, asset_name)
        dest = os.path.join(site_root, dest_rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        print(f"  Installed {dest_rel}")

    print()
    print("Add this line to layouts/post/single.html where you want the player:")
    print('  {{ partial "tts-player.html" . }}')


def cmd_hugo_status(site_root):
    """Show TTS status for all posts."""
    _validate_site(site_root)
    posts = find_tts_posts(site_root)

    if not posts:
        print("No posts with tts: true found.")
        return

    for post in posts:
        status = "audio ready" if post["has_audio"] else "needs generation"
        print(f"  [{status}] {post['slug']} — {post['title']}")

    ready = sum(1 for p in posts if p["has_audio"])
    pending = len(posts) - ready
    print(f"\n{len(posts)} posts with tts: true ({ready} ready, {pending} pending)")


def cmd_hugo_generate(site_root, force=False, dry_run=False, post_slug=None):
    """Generate TTS narration for Hugo posts.

    Args:
        site_root: Path to Hugo site root.
        force: Regenerate even if narration.opus exists.
        dry_run: Show what would be done without generating.
        post_slug: Generate for a single post only.

    Returns:
        Dict with counts: generated, skipped, errors, pending (dry_run only).
    """
    _validate_site(site_root)
    _check_ffmpeg()

    posts = find_tts_posts(site_root)
    if post_slug:
        posts = [p for p in posts if p["slug"] == post_slug]

    if not posts:
        print("No matching posts with tts: true found.")
        return {"generated": 0, "skipped": 0, "errors": 0}

    # Filter to pending posts
    if force:
        pending = posts
    else:
        pending = [p for p in posts if not p["has_audio"]]
        skipped = len(posts) - len(pending)

    if dry_run:
        for p in pending:
            print(f"  [pending] {p['slug']} — {p['title']}")
        print(f"\n{len(pending)} posts would be generated.")
        return {"generated": 0, "skipped": len(posts) - len(pending), "errors": 0, "pending": len(pending)}

    if not pending:
        print("All posts already have narration. Use --force to regenerate.")
        return {"generated": 0, "skipped": len(posts), "errors": 0}

    # Lazy imports — only load the model when we actually generate
    from narro import Narro
    from narro.alignment import extract_alignment_from_encoded, save_alignment

    print("Loading Narro model...")
    tts = Narro()

    generated = 0
    errors = 0
    for post in pending:
        print(f"\n  Generating: {post['slug']} — {post['title']}")
        try:
            # Extract speakable text
            prose = extract_prose(post["body"])
            if not prose.strip():
                print(f"    Skipped: no speakable text found")
                continue

            # Generate audio with alignment
            wav_path = os.path.join(post["dir"], "narration.wav")
            opus_path = os.path.join(post["dir"], "narration.opus")
            json_path = os.path.join(post["dir"], "narration.json")

            encoded = tts.encode(prose, include_attention=True)
            tts.decode_to_wav(encoded, wav_path)

            # Extract and save alignment
            alignment = extract_alignment_from_encoded(encoded)
            save_alignment(alignment, json_path)

            # Convert to Opus
            subprocess.run(
                ["ffmpeg", "-i", wav_path, "-c:a", "libopus", "-b:a", "32k", "-y", opus_path],
                capture_output=True, check=True,
            )

            # Clean up WAV
            os.remove(wav_path)

            generated += 1
            print(f"    Done: narration.opus + narration.json")

        except Exception as e:
            errors += 1
            logger.error("Failed to generate for %s: %s", post['slug'], e)
            print(f"    Error: {e}")
            # Clean up partial files
            for f in ("narration.wav", "narration.opus", "narration.json"):
                path = os.path.join(post["dir"], f)
                if os.path.isfile(path):
                    os.remove(path)

    skipped_count = len(posts) - len(pending)
    print(f"\nDone: {generated} generated, {skipped_count} skipped, {errors} errors")
    return {"generated": generated, "skipped": skipped_count, "errors": errors}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hugo_cli.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add narro/hugo/cli.py tests/test_hugo_cli.py
git commit -m "feat(hugo): add generate, install, status commands"
```

---

### Task 5: Wire hugo subcommand into main CLI

Register the `hugo` subcommand group in the existing argparse-based CLI.

**Files:**
- Modify: `narro/cli.py`

**Step 1: Add hugo to the subcommand set and create sub-parsers**

In `narro/cli.py`, make these changes:

1. Add `'hugo'` to `_subcommands` set (line 89).
2. Add three new `cmd_hugo_*` wrapper functions.
3. Add the `hugo` subparser group after the `decode` parser.

Update `_subcommands`:

```python
_subcommands = {'speak', 'encode', 'decode', 'hugo'}
```

Add wrapper functions (after `cmd_decode`):

```python
def cmd_hugo(args):
    """Dispatch hugo subcommands."""
    if args.hugo_command == 'generate':
        from narro.hugo.cli import cmd_hugo_generate
        cmd_hugo_generate(
            args.site_root,
            force=args.force,
            dry_run=args.dry_run,
            post_slug=args.post,
        )
    elif args.hugo_command == 'install':
        from narro.hugo.cli import cmd_hugo_install
        cmd_hugo_install(args.site_root)
    elif args.hugo_command == 'status':
        from narro.hugo.cli import cmd_hugo_status
        cmd_hugo_status(args.site_root)
    else:
        args._parser.print_help()
```

Add hugo parser group (after the decode parser block):

```python
    # --- hugo ---
    hugo_parser = subparsers.add_parser('hugo', help='Hugo site integration')
    hugo_subparsers = hugo_parser.add_subparsers(dest='hugo_command')
    hugo_parser.set_defaults(func=cmd_hugo, _parser=hugo_parser)

    # hugo generate
    gen_parser = hugo_subparsers.add_parser('generate', help='Generate TTS for posts')
    gen_parser.add_argument('site_root', help='Path to Hugo site root')
    gen_parser.add_argument('--force', action='store_true',
                            help='Regenerate even if audio exists')
    gen_parser.add_argument('--dry-run', action='store_true',
                            help='Show what would be generated')
    gen_parser.add_argument('--post', help='Generate for a single post slug')

    # hugo install
    inst_parser = hugo_subparsers.add_parser('install', help='Install player assets')
    inst_parser.add_argument('site_root', help='Path to Hugo site root')

    # hugo status
    stat_parser = hugo_subparsers.add_parser('status', help='Show TTS status')
    stat_parser.add_argument('site_root', help='Path to Hugo site root')
```

**Step 2: Verify CLI help works**

Run: `narro hugo --help`
Expected: Shows generate/install/status subcommands.

Run: `narro hugo generate --help`
Expected: Shows site_root, --force, --dry-run, --post options.

**Step 3: Verify install works on a test site**

Run: `narro hugo install /tmp/test-hugo-site` (create minimal site first)
Expected: Error about not being a Hugo site.

**Step 4: Commit**

```bash
git add narro/cli.py
git commit -m "feat(hugo): wire hugo subcommand into main CLI"
```

---

### Task 6: Run full test suite and coverage

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All existing tests + new hugo tests pass.

**Step 2: Run coverage**

Run: `pytest tests/ --cov=narro --cov-report=term-missing`
Expected: Coverage stays at 90%+ overall, new hugo modules well-covered.

**Step 3: Fix any issues**

If tests fail, fix and re-run.

**Step 4: Final commit**

```bash
git add -A
git commit -m "test: full suite passes with hugo integration"
```

---

### Task 7: Update CLAUDE.md and pyproject.toml version

**Files:**
- Modify: `CLAUDE.md`
- Modify: `pyproject.toml:7`

**Step 1: Add Hugo integration section to CLAUDE.md**

Add after the "Hallucination Detection" section:

```markdown
### Hugo Integration

`narro/hugo/` provides Hugo site TTS integration:

- `extract.py`: `extract_prose()` strips markdown to speakable text (frontmatter, code, math, images, shortcodes, HTML)
- `cli.py`: `cmd_hugo_generate()`, `cmd_hugo_install()`, `cmd_hugo_status()` — called from `narro hugo <subcommand>`
- `assets/`: Hugo partial (`tts-player.html`), JS player (`tts-player.js`), CSS (`tts-player.css`)

Usage: `narro hugo install <site>` copies assets, then `narro hugo generate <site>` creates narration for posts with `tts: true`.
```

**Step 2: Bump version**

In `pyproject.toml`, bump version from `"0.3.0"` to `"0.4.0"`.

**Step 3: Commit**

```bash
git add CLAUDE.md pyproject.toml
git commit -m "docs: update CLAUDE.md for hugo integration, bump to 0.4.0"
```
