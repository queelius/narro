# Metafunctor TTS Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add TTS narration with synchronized word highlighting to metafunctor.com blog posts, triggered by `tts: true` in frontmatter.

**Architecture:** An `mf content tts` subcommand finds posts with `tts: true`, generates audio via narro + ffmpeg, and places `.opus` + `.json` files in page bundles. The Hugo post layout auto-injects a player that highlights words during playback using the alignment JSON.

**Tech Stack:** Python (Click CLI), narro (TTS), ffmpeg (WAV→Opus), Hugo (templates), vanilla JS (~80 lines), CSS.

**Repos involved:**
- `~/github/repos/narro/` — no code changes, only this plan doc
- `~/github/repos/metafunctor/` — all implementation work happens here

---

### Task 1: Text extraction utility

**Files:**
- Create: `scripts/mf/src/mf/content/tts.py`
- Create: `scripts/mf/tests/test_tts.py`

**Step 1: Write the failing tests**

Create `scripts/mf/tests/test_tts.py`:

```python
"""Tests for TTS text extraction."""
import textwrap
from mf.content.tts import extract_prose


def test_strips_frontmatter():
    md = textwrap.dedent("""\
        ---
        title: Hello
        date: 2025-01-01
        ---
        This is the body.
    """)
    assert extract_prose(md) == "This is the body."


def test_strips_fenced_code_blocks():
    md = textwrap.dedent("""\
        Some text before.

        ```python
        x = 1
        ```

        Some text after.
    """)
    result = extract_prose(md)
    assert "x = 1" not in result
    assert "Some text before." in result
    assert "Some text after." in result


def test_strips_latex_math():
    md = r"Inline math \(x^2\) and block $$\sum_{i=1}^{n} i$$."
    result = extract_prose(md)
    assert "x^2" not in result
    assert "sum" not in result
    assert "Inline math" in result


def test_strips_images_keeps_link_text():
    md = "See ![alt text](image.png) and [click here](url)."
    result = extract_prose(md)
    assert "alt text" not in result
    assert "click here" in result
    assert "image.png" not in result
    assert "url" not in result


def test_strips_html_tags():
    md = "Hello <strong>world</strong> and <br/> done."
    result = extract_prose(md)
    assert "<strong>" not in result
    assert "world" in result


def test_strips_hugo_shortcodes():
    md = 'Before {{< tts src="x" >}} and {{% figure %}} after.'
    result = extract_prose(md)
    assert "{{" not in result
    assert "Before" in result
    assert "after." in result


def test_collapses_whitespace():
    md = "Hello   world\n\n\nfoo   bar"
    result = extract_prose(md)
    assert result == "Hello world foo bar"


def test_strips_headings_markup():
    md = "## Section Title\n\nParagraph text."
    result = extract_prose(md)
    assert "##" not in result
    assert "Section Title" in result
    assert "Paragraph text." in result
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/github/repos/metafunctor && python -m pytest scripts/mf/tests/test_tts.py -v`
Expected: ImportError — `mf.content.tts` does not exist yet.

**Step 3: Implement extract_prose**

Create `scripts/mf/src/mf/content/tts.py`:

```python
"""TTS utilities for generating narration of blog posts."""
from __future__ import annotations

import re


def extract_prose(markdown: str) -> str:
    """Extract speakable plain text from a markdown string.

    Strips frontmatter, code blocks, math, images, HTML, shortcodes,
    and heading markers. Keeps link text. Collapses whitespace.
    """
    text = markdown

    # Strip YAML frontmatter
    text = re.sub(r'\A---\n.*?\n---\n?', '', text, flags=re.DOTALL)

    # Strip fenced code blocks (``` or ~~~)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'~~~.*?~~~', '', text, flags=re.DOTALL)

    # Strip Hugo shortcodes {{< ... >}} and {{% ... %}}
    text = re.sub(r'\{\{[<%].*?[%>]\}\}', '', text, flags=re.DOTALL)

    # Strip LaTeX math: $$...$$, \[...\], \(...\)
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', '', text)

    # Strip images ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Convert links [text](url) to just text
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)

    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Strip heading markers (## etc.)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Strip bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/github/repos/metafunctor && python -m pytest scripts/mf/tests/test_tts.py -v`
Expected: All 8 tests PASS.

**Step 5: Commit**

```bash
cd ~/github/repos/metafunctor
git add scripts/mf/src/mf/content/tts.py scripts/mf/tests/test_tts.py
git commit -m "feat(mf): add extract_prose for TTS text extraction"
```

---

### Task 2: `mf content tts` subcommand

**Files:**
- Modify: `scripts/mf/src/mf/content/commands.py` (add `tts` command)
- Modify: `scripts/mf/src/mf/content/tts.py` (add orchestration functions)
- Create: `scripts/mf/tests/test_tts_command.py`

**Step 1: Write the failing test**

Create `scripts/mf/tests/test_tts_command.py`:

```python
"""Tests for mf content tts command."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from mf.content.commands import content


def make_post(tmp_path, slug="test-post", tts=True, has_audio=False):
    """Create a fake post directory with index.md."""
    post_dir = tmp_path / "content" / "post" / slug
    post_dir.mkdir(parents=True)
    fm = f"---\ntitle: Test\ndate: 2025-01-01\ntts: {str(tts).lower()}\n---\n\nHello world."
    (post_dir / "index.md").write_text(fm)
    if has_audio:
        (post_dir / "narration.opus").write_bytes(b"fake")
        (post_dir / "narration.json").write_text('[{"word":"Hello","start":0.0,"end":0.5}]')
    return post_dir


def test_dry_run_lists_posts(tmp_path):
    make_post(tmp_path, "my-post", tts=True)
    runner = CliRunner()
    result = runner.invoke(content, ["tts", "--dry-run", "--content-dir", str(tmp_path / "content")])
    assert result.exit_code == 0
    assert "my-post" in result.output
    assert "would generate" in result.output


def test_skips_posts_without_tts(tmp_path):
    make_post(tmp_path, "no-tts", tts=False)
    runner = CliRunner()
    result = runner.invoke(content, ["tts", "--dry-run", "--content-dir", str(tmp_path / "content")])
    assert result.exit_code == 0
    assert "no-tts" not in result.output


def test_skips_existing_audio(tmp_path):
    make_post(tmp_path, "has-audio", tts=True, has_audio=True)
    runner = CliRunner()
    result = runner.invoke(content, ["tts", "--dry-run", "--content-dir", str(tmp_path / "content")])
    assert result.exit_code == 0
    assert "skip" in result.output


def test_force_regenerates(tmp_path):
    make_post(tmp_path, "has-audio", tts=True, has_audio=True)
    runner = CliRunner()
    result = runner.invoke(content, ["tts", "--dry-run", "--force", "--content-dir", str(tmp_path / "content")])
    assert result.exit_code == 0
    assert "would generate" in result.output
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/github/repos/metafunctor && python -m pytest scripts/mf/tests/test_tts_command.py -v`
Expected: FAIL — `tts` command doesn't exist yet.

**Step 3: Add orchestration functions to tts.py**

Add to `scripts/mf/src/mf/content/tts.py`:

```python
import shutil
import subprocess
from pathlib import Path

import yaml


def find_posts_with_tts(content_dir: Path) -> list[Path]:
    """Find all posts with tts: true in frontmatter."""
    posts = []
    for index_md in sorted(content_dir.glob("post/**/index.md")):
        text = index_md.read_text()
        # Quick parse of YAML frontmatter
        if text.startswith("---"):
            end = text.index("---", 3)
            fm = yaml.safe_load(text[3:end])
            if fm and fm.get("tts"):
                posts.append(index_md)
    return posts


def generate_narration(post_path: Path, force: bool = False, dry_run: bool = False) -> str | None:
    """Generate TTS narration for a single post.

    Returns a status string or None if skipped.
    """
    bundle_dir = post_path.parent
    opus_path = bundle_dir / "narration.opus"

    if opus_path.exists() and not force:
        return "skip"

    if dry_run:
        return "would generate"

    # Check dependencies
    if not shutil.which("narro"):
        raise RuntimeError("narro CLI not found. Install with: pip install narro")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install with your package manager.")

    text = extract_prose(post_path.read_text())
    if not text.strip():
        return "skip (empty)"

    wav_path = bundle_dir / "narration.wav"
    json_path = bundle_dir / "narration.json"

    # Generate audio + alignment
    subprocess.run(
        ["narro", text, "-o", str(wav_path), "--align", str(json_path)],
        check=True,
    )

    # Convert WAV to Opus
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-c:a", "libopus", "-b:a", "32k", str(opus_path)],
        check=True,
        capture_output=True,
    )

    # Clean up WAV
    wav_path.unlink()

    return "done"
```

**Step 4: Add the tts command to commands.py**

Add to the bottom of `scripts/mf/src/mf/content/commands.py`:

```python
@content.command()
@click.option("--force", is_flag=True, help="Regenerate even if audio exists")
@click.option("--dry-run", is_flag=True, help="Show what would be generated")
@click.option("--content-dir", type=click.Path(exists=True, path_type=Path),
              default=None, help="Content directory (default: auto-detect)")
def tts(force: bool, dry_run: bool, content_dir: Path | None) -> None:
    """Generate TTS narration for posts with tts: true."""
    from mf.content.tts import find_posts_with_tts, generate_narration

    if content_dir is None:
        content_dir = Path("content")

    posts = find_posts_with_tts(content_dir)
    if not posts:
        click.echo("No posts with tts: true found.")
        return

    click.echo(f"Found {len(posts)} post(s) with tts: true")
    for post_path in posts:
        slug = post_path.parent.name
        status = generate_narration(post_path, force=force, dry_run=dry_run)
        click.echo(f"  {status}: {slug}")
```

Add `from pathlib import Path` to the imports at the top of `commands.py`.

**Step 5: Run tests to verify they pass**

Run: `cd ~/github/repos/metafunctor && python -m pytest scripts/mf/tests/test_tts_command.py -v`
Expected: All 4 tests PASS.

**Step 6: Commit**

```bash
cd ~/github/repos/metafunctor
git add scripts/mf/src/mf/content/tts.py scripts/mf/src/mf/content/commands.py scripts/mf/tests/test_tts_command.py
git commit -m "feat(mf): add mf content tts command"
```

---

### Task 3: Hugo layout — TTS player injection

**Files:**
- Modify: `layouts/post/single.html` (add player block)

**Step 1: Add the TTS player block**

In `layouts/post/single.html`, insert between the series navigation (line 55) and the content div (line 57):

```html
      {{ partial "series-navigation.html" . }}

      {{ if .Params.tts }}
      {{ $audio := .Resources.GetMatch "narration.opus" }}
      {{ $align := .Resources.GetMatch "narration.json" }}
      {{ if $audio }}
      <div class="tts-player" id="tts-player"
           {{ if $align }}data-align="{{ $align.RelPermalink }}"{{ end }}>
        <audio id="tts-audio" src="{{ $audio.RelPermalink }}" preload="metadata"></audio>
        <button class="tts-play" id="tts-play-btn" aria-label="Play narration">
          <span class="tts-play-icon">&#9654;</span> Listen to this post
        </button>
        <span class="tts-time" id="tts-time"></span>
      </div>
      <link rel="stylesheet" href="/css/tts-player.css">
      {{ end }}
      {{ end }}

      <div class="content" itemprop="articleBody">
        {{ .Content }}
      </div>
```

And add the JS at the bottom of the template, just before the final `{{ end }}`:

```html
      {{ if .Params.tts }}
      {{ $audio := .Resources.GetMatch "narration.opus" }}
      {{ if $audio }}
      <script src="/js/tts-player.js" defer></script>
      {{ end }}
      {{ end }}
```

**Step 2: Verify locally**

Run: `cd ~/github/repos/metafunctor && make serve`
Visit a test post. Without `tts: true`, no player appears. This is a visual check.

**Step 3: Commit**

```bash
cd ~/github/repos/metafunctor
git add layouts/post/single.html
git commit -m "feat: add TTS player injection in post layout"
```

---

### Task 4: JS player with word highlighting

**Files:**
- Create: `static/js/tts-player.js`

**Step 1: Create the player JS**

Create `static/js/tts-player.js`. Uses safe DOM methods only (no innerHTML):

```javascript
(function () {
  'use strict';

  var player = document.getElementById('tts-player');
  if (!player) return;

  var audio = document.getElementById('tts-audio');
  var playBtn = document.getElementById('tts-play-btn');
  var timeDisplay = document.getElementById('tts-time');
  var alignUrl = player.getAttribute('data-align');
  var alignment = null;
  var wordSpans = [];
  var activeIdx = -1;

  // Safe button text update (no innerHTML)
  function setButtonState(playing) {
    // Clear button contents safely
    while (playBtn.firstChild) playBtn.removeChild(playBtn.firstChild);
    var icon = document.createElement('span');
    icon.className = 'tts-play-icon';
    if (playing) {
      icon.textContent = '\u23F8';  // pause symbol
      playBtn.appendChild(icon);
      playBtn.appendChild(document.createTextNode(' Pause'));
    } else {
      icon.textContent = '\u25B6';  // play symbol
      playBtn.appendChild(icon);
      playBtn.appendChild(document.createTextNode(' Listen to this post'));
    }
  }

  // Format seconds as m:ss
  function formatTime(s) {
    var m = Math.floor(s / 60);
    var sec = Math.floor(s % 60);
    return m + ':' + (sec < 10 ? '0' : '') + sec;
  }

  // Binary search for current word index
  function findWordAt(time) {
    if (!alignment || alignment.length === 0) return -1;
    var lo = 0, hi = alignment.length - 1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (alignment[mid].end < time) lo = mid + 1;
      else if (alignment[mid].start > time) hi = mid - 1;
      else return mid;
    }
    return -1;
  }

  // Wrap text nodes in the content area with word spans
  function wrapWords(container) {
    var walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        var tag = node.parentElement ? node.parentElement.tagName : '';
        if (tag === 'CODE' || tag === 'PRE' || tag === 'SCRIPT' || tag === 'STYLE' || tag === 'KBD') {
          return NodeFilter.FILTER_REJECT;
        }
        if (node.textContent.trim().length === 0) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);

    nodes.forEach(function (textNode) {
      var frag = document.createDocumentFragment();
      var parts = textNode.textContent.split(/(\s+)/);
      parts.forEach(function (part) {
        if (/^\s+$/.test(part)) {
          frag.appendChild(document.createTextNode(part));
        } else if (part.length > 0) {
          var span = document.createElement('span');
          span.className = 'tts-word';
          span.textContent = part;
          frag.appendChild(span);
        }
      });
      textNode.parentNode.replaceChild(frag, textNode);
    });
  }

  // Match alignment words to DOM spans sequentially
  function matchWords() {
    var domWords = document.querySelectorAll('.content .tts-word');
    if (!alignment || domWords.length === 0) return;

    wordSpans = new Array(alignment.length);
    var domIdx = 0;
    for (var i = 0; i < alignment.length; i++) {
      var target = alignment[i].word.replace(/[^\w]/g, '').toLowerCase();
      while (domIdx < domWords.length) {
        var domText = domWords[domIdx].textContent.replace(/[^\w]/g, '').toLowerCase();
        if (domText === target) {
          wordSpans[i] = domWords[domIdx];
          domWords[domIdx].setAttribute('data-tts-idx', i);
          domIdx++;
          break;
        }
        domIdx++;
      }
    }
  }

  // Highlight the word at the given alignment index
  function highlight(idx) {
    if (idx === activeIdx) return;
    if (activeIdx >= 0 && wordSpans[activeIdx]) {
      wordSpans[activeIdx].classList.remove('tts-active');
    }
    activeIdx = idx;
    if (idx >= 0 && wordSpans[idx]) {
      wordSpans[idx].classList.add('tts-active');
    }
  }

  // Play/pause toggle
  playBtn.addEventListener('click', function () {
    if (audio.paused) {
      audio.play();
      setButtonState(true);
    } else {
      audio.pause();
      setButtonState(false);
    }
  });

  // Time update — highlight current word
  audio.addEventListener('timeupdate', function () {
    var t = audio.currentTime;
    timeDisplay.textContent = formatTime(t) + ' / ' + formatTime(audio.duration || 0);
    highlight(findWordAt(t));
  });

  // Reset on end
  audio.addEventListener('ended', function () {
    setButtonState(false);
    highlight(-1);
  });

  // Click word to seek
  document.querySelector('.content').addEventListener('click', function (e) {
    var span = e.target.closest('.tts-word[data-tts-idx]');
    if (!span) return;
    var idx = parseInt(span.getAttribute('data-tts-idx'), 10);
    if (alignment[idx]) {
      audio.currentTime = alignment[idx].start;
      if (audio.paused) audio.play();
      setButtonState(true);
    }
  });

  // Initialize
  var contentEl = document.querySelector('.content[itemprop="articleBody"]');
  if (contentEl) wrapWords(contentEl);

  if (alignUrl) {
    fetch(alignUrl)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        alignment = data;
        matchWords();
      })
      .catch(function (err) {
        console.warn('TTS: Failed to load alignment:', err);
      });
  }
})();
```

**Step 2: Commit**

```bash
cd ~/github/repos/metafunctor
git add static/js/tts-player.js
git commit -m "feat: add TTS player JS with word highlighting"
```

---

### Task 5: CSS for TTS player and word highlighting

**Files:**
- Create: `static/css/tts-player.css`

**Step 1: Create the CSS**

Create `static/css/tts-player.css`:

```css
/* TTS Player Bar */
.tts-player {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  margin-bottom: 1.5rem;
  border-radius: 8px;
  background: var(--bg-secondary, #f8f9fa);
  border: 1px solid var(--border-color, #e0e0e0);
}

.tts-play {
  cursor: pointer;
  border: 1px solid var(--accent-color, #2563eb);
  background: transparent;
  color: var(--accent-color, #2563eb);
  padding: 0.4rem 1rem;
  border-radius: 6px;
  font-size: 0.9rem;
  font-family: inherit;
  transition: background-color 0.15s, color 0.15s;
  white-space: nowrap;
}

.tts-play:hover {
  background: var(--accent-color, #2563eb);
  color: #fff;
}

.tts-play-icon {
  font-size: 0.8em;
}

.tts-time {
  font-size: 0.8rem;
  color: var(--text-muted, #6b7280);
  margin-left: auto;
  font-variant-numeric: tabular-nums;
}

/* Word Highlighting */
.tts-word {
  cursor: default;
  border-radius: 2px;
  transition: background-color 0.12s ease;
}

.tts-word[data-tts-idx] {
  cursor: pointer;
}

.tts-word.tts-active {
  background-color: rgba(37, 99, 235, 0.15);
  box-shadow: 0 1px 3px rgba(37, 99, 235, 0.1);
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
  .tts-player {
    background: var(--bg-secondary, #1e293b);
    border-color: var(--border-color, #334155);
  }

  .tts-word.tts-active {
    background-color: rgba(96, 165, 250, 0.2);
  }
}

/* Responsive */
@media (max-width: 640px) {
  .tts-player {
    flex-wrap: wrap;
  }

  .tts-time {
    margin-left: 0;
    width: 100%;
    text-align: center;
  }
}
```

**Step 2: Commit**

```bash
cd ~/github/repos/metafunctor
git add static/css/tts-player.css
git commit -m "feat: add TTS player and word highlight CSS"
```

---

### Task 6: End-to-end test with a real post

**Files:**
- Modify: one test post's frontmatter (add `tts: true`)

**Step 1: Pick a short post and add `tts: true`**

Find a short post (~200 words) in `content/post/`. Add `tts: true` to its frontmatter.

**Step 2: Run `mf content tts --dry-run`**

Run: `cd ~/github/repos/metafunctor && mf content tts --dry-run`
Expected: Shows the selected post as "would generate".

**Step 3: Generate narration**

Run: `cd ~/github/repos/metafunctor && mf content tts`
Expected: Generates `narration.opus` and `narration.json` in the post's page bundle.

**Step 4: Verify the generated files**

```bash
ls -la content/post/<slug>/narration.*
cat content/post/<slug>/narration.json | python -m json.tool | head -20
```

Expected: `.opus` file exists (few hundred KB), `.json` has word/start/end entries.

**Step 5: Test locally in browser**

Run: `cd ~/github/repos/metafunctor && make serve`
Visit the post. Verify:
- Player bar appears above content
- Play button works and audio plays
- Words highlight during playback
- Clicking a word seeks to that position
- Time display updates

**Step 6: Commit**

```bash
cd ~/github/repos/metafunctor
git add content/post/<slug>/narration.opus content/post/<slug>/narration.json content/post/<slug>/index.md
git commit -m "feat: add TTS narration to test post"
```

---

### Task 7: Makefile integration and cleanup

**Files:**
- Modify: `Makefile` (add `tts` target)

**Step 1: Add Makefile target**

Add to `Makefile`:

```makefile
tts:  ## Generate TTS for posts with tts: true
	mf content tts

tts-force:  ## Regenerate all TTS narrations
	mf content tts --force
```

**Step 2: Verify**

Run: `cd ~/github/repos/metafunctor && make tts`
Expected: Skips already-generated posts or processes new ones.

**Step 3: Commit**

```bash
cd ~/github/repos/metafunctor
git add Makefile
git commit -m "feat: add make tts target for narration generation"
```

---

## Summary

| Task | What | Where |
|------|------|-------|
| 1 | Text extraction (`extract_prose`) + tests | `scripts/mf/src/mf/content/tts.py` |
| 2 | `mf content tts` command + tests | `scripts/mf/src/mf/content/commands.py` |
| 3 | Hugo layout player injection | `layouts/post/single.html` |
| 4 | JS player with word highlighting | `static/js/tts-player.js` |
| 5 | CSS for player and highlights | `static/css/tts-player.css` |
| 6 | End-to-end test with real post | Manual verification |
| 7 | Makefile integration | `Makefile` |

Tasks 1-2 are TDD (tests first). Tasks 3-5 are frontend (visual verification). Task 6 is integration testing. Task 7 is wiring.
