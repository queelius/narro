# Metafunctor TTS Integration Design

## Summary

Add TTS narration with word-level highlighting to metafunctor.com blog posts
using narro. Authors opt in via frontmatter (`tts: true`), a build step
generates audio + alignment data, and the Hugo layout auto-injects a player
with synchronized word highlighting.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Build step | `mf content tts` subcommand | Fits existing mf CLI patterns; leverages FrontMatterEditor |
| Audio format | Opus via ffmpeg | ~50KB/min vs ~3MB/min for WAV; universal browser support |
| File location | Page bundles | Assets alongside index.md; Hugo resource pipeline |
| Player UX | Auto-inject in layout | No shortcode needed; `tts: true` is the only author action |
| Highlighting | Hybrid (Approach C) | JS walks DOM text nodes, matches words from alignment JSON sequentially |
| Git strategy | Commit audio to repo | Matches existing GitHub Pages deployment model (docs/ committed) |

## Content Pipeline

```
Author sets tts: true in frontmatter
         │
         ▼
  mf content tts
         │
         ▼
  Find posts with tts: true (glob content/post/**/index.md)
         │
         ▼
  For each post missing narration.opus (or --force):
    1. Extract plain text from markdown
    2. Run: narro "<text>" -o narration.wav --align narration.json
    3. Run: ffmpeg -i narration.wav -c:a libopus -b:a 32k narration.opus
    4. Delete narration.wav
    5. Assets placed in page bundle directory
         │
         ▼
  Hugo build picks up narration.opus + narration.json as page resources
         │
         ▼
  post/single.html auto-injects player when tts: true
```

The command is **idempotent** — skips posts that already have `narration.opus`.
Pass `--force` to regenerate. Pass `--dry-run` to preview.

## Components

### 1. `mf content tts` subcommand

Location: `scripts/mf/commands/content.py`

```python
@content.command()
@click.option("--force", is_flag=True, help="Regenerate even if audio exists")
@click.option("--dry-run", is_flag=True, help="Show what would be generated")
def tts(force, dry_run):
    """Generate TTS narration for posts with tts: true."""
    posts = find_posts_with_tts()
    for post_path in posts:
        bundle_dir = post_path.parent
        opus_path = bundle_dir / "narration.opus"
        if opus_path.exists() and not force:
            click.echo(f"  skip: {post_path.name} (already has audio)")
            continue
        if dry_run:
            click.echo(f"  would generate: {bundle_dir}")
            continue
        text = extract_prose(post_path)
        wav_path = bundle_dir / "narration.wav"
        json_path = bundle_dir / "narration.json"
        subprocess.run(["narro", text, "-o", str(wav_path), "--align", str(json_path)], check=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_path), "-c:a", "libopus", "-b:a", "32k",
                        str(opus_path)], check=True)
        wav_path.unlink()
        click.echo(f"  done: {bundle_dir.name}")
```

### 2. Text extraction (`extract_prose`)

Strips markdown to speakable plain text:

- Remove YAML frontmatter (`---` blocks)
- Remove fenced code blocks (``` and ~~~)
- Remove indented code blocks
- Remove LaTeX math (`$$...$$`, `\[...\]`, `\(...\)`)
- Remove images (`![alt](url)`)
- Keep link text, remove URL (`[text](url)` → `text`)
- Remove HTML tags
- Collapse whitespace
- Strip Hugo shortcodes (`{{< ... >}}`, `{{% ... %}}`)

### 3. Hugo layout changes (`layouts/post/single.html`)

Conditional block when `tts: true`:

```html
{{ if .Params.tts }}
  {{ $audio := .Resources.GetMatch "narration.opus" }}
  {{ $align := .Resources.GetMatch "narration.json" }}
  {{ if $audio }}
    <div class="tts-player" data-align="{{ if $align }}{{ $align.RelPermalink }}{{ end }}">
      <audio src="{{ $audio.RelPermalink }}" preload="metadata"></audio>
      <button class="tts-play" aria-label="Play narration">
        ▶ Listen to this post
      </button>
      <span class="tts-time"></span>
    </div>
    <script src="/js/tts-player.js" defer></script>
  {{ end }}
{{ end }}
```

Placed above the article content. The `.article-content` div (or equivalent
prose container) is the target for word wrapping.

### 4. JS player (`static/js/tts-player.js`)

~80 lines. Three phases:

**Initialization:**
- Fetch alignment JSON from `data-align` URL
- Walk text nodes inside the prose container
- Split each text node on word boundaries (`/\S+/g`)
- Replace each word with `<span class="tts-word" data-idx="N">word</span>`
- Build a flat array mapping idx → alignment entry

**Sequential word matching:**
- Alignment JSON words are matched to DOM text words in order
- Non-text elements (code, images, SVG) are skipped
- If a word doesn't match, advance the DOM pointer (tolerate formatting diffs)
- Unmatched alignment words are silently skipped

**Playback:**
- `timeupdate` event on `<audio>` fires ~4x/sec
- Binary search alignment array for current timestamp
- Add `.tts-active` to current word span, remove from previous
- Update time display

**Controls:**
- Click play/pause button toggles playback
- Click on a word seeks audio to that word's timestamp
- Keyboard: space = play/pause

### 5. CSS (`static/css/tts-player.css`)

```css
.tts-player {
  /* Inline bar above content */
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  margin-bottom: 1.5rem;
  border-radius: 8px;
  background: var(--surface-subtle, #f8f9fa);
  border: 1px solid var(--border-color, #e0e0e0);
}

.tts-play {
  /* Minimal button matching site theme */
  cursor: pointer;
  border: none;
  background: none;
  font-size: 0.9rem;
  color: var(--accent-color, #2563eb);
}

.tts-word.tts-active {
  background-color: rgba(37, 99, 235, 0.12);
  border-radius: 2px;
  transition: background-color 0.15s ease;
}

.tts-time {
  font-size: 0.8rem;
  color: var(--text-muted, #6b7280);
  margin-left: auto;
}
```

## File Changes

### Metafunctor repo (`metafunctor/`)

| File | Action |
|------|--------|
| `scripts/mf/commands/content.py` | Add `tts` subcommand |
| `layouts/post/single.html` | Add TTS player conditional block |
| `static/js/tts-player.js` | New — player with word highlighting |
| `static/css/tts-player.css` | New — player and highlight styles |
| `content/post/*/narration.opus` | Generated — audio files |
| `content/post/*/narration.json` | Generated — alignment data |

### Narro repo (`narro/`)

No changes needed. The existing `--align` flag and Python API are sufficient.

## Open Considerations

- **Long posts:** narro processes text sentence-by-sentence. Very long posts
  (5000+ words) may take several minutes. The `mf content tts` command should
  show progress per post.
- **Text changes:** If a post is edited after TTS generation, the audio and
  alignment will be stale. The author must re-run `mf content tts --force` or
  delete the opus file. No automatic staleness detection.
- **ffmpeg dependency:** Required on the build machine. The command should check
  for ffmpeg and give a clear error if missing.
- **Repo size:** A 5-minute narration in Opus at 32kbps is ~1.2MB. With 10-20
  narrated posts, this adds 12-24MB to the repo — manageable. Git LFS can be
  added later if needed.
