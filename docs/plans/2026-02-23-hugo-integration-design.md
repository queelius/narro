# Narro Hugo Integration Design

## Overview

Add a `narro hugo` CLI subcommand group that generates TTS narration for Hugo blog posts and installs the required Hugo assets (JS player, CSS, layout partial) into a Hugo site.

This lives entirely in the narro repo — not in mf or metafunctor. The `mf` CLI has a tight "sync external sources → DB → Hugo pages" design; TTS generation flows the other direction (Hugo content → audio artifact) and doesn't belong there.

## CLI Surface

```bash
# Generate TTS for all posts with tts: true
narro hugo generate <site-root> [--force] [--dry-run] [--post <slug>]

# Install JS/CSS/partial into a Hugo site
narro hugo install <site-root>

# Show status: which posts have tts:true, which have audio
narro hugo status <site-root>
```

## Generate Pipeline

```
1. Validate site-root (hugo.toml exists)
2. Check ffmpeg available
3. Load Narro() model once
4. Walk content/post/**/index.md
5. For each post:
   a. Parse YAML frontmatter — skip if tts != true
   b. Check if narration.opus exists — skip unless --force
   c. extract_prose(markdown) → plain text
   d. tts.speak() with alignment → narration.wav + narration.json
   e. ffmpeg -i narration.wav -c:a libopus -b:a 32k narration.opus
   f. Delete narration.wav
6. Print summary: N generated, M skipped, K errors
```

Options:
- `--force` — regenerate even if narration.opus exists
- `--dry-run` — show what would be generated, don't run TTS
- `--post <slug>` — generate for a single post only

Generated files land in the page bundle alongside index.md:

```
content/post/2026-02-15-my-post/
├── index.md
├── narration.opus    ← generated
└── narration.json    ← generated
```

## Text Extraction

`extract_prose(markdown_content) → str` strips markdown to speakable text. Processing order matters — earlier steps prevent false matches in later steps:

1. Strip YAML frontmatter (`---...---`)
2. Remove fenced code blocks (``` ```...``` ```)
3. Remove LaTeX math (`$$...$$`, `\[...\]`, `\(...\)`)
4. Remove images (`![alt](url)`)
5. Remove Hugo shortcodes (`{{< ... >}}`, `{{< ... />}}`)
6. Remove HTML tags
7. Convert markdown links to text (`[text](url)` → `text`)
8. Strip heading markers (`## Foo` → `Foo`)
9. Collapse whitespace

Lives in `narro/hugo/extract.py`, testable independently.

## Hugo Assets

Three files installed by `narro hugo install <site-root>`:

### `layouts/partials/tts-player.html`

Hugo partial that auto-injects when `tts: true` and `narration.opus` exists:

```html
{{ if .Params.tts }}
  {{ $audio := .Resources.GetMatch "narration.opus" }}
  {{ $align := .Resources.GetMatch "narration.json" }}
  {{ if $audio }}
    <link rel="stylesheet" href="/css/tts-player.css">
    <div class="tts-player" data-align="{{ if $align }}{{ $align.RelPermalink }}{{ end }}">
      <audio src="{{ $audio.RelPermalink }}" preload="metadata"></audio>
      <button class="tts-play" aria-label="Play narration">&#9654; Listen</button>
      <span class="tts-time"></span>
    </div>
    <script src="/js/tts-player.js" defer></script>
  {{ end }}
{{ end }}
```

User adds one line to their `layouts/post/single.html`:

```
{{ partial "tts-player.html" . }}
```

### `static/js/tts-player.js` (~80 lines)

- On load: fetch alignment JSON, walk article DOM text nodes, wrap words in `<span class="tts-word" data-idx="N">` via sequential matching
- On `timeupdate` (~4Hz): binary search alignment array, toggle `.tts-active` on matching span
- Controls: play/pause button, click-word-to-seek, spacebar toggle
- Graceful degradation: no alignment JSON → plain audio player without highlighting

Word wrapping happens client-side (JS walks rendered DOM), keeping markdown source clean.

### `static/css/tts-player.css`

```css
.tts-player { display: flex; align-items: center; padding: 0.75rem 1rem; border-radius: 8px; }
.tts-play { cursor: pointer; border: none; background: none; font-size: 1rem; }
.tts-word.tts-active {
  background-color: rgba(37, 99, 235, 0.12);
  border-radius: 2px;
  transition: background-color 0.15s ease;
}
.tts-time { font-size: 0.8rem; color: var(--text-muted); margin-left: auto; }

@media (prefers-reduced-motion: reduce) {
  .tts-word.tts-active { transition: none; }
}
```

## Package Structure

New files in narro repo:

```
narro/
├── hugo/
│   ├── __init__.py
│   ├── cli.py          # Click commands: generate, install, status
│   ├── extract.py      # extract_prose()
│   └── assets/         # Copied by install
│       ├── tts-player.html
│       ├── tts-player.js
│       └── tts-player.css
tests/
├── test_hugo_extract.py    # extract_prose unit tests
├── test_hugo_cli.py        # CLI integration tests (mocked TTS)
```

The `hugo` subcommand group registers in `narro/cli.py` alongside existing `speak`/`encode`/`decode`.

## Dependencies

- No new pip dependencies
- ffmpeg: system requirement, checked at runtime with clear error message
- Audio: Opus at 32kbps (~50KB/min of narration)

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Where it lives | narro repo | mf has a different data flow pattern; TTS is content → audio, not source → DB → Hugo |
| Install method | File copy | Simple, user owns files after install |
| Player injection | Auto-inject via partial | Minimal author friction (one-line layout edit, then just `tts: true`) |
| Word highlighting | Client-side DOM walking | Keeps markdown source clean |
| Batch model loading | Single load, iterate posts | ~10s startup once vs per-post |
| Skip logic | File existence check | Simple; `--force` for re-generation |
| Audio format | Opus 32kbps | ~50KB/min, browser-native playback |
