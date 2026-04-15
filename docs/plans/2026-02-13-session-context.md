# Session Context — Resume After Directory Move

## What Just Happened

We completed the full narro rename and implementation plan (9/9 tasks done). The repo is at `queelius/narro` on GitHub. All code is committed and pushed.

## Current State

- **Branch**: `main`
- **HEAD**: `0bf8a30` — "fix: alignment duplicate word collapse, zero-spread, CLI cleanup"
- **Tests**: 263 pass, 5 skipped, 96% coverage
- **Package**: `narro` installed editable (`pip install -e .`)
- **Remote**: `origin` -> `https://github.com/queelius/narro.git`
- **All code pushed**: Yes, up to date with origin/main

## What's Done

1. Renamed `soprano` -> `narro` (package, module dir, class, CLI, pyproject.toml)
2. `SopranoTTS` -> `Narro` everywhere; `SopranoDecoder` stays (upstream model name)
3. `.soprano` file format extension preserved (it's a format name, not a package name)
4. Deleted `js/` (browser decoder) and `exports/` (ONNX export)
5. Created fresh `queelius/narro` repo on GitHub (not a fork)
6. Added `--align` feature: word-level timestamps from LLM attention weights
   - `narro/alignment.py` — center-of-mass algorithm
   - CLI: `narro "text" -o out.wav --align alignment.json`
   - Handles duplicate words (instance tracking by adjacency)
   - Minimum spread width (token_duration/2)
7. Updated CLAUDE.md, README.md, auto-memory

## What's Next

### Immediate: Move directory
```bash
mv ~/github/repos/tts/soprano ~/github/repos/tts/narro
cd ~/github/repos/tts/narro
pip install -e .
```

### Then: Metafunctor integration (the `mf` workflow)

The user wants to use narro to generate TTS audio for blog posts on metafunctor.com (Hugo site). NOT registering narro as a project page — using it as a content tool.

**Discussed design (from `docs/plans/2026-02-13-narro-rename-design.md`):**

- Hugo frontmatter opt-in: `tts: true`
- Hugo shortcode: `{{< tts src="audio/my-post.opus" align="audio/my-post.json" >}}`
- Build step: run narro on posts with `tts: true` to generate audio + alignment JSON
- JS player (~50 lines): listens to `timeupdate`, binary searches alignment array, toggles CSS class on word `<span>` elements

**Open questions for next session:**
1. Should `mf` get a new subcommand (e.g., `mf content tts`) that finds posts with `tts: true` and runs narro?
2. Or is this a separate build script / Hugo pipe?
3. Audio format: WAV is huge — need to convert to Opus for serving. Where does that happen?
4. How does the shortcode work? Does it wrap the post text in `<span>` elements for highlighting?
5. The `--align` feature currently requires `include_attention=True` during encoding. This doubles memory usage. For a build-time tool this is fine, but worth noting.

### Also pending
- The old `soprano` project entry in `mf projects` database needs cleanup (run `mf projects sync` from metafunctor site root)
- PyPI publish of `narro` (not yet done)
- CLAUDE.md in the repo still references some old paths/names — should be audited after the directory move

## Key Files

| File | Purpose |
|------|---------|
| `narro/tts.py` | Main `Narro` class |
| `narro/alignment.py` | Word-level timestamp extraction |
| `narro/cli.py` | CLI with speak/encode/decode + --align |
| `narro/encoded.py` | EncodedSpeech IR dataclasses |
| `narro/decode_only.py` | Standalone decoder (no LLM) |
| `docs/plans/2026-02-13-narro-rename-design.md` | Design doc |
| `docs/plans/2026-02-13-narro-implementation.md` | Implementation plan (completed) |

## Commit History (recent)

```
0bf8a30 fix: alignment duplicate word collapse, zero-spread, CLI cleanup
0b3ada9 feat: add --align flag for word-level timestamp extraction
a8e852b Update CLAUDE.md and README.md for narro rename
8dd48b8 Fix: revert .soprano file extension (format name, not package name)
7b94bee Update all test imports soprano -> narro
2ba6bc4 Update all internal imports soprano -> narro, rename SopranoTTS -> Narro
5fa2c6b Rename module directory soprano/ -> narro/
0b37049 Remove JS browser decoder and ONNX exports
```
