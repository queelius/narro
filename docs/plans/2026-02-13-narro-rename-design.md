# Narro: Rename and Simplification Design

## What narro is

A lightweight CPU text-to-speech tool. Text in, audio out. Wraps the
Soprano-1.1-80M model (Qwen3 LLM + Vocos decoder). Available on PyPI as `narro`.

Latin: "I narrate, I tell."

## Primary interface

### CLI

```bash
narro "Hello world" -o output.wav
narro "Hello world" -o output.wav --align alignment.json
```

Default subcommand is `speak`. Encode/decode subcommands stay for power users.

### Python API

```python
from narro import Narro

tts = Narro()
tts.speak("Hello world", out="output.wav")
tts.speak("Hello world", out="output.wav", align="alignment.json")
```

## What gets removed

- `js/` directory (browser decoder: demo.html, soprano-decoder.mjs, fft.mjs, tests)
- `exports/` directory (ONNX export script + model files)
- These served the IR-in-browser story which is no longer the direction.

## What stays (internal/advanced)

- `encode()` / `decode()` methods for development and power users
- `.soprano` format (npz) for offline encode/decode workflows
- Streaming (`infer_stream`) for real-time applications
- Hallucination detection

## Rename scope

| Before | After |
|--------|-------|
| Package: `soprano` | Package: `narro` |
| Module dir: `soprano/` | Module dir: `narro/` |
| Class: `SopranoTTS` | Class: `Narro` |
| CLI: `soprano` | CLI: `narro` |
| Repo: `queelius/soprano` | Repo: `queelius/narro` (fresh, not a fork) |
| PyPI: (none) | PyPI: `narro` |

Internal class `SopranoDecoder` stays as-is (it's the Vocos decoder, named after
the upstream model it wraps).

## New feature: `--align`

Extract word-level timestamps from the LLM's attention weights during generation.
Writes a sidecar JSON file:

```json
[
  {"word": "Hello", "start": 0.0, "end": 0.32},
  {"word": "world", "start": 0.35, "end": 0.71}
]
```

Generated server-side during `speak()`. No browser-side decoder needed. Designed
for Hugo shortcode integration — a small JS player reads the JSON and highlights
words during `<audio>` playback.

### Hugo integration (future)

Frontmatter opt-in:

```yaml
---
title: "My Post"
tts: true
---
```

Hugo shortcode renders an audio player with word highlighting:

```
{{< tts src="audio/my-post.opus" align="audio/my-post.json" >}}
```

The player JS is ~50 lines: listen to `timeupdate` events, binary search the
alignment array, toggle a CSS class on word `<span>` elements.

## GitHub repo setup

1. Delete `queelius/soprano` (fork of ekwek1/soprano)
2. Create fresh `queelius/narro` (standalone, not a fork)
3. Push full history
4. Register `narro` on PyPI

## File structure after rename

```
narro/
  __init__.py          # exports Narro, EncodedSpeech, etc.
  tts.py               # Narro class (was SopranoTTS)
  cli.py               # CLI entry point
  encoded.py           # EncodedSpeech, save/load (.soprano format)
  decode_only.py       # Standalone decoder loading
  backends/
    base.py            # BaseModel with infer/stream_infer
    transformers.py    # TransformersModel (HuggingFace)
  vocos/
    decoder.py         # SopranoDecoder (Vocos architecture)
    models.py          # VocosBackbone
    modules.py         # ConvNeXtBlock, ChannelsFirstLayerNorm
    heads.py           # ISTFTHead
    spectral_ops.py    # ISTFT implementation
    migrate_weights.py # Old->new checkpoint migration
  utils/
    text_normalizer.py
    text_splitter.py
tests/
  test_tts_coverage.py
  test_performance.py
  test_encoded.py
  test_encode_decode.py
benchmarks/
  bench.py
```

## Implementation sequence

1. Remove `js/` and `exports/` directories
2. Rename `soprano/` to `narro/` (module directory)
3. Rename `SopranoTTS` to `Narro` across codebase
4. Update all imports (`from soprano.` -> `from narro.`)
5. Update pyproject.toml (package name, entry point, metadata)
6. Update CLAUDE.md, README.md
7. Update all tests
8. Delete GitHub fork, create fresh repo, push
9. Implement `--align` feature
10. Publish to PyPI
