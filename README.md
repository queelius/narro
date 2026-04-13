# Muse

Model-agnostic multi-modality generation server and client. Speaks
OpenAI-compatible HTTP: text-to-speech on `/v1/audio/speech`, text-to-image
on `/v1/images/generations`. Add a modality by dropping in a router,
a protocol, and a catalog entry (no shared base class, no coupling
between modalities).

## Install

```bash
pip install -e ".[server,audio,images]"
```

Optional extras:
- `audio`: PyTorch + transformers for TTS backends
- `audio-kokoro`: Kokoro TTS (needs system `espeak-ng`)
- `images`: diffusers + Pillow for SD-Turbo and future image backends
- `server`: FastAPI + uvicorn + sse-starlette (only needed on the serving host)
- `dev`: pytest + coverage tools

## Quick start

```bash
# Pull a model (installs pip deps, downloads HF weights, records in catalog)
muse pull soprano-80m
muse pull sd-turbo

# Start the server (loads all pulled models; serves matching modality endpoints)
muse serve --host 0.0.0.0 --port 8000

# Synthesize speech (defaults to localhost:8000; override with MUSE_SERVER env var)
muse speak "Hello world" -o hello.wav

# Generate an image
muse imagine "a cat on mars, cinematic" -o cat.png
```

## CLI hierarchy

| Command | Description |
|---|---|
| `muse serve` | start HTTP server |
| `muse pull <model-id>` | download weights + install deps |
| `muse audio speech models list` | list audio.speech models |
| `muse audio speech models info <id>` | show catalog metadata |
| `muse audio speech create "text" -o f.wav` | generate speech (long form) |
| `muse images generations models list` | list images.generations models |
| `muse images generations create "prompt" -o f.png` | generate image (long form) |
| `muse speak` / `muse imagine` | short-form aliases for the create commands |

## HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + enabled modalities |
| `GET /v1/models` | all registered models, aggregated |
| `POST /v1/audio/speech` | synthesize speech (OpenAI-compatible) |
| `GET /v1/audio/speech/voices` | list voices for the current model |
| `POST /v1/images/generations` | generate images (OpenAI-compatible) |

## Architecture

- `muse.core`: modality-agnostic registry, catalog, HF downloader + pip auto-install, FastAPI app factory
- `muse.audio.speech`: text-to-speech (Soprano, Kokoro, Bark backends)
- `muse.images.generations`: text-to-image (SD-Turbo backend)

See `CLAUDE.md` for implementation details and contribution guide.

## License

MIT
