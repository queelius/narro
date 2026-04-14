# Muse

Model-agnostic multi-modality generation server. OpenAI-compatible HTTP is the canonical interface: text-to-speech on `/v1/audio/speech`, text-to-image on `/v1/images/generations`, more modalities landing the same way (embeddings, transcriptions, video). Add a modality by dropping in a router, a protocol, and a catalog entry (no shared base class, no coupling between modalities).

The CLI is deliberately admin-only (`serve`, `pull`, `models`). Generation is reached via the HTTP API, consumed by Python clients, `curl`, or future wrappers like `muse mcp`.

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
# Pull a model (creates a dedicated venv + installs its pip deps + downloads HF weights)
muse pull soprano-80m
muse pull sd-turbo

# Admin: list what's in the catalog
muse models list

# Start the server (loads pulled models; serves OpenAI-compatible endpoints)
muse serve --host 0.0.0.0 --port 8000
```

From any client, generation is an HTTP call:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","model":"soprano-80m"}' \
  --output hello.wav
```

```python
from muse.audio.speech import SpeechClient
from muse.images.generations import GenerationsClient

# MUSE_SERVER env var sets the base URL for remote use; default http://localhost:8000
wav_bytes = SpeechClient().infer("Hello world")
pngs = GenerationsClient().generate("a cat on mars, cinematic", n=1)
```

## CLI (admin-only)

| Command | Description |
|---|---|
| `muse serve` | start the HTTP server |
| `muse pull <model-id>` | download weights + install deps |
| `muse models list [--modality X]` | list known/pulled models |
| `muse models info <model-id>` | show catalog entry |
| `muse models remove <model-id>` | unregister from catalog |

No per-modality subcommands (`muse speak`, `muse audio ...`). Those would be hardcoded modality-to-verb mappings that grow with every new modality. Keeping the CLI modality-agnostic means embeddings, transcriptions, and video land without CLI churn.

## HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + enabled modalities |
| `GET /v1/models` | all registered models, aggregated |
| `POST /v1/audio/speech` | synthesize speech (OpenAI-compatible) |
| `GET /v1/audio/speech/voices` | list voices for a model |
| `POST /v1/images/generations` | generate images (OpenAI-compatible) |

Error shape is uniform: `{"error": {"code", "message", "type"}}` across 404 (model not found) and 422 (validation). Matches OpenAI's envelope so clients written against their API work against muse.

## Architecture

- `muse.core`: modality-agnostic registry, catalog, venv management, HF downloader, pip auto-install, FastAPI app factory
- `muse.cli_impl`: `serve` (supervisor), `worker` (single-venv process), `gateway` (HTTP proxy by model-id)
- `muse.audio.speech`: text-to-speech (Soprano, Kokoro, Bark backends)
- `muse.images.generations`: text-to-image (SD-Turbo backend)

`muse serve` is a supervisor process. It spawns one worker subprocess per venv (each model has its own venv with its own deps) and runs a gateway that proxies requests by the request's `model` field. Dep conflicts between models are structurally impossible.

See `CLAUDE.md` for implementation details and contribution guide.

## License

MIT
