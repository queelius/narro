# Muse

[![tests](https://github.com/queelius/muse/actions/workflows/tests.yml/badge.svg)](https://github.com/queelius/muse/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/museq)](https://pypi.org/project/museq/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/museq)](https://pypi.org/project/museq/)
![fresh-venv-smoke](https://github.com/queelius/muse/actions/workflows/fresh-venv-smoke.yml/badge.svg)

Model-agnostic multi-modality generation server. OpenAI-compatible HTTP is the canonical interface:
- text-to-speech on `/v1/audio/speech`
- speech-to-text on `/v1/audio/transcriptions` and `/v1/audio/translations`
- audio event / emotion / language classification on `/v1/audio/classifications`
- speech naturalness and production-quality scoring on `/v1/audio/quality`
- reference-transcript word alignment on `/v1/audio/alignments`
- text-to-music on `/v1/audio/music` and text-to-sound-effects on `/v1/audio/sfx`
- text-to-image on `/v1/images/generations`, image inpainting on `/v1/images/edits`, image variations on `/v1/images/variations`
- image-to-image super-resolution on `/v1/images/upscale`
- raster-to-editable-SVG vectorization on `/v1/images/vectorize`
- image depth, keypoints, and object detection on `/v1/images/depth`, `/v1/images/keypoints`, and `/v1/images/detect`
- image OCR on `/v1/images/ocr`
- promptable segmentation on `/v1/images/segment`
- text-to-animation on `/v1/images/animations`
- text-to-video on `/v1/video/generations`
- image-to-vector on `/v1/images/embeddings`
- audio-to-vector on `/v1/audio/embeddings`
- text-to-vector on `/v1/embeddings`
- text-to-text (LLM, tool calls, streaming) on `/v1/chat/completions`
- text moderation/classification on `/v1/moderations`
- text rerank (Cohere-compat) on `/v1/rerank`
- text summarization (Cohere-compat) on `/v1/summarize`
- text translation (LibreTranslate-compat) on `/v1/translate` (+ bare `/translate` alias, `GET /languages`)
- image-to-3D and text-to-3D on `/v1/3d/from-image` and `/v1/3d/generations`

Modality tags are MIME-style (`3d/generation`, `audio/alignment`, `audio/classification`, `audio/embedding`, `audio/generation`, `audio/quality`, `audio/speech`, `audio/transcription`, `chat/completion`, `embedding/text`, `image/animation`, `image/cv`, `image/embedding`, `image/generation`, `image/ocr`, `image/segmentation`, `image/upscale`, `image/vectorization`, `text/classification`, `text/rerank`, `text/summarization`, `text/translation`, `video/generation`).

Three ways to add a model, in order of how often you'll reach for them:

1. **Pull a GGUF or sentence-transformers model from HuggingFace by URI.** No script, no edits:
   ```bash
   muse search qwen3 --modality chat/completion --max-size-gb 10
   muse pull hf://Qwen/Qwen3-8B-GGUF@q4_k_m
   ```
2. **Drop a `.py` script into `~/.muse/models/`** for a one-off model with custom code (see `docs/MODEL_SCRIPTS.md`).
3. **Add a whole new modality** (rare) by dropping a subpackage into
   `src/muse/modalities/` or `$MUSE_MODALITIES_DIR`. The subpackage
   exports `MODALITY` + `build_router` and discovery picks it up.
   Optional: drop a `hf.py` next to `__init__.py` exporting an
   `HF_PLUGIN` dict; muse's HF resolver picks it up the same way and
   `muse search`/`muse pull hf://...` work for the new modality.

All three surfaces are discovered at runtime; there is no hardcoded catalog, no allowlist, and no registration calls.

The CLI is deliberately admin-only (`serve`, `pull`, `search`, `models`). Generation is reached via the HTTP API, consumed by Python clients, `curl`, or future wrappers like `muse mcp`.

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
# Pull bundled models by id (creates a dedicated venv + installs deps + downloads weights)
muse pull soprano-80m
muse pull sd-turbo

# Or pull anything resolvable from HuggingFace by URI
muse pull hf://Qwen/Qwen3-8B-GGUF@q4_k_m
muse pull hf://sentence-transformers/all-MiniLM-L6-v2

# Admin: list what's in the catalog
muse models list

# Start the server (instant boot; serves OpenAI-compatible endpoints).
# As of v0.40.0 muse is lazy-load: enabled models stay on disk until
# the first request that names them, then spawn a worker on demand.
muse serve --host 0.0.0.0 --port 8000

# Optional: pre-warm a model so the first real request is hot
muse models warmup soprano-80m
```

From any client, generation is an HTTP call:

```bash
# Text-to-speech
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","model":"soprano-80m"}' \
  --output hello.wav

# Embeddings (accepts single string or list)
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"hello world","model":"all-minilm-l6-v2"}'

# Image embeddings (input is data: URL or http(s):// URL; mirrors /v1/embeddings)
IMG_B64=$(base64 -w0 cat.png)
curl -X POST http://localhost:8000/v1/images/embeddings \
  -H "Content-Type: application/json" \
  -d "{\"input\":\"data:image/png;base64,${IMG_B64}\",\"model\":\"dinov2-small\"}"

# Audio embeddings (multipart upload; one or more `file` parts; mirrors /v1/embeddings envelope)
# The bundled MERT default is CC-BY-NC-4.0 and restricted to non-commercial use.
curl -X POST http://localhost:8000/v1/audio/embeddings \
  -F "file=@clip.wav" \
  -F "model=mert-v1-95m"

# Speech/audio quality assessment (named, scaled axes)
# Long clips are scored in bounded 10-second windows; metadata includes
# every segment, the literal worst segment, and a `worst_review_segment`
# that ignores subsecond tail windows when a longer candidate exists.
# The decoded-duration cap defaults to 600 seconds
# (MUSE_AUDIO_QUALITY_MAX_DURATION_SECONDS).
muse pull utmos
curl -X POST http://localhost:8000/v1/audio/quality \
  -F "file=@narration.wav" \
  -F "model=utmos"

# Forced alignment: timestamp a trusted transcript without running ASR.
# `language` accepts a supported name or common code and may be omitted.
muse pull qwen3-forced-aligner-0.6b
curl -X POST http://localhost:8000/v1/audio/alignments \
  -F "file=@narration.wav" \
  -F "text=Once upon a time, there was a curious fox." \
  -F "language=en" \
  -F "model=qwen3-forced-aligner-0.6b"

# Chat (OpenAI-compatible incl. tools and streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b-gguf-q4-k-m","messages":[{"role":"user","content":"Capital of France?"}]}'

# Rerank (Cohere-compat); pulls bge-reranker-v2-m3 by default
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is muse?",
    "documents": [
      "muse is an audio server",
      "muse is a multi-modality generation server",
      "muse is the goddess of inspiration"
    ],
    "model": "bge-reranker-v2-m3",
    "top_n": 2,
    "return_documents": true
  }'

# Summarize (Cohere-compat); pulls bart-large-cnn by default
curl -X POST http://localhost:8000/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "muse is a model-agnostic multi-modality generation server. It hosts text, image, audio, and video models behind a unified HTTP API that mirrors OpenAI where possible.",
    "length": "short",
    "format": "paragraph",
    "model": "bart-large-cnn"
  }'

# Music generation (capability-gated; default model: stable-audio-open-1.0)
curl -X POST http://localhost:8000/v1/audio/music \
  -H "Content-Type: application/json" \
  -d '{"prompt":"ambient piano with light rain","model":"stable-audio-open-1.0","duration":10.0}' \
  --output music.wav

# Sound effects generation (same model, different intent)
curl -X POST http://localhost:8000/v1/audio/sfx \
  -H "Content-Type: application/json" \
  -d '{"prompt":"footsteps on gravel","model":"stable-audio-open-1.0","duration":3.0}' \
  --output footsteps.wav

# Full song with lyrics (ACE-Step; GPU-required, 48kHz stereo).
# Omit "lyrics" (or leave it empty) for an instrumental backing track.
curl -X POST http://localhost:8000/v1/audio/music \
  -H "Content-Type: application/json" \
  -d '{"prompt":"dreamy synthpop, female vocal","model":"ace-step-v1-3.5b","duration":120.0,"lyrics":"[verse]\nneon rain on empty streets\n[chorus]\nwe are electric"}' \
  --output song.wav

# Image inpainting (multipart: image + mask + prompt)
# White mask pixels are regenerated; black pixels are kept.
curl -X POST http://localhost:8000/v1/images/edits \
  -F "image=@scene.png" \
  -F "mask=@mask.png" \
  -F "prompt=add a moon to the sky" \
  -F "model=sd-turbo" \
  -F "size=512x512" \
  -F "n=1"

# Image variations (multipart: image only, no prompt)
curl -X POST http://localhost:8000/v1/images/variations \
  -F "image=@scene.png" \
  -F "model=sd-turbo" \
  -F "size=512x512" \
  -F "n=2"

# Image upscale (multipart: 4x super-resolution; SD x4 supports scale=4 only)
curl -s -X POST http://localhost:8000/v1/images/upscale \
  -F "image=@source.png" \
  -F "model=stable-diffusion-x4-upscaler" \
  -F "scale=4" \
  -F "prompt=high detail" \
  | jq -r '.data[0].b64_json' \
  | base64 -d > upscaled.png

# Raster-to-SVG vectorization. Raw SVG output is convenient for Manim;
# use response_format=json to also receive dimensions, seed, and usage.
muse pull starvector-1b-im2svg
curl -s -X POST http://localhost:8000/v1/images/vectorize \
  -F "image=@diagram.png" \
  -F "model=starvector-1b-im2svg" \
  -F "response_format=svg" \
  --output diagram.svg

# Image segmentation (multipart: SAM-2 promptable masks)
# Mode 1: automatic (sweep grid of point prompts internally)
curl -s -X POST http://localhost:8000/v1/images/segment \
  -F "image=@scene.png" \
  -F "model=sam2-hiera-tiny" \
  -F "mode=auto" \
  -F "max_masks=8"

# Mode 2: foreground click points
curl -s -X POST http://localhost:8000/v1/images/segment \
  -F "image=@scene.png" \
  -F "model=sam2-hiera-tiny" \
  -F "mode=points" \
  -F 'points=[[150, 200]]'

# Mode 3: bounding boxes
curl -s -X POST http://localhost:8000/v1/images/segment \
  -F "image=@scene.png" \
  -F "model=sam2-hiera-tiny" \
  -F "mode=boxes" \
  -F 'boxes=[[50, 60, 250, 240]]' \
  -F "mask_format=rle"

# Video generation (since v0.27.0; GPU-required, 8GB+ VRAM tight)
# Default response_format=mp4; "webm" and "frames_b64" also supported.
curl -s -X POST http://localhost:8000/v1/video/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a flag waving in the wind",
    "model": "wan2-1-t2v-1-3b",
    "duration_seconds": 5.0,
    "fps": 5,
    "size": "832x480",
    "steps": 30
  }' \
  | jq -r '.data[0].b64_json' \
  | base64 -d > flag.mp4
```

```python
from muse.modalities.audio_speech import SpeechClient
from muse.modalities.audio_alignment import AudioAlignmentClient
from muse.modalities.image_generation import (
    GenerationsClient, ImageEditsClient, ImageVariationsClient,
)
from muse.modalities.embedding_text import EmbeddingsClient
from muse.modalities.chat_completion import ChatClient

# MUSE_SERVER env var sets the base URL for remote use; default http://localhost:8000
wav_bytes = SpeechClient().infer("Hello world")
alignment = AudioAlignmentClient().align(
    "narration.wav", "Hello world", language="en",
    model="qwen3-forced-aligner-0.6b",
)
pngs = GenerationsClient().generate("a cat on mars, cinematic", n=1)
# LoRA adapters: muse pull hf://nerijs/pixel-art-xl (or curated pixel-art-xl),
# optional --base <muse-id-or-hf-repo>, per-request lora_scale via extra_body.
vectors = EmbeddingsClient().embed(["alpha", "beta"])   # list[list[float]]
chat = ChatClient().chat(
    model="qwen3-8b-gguf-q4-k-m",
    messages=[{"role": "user", "content": "Capital of France?"}],
)

# Image inpainting and variations (since v0.21.0)
src = open("scene.png", "rb").read()
msk = open("mask.png", "rb").read()
edited = ImageEditsClient().edit(
    "add a moon to the sky", image=src, mask=msk, model="sd-turbo",
)
variants = ImageVariationsClient().vary(image=src, model="sd-turbo", n=2)

# Image upscale (since v0.25.0): 4x super-resolution
from muse.modalities.image_upscale import ImageUpscaleClient
from muse.modalities.image_vectorization import VectorizationClient
from pathlib import Path
upscaled = ImageUpscaleClient().upscale(
    image=Path("source.png").read_bytes(),
    model="stable-diffusion-x4-upscaler",
    scale=4,
    prompt="razor sharp detail",
)
Path("upscaled.png").write_bytes(upscaled[0])

# Raster-to-SVG vectorization (v0.59.0)
svg = VectorizationClient().vectorize(
    Path("diagram.png"), model="starvector-1b-im2svg",
    response_format="svg", seed=0,
)
Path("diagram.svg").write_text(svg)

# Image segmentation (since v0.26.0): SAM-2 promptable masks
from muse.modalities.image_segmentation import ImageSegmentationClient
seg = ImageSegmentationClient()
src_bytes = Path("scene.png").read_bytes()
result_auto = seg.segment(
    image=src_bytes, model="sam2-hiera-tiny", mode="auto", max_masks=8,
)
result_points = seg.segment(
    image=src_bytes, model="sam2-hiera-tiny", mode="points",
    points=[[150, 200]],
)
result_boxes = seg.segment(
    image=src_bytes, model="sam2-hiera-tiny", mode="boxes",
    boxes=[[50, 60, 250, 240]], mask_format="rle",
)
# Each result is a dict {id, model, mode, image_size, masks: [...]}
# masks[i]["mask"] is a base64 PNG (mask_format=png_b64) or
# a {"size": [H, W], "counts": str} dict (mask_format=rle)

# Video generation (since v0.27.0): GPU-required, 8GB+ VRAM tight
# Wan2.1 T2V 1.3B (~3GB at fp16) is the default low-VRAM bundle;
# CogVideoX-2b (~9GB) and LTX-Video (~16GB) are curated additions.
from muse.modalities.video_generation import VideoGenerationClient
vid = VideoGenerationClient()
mp4_bytes = vid.generate(
    "a flag waving in the wind",
    model="wan2-1-t2v-1-3b",
    duration_seconds=5.0,
    fps=5,
    size="832x480",
    steps=30,
)
Path("flag.mp4").write_bytes(mp4_bytes)
```

### Audio alignment

`audio/alignment` uses the curated Qwen3 ForcedAligner 0.6B model to locate
words from a known transcript; it does not transcribe or rewrite the text.
The multipart fields are `file` and `text` (required), plus `model` and
`language` (optional). English, Chinese, Cantonese, French, German, Italian,
Japanese, Korean, Portuguese, Russian, and Spanish are supported. Common
codes such as `en`, `zh`, and `es` are accepted.

The response includes `duration_seconds` and ordered `words` with `word`,
`start`, `end`, and `confidence`. Confidence is the mean probability of the
word's start/end timestamp tokens; it is useful for ranking suspicious spans
but is not a calibrated speech-quality or pronunciation score. Uploads default
to 50 MB, five decoded minutes, and 50,000 transcript characters; the Qwen
runtime additionally rejects more than 2,048 alignable words or 8,192 prepared
input tokens before inference. Configure the corresponding
`MUSE_AUDIO_ALIGNMENT_MAX_*` variables to lower the request-level caps.

VRAM caveats for `video/generation`: even Wan 1.3B at fp16 is tight on 8GB cards; 12GB+ recommended for headroom. CogVideoX-2b realistically wants 16GB. LTX-Video needs 16GB+. Mochi-1 (24GB+) and HunyuanVideo (60GB+) are documented but not curated; their dedicated runtimes ship in v1.next.

The OpenAI Python SDK works against muse with no modifications:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
client.chat.completions.create(model="qwen3-8b-gguf-q4-k-m", messages=[...])
```

**Vision (v0.42.0+):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

with open("photo.png", "rb") as f:
    import base64
    data_url = "data:image/png;base64," + base64.b64encode(f.read()).decode()

r = client.chat.completions.create(
    model="smolvlm-256m-instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }],
)
print(r.choices[0].message.content)
```

`muse serve` auto-restarts crashed worker processes with exponential backoff.
Individual model failures don't take down the server or other modalities.

As of v0.40.0 muse is **lazy-load by default**. `muse serve` brings
the gateway up instantly with zero workers running. The first request
to each model triggers a cold load (worker spawn + weights), so
expect 5-30s of latency on that first hit; subsequent requests are
hot. Memory pressure is handled by on-demand LRU eviction backed by
live `pynvml` + `psutil` measurements: a 12GB GPU can have 30 models
catalog-enabled and serve them all, just not simultaneously. Operators
who want eager-boot semantics put a warmup loop in their startup
script:

```bash
muse serve &
sleep 1
for m in $(muse models list --json | jq -r '.[].id'); do
    muse models warmup "$m"
done
```

`muse models list` shows a five-state status indicator: `enabled_loaded`
(filled circle) for resident workers, `enabled_unloaded` (half circle)
for catalog-enabled-but-unloaded, plus the existing `disabled`,
`recommended`, and `available` states. `/v1/models` gains `loaded`,
`last_loaded_at`, and `unservable_reason` per entry. Headroom margins
are tunable via `MUSE_GPU_HEADROOM_GB` (default 1.0) and
`MUSE_CPU_HEADROOM_GB` (default 2.0). Optional `MUSE_GPU_BUDGET_GB` and
`MUSE_CPU_BUDGET_GB` values cap Muse's total resident working set;
physical capacity remains the hard ceiling, while current free memory
governs each load's admission and any same-pool eviction.

## CLI (admin-only)

| Command | Description |
|---|---|
| `muse serve` | start the HTTP server (instant boot; lazy-load on first request) |
| `muse pull <model-id-or-uri>` | download weights + install deps + run probe (accepts bundled id OR resolver URI like `hf://org/repo@variant`; `--no-probe` opts out) |
| `muse search <query> [--modality M]` | search HuggingFace for pullable GGUF / sentence-transformers models |
| `muse models list [--modality X]` | list known/pulled models with five-state status (enabled_loaded / enabled_unloaded / disabled / recommended / available) |
| `muse models info <model-id>` | show catalog entry |
| `muse models remove <model-id>` | unregister from catalog |
| `muse models enable <model-id>` | mark a pulled model active in the catalog (allowed to lazy-load) |
| `muse models disable <model-id>` | mark a pulled model inactive in the catalog (refuses to lazy-load) |
| `muse models warmup <model-id>` | pre-load a model into a worker without serving traffic; first real request is hot |
| `muse models refresh <id> \| --all \| --enabled` | re-install museq[server,extras] into per-model venv(s) (after `pip install -U museq`) |
| `muse config generate \| show \| path \| get \| set \| unset` | manage `~/.muse/config.yaml` (see Configuration below) |
| `muse doctor resources [--repair]` | inspect Muse's owned-resource registry; optionally clean verified stale/orphan records after an unclean exit |
| `muse mcp [--http]` | run an MCP server bridging muse to LLM clients (31 tools) |

No per-modality subcommands (`muse speak`, `muse audio ...`). Those would be hardcoded modality-to-verb mappings that grow with every new modality. Keeping the CLI modality-agnostic means embeddings, transcriptions, and video land without CLI churn.

`muse doctor resources` is read-only by default and does not scan unrelated
host processes. Use `--repair` after a crash or forced shutdown to remove stale
records and terminate only orphan Muse process leaders whose recorded identity
can still be verified; unverifiable or changed identities are refused and
reported for manual investigation. Normal `Ctrl+C` shutdown should release the
same resources without needing repair.

## Configuration

Every muse server setting lives in one registry and resolves by precedence
**CLI flag > `MUSE_*` env var > `~/.muse/config.yaml` > built-in default**. Any
`MUSE_*` env var can equivalently be set in the config file.

```bash
muse config generate            # write a commented ~/.muse/config.yaml
muse config show                # effective value + source for every setting
muse config set server.idle_timeout_seconds 0   # e.g. disable idle eviction
```

The config file holds SERVER / global settings (memory budgets, request-size
limits, idle-timeout, paths, fetch policy). Per-model state (enable/disable,
device pin, memory measurements) lives in `~/.muse/catalog.json` and is managed
with `muse models ...`. See [docs/CONFIG.md](docs/CONFIG.md) for the full
settings inventory and precedence rules.

## HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + enabled modalities |
| `GET /v1/models` | all registered models, aggregated |
| `POST /v1/audio/speech` | synthesize speech (OpenAI-compatible) |
| `GET /v1/audio/speech/voices` | list voices for a model |
| `POST /v1/audio/transcriptions` | transcribe audio to text (OpenAI-compatible) |
| `POST /v1/audio/translations` | transcribe + translate audio to English (OpenAI-compatible) |
| `POST /v1/audio/alignments` | align trusted reference text to audio as word timestamps (muse-native multipart) |
| `POST /v1/images/generations` | generate images (OpenAI-compatible; supports img2img via `image` + `strength`) |
| `POST /v1/images/edits` | inpaint masked regions (OpenAI-compatible; multipart with image+mask+prompt) |
| `POST /v1/images/variations` | generate alternates of one image (OpenAI-compatible; multipart, no prompt) |
| `POST /v1/images/vectorize` | convert a raster icon/diagram to validated static SVG (multipart; JSON or image/svg+xml) |
| `POST /v1/embeddings` | text embeddings (OpenAI-compatible) |
| `POST /v1/images/embeddings` | image embeddings (OpenAI-shape envelope mirroring /v1/embeddings) |
| `POST /v1/audio/embeddings` | audio embeddings (multipart upload + OpenAI-shape envelope mirroring /v1/embeddings) |
| `POST /v1/chat/completions` | chat (OpenAI-compatible incl. tools, structured output, streaming) |
| `POST /v1/moderations` | text moderation/classification (OpenAI-compatible) |
| `POST /v1/rerank` | text rerank (Cohere-compat) |
| `POST /v1/summarize` | text summarization (Cohere-compat) |
| `POST /v1/translate` (+ bare `/translate` alias) | text translation (LibreTranslate-compat) |
| `GET /languages` | supported translation languages (LibreTranslate-compat) |
| `POST /v1/audio/music` | music generation (capability-gated; muse-native shape) |
| `POST /v1/audio/sfx` | sound-effect generation (capability-gated; muse-native shape) |
| `POST /v1/video/generations` | text-to-video generation (mp4/webm/frames_b64; GPU-required) |

Error shape is uniform: `{"error": {"code", "message", "type"}}` across 404 (model not found) and 422 (validation). Matches OpenAI's envelope so clients written against their API work against muse.

### Admin endpoints (v0.28.0+)

Eleven endpoints under `/v1/admin/*` let you enable, disable, probe, pull, and remove models on a running supervisor without restarting it. The admin surface is closed-by-default: set `MUSE_ADMIN_TOKEN` to any non-empty value to enable it, then send `Authorization: Bearer <token>` on every request.

| Endpoint | Purpose |
|---|---|
| `POST /v1/admin/models/{id}/enable` | spawn a worker (or restart-in-place) hosting `id`; returns 202 + job_id |
| `POST /v1/admin/models/{id}/disable` | unload `id` from its worker; sync |
| `POST /v1/admin/models/{id}/probe` | run `muse models probe` in the model's venv; returns 202 + job_id |
| `POST /v1/admin/models/_/pull` | pull from a curated alias or resolver URI in the body; returns 202 + job_id |
| `DELETE /v1/admin/models/{id}?purge=bool` | remove from catalog (refuses 409 if loaded) |
| `GET /v1/admin/models/{id}/status` | merged catalog + live worker view |
| `GET /v1/admin/workers` | spawned workers + pid/uptime/restart-count |
| `POST /v1/admin/workers/{port}/restart` | SIGTERM by port; auto-restart monitor handles bringup |
| `GET /v1/admin/memory` | per-device aggregate + per-model breakdown |
| `GET /v1/admin/jobs/{job_id}` | one async-job record (404 if reaped) |
| `GET /v1/admin/jobs` | recent jobs newest-first |

Auth setup:

```bash
export MUSE_ADMIN_TOKEN="$(openssl rand -hex 32)"  # or any non-empty value
muse serve  # admin endpoints now active under the same port
```

Five auth scenarios:
- env var unset, any header: `503 admin_disabled`
- env var set, no header: `401 missing_token`
- env var set, malformed header: `401 missing_token`
- env var set, wrong bearer: `403 invalid_token`
- env var set, correct bearer: route runs

curl examples:

```bash
TOKEN="$MUSE_ADMIN_TOKEN"
H="Authorization: Bearer $TOKEN"

# enable a pulled model (worker spawns or joins existing venv-group)
curl -s -X POST -H "$H" http://localhost:8000/v1/admin/models/kokoro-82m/enable

# poll the returned job
curl -s -H "$H" http://localhost:8000/v1/admin/jobs/<job_id>

# disable a loaded model (sync)
curl -s -X POST -H "$H" http://localhost:8000/v1/admin/models/kokoro-82m/disable

# merged status
curl -s -H "$H" http://localhost:8000/v1/admin/models/kokoro-82m/status

# memory aggregate (psutil + pynvml)
curl -s -H "$H" http://localhost:8000/v1/admin/memory
```

Python (use the AdminClient):

```python
from muse.admin.client import AdminClient

# Reads MUSE_SERVER and MUSE_ADMIN_TOKEN from env when unset.
admin = AdminClient()

job = admin.enable("kokoro-82m")
final = admin.wait(job["job_id"])
print(final["state"], final.get("result"))

print(admin.status("kokoro-82m"))
print(admin.workers())
print(admin.memory())
```

The `muse models enable/disable` CLI commands route through this admin API automatically when `MUSE_ADMIN_TOKEN` is set and the supervisor is reachable, falling back to a catalog-only mutation (effective on next `muse serve`) otherwise.

### Observability dashboard

`muse serve` also ships a lightweight `/dashboard` page: loaded models, request-rate and latency charts, and a live per-model log tail, all served from a single self-contained HTML file with no build step. It is on by default (`telemetry.enabled` / `MUSE_TELEMETRY_ENABLED`, default `true`); set it to `false` to turn recording off entirely. The dashboard's data endpoints are closed-by-default and reuse the same `MUSE_ADMIN_TOKEN` as the admin API above; the page itself always loads and prompts for a token.

```bash
export MUSE_ADMIN_TOKEN="$(openssl rand -hex 32)"
muse serve
# open http://localhost:8000/dashboard and paste the token in
```

## MCP server (since v0.29.0)

`muse mcp` runs a Model Context Protocol server that exposes muse to LLM clients (Claude Desktop, Cursor, etc.) as 31 structured tools: 11 admin tools (gated by `MUSE_ADMIN_TOKEN`) plus 20 inference tools. Stdio mode is the default (for desktop apps); HTTP+SSE mode (`--http --port 8088`) is available for remote / web embedders.

```bash
muse mcp                                  # stdio mode
muse mcp --http --port 8088               # HTTP+SSE
muse mcp --filter inference               # only inference tools (no admin)
muse mcp --filter admin                   # only admin tools (control panel)
muse mcp --server http://other:8000       # connect to a remote muse server
```

Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "muse": {
      "command": "muse",
      "args": ["mcp"],
      "env": {
        "MUSE_SERVER": "http://localhost:8000",
        "MUSE_ADMIN_TOKEN": "your-admin-token-here"
      }
    }
  }
}
```

Tools split into two groups:

**Admin (11):** `muse_list_models`, `muse_get_model_info`, `muse_search_models`, `muse_pull_model`, `muse_remove_model`, `muse_enable_model`, `muse_disable_model`, `muse_probe_model`, `muse_get_memory_status`, `muse_get_workers`, `muse_get_jobs`. Long-running ops (pull, probe, enable) return a `job_id` and the LLM polls `muse_get_jobs` to track progress.

**Inference (20):** `muse_chat`, `muse_summarize`, `muse_rerank`, `muse_classify`, `muse_embed_text`, `muse_translate`, `muse_generate_image`, `muse_edit_image`, `muse_vary_image`, `muse_upscale_image`, `muse_segment_image`, `muse_vectorize_image`, `muse_generate_animation`, `muse_embed_image`, `muse_speak`, `muse_transcribe`, `muse_generate_music`, `muse_generate_sfx`, `muse_embed_audio`, `muse_generate_video`.

Binary inputs accept `<name>_b64` (base64), `<name>_url` (data: or http URL), or `<name>_path` (local file). Image and audio outputs return as MCP `ImageContent` / `AudioContent` blocks plus a JSON summary.

## Federation

`muse federate` runs a thin coordinator in front of a fixed list of
unmodified muse `serve` nodes, so one OpenAI-compatible endpoint fronts
your whole cluster instead of clients having to know which box has
which model.

```bash
muse federate --port 8100 \
    --node http://192.168.0.204:8000 \
    --node http://192.168.0.50:8000
# or: muse federate --config ~/.muse/federation.yaml
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://coordinator:8100/v1", api_key="not-used")
client.chat.completions.create(model="qwen3.5-4b-q4", messages=[...])
```

Requests are routed by model-locality (a node that already has the
model loaded wins over one that would have to cold-load it), with an
in-flight-count tie-break. `GET /v1/models`, `/health`, and
`/v1/federation/nodes` on the coordinator aggregate across the cluster.
See the "Federation" section of `CLAUDE.md` for the full design.

## Architecture

- `muse.core`: modality-agnostic discovery, registry, catalog, venv management, HF downloader, pip auto-install, FastAPI app factory.
- `muse.cli_impl`: `serve` (supervisor), `worker` (single-venv process), `gateway` (HTTP proxy routing by request's `model` field).
- `muse.modalities/`: one subpackage per modality (wire contract: protocol + routes + codec + client).
  - `audio_alignment/` (MODALITY `"audio/alignment"`; Qwen3 ForcedAligner reference text to word timestamps)
  - `audio_classification/` (MODALITY `"audio/classification"`; multipart event/emotion/language classification)
  - `audio_embedding/` (MODALITY `"audio/embedding"`; multipart upload + OpenAI-shape envelope; includes `runtimes/transformers_audio.py`)
  - `audio_generation/` (MODALITY `"audio/generation"`; mounts both `/v1/audio/music` and `/v1/audio/sfx` on one MIME tag with per-route capability gates)
  - `audio_quality/` (MODALITY `"audio/quality"`; bounded windowed UTMOS naturalness MOS + Audiobox Aesthetics quality axes)
  - `audio_speech/` (MODALITY `"audio/speech"`)
  - `audio_transcription/` (MODALITY `"audio/transcription"`; multipart/form-data upload, OpenAI Whisper wire shape)
  - `chat_completion/` (MODALITY `"chat/completion"`; includes `runtimes/llama_cpp.py`)
  - `embedding_text/` (MODALITY `"embedding/text"`; includes `runtimes/sentence_transformers.py`)
  - `image_embedding/` (MODALITY `"image/embedding"`; includes `runtimes/transformers_image.py`)
  - `image_generation/` (MODALITY `"image/generation"`)
  - `text_classification/` (MODALITY `"text/classification"`; OpenAI `/v1/moderations` wire shape)
  - `text_rerank/` (MODALITY `"text/rerank"`; Cohere `/v1/rerank` wire shape)
  - `text_summarization/` (MODALITY `"text/summarization"`; Cohere `/v1/summarize` wire shape)
  - `text_translation/` (MODALITY `"text/translation"`; LibreTranslate `/v1/translate` + `/translate` alias + `/languages` wire shape)
  - `video_generation/` (MODALITY `"video/generation"`; includes `runtimes/wan_runtime.py` and `runtimes/cogvideox_runtime.py`)
- `muse.models/`: flat directory of drop-in model scripts, one file per model (MANIFEST + Model class).
  - `soprano_80m.py`, `kokoro_82m.py`, `bark_small.py`, `supertonic_3.py` (audio/speech; `supertonic_3` is ONNX on-device CPU, 31 languages)
  - `nv_embed_v2.py` (embedding/text; MiniLM and Qwen3-Embedding are now resolver-pulled via the generic runtime, see `curated.yaml`)
  - `sd_turbo.py` (image/generation)
  - `bge_reranker_v2_m3.py` (text/rerank)
  - `stable_audio_open_1_0.py` (audio/generation; Stable Audio Open 1.0, Apache 2.0)
  - `ace_step_v1_3_5b.py` (audio/generation; ACE-Step v1 3.5B full songs + lyrics, Apache 2.0, GPU-required)
  - `bart_large_cnn.py` (text/summarization; facebook/bart-large-cnn, Apache 2.0, ~400MB CPU-friendly)
  - `m2m100_418m.py` (text/translation; facebook/m2m100_418M, MIT, ~2GB, all-pairs across 100 languages)
  - `dinov2_small.py` (image/embedding; facebook/dinov2-small, Apache 2.0, 88MB, 384-dim CPU-friendly)
  - `mert_v1_95m.py` (audio/embedding; m-a-p/MERT-v1-95M, CC-BY-NC-4.0 non-commercial, 95M parameters, 768-dim music understanding via mean-pool over time)
  - `wan2_1_t2v_1_3b.py` (video/generation; Wan-AI/Wan2.1-T2V-1.3B, Apache 2.0, ~3GB at fp16, 5s clips at 832x480, GPU-required)
- `muse.core.resolvers`: URI -> ResolvedModel dispatch for `muse pull hf://...`.
  - `resolvers_hf` registers the `hf://` resolver for HuggingFace GGUF + sentence-transformers repos.

`muse serve` is a supervisor process. It spawns one worker subprocess per venv (each pulled model has its own venv with its own deps) and runs a gateway that proxies by the `model` field. Dep conflicts between models are structurally impossible.

Three ways to extend muse:
1. **Resolver URI**: `muse pull hf://Qwen/Qwen3-8B-GGUF@q4_k_m` for any GGUF or sentence-transformers HF repo. See `docs/RESOLVERS.md`.
2. **Model script**: drop a `.py` into `~/.muse/models/` for one-off models with custom code. See `docs/MODEL_SCRIPTS.md`.
3. **Modality subpackage**: drop into `src/muse/modalities/` or `$MUSE_MODALITIES_DIR` for a whole new modality.

See `CLAUDE.md` for implementation details and contribution guide,
`docs/MODEL_SCRIPTS.md` for writing your own model scripts,
`docs/RESOLVERS.md` for adding a new URI scheme, and
`docs/CHAT_COMPLETION.md` for the chat endpoint specification.

## License

MIT
