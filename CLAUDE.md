# CLAUDE.md

Guidance for Claude Code when working on Muse.

## Project overview

Muse is a multi-modality generation server and client. It currently supports
two modalities:

- **audio.speech**: text-to-speech via `/v1/audio/speech` (Soprano, Kokoro, Bark)
- **images.generations**: text-to-image via `/v1/images/generations` (SD-Turbo)

The package structure mirrors OpenAI's URL hierarchy. Each modality owns its
protocol, routes, CLI subcommands, and backends. A modality-agnostic core
holds the registry, HF downloader, pip auto-install, and FastAPI app factory.

## Architecture

```
HTTP API (/v1/audio/speech, /v1/images/generations, /v1/models, /health)
    |
    v
muse.core.server   (FastAPI factory, mounts per-modality routers)
    |
    v
muse.core.registry (ModalityRegistry: {modality: {model_id: Model}})
    |
    v
Modality backends implementing modality-specific protocols
```

### Key modules

- `muse.core.registry.ModalityRegistry`: keyed by `(modality, model_id)`.
  First registered model per modality is the default for that modality. No
  shared protocol base across modalities.
- `muse.core.catalog.KNOWN_MODELS`: static dict of `CatalogEntry`. Each entry
  carries `modality`, `backend_path`, `hf_repo`, `pip_extras`,
  `system_packages`. `pull()` installs pip deps, warns on missing system
  packages, and downloads weights from HF. Catalog state lives at
  `~/.muse/catalog.json` (or `MUSE_CATALOG_DIR` env override); writes are
  atomic (write-then-rename).
- `muse.core.server.create_app(registry, routers)`: builds the FastAPI app
  with shared `/health` and `/v1/models`, mounts per-modality routers, and
  registers the `ModelNotFoundError` exception handler so 404s use the
  OpenAI-style `{"error":{...}}` envelope instead of FastAPI's `{"detail":...}`.

### Modality conventions

Each modality subpackage contains:
- `protocol.py`: Protocol + Result dataclass(es) for this modality
- `routes.py`: `build_router(registry) -> APIRouter`
- `client.py`: HTTP client for this modality's endpoints
- `codec.py`: modality-specific encoding (wav/opus for audio; png/jpeg for images)
- `backends/`: concrete model adapters

Each backend class:
- Satisfies the modality's Protocol structurally (no base class required)
- Accepts `hf_repo=`, `local_dir=`, `device=`, `**_` in its constructor (the
  catalog loader calls with those kwargs; `**_` absorbs future additions)
- Prefers `local_dir` over `hf_repo` when loading weights
- Defers heavy imports (transformers, diffusers) to module top-level behind a
  try/except so `muse --help` stays instant

### No shared supertype across modalities

`AudioResult` and `ImageResult` do NOT share a common base. Streaming semantics
differ (audio chunks are time-ordered and playable immediately; diffusion steps
are progressive refinement of one frame). A `GenerationModel` abstract base
would be a leaky abstraction. Instead, `ModalityRegistry` treats models as
`Any`, and each modality's router + codec knows its own types.

## Development commands

```bash
# Install (dev)
pip install -e ".[dev,server,audio,images]"

# Run all tests
pytest tests/

# Run tests for one modality
pytest tests/audio/speech/
pytest tests/images/generations/

# Coverage
pytest tests/ --cov=muse

# Start server
muse serve --device cuda

# Generation is over HTTP (Python client, curl, or future muse mcp).
# There are deliberately no `muse speak` / `muse imagine` subcommands.
# The CLI is admin-only (serve / pull / models) so new modalities land
# without CLI churn.
python - <<'PY'
from muse.audio.speech import SpeechClient
from muse.images.generations import GenerationsClient
SpeechClient().infer("hello")           # → WAV bytes (MUSE_SERVER env sets base URL)
GenerationsClient().generate("a cat")   # → list[bytes] (PNGs)
PY
```

## Project-specific conventions

- **Deferred imports:** `src/muse/__init__.py` and `src/muse/cli.py` MUST NOT
  import heavy libs (torch, diffusers, transformers). Each backend imports
  its heavy deps at module top-level inside a try/except so import of the
  backend module succeeds even without the deps. Tests mock at the module
  path where the library is imported. `muse --help` and `muse pull` work
  without any ML deps installed; pulling a model installs them on demand.
- **FakeModel-pattern tests:** Server and router tests use plain classes that
  satisfy the modality protocol, no real weights. Backend tests also mock
  heavy libs (see `tests/images/generations/test_sd_turbo.py`).
- **Registry is a singleton at module level** (`muse.core.registry.registry`),
  but tests create their own `ModalityRegistry()` instances to avoid coupling.
- **Audio is float32 in `[-1, 1]`** at the protocol boundary; codec converts
  to int16 PCM at output. Scaling uses `* 32768` + `np.clip` to reach full
  int16 range `[-32768, 32767]`.
- **Images are `Any`** at the protocol boundary; codec normalizes PIL / numpy /
  torch to PIL before encoding.
- **OpenAI error envelopes:** Use `raise ModelNotFoundError(model_id, modality)`
  from `muse.core.errors`, not `HTTPException(detail=...)`. The former gives
  `{"error":{"code","message","type"}}`; the latter gives `{"detail":...}`.
- **Streaming uses producer thread + `asyncio.Queue`**, not `list(generator)`.
  Synthesis chunks must dispatch as they're produced, not after full generation.
- **Env vars:** `MUSE_SERVER` (client base URL), `MUSE_CATALOG_DIR` (catalog
  location, defaults `~/.muse/`), `MUSE_HOME` (voices dir base).

## Adding a new modality

1. Create `src/muse/<family>/<op>/` (e.g., `muse/audio/transcriptions/`).
2. Write `protocol.py` with the backend Protocol and Result dataclass.
3. Write `routes.py` exposing `build_router(registry) -> APIRouter`.
4. Write `client.py` with an HTTP client.
5. Add backends under `backends/`.
6. Add `CatalogEntry`s to `muse.core.catalog.KNOWN_MODELS`.
7. Wire up the CLI subtree in `src/muse/cli.py`.
8. Wire the router into `src/muse/cli_impl/serve.py`.
9. Add matching tests in `tests/<family>/<op>/`.
