"""Muse: model-agnostic multi-modality generation server.

The authoritative list of supported modalities lives in
`muse.core.discovery.discover_modalities()`, which scans
`src/muse/modalities/` plus any user-configured dirs. As of v0.32.0
the bundled modalities are:

  - audio/alignment: /v1/audio/alignments (Qwen3 ForcedAligner; trusted transcript to word timestamps)
  - audio/embedding: /v1/audio/embeddings (transformers AutoModel + librosa; MERT, CLAP, wav2vec; multipart upload, OpenAI-shape envelope)
  - audio/generation: /v1/audio/music, /v1/audio/sfx (Stable Audio Open 1.0; capability-gated)
  - audio/quality: /v1/audio/quality (UTMOS naturalness + Audiobox Aesthetics quality axes)
  - audio/speech: /v1/audio/speech (TTS: Soprano, Kokoro, Bark)
  - audio/transcription: /v1/audio/transcriptions, /v1/audio/translations (faster-whisper)
  - chat/completion: /v1/chat/completions (llama-cpp-python over GGUF)
  - embedding/text: /v1/embeddings (sentence-transformers)
  - image/animation: /v1/images/animations (AnimateDiff; short looping clips, animated WebP/GIF/MP4)
  - image/embedding: /v1/images/embeddings (transformers AutoModel; CLIP, SigLIP, DINOv2)
  - image/generation: /v1/images/generations, /v1/images/edits (inpaint), /v1/images/variations (diffusers)
  - image/segmentation: /v1/images/segment (SAM-2; multipart upload, mode-aware: auto/points/boxes/text; PNG or COCO RLE masks)
  - image/upscale: /v1/images/upscale (StableDiffusionUpscalePipeline; SD x4; multipart upload)
  - image/vectorization: /v1/images/vectorize (StarVector-1B; raster-to-static-SVG; multipart upload)
  - text/classification: /v1/moderations (HF text-classification)
  - text/rerank: /v1/rerank (sentence-transformers CrossEncoder; Cohere-compat)
  - text/summarization: /v1/summarize (transformers AutoModelForSeq2SeqLM; Cohere-compat)
  - video/generation: /v1/video/generations (Wan, CogVideoX; narrative clips, mp4/webm/frames_b64; GPU-required)

v0.32.0 adds CI smoke-tests of fresh per-model venvs (#124). The
workflow `.github/workflows/fresh-venv-smoke.yml` matrix-tests five
lightweight bundled models (kokoro-82m, dinov2-small, bart-large-cnn,
bge-reranker-v2-m3, mert-v1-95m) on every push to main and every PR;
each job creates a fresh venv, installs only what `muse pull` would
install, and verifies the model loads via the in-venv probe worker
(no inference; that's GPU-bound and out of scope). Catches the
production failure mode where a bundled script's `pip_extras` misses
a transitive dep that `from_pretrained` (or sentence-transformers, or
diffusers) pulls in at load time, complementing the v0.30.0 static
audit (#110) which can only flag direct-import gaps via AST scan.
Heavy / GPU-only models deferred until paid runner budget allows.
Local repro: `python scripts/smoke_fresh_venv.py --model_id <id>`.

v0.31.0 consolidates cross-runtime utilities into
`muse.core.runtime_helpers`: `select_device` (cuda/mps/cpu auto-detect),
`dtype_for_name` (string-to-torch.dtype map with `fp16`/`bf16`/`fp32`
aliases), `set_inference_mode` (no-grad switch with the literal
method-name token kept out of caller bodies), and `LoadTimer` (opt-in
load-time logging context). Removes ~30 per-runtime copies; an AST-based
meta-test (`tests/core/test_runtime_helpers_meta.py`) walks every
runtime and bundled script to flag re-implementations. Behavior-
preserving; the existing 2150 fast-lane tests pass without modification.

v0.30.0 bundles three operational improvements:
  - the supervisor starts the gateway after the FIRST worker is healthy
    (was: ALL workers), so clients can hit the fast workers while slow
    ones still load. Remaining workers promote on a daemon thread.
  - bundled scripts in `muse/models/` got a `pip_extras` audit; missing
    transitive deps (torch, numpy) added to seven manifests; a static
    regression-guard test parametrized over every bundled script
    catches future gaps.
  - new `muse models refresh <id> | --all | --enabled` re-installs
    `museq[server,<extras>]` plus the model's `pip_extras` into per-model
    venvs; use after `pip install -U museq` to propagate new server-side
    deps.

v0.29.0 adds `muse mcp`: an MCP (Model Context Protocol) server that
exposes muse to LLM clients (Claude Desktop, Cursor, etc.) as 29
structured tools. 11 admin tools wrap `/v1/admin/*` (gated by
`MUSE_ADMIN_TOKEN`); 18 inference tools wrap the generation routes.
Stdio mode is the default; HTTP+SSE mode is available for remote /
web embedders. Filter mode lets ops pin to admin-only or
inference-only. See CLAUDE.md "Using muse from Claude Desktop".

v0.28.0 added an admin REST API under `/v1/admin/*` for runtime model
control (enable/disable/probe/pull/remove without restarting `muse
serve`). Closed-by-default behind `MUSE_ADMIN_TOKEN`. See README.md
"Admin endpoints" and CLAUDE.md "Admin REST API" for the full surface.

Heavy backends (transformers, diffusers, faster-whisper, llama-cpp,
sentence-transformers) are imported lazily inside per-modality runtime
modules to keep `muse --help` and `muse pull` instant. Each pulled
model lives in its own venv at `~/.muse/venvs/<model-id>/`.

`__version__` is read from pyproject.toml at install time; this
fallback covers in-tree imports without an installed muse.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("museq")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
