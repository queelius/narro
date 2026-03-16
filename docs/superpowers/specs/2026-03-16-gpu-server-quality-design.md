# Narro: GPU Server, Quality Pipeline, and Benchmark Framework

## Overview

Narro is currently CPU-only and uses mechanical regex-based text normalization. This design adds GPU acceleration via a RESTful API server, LLM-powered text rewriting for better TTS quality, improved failure detection, and a benchmark framework for iterating on performance.

## Architecture

Two parallel tracks that share almost no code:

- **Track 1 (Infrastructure)**: GPU device support, FastAPI server, HTTP client, benchmark framework, CPU micro-optimizations
- **Track 2 (Quality)**: `clean_text()` improvements, garbled audio detection, LLM rewriting layer

## Track 1: Infrastructure

### 1. GPU Device Support

Restore multi-device support (removed during the narro simplification). The original GPU implementation exists in git history (commit `162468b`).

**`Narro.__init__`** — New `device` parameter with auto-detection:

```python
def __init__(self, device='auto', ...):
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    self.device = device
```

**`TransformersModel.__init__`** — Accept `device`, move model to device:

```python
self.model = AutoModelForCausalLM.from_pretrained(...).to(device)
```

**`TransformersModel.infer`** — Move inputs to device:

```python
inputs = self.tokenizer(...).to(self.device)
```

**Decoder** — `.to(device)` on the nn.Module, `torch.load(..., map_location=device)` for weights.

**Quantization constraint**: `torch.quantization.quantize_dynamic` is CPU-only. Skip on GPU (GPU doesn't need INT8 quantization for speed).

**Files modified**: `narro/tts.py`, `narro/backends/transformers.py`, `narro/backends/base.py`, `narro/decode_only.py`

### 2. Server Architecture (`narro serve`)

FastAPI server exposing the Narro pipeline over HTTP. OpenAI-compatible endpoint.

**New file**: `narro/server.py`

**CLI subcommand**:

```bash
narro serve                    # default: 0.0.0.0:8000, auto device
narro serve --port 9000        # custom port
narro serve --device cuda      # explicit device
narro serve --host 127.0.0.1   # localhost only
```

**Endpoint: `POST /v1/audio/speech`**

Request body (OpenAI-compatible):

```json
{
  "input": "Text to synthesize",
  "model": "narro",
  "voice": "default",
  "response_format": "opus",
  "stream": false,
  "align": false
}
```

- `model`, `voice` — accepted but ignored (single model, single voice). For client compatibility.
- `response_format` — `"wav"` or `"opus"` (default `"opus"`). Opus requires ffmpeg on the server.
- `stream` — if `true`, returns SSE stream using `infer_stream()`.
- `align` — if `true`, include paragraph alignment data in response.

**Non-streaming response**: Raw audio bytes with appropriate `Content-Type`. If `align: true`, alignment JSON in `X-Alignment` header.

**Streaming response** (SSE):

```
data: {"audio": "<base64-wav-chunk>", "type": "speech.audio.delta"}

data: {"audio": "<base64-wav-chunk>", "type": "speech.audio.done", "alignment": [...]}
```

**Additional endpoints**:

- `GET /health` — `{"status": "ok", "device": "cuda", "model": "ekwek/Soprano-1.1-80M"}`

**Dependencies**: `fastapi` and `uvicorn` as optional extras:

```toml
[project.optional-dependencies]
server = ["fastapi", "uvicorn"]
```

`pip install narro[server]` gets the server. Plain `pip install narro` stays lightweight.

**Startup**: Model loaded once at startup. Stays in memory across requests.

### 3. Client Layer

When `NARRO_SERVER` is set or `--server` is passed, the CLI sends HTTP requests instead of loading the model locally.

**New file**: `narro/client.py`

```python
class NarroClient:
    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')

    def infer(self, text, out_path=None, response_format='wav', stream=False):
        """Send text to server, return audio bytes or stream chunks."""
        ...

    def health(self):
        """Check server availability. Returns dict or raises."""
        ...
```

**Discovery order**:

1. `--server` CLI flag (highest priority)
2. `NARRO_SERVER` environment variable
3. Local inference (fallback)

**CLI integration** — In `cli.py`, before loading the local model:

```python
server_url = args.server or os.environ.get('NARRO_SERVER')
if server_url:
    from narro.client import NarroClient
    client = NarroClient(server_url)
else:
    from narro.tts import Narro
    tts = Narro()
```

**Hugo integration** — Same pattern in `hugo/cli.py`. The `_lazy_import()` function checks for `NARRO_SERVER`.

**Failure behavior**: If server is configured but unreachable, fail loudly with a clear error. No silent fallback to local CPU — a sudden 10x slowdown with no explanation would be confusing.

**HTTP client**: Uses `requests` (already a transitive dep via `huggingface_hub`). No new dependencies.

**Files modified**: `narro/cli.py`, `narro/hugo/cli.py`
**Files created**: `narro/client.py`

### 4. Benchmark Framework

Replace the existing `benchmarks/bench.py` with a proper framework. Add `narro bench` as a CLI subcommand.

**Stages measured independently**:

1. Startup — model load + warmup
2. Preprocessing — `clean_text()` + sentence splitting
3. LLM inference — `encode_batch()` (the bottleneck)
4. Decoding — Vocos decoder
5. End-to-end — full pipeline including I/O

**Test corpus**: Short, medium, long, and a real blog paragraph with numbers, abbreviations, mixed formatting.

**Output modes**:

```bash
narro bench                    # human-readable table
narro bench --json             # structured JSONL for comparison
narro bench --runs 10          # more iterations
narro bench --device cuda      # benchmark GPU
narro bench --no-compile       # compare compiled vs not
```

**JSON schema per run**:

```json
{
  "timestamp": "2026-03-16T10:30:00",
  "device": "cpu",
  "compile": true,
  "quantize": false,
  "num_threads": 8,
  "texts": {
    "short": {
      "input_chars": 22,
      "tokens": 12,
      "audio_sec": 0.77,
      "preprocess_ms": 2,
      "encode_ms": 450,
      "decode_ms": 30,
      "total_ms": 482,
      "rtf": 0.62
    }
  }
}
```

Primary metric: **Real-Time Factor (RTF)** = `inference_time / audio_duration`. RTF < 1.0 means faster than real-time.

**Files modified**: `narro/cli.py` (add `bench` subcommand)
**Files created**: `narro/bench.py` (replaces `benchmarks/bench.py`)

### 5. CPU Micro-optimizations

No new dependencies. Measured via benchmark framework — only keep changes that measurably help.

**5a. Threading defaults**: Auto-tune `torch.set_num_threads` based on core count, capped at 8 (diminishing returns beyond that).

**5b. `torch.inference_mode()`**: Normalize from `torch.no_grad()` to `torch.inference_mode()` everywhere. Faster because it disables more tracking.

**5c. KV-cache prefix reuse**: All sentences share `[STOP][TEXT]` prefix tokens. Pre-compute prefix KV cache once, reuse across sentences in a batch. Modest savings (prefix is ~10 tokens) but free.

**5d. Tighter `max_new_tokens`**: Estimate bound from input length: `max_new_tokens = min(512, len(input_tokens) * 8)`. Garbled sentences fail faster instead of generating 512 garbage tokens.

**5e. `torch.compile` mode tuning**: Compare `reduce-overhead` vs `max-autotune` on CPU via benchmark framework. Use whichever is faster.

**Files modified**: `narro/tts.py`, `narro/backends/base.py`

## Track 2: Quality

### 6. Improved `clean_text()` Fallback

Even with LLM rewriting as the primary path, the mechanical fallback should stop making things actively worse. Theme: **do less, not more** — the model is an LLM, it handles natural text.

**6a. Stop converting colons to periods.** `(':' → '.')` fragments sentences. Keep the colon or strip it — don't create artificial sentence boundaries.

**6b. Drop URLs entirely** instead of spelling them out letter by letter. If a URL leaks through `extract_prose()`, remove it. `h t t p s colon slash slash` is catastrophic for the model.

**6c. Audit parentheses handling.** Current transform `(text)` → `, text,` may be fine, but verify against actual model output.

**6d. Fix `normalize_newlines()` period insertion.** Stop adding `.` to lines without terminal punctuation. Now that we split on `\n\n` into paragraphs first, within-paragraph newlines are soft wraps, not sentence boundaries.

**6e. Reconsider CamelCase splitting.** `MyClass` → `My Class` also fires on proper nouns like "McDonald" → "Mc Donald". Consider removing this transform entirely — code identifiers should already be stripped by `extract_prose()`.

**6f. Audit the full chain.** Run the benchmark corpus through `clean_text()`, manually inspect output. Find transforms that make text worse for TTS. Some were designed for the Tortoise TTS tokenizer this code was adapted from, not for a Qwen LLM backbone.

**Files modified**: `narro/utils/text_normalizer.py`

### 7. Garbled Audio Detection

The current `hallucination_detector` only catches repetition (consecutive similar hidden states). Garbled/distorted audio has a different signature — degenerate states, high entropy, erratic output.

**Replace `hallucination_detector` with `quality_check`** that tests multiple signals:

```python
def quality_check(self, response, input_text):
    """Check for repetition, garbled output, or degenerate generation."""
    hidden_states = response['hidden_state']
    entropy = response['token_entropy']

    # Existing: repetition detection
    if self._detect_repetition(hidden_states):
        return 'repetition'

    # New: entropy spike detection
    if entropy.mean() > ENTROPY_THRESHOLD:
        return 'garbled'

    # New: abnormal length ratio
    ratio = len(hidden_states) / max(len(input_text), 1)
    if ratio > MAX_TOKEN_RATIO or ratio < MIN_TOKEN_RATIO:
        return 'length_anomaly'

    # Existing: didn't finish
    if response['finish_reason'] == 'length':
        return 'truncated'

    return None  # passed
```

**Three new signals (all use data already computed)**:

- **Entropy monitoring**: Mean token entropy exceeding a threshold indicates the model is guessing randomly. The data (`token_entropy`) is already computed and stored — just not checked.
- **`finish_reason='length'`**: Currently logged as a warning but output is kept. Should trigger retry.
- **Token/input length ratio**: Normal speech has a predictable output-to-input ratio. Extreme outliers (50-char input producing 400 tokens, or 3 tokens) indicate garbage.

**Default retries**: Change from `retries=0` to `retries=1`. With better detection, retry is more likely to help.

**Threshold calibration**: Run benchmark corpus, collect entropy distributions and length ratios for good vs bad output. Set thresholds at the boundary. The benchmark framework enables this.

**Files modified**: `narro/tts.py`

### 8. LLM Rewriting Layer

Three tiers from most to least intelligent:

**Tier 1: Claude Code orchestration (primary workflow)**

No code changes to narro. When Claude Code orchestrates narration:

1. Read paragraphs via `extract_prose()` + split on `\n\n`
2. Rewrite each paragraph to conversational speech prose
3. Pass rewritten paragraphs to `encode_batch()`

Paragraph index alignment preserved because rewrite is 1:1 per paragraph. This is a workflow pattern, not a code feature.

Helper function `extract_paragraphs(body)` provided for convenience — returns the list of paragraphs ready for rewriting.

**Tier 2: Built-in endpoint rewriting (`narro hugo generate --rewrite`)**

For autonomous operation without Claude Code. Calls a configurable `/v1/chat/completions` endpoint.

Configuration via envvars:

```
NARRO_LLM_URL=http://localhost:11434/v1
NARRO_LLM_KEY=sk-...
NARRO_LLM_MODEL=claude-sonnet-4-5-20250514
```

**New file**: `narro/rewrite.py`

```python
def rewrite_paragraphs(paragraphs, api_url, api_key=None, model=None):
    """Rewrite paragraphs to conversational speech prose via LLM.

    Returns list of rewritten paragraphs, same length as input.
    """
```

System prompt:
> "Rewrite the following paragraph as natural spoken prose for text-to-speech narration. Preserve the meaning. Expand abbreviations. Spell out symbols. Remove anything that doesn't make sense when read aloud. Keep it the same length — don't add commentary or meta-text. Return only the rewritten paragraph."

Each paragraph rewritten independently to preserve 1:1 index mapping.

Uses `requests` (transitive dep). No new dependencies.

**Tier 3: Improved `clean_text()` fallback**

Covered in Section 6. Used when no LLM is available.

**Hugo CLI flow with `--rewrite`**:

```python
paragraphs = extract_paragraphs(body)

if rewrite and llm_configured:
    paragraphs = rewrite_paragraphs(paragraphs, ...)

encoded = tts.encode_batch(paragraphs)
alignment = extract_paragraph_alignment(encoded)
```

Alignment stays structural — paragraph indices, not text matching. Rewriting doesn't break it.

**Files modified**: `narro/hugo/cli.py`, `narro/hugo/extract.py`
**Files created**: `narro/rewrite.py`

## New Dependencies

| Dependency | Scope | Purpose |
|---|---|---|
| `fastapi` | `narro[server]` optional extra | Server framework |
| `uvicorn` | `narro[server]` optional extra | ASGI server |

No new required dependencies. `requests` (used for client and LLM rewriting) is already a transitive dependency via `huggingface_hub`.

## New CLI Subcommands

| Command | Purpose |
|---|---|
| `narro serve` | Start the TTS API server |
| `narro bench` | Run performance benchmarks |

## New Environment Variables

| Variable | Purpose |
|---|---|
| `NARRO_SERVER` | URL of a narro server for remote inference |
| `NARRO_LLM_URL` | LLM endpoint for text rewriting |
| `NARRO_LLM_KEY` | API key for LLM endpoint (optional) |
| `NARRO_LLM_MODEL` | Model name for LLM rewriting |

## File Summary

**New files**:
- `narro/server.py` — FastAPI server
- `narro/client.py` — HTTP client for remote inference
- `narro/bench.py` — Benchmark framework
- `narro/rewrite.py` — LLM rewriting layer

**Modified files**:
- `narro/tts.py` — Device support, quality_check, threading, retries
- `narro/backends/transformers.py` — Device parameter
- `narro/backends/base.py` — Device-aware inference, inference_mode
- `narro/decode_only.py` — Device-aware decoder loading
- `narro/cli.py` — `serve`, `bench` subcommands, `--server` flag
- `narro/hugo/cli.py` — Client integration, `--rewrite` flag
- `narro/hugo/extract.py` — `extract_paragraphs()` helper
- `narro/utils/text_normalizer.py` — clean_text improvements
- `pyproject.toml` — `[server]` optional extra
- `benchmarks/bench.py` — Replaced by `narro/bench.py`
