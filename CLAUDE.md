# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Narro is a model-agnostic text-to-speech server and client. It defines a `TTSModel` protocol that any TTS backend can implement, a model registry for multi-model serving, and an HTTP API (OpenAI-compatible `/v1/audio/speech`). The first supported backend is Soprano-1.1-80M (80M params, 32kHz audio, Qwen3 LLM + Vocos decoder).

## Architecture

### Three-Layer Design

```
HTTP API (/v1/audio/speech, /v1/models)     ← client.py talks to this
    ↓
Server (server.py) + Model Registry          ← model-agnostic
    ↓
Model Backends (models/soprano.py, ...)      ← implement TTSModel protocol
```

- **Protocol** (`protocol.py`): `TTSModel` Protocol with `synthesize()` and `synthesize_stream()`. Returns `AudioResult` (numpy float32 audio + metadata dict) or yields `AudioChunk`.
- **Registry** (`models/__init__.py`): `ModelRegistry` singleton. First registered model is the default. Models looked up by `model_id`.
- **Server** (`server.py`): FastAPI app, model-agnostic. Routes requests to models via registry. `configure_app()` registers models; `serve()` is the CLI entry point.
- **Client** (`client.py`): Thin HTTP client against the API contract. Works with any compatible server.
- **CLI** (`cli.py`): `narro serve` (starts server), `narro speak` (client, requires server).

### Soprano Backend (models/soprano.py)

Wraps the `Narro` class from `tts.py` to implement `TTSModel`. Model ID: `soprano-80m`.

The Soprano inference pipeline (internal to the backend):

1. **Text preprocessing** (`tts.py:_preprocess_text`): `clean_text()` normalizes numbers/abbreviations/special chars, `split_and_recombine_text()` splits into sentences, short sentences are merged (min 30 chars). Each sentence is wrapped as `[STOP][TEXT]...[START]`.

2. **LLM backbone** (`backends/transformers.py`): HuggingFace causal LM (`ekwek/Soprano-1.1-80M`) generates hidden states (not text tokens). Shape: `(seq_len, 512)`. Supports batch inference and streaming.

3. **Vocos decoder** (`vocos/`): ConvNeXt blocks process hidden states, then `ISTFTHead` predicts magnitude + phase for inverse STFT. Each hidden state token maps to 2048 audio samples at 32kHz.

### Key Constants (in `tts.py`)

- `SAMPLE_RATE = 32000`: output audio sample rate
- `TOKEN_SIZE = 2048`: audio samples per decoder token
- `HIDDEN_DIM = 512`: LLM hidden state dimension
- `RECEPTIVE_FIELD = 4`: decoder context window for streaming

## Development Commands

```bash
# Install for development
pip install -e ".[server]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=narro

# Start server
narro serve --device cuda

# Synthesize (requires running server)
export NARRO_SERVER=http://localhost:8000
narro "Hello world" -o output.wav
```

## Testing

Tests use `pytest` with `unittest.mock`. Server tests use `FakeModel` (a plain class satisfying the `TTSModel` protocol): no real model loading. The `_clean_registry` autouse fixture ensures test isolation. Benchmark tests are `@pytest.mark.skip` by default.

## Project-Specific Conventions

- The `Narro` constructor runs a warmup inference (`self.infer("Hello world!")`): tests that construct it must mock this or patch the class.
- Audio is float32 `[-1, 1]` internally (numpy arrays at the protocol boundary); converted to int16 PCM at output (WAV files).
- The `ISTFTHead.forward` is decorated with `@torch.compiler.disable` because torch.compile doesn't support complex FFT operations.
- `temperature=0.0` in the public API is silently clamped to `0.001` in the backend to avoid division by zero.
- Synchronous model inference runs in `run_in_executor` / `asyncio.to_thread` to avoid blocking the event loop.
- The `.soprano` file format and `encoded.py` / `decode_only.py` are Soprano-internal: not exposed via CLI or API.
