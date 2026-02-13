# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Soprano is an ultra-lightweight text-to-speech (TTS) core library (80M params) that generates 32kHz audio. It uses an LLM to produce hidden states from text, then a Vocos-based decoder converts those hidden states to audio via ISTFT. CPU-only.

## Development Commands

```bash
# Install for development
pip install -e .

# Run tests
pytest tests/

# Run a single test
pytest tests/test_performance.py::TestWeightMigration::test_migrate_pwconv_weights

# Run benchmark tests (skipped by default)
pytest tests/ -m benchmark

# Run with coverage
pytest tests/ --cov=soprano

# CLI usage
soprano "Hello world" -o output.wav
```

## Architecture

### Inference Pipeline

Text flows through three stages:

1. **Text preprocessing** (`tts.py:_preprocess_text`): `clean_text()` normalizes numbers/abbreviations/special chars, `split_and_recombine_text()` splits into sentences, short sentences are merged (min 30 chars). Each sentence is wrapped as `[STOP][TEXT]...[START]`.

2. **LLM backbone** (`backends/transformers.py`): A HuggingFace causal LM (`ekwek/Soprano-1.1-80M`) generates hidden states — not text tokens. The last hidden layer's per-token vectors are extracted and stacked into a tensor of shape `(seq_len, 512)`. Supports batch inference and token-by-token streaming.

3. **Vocos decoder** (`vocos/`): ConvNeXt blocks process hidden states, then `ISTFTHead` predicts magnitude + phase for inverse STFT to produce waveform audio. The decoder upsamples by 4x via linear interpolation before the ConvNeXt backbone. Each hidden state token maps to 2048 audio samples.

### Key Constants (in `tts.py`)

- `SAMPLE_RATE = 32000` — output audio sample rate
- `TOKEN_SIZE = 2048` — audio samples per decoder token
- `HIDDEN_DIM = 512` — LLM hidden state dimension
- `RECEPTIVE_FIELD = 4` — decoder context window for streaming

### Decoder Weight Migration

The decoder's `ConvNeXtBlock` was refactored from `nn.Linear` to `nn.Conv1d` (kernel_size=1) to eliminate transpose operations. Old checkpoints have 2D weights `(out, in)` that need migration to 3D `(out, in, 1)`. The `vocos/migrate_weights.py` module handles this transparently via `load_with_migration()`.

### Hallucination Detection

`tts.py:hallucination_detector` monitors consecutive hidden states — if the L1 difference between adjacent states stays below `DIFF_THRESHOLD=300` for more than `MAX_RUNLENGTH=16` steps, it flags hallucination and can trigger regeneration (controlled by `retries` parameter).

## Testing

Tests use `pytest` with `unittest.mock`. Performance tests validate tensor shapes and weight migration without loading the actual model. Benchmark tests are `@pytest.mark.skip` by default.

## Project-Specific Conventions

- The `SopranoTTS` constructor runs a warmup inference (`self.infer("Hello world!")`) — tests that construct it must mock this or patch the class.
- Audio tensors are float32 in `[-1, 1]` range internally; converted to int16 PCM at output boundaries (WAV files).
- The `ISTFTHead.forward` is decorated with `@torch.compiler.disable` because torch.compile doesn't support complex FFT operations.
- `temperature=0.0` in the public API is silently clamped to `0.001` in the backend to avoid division by zero.
