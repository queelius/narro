# Narro Rename and Simplification — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename soprano → narro, remove dead code (JS decoder, ONNX exports), create fresh GitHub repo, and add `--align` feature for word-level timestamps.

**Architecture:** Mechanical rename of module directory and all references, followed by a new alignment extraction feature that uses LLM attention weights to produce word-level timestamps during TTS generation.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers, pytest

---

### Task 1: Remove js/ and exports/ directories

**Files:**
- Delete: `js/` (entire directory)
- Delete: `exports/` (entire directory)
- Modify: `.gitignore` (remove `*.onnx` rule, no longer needed)

**Step 1: Delete the directories**

```bash
rm -rf js/ exports/
```

**Step 2: Clean up .gitignore**

Remove `*.onnx` line (no longer relevant). Keep other rules.

**Step 3: Run tests to confirm nothing depended on them**

Run: `pytest tests/ -v`
Expected: All 245 pass (these dirs were not imported by anything in soprano/)

**Step 4: Commit**

```bash
git add -A
git commit -m "Remove JS browser decoder and ONNX exports"
```

---

### Task 2: Rename module directory soprano/ → narro/

**Files:**
- Rename: `soprano/` → `narro/` (entire directory)

**Step 1: Git mv the directory**

```bash
git mv soprano/ narro/
```

This preserves git history for all files.

**Step 2: Verify directory structure**

```bash
ls narro/
# Should show: __init__.py backends/ cli.py decode_only.py encoded.py tts.py utils/ vocos/
```

**Step 3: Commit (tests will fail — imports not updated yet)**

```bash
git add -A
git commit -m "Rename module directory soprano/ -> narro/"
```

---

### Task 3: Update all internal imports

**Files:**
- Modify: `narro/__init__.py`
- Modify: `narro/tts.py`
- Modify: `narro/cli.py`
- Modify: `narro/decode_only.py`
- Modify: `narro/encoded.py`
- Modify: `narro/backends/transformers.py`
- Modify: `narro/vocos/decoder.py` (if it imports from soprano)
- Modify: `benchmarks/bench.py`

**Step 1: Find-and-replace `from soprano.` → `from narro.` in narro/ directory**

In every `.py` file under `narro/`, replace:
- `from .backends` → unchanged (relative imports stay)
- `from soprano.` → `from narro.` (only absolute imports)
- `import soprano` → `import narro`

The internal files mostly use relative imports (`.backends`, `.encoded`, etc.) so only a few need changing. Check each file:

- `narro/__init__.py`: Uses relative imports (`.tts`, `.encoded`) — no change needed
- `narro/tts.py`: Uses relative imports — no change needed
- `narro/cli.py`: `from soprano import SopranoTTS` → `from narro import Narro`; `from soprano.encoded import` → `from narro.encoded import`; `from soprano.decode_only import` → `from narro.decode_only import`
- `narro/decode_only.py`: Uses relative imports — no change needed
- `narro/encoded.py`: Uses no soprano imports — no change needed
- `narro/backends/transformers.py`: Uses relative import — no change needed
- `benchmarks/bench.py`: `from soprano` → `from narro`

**Step 2: Rename SopranoTTS → Narro**

In `narro/tts.py`:
- `class SopranoTTS:` → `class Narro:`
- Docstring: update class description

In `narro/__init__.py`:
- `from .tts import SopranoTTS as SopranoTTS` → `from .tts import Narro as Narro`

In `narro/cli.py`:
- All references to `SopranoTTS` → `Narro`
- `from narro import Narro` (or `from narro import Narro`)

**Step 3: Update pyproject.toml**

```toml
[project]
name = "narro"
version = "0.3.0"
authors = [
  { name="Alex Towell", email="lex@metafunctor.com" },
]
description = "Narro: lightweight CPU text-to-speech"

[project.urls]
Homepage = "https://github.com/queelius/narro"
Issues = "https://github.com/queelius/narro/issues"

[tool.setuptools.packages.find]
include = ["narro*"]

[project.scripts]
narro = "narro.cli:main"
```

**Step 4: Run tests (will still fail — test imports not updated)**

Run: `pytest tests/ -v 2>&1 | head -20`
Expected: ImportError for `soprano`

**Step 5: Commit**

```bash
git add -A
git commit -m "Update all internal imports soprano -> narro, rename SopranoTTS -> Narro"
```

---

### Task 4: Update all test imports

**Files:**
- Modify: `tests/test_tts_coverage.py` (~139 soprano references, ~15 SopranoTTS references)
- Modify: `tests/test_encode_decode.py` (~13 soprano references, ~4 SopranoTTS references)
- Modify: `tests/test_encoded.py` (~2 soprano references)
- Modify: `tests/test_performance.py` (~8 soprano references, ~3 SopranoTTS references)
- Modify: `tests/test_benchmarks.py` (~2 soprano references, ~2 SopranoTTS references)

**Step 1: Global find-and-replace in tests/**

For each test file, replace:
- `from soprano.` → `from narro.`
- `from soprano import` → `from narro import`
- `import soprano` → `import narro`
- `soprano.` in mock patch paths → `narro.` (e.g., `patch('soprano.tts.TransformersModel')` → `patch('narro.tts.TransformersModel')`)
- `SopranoTTS` → `Narro` everywhere

**Critical: mock patch paths must match**. Every `@patch('soprano.xxx')` or `patch('soprano.xxx')` must become `@patch('narro.xxx')` or `patch('narro.xxx')`.

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All 245 pass, 5 skipped

**Step 3: Commit**

```bash
git add tests/
git commit -m "Update all test imports soprano -> narro"
```

---

### Task 5: Update documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `narro/tts.py` (docstrings)
- Modify: `narro/decode_only.py` (docstrings)
- Modify: `narro/encoded.py` (docstrings)
- Modify: `narro/cli.py` (help text, epilog)

**Step 1: Update CLAUDE.md**

Replace all references:
- "Soprano" → "Narro" (when referring to this project)
- "Soprano-1.1-80M" stays (it's the upstream model name)
- `soprano/` paths → `narro/` paths
- `SopranoTTS` → `Narro`
- CLI examples: `soprano "text"` → `narro "text"`

**Step 2: Update README.md**

Rewrite to reflect narro identity:
- Title: "Narro"
- Description: "Lightweight CPU text-to-speech"
- Installation: `pip install narro`
- CLI: `narro "Hello world" -o output.wav`
- Python: `from narro import Narro`

**Step 3: Update CLI help text**

In `narro/cli.py`:
- Parser description: `'Narro Text-to-Speech CLI'`
- Epilog examples: `narro "Hello world"` etc.

**Step 4: Update module docstrings**

In `narro/encoded.py`: "Intermediate representation for Narro TTS..."
In `narro/decode_only.py`: "Lightweight decode-only module for Narro TTS..."

**Step 5: Run tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 6: Commit**

```bash
git add -A
git commit -m "Update documentation for narro rename"
```

---

### Task 6: Reinstall and verify

**Step 1: Reinstall in development mode**

```bash
pip install -e .
```

**Step 2: Verify CLI works**

```bash
narro --help
# Should show "Narro Text-to-Speech CLI"
```

**Step 3: Verify Python import works**

```bash
python -c "from narro import Narro; print('OK')"
```

**Step 4: Run full test suite with coverage**

```bash
pytest tests/ --cov=narro -v
```

Expected: All pass, coverage report shows `narro/` paths

**Step 5: Commit any remaining fixes**

---

### Task 7: Create fresh GitHub repo

**Step 1: Delete the fork on GitHub**

```bash
gh repo delete queelius/soprano --yes
```

**Step 2: Create fresh narro repo**

```bash
gh repo create queelius/narro --public --description "Narro: lightweight CPU text-to-speech"
```

**Step 3: Update git remote and push**

```bash
git remote set-url origin https://github.com/queelius/narro.git
git push -u origin main
```

**Step 4: Verify**

```bash
gh api repos/queelius/narro --jq '.fork'
# Should print: false
```

---

### Task 8: Add --align feature (TDD)

**Files:**
- Create: `narro/alignment.py`
- Modify: `narro/tts.py` (add align parameter to speak/infer)
- Modify: `narro/cli.py` (add --align flag)
- Create: `tests/test_alignment.py`

**Step 1: Write failing test for alignment extraction**

Create `tests/test_alignment.py`:

```python
"""Tests for word-level alignment extraction."""
import numpy as np
import pytest

from narro.alignment import extract_alignment


class TestExtractAlignment:
    def test_basic_alignment(self):
        """Two words, 10 generated tokens, 6 input tokens."""
        # Input: "[STOP][TEXT]Hello world[START]"
        # Tokens: [STOP], [TEXT], Hel, lo, _world, [START]
        # Words: "Hello" (tokens 2,3), "world" (token 4)
        # Special tokens (0,1,5) are not words.
        T = 10  # generated audio tokens
        input_len = 6
        attention = np.zeros((T, input_len), dtype=np.float32)
        # First 5 generated tokens attend to "Hello" tokens (2,3)
        attention[0:5, 2] = 0.4
        attention[0:5, 3] = 0.4
        # Last 5 generated tokens attend to "world" token (4)
        attention[5:10, 4] = 0.8

        token_to_word = {2: "Hello", 3: "Hello", 4: "world"}
        token_duration = 0.064  # 2048 / 32000

        result = extract_alignment(attention, token_to_word, token_duration)

        assert len(result) == 2
        assert result[0]["word"] == "Hello"
        assert result[1]["word"] == "world"
        assert result[0]["start"] < result[1]["start"]
        assert result[0]["end"] <= result[1]["start"] + 0.01  # small overlap ok
        # Timestamps should be in seconds
        assert result[1]["end"] <= T * token_duration + 0.01

    def test_empty_attention(self):
        """No attention weights produces empty alignment."""
        attention = np.zeros((0, 0), dtype=np.float32)
        result = extract_alignment(attention, {}, 0.064)
        assert result == []

    def test_single_word(self):
        """Single word maps to full duration."""
        T = 5
        attention = np.zeros((T, 3), dtype=np.float32)
        attention[:, 1] = 1.0
        token_to_word = {1: "Hello"}

        result = extract_alignment(attention, token_to_word, 0.064)
        assert len(result) == 1
        assert result[0]["word"] == "Hello"

    def test_timestamps_are_rounded(self):
        """Timestamps are rounded to 3 decimal places."""
        T = 10
        attention = np.ones((T, 3), dtype=np.float32)
        token_to_word = {1: "word"}

        result = extract_alignment(attention, token_to_word, 0.064)
        for entry in result:
            # Check 3 decimal places
            assert entry["start"] == round(entry["start"], 3)
            assert entry["end"] == round(entry["end"], 3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_alignment.py -v`
Expected: FAIL with ImportError (alignment module doesn't exist)

**Step 3: Implement extract_alignment**

Create `narro/alignment.py`:

```python
"""Word-level alignment extraction from attention weights."""

import json
from collections import defaultdict

import numpy as np


def extract_alignment(attention, token_to_word, token_duration):
    """Extract word-level timestamps from attention weights.

    Uses the attention matrix to find when each word was being "spoken"
    by computing the center of mass of attention over generated tokens.

    Args:
        attention: numpy array (T, input_len) — attention from generated
            tokens (rows) to input tokens (columns).
        token_to_word: dict mapping input token index -> word string.
            Only tokens that correspond to actual words should be included
            (not special tokens like [STOP], [TEXT], [START]).
        token_duration: seconds per generated token (TOKEN_SIZE / SAMPLE_RATE).

    Returns:
        List of dicts with 'word', 'start', 'end' keys, ordered by start time.
    """
    T, input_len = attention.shape
    if T == 0 or not token_to_word:
        return []

    # Group input token indices by word, preserving order
    word_tokens = defaultdict(list)
    word_order = []
    seen = set()
    for tok_idx in sorted(token_to_word.keys()):
        word = token_to_word[tok_idx]
        word_tokens[word].append(tok_idx)
        if word not in seen:
            word_order.append(word)
            seen.add(word)

    # For each word, compute weighted center-of-mass over generated tokens
    gen_times = np.arange(T, dtype=np.float32) * token_duration
    result = []

    for word in word_order:
        tok_indices = word_tokens[word]
        # Sum attention over all input tokens belonging to this word
        word_attn = attention[:, tok_indices].sum(axis=1)  # shape (T,)
        total = word_attn.sum()
        if total < 1e-10:
            continue

        # Weighted mean = center of mass
        center = float(np.dot(gen_times, word_attn) / total)
        # Spread = weighted std dev, gives approximate duration
        variance = float(np.dot((gen_times - center) ** 2, word_attn) / total)
        spread = max(variance ** 0.5, token_duration)  # at least one token wide

        start = round(max(0.0, center - spread), 3)
        end = round(min(T * token_duration, center + spread), 3)

        result.append({"word": word, "start": start, "end": end})

    return result


def save_alignment(alignment, path):
    """Save alignment to a JSON file.

    Args:
        alignment: List of dicts from extract_alignment().
        path: Output JSON file path.
    """
    with open(path, 'w') as f:
        json.dump(alignment, f, indent=2)
```

**Step 4: Run tests**

Run: `pytest tests/test_alignment.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add narro/alignment.py tests/test_alignment.py
git commit -m "Add word-level alignment extraction from attention weights"
```

**Step 6: Write failing test for tokenizer-to-word mapping**

Add to `tests/test_alignment.py`:

```python
from narro.alignment import build_token_to_word_map


class TestBuildTokenToWordMap:
    def test_basic_mapping(self):
        """Maps tokenizer output back to words."""
        # Simulated tokenizer offset mapping for "[STOP][TEXT]Hello world[START]"
        # The text portion is "Hello world" starting at some offset
        text = "Hello world"
        # offsets are character positions in the FULL prompt
        # Assume prompt is "[STOP][TEXT]Hello world[START]"
        # "[STOP][TEXT]" = 12 chars, so text starts at 12
        prefix_len = 12  # len("[STOP][TEXT]")
        offsets = [
            (0, 6),      # [STOP] — special
            (6, 12),     # [TEXT] — special
            (12, 15),    # "Hel"
            (15, 17),    # "lo"
            (17, 18),    # " "  (space — not a word)
            (18, 23),    # "world"
            (23, 30),    # [START] — special
        ]

        result = build_token_to_word_map(text, offsets, prefix_len)

        # Tokens 2,3 -> "Hello", token 5 -> "world"
        # Token 4 is just a space, should be attached to adjacent word
        assert 2 in result
        assert 5 in result
        assert result[2] == "Hello"
        assert result[5] == "world"
```

**Step 7: Implement build_token_to_word_map**

Add to `narro/alignment.py`:

```python
def build_token_to_word_map(text, token_offsets, prefix_len):
    """Map tokenizer token indices to words in the original text.

    Args:
        text: Original sentence text (before wrapping with [STOP][TEXT]...[START]).
        token_offsets: List of (start, end) character offsets from the tokenizer,
            relative to the full prompt string.
        prefix_len: Number of characters in the prompt prefix before the text
            (e.g., len("[STOP][TEXT]") = 12).

    Returns:
        Dict mapping token index -> word string, only for tokens that
        correspond to actual words in the text.
    """
    text_start = prefix_len
    text_end = text_start + len(text)
    words = text.split()

    # Build character-position-to-word index
    char_to_word = {}
    pos = 0
    for word in words:
        idx = text.find(word, pos)
        for c in range(idx, idx + len(word)):
            char_to_word[c] = word
        pos = idx + len(word)

    # Map each token to a word
    token_to_word = {}
    for tok_idx, (start, end) in enumerate(token_offsets):
        if start < text_start or end > text_end:
            continue  # skip special tokens outside text region
        # Map to text-relative position
        text_pos = start - text_start
        if text_pos in char_to_word:
            token_to_word[tok_idx] = char_to_word[text_pos]

    return token_to_word
```

**Step 8: Run tests**

Run: `pytest tests/test_alignment.py -v`
Expected: All pass

**Step 9: Commit**

```bash
git add narro/alignment.py tests/test_alignment.py
git commit -m "Add token-to-word mapping for alignment"
```

**Step 10: Write failing test for CLI --align flag**

Add to `tests/test_alignment.py`:

```python
import argparse
from unittest.mock import patch, MagicMock


class TestCLIAlignFlag:
    def test_speak_parser_has_align(self):
        """The speak subcommand accepts --align."""
        from narro.cli import main
        with patch('sys.argv', ['narro', 'speak', 'Hello', '--align', 'out.json']):
            # Just parse args, don't execute
            import narro.cli as cli_mod
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest='command')
            speak_parser = subparsers.add_parser('speak')
            cli_mod._add_speak_args(speak_parser)
            cli_mod._add_common_args(speak_parser)
            args = parser.parse_args(['speak', 'Hello', '--align', 'out.json'])
            assert args.align == 'out.json'
```

**Step 11: Add --align to CLI**

In `narro/cli.py`, in `_add_speak_args()`:

```python
parser.add_argument('--align', '-a',
                    help='Output word-alignment JSON file path')
```

In `cmd_speak()`, after `tts.infer(...)`:

```python
if args.align:
    # Alignment requires encode with attention, then decode
    encoded = tts.encode(args.text, include_attention=True)
    from narro.alignment import extract_alignment_from_encoded, save_alignment
    alignment = extract_alignment_from_encoded(encoded)
    save_alignment(alignment, args.align)
```

**Step 12: Implement extract_alignment_from_encoded**

Add to `narro/alignment.py`:

```python
def extract_alignment_from_encoded(encoded, tokenizer=None):
    """Extract word alignment from an EncodedSpeech with attention weights.

    Convenience function that handles the full pipeline:
    attention weights + tokenizer offsets -> word timestamps.

    Args:
        encoded: EncodedSpeech with attention_weights populated.
        tokenizer: HuggingFace tokenizer (loaded from model if None).

    Returns:
        List of alignment dicts across all sentences, with cumulative timestamps.
    """
    from .tts import TOKEN_SIZE, SAMPLE_RATE

    token_duration = TOKEN_SIZE / SAMPLE_RATE
    all_alignment = []
    time_offset = 0.0

    for sentence in encoded.sentences:
        if sentence.attention_weights is None:
            time_offset += len(sentence.hidden_states) * token_duration
            continue

        # Build token-to-word map from the sentence text
        prefix = "[STOP][TEXT]"
        text = sentence.text
        # Simple word-based alignment without tokenizer offset mapping
        # Each input token's attention tells us when it was spoken
        # For now, use a simplified approach based on attention peaks
        words = text.split()
        if not words:
            time_offset += len(sentence.hidden_states) * token_duration
            continue

        T = sentence.attention_weights.shape[0]
        input_len = sentence.attention_weights.shape[1]

        # Divide input positions evenly among words (simplified)
        # Skip first 2 and last 1 positions ([STOP], [TEXT], [START])
        usable_start = 2
        usable_end = input_len - 1
        usable_len = usable_end - usable_start
        if usable_len <= 0:
            time_offset += T * token_duration
            continue

        tokens_per_word = max(1, usable_len // len(words))
        token_to_word = {}
        for i, word in enumerate(words):
            start_tok = usable_start + i * tokens_per_word
            end_tok = min(usable_start + (i + 1) * tokens_per_word, usable_end)
            for t in range(start_tok, end_tok):
                token_to_word[t] = word

        alignment = extract_alignment(
            sentence.attention_weights, token_to_word, token_duration
        )

        # Apply cumulative time offset
        for entry in alignment:
            entry["start"] = round(entry["start"] + time_offset, 3)
            entry["end"] = round(entry["end"] + time_offset, 3)

        all_alignment.extend(alignment)
        time_offset += T * token_duration

    return all_alignment
```

**Step 13: Run all tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 14: Commit**

```bash
git add -A
git commit -m "Add --align CLI flag for word-level timestamps"
```

---

### Task 9: Final verification and push

**Step 1: Run full test suite with coverage**

```bash
pytest tests/ --cov=narro -v
```

Expected: All pass, good coverage

**Step 2: Verify CLI end-to-end**

```bash
narro --help
python -c "from narro import Narro; print('import OK')"
```

**Step 3: Push to GitHub**

```bash
git push origin main
```
