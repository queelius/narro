# Embeddings Modality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `/v1/embeddings` as muse's third modality, mirroring OpenAI's request/response shape so the OpenAI Python SDK can drop in against muse as a base URL. First backend is `sentence-transformers/all-MiniLM-L6-v2` (22MB, 384-dim, CPU-friendly). Proves the claim that new modalities are additive: zero gateway changes needed because the gateway routes by request-body `model` field, not URL path.

**Architecture:** New subpackage `muse.embeddings/` mirrors `muse.images.generations/` in shape (protocol, router, codec, client, backends). Single-level (no family.op) because OpenAI's URL is `/v1/embeddings` (flat), and the gateway's body-based routing makes URL hierarchy irrelevant for dispatch. Modality key is the string `"embeddings"`. Catalog gains one entry; worker.py gains one `build_router` import + mount; everything else in muse is untouched. sentence-transformers lazy-imported behind try/except so the module loads without it installed; tests mock the library entirely.

**Tech Stack:** Python 3.10+, FastAPI + Pydantic (router), sentence-transformers + torch + numpy (backend, lazy-imported), httpx (client request, optional — we use `requests` to match the other clients' style).

---

## File Structure (final)

```
src/muse/embeddings/                         NEW subpackage
├── __init__.py                              exports EmbeddingsClient, EmbeddingsModel, EmbeddingResult
├── protocol.py                              EmbeddingsModel Protocol + EmbeddingResult dataclass
├── routes.py                                build_router(registry) -> APIRouter, EmbeddingsRequest
├── codec.py                                 embedding_to_base64, base64_to_embedding helpers
├── client.py                                EmbeddingsClient (thin HTTP wrapper)
└── backends/
    ├── __init__.py                          (empty)
    └── minilm.py                            MiniLMBackend (sentence-transformers wrapper)

src/muse/cli_impl/worker.py                  MODIFIED: add one build_router import + mount
src/muse/core/catalog.py                     MODIFIED: add one KNOWN_MODELS entry
pyproject.toml                               MODIFIED: add `embeddings` extras group

tests/embeddings/                            NEW test subtree
├── __init__.py
├── test_protocol.py
├── test_codec.py
├── test_routes.py
├── test_client.py
└── test_minilm.py

README.md, CLAUDE.md                         MODIFIED: mention the new modality
```

---

## Key design decisions

1. **Modality key is `"embeddings"`, not `"text.embeddings"`**. Matches OpenAI's URL. The gateway routes by `body.model`, so internal naming doesn't affect request dispatch. Future `audio.transcriptions` or `text.completions` will use the dotted form when they belong to a family with siblings.

2. **`EmbeddingResult.embeddings` is `list[list[float]]`, not `np.ndarray`**. JSON-serializable at the protocol boundary; no numpy dep leaks into the protocol. Backends may operate on numpy internally and convert via `.tolist()` at the boundary.

3. **`input` accepts `str | list[str]`** per OpenAI. The router normalizes to list internally so backends always see a list. This is structural validation at the Pydantic layer (a Union type).

4. **`encoding_format` defaults to `"float"`**. Setting `"base64"` encodes each embedding as little-endian float32 bytes → base64 (matches OpenAI's format exactly so `openai-python` clients work).

5. **`dimensions` truncation is naive (truncate + L2-normalize)**. MiniLM is NOT a matryoshka model, so truncating destroys information. We still support it for OpenAI SDK compatibility; quality is caller's responsibility. Future matryoshka-capable backends (e.g., `text-embedding-3-small`-family) override this with proper scaling.

6. **Usage reporting is approximate**. OpenAI reports `prompt_tokens` and `total_tokens`. We sum attention_mask lengths from sentence-transformers' tokenizer. Not a billing source; just populated for client-side compatibility.

7. **No streaming**. OpenAI's `/v1/embeddings` doesn't stream (the response is synchronous). Our router is pure sync-then-return, no SSE, no `asyncio.Queue`.

8. **sentence-transformers pulls in torch + transformers + huggingface_hub**. All already in muse's ecosystem. The `[embeddings]` extras group declares only `torch>=2.1.0` and `sentence-transformers>=2.2.0`; pip resolves transitively.

---

## Task graph

Parts A/B/C/D/E are independent of each other except through the protocol (A is prerequisite for B, C, D, E). F depends on everything; G depends on C; H depends on F; I depends on H.

```
A (protocol) -> B (codec) -+
             -> C (router)  -> F (catalog + pyproject + __init__) -> G (worker mount) -> H (docs) -> I (verify + merge)
             -> D (backend) -+
             -> E (client)  -+
```

9 tasks total. Parts B-E can be implemented in any order after A.

---

## Part A — Protocol

### Task A1: `muse.embeddings.protocol`

**Files:**
- Create: `src/muse/embeddings/__init__.py` (empty stub for now; populated in F)
- Create: `src/muse/embeddings/protocol.py`
- Create: `src/muse/embeddings/backends/__init__.py` (empty)
- Create: `tests/embeddings/__init__.py` (empty)
- Create: `tests/embeddings/test_protocol.py`

- [ ] **Step 1: Create the directory skeleton**

```bash
cd /home/spinoza/github/repos/muse
mkdir -p src/muse/embeddings/backends
mkdir -p tests/embeddings
touch src/muse/embeddings/__init__.py
touch src/muse/embeddings/backends/__init__.py
touch tests/embeddings/__init__.py
```

- [ ] **Step 2: Write failing tests**

File: `tests/embeddings/test_protocol.py`

```python
"""Tests for EmbeddingsModel protocol."""
import pytest

from muse.embeddings.protocol import EmbeddingsModel, EmbeddingResult


def test_embedding_result_stores_all_fields():
    r = EmbeddingResult(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        dimensions=2,
        model_id="fake-embed",
        prompt_tokens=10,
        metadata={"prompt": "hi"},
    )
    assert r.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert r.dimensions == 2
    assert r.model_id == "fake-embed"
    assert r.prompt_tokens == 10
    assert r.metadata["prompt"] == "hi"


def test_embedding_result_metadata_defaults_empty():
    r = EmbeddingResult(
        embeddings=[[0.1]], dimensions=1, model_id="x", prompt_tokens=0,
    )
    assert r.metadata == {}


def test_embedding_result_metadata_is_independent_per_instance():
    """Regression: shared mutable default would leak state across instances."""
    a = EmbeddingResult(embeddings=[[0.1]], dimensions=1, model_id="x", prompt_tokens=0)
    b = EmbeddingResult(embeddings=[[0.2]], dimensions=1, model_id="y", prompt_tokens=0)
    a.metadata["k"] = 1
    assert "k" not in b.metadata


def test_embeddings_model_protocol_accepts_structural_impl():
    class MyModel:
        model_id = "fake-embed"
        dimensions = 384
        def embed(self, input, **kwargs):
            ...

    assert isinstance(MyModel(), EmbeddingsModel)


def test_embeddings_model_protocol_rejects_incomplete():
    class Missing:
        pass

    assert not isinstance(Missing(), EmbeddingsModel)
```

- [ ] **Step 3: Run — verify fail**

```bash
pytest tests/embeddings/test_protocol.py -v
```

Expected: `ModuleNotFoundError: No module named 'muse.embeddings.protocol'`.

- [ ] **Step 4: Implement `src/muse/embeddings/protocol.py`**

```python
"""Muse embeddings modality protocol.

Defines EmbeddingsModel (backend contract) and EmbeddingResult
(synthesis return). The modality is flat (no family.op) because
OpenAI's endpoint is /v1/embeddings — single level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class EmbeddingResult:
    """N input texts in, N embedding vectors out, plus provenance.

    `embeddings` is a plain list[list[float]] at the protocol boundary
    so no numpy dep is needed by consumers. Backends may use numpy
    internally and convert via `.tolist()` before returning.
    """
    embeddings: list[list[float]]
    dimensions: int
    model_id: str
    prompt_tokens: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class EmbeddingsModel(Protocol):
    """Protocol for text-to-embedding backends."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(
        self,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        **kwargs,
    ) -> EmbeddingResult: ...
```

- [ ] **Step 5: Run — pass**

```bash
pytest tests/embeddings/test_protocol.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/muse/embeddings/__init__.py \
        src/muse/embeddings/protocol.py \
        src/muse/embeddings/backends/__init__.py \
        tests/embeddings/__init__.py \
        tests/embeddings/test_protocol.py
git commit -m "feat(embeddings): add EmbeddingsModel protocol + EmbeddingResult

Modality is flat ('embeddings'), matching OpenAI's /v1/embeddings URL.
EmbeddingResult.embeddings is list[list[float]] (no numpy at protocol
boundary). Backends implement .embed(input, dimensions=None) where
input is str or list[str]."
```

---

## Part B — Codec

### Task B1: base64 encoding helpers

**Files:**
- Create: `src/muse/embeddings/codec.py`
- Create: `tests/embeddings/test_codec.py`

OpenAI's `encoding_format="base64"` returns embeddings as little-endian float32 bytes encoded base64. We need helpers in both directions (encode for the router, decode for the client).

- [ ] **Step 1: Write failing tests**

File: `tests/embeddings/test_codec.py`

```python
"""Tests for embeddings base64 codec."""
import base64
import struct

import pytest

from muse.embeddings.codec import embedding_to_base64, base64_to_embedding


def test_round_trip_preserves_values_approximately():
    original = [0.1, -0.5, 3.14159, 0.0, 1.0]
    encoded = embedding_to_base64(original)
    decoded = base64_to_embedding(encoded)
    assert len(decoded) == len(original)
    # float32 round-trip is lossy; use approx equality
    for a, b in zip(original, decoded):
        assert abs(a - b) < 1e-6


def test_format_is_little_endian_float32():
    """OpenAI's format is little-endian float32 bytes -> base64.

    The openai-python SDK decodes with `np.frombuffer(bytes, dtype='<f4')`
    so our encoding must match that exactly.
    """
    vec = [1.0, 2.0]
    encoded = embedding_to_base64(vec)
    raw = base64.b64decode(encoded)
    # Two little-endian float32 values = 8 bytes
    assert len(raw) == 8
    # Unpack with struct to confirm byte order
    v0 = struct.unpack("<f", raw[0:4])[0]
    v1 = struct.unpack("<f", raw[4:8])[0]
    assert abs(v0 - 1.0) < 1e-6
    assert abs(v1 - 2.0) < 1e-6


def test_empty_embedding_encodes_to_empty_base64():
    assert embedding_to_base64([]) == ""
    assert base64_to_embedding("") == []


def test_large_embedding_round_trips():
    # 384-dim (MiniLM size) vector
    original = [i * 0.01 for i in range(384)]
    encoded = embedding_to_base64(original)
    decoded = base64_to_embedding(encoded)
    assert len(decoded) == 384
    for a, b in zip(original, decoded):
        assert abs(a - b) < 1e-5


def test_base64_to_embedding_rejects_invalid_length():
    """Raw bytes must be a multiple of 4 (float32 size)."""
    # 5 bytes = not a whole number of float32s
    bad_bytes = base64.b64encode(b"\x00" * 5).decode()
    with pytest.raises(ValueError, match="multiple of 4"):
        base64_to_embedding(bad_bytes)


def test_encoded_is_pure_ascii_base64():
    encoded = embedding_to_base64([1.0, 2.0, 3.0])
    # Should be valid base64 string (ascii-safe)
    encoded.encode("ascii")  # raises if not ascii
    # And should round-trip through base64.b64decode
    base64.b64decode(encoded)
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/embeddings/test_codec.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/muse/embeddings/codec.py`**

```python
"""Base64 encoding for embeddings.

OpenAI's /v1/embeddings with encoding_format='base64' returns
little-endian float32 bytes encoded as base64. Match that format
exactly so openai-python SDK clients round-trip cleanly via
`np.frombuffer(decoded_bytes, dtype='<f4')`.
"""
from __future__ import annotations

import base64

import numpy as np


def embedding_to_base64(embedding: list[float]) -> str:
    """Encode a float vector as little-endian float32 bytes -> base64.

    Empty input yields empty string.
    """
    if not embedding:
        return ""
    arr = np.asarray(embedding, dtype="<f4")  # little-endian float32
    return base64.b64encode(arr.tobytes()).decode("ascii")


def base64_to_embedding(encoded: str) -> list[float]:
    """Decode base64 -> little-endian float32 bytes -> list[float].

    Empty input yields empty list. Raises ValueError if the decoded
    byte length is not a multiple of 4 (float32 size).
    """
    if not encoded:
        return []
    raw = base64.b64decode(encoded)
    if len(raw) % 4 != 0:
        raise ValueError(
            f"decoded byte length {len(raw)} is not a multiple of 4 "
            f"(float32 size); data is corrupt"
        )
    arr = np.frombuffer(raw, dtype="<f4")
    return arr.tolist()
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/embeddings/test_codec.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/muse/embeddings/codec.py tests/embeddings/test_codec.py
git commit -m "feat(embeddings): add base64 codec for OpenAI wire format

Little-endian float32 bytes -> base64, matching OpenAI's encoding_format=base64
output exactly. openai-python SDK round-trips via np.frombuffer(dtype='<f4')."
```

---

## Part C — Router

### Task C1: `/v1/embeddings` FastAPI router

**Files:**
- Create: `src/muse/embeddings/routes.py`
- Create: `tests/embeddings/test_routes.py`

- [ ] **Step 1: Write failing tests**

File: `tests/embeddings/test_routes.py`

```python
"""Tests for /v1/embeddings FastAPI router."""
import base64
import struct

import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.embeddings.protocol import EmbeddingResult
from muse.embeddings.routes import build_router


class FakeEmbeddingsModel:
    """Protocol-compatible stub that returns deterministic vectors."""
    model_id = "fake-embed"
    dimensions = 4

    def embed(self, input, *, dimensions=None, **_):
        texts = [input] if isinstance(input, str) else list(input)
        vectors = [[float(i), 0.0, 0.0, 0.0] for i in range(len(texts))]
        if dimensions is not None and dimensions < self.dimensions:
            vectors = [v[:dimensions] for v in vectors]
            out_dim = dimensions
        else:
            out_dim = self.dimensions
        return EmbeddingResult(
            embeddings=vectors,
            dimensions=out_dim,
            model_id=self.model_id,
            prompt_tokens=sum(len(t.split()) for t in texts),
        )


@pytest.fixture
def client():
    reg = ModalityRegistry()
    reg.register("embeddings", FakeEmbeddingsModel())
    app = create_app(registry=reg, routers={"embeddings": build_router(reg)})
    return TestClient(app)


def test_single_string_input_returns_one_embedding(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "model": "fake-embed"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["object"] == "embedding"
    assert entry["index"] == 0
    assert entry["embedding"] == [0.0, 0.0, 0.0, 0.0]
    assert body["model"] == "fake-embed"


def test_list_input_returns_embeddings_in_order(client):
    r = client.post("/v1/embeddings", json={
        "input": ["first", "second", "third"],
        "model": "fake-embed",
    })
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 3
    # FakeEmbeddingsModel returns [i, 0, 0, 0] for the i-th input
    for i, entry in enumerate(data):
        assert entry["index"] == i
        assert entry["embedding"][0] == float(i)


def test_response_includes_usage(client):
    r = client.post("/v1/embeddings", json={
        "input": ["hello world", "foo bar baz"],
        "model": "fake-embed",
    })
    body = r.json()
    assert "usage" in body
    assert body["usage"]["prompt_tokens"] == 5  # 2 + 3 split words
    assert body["usage"]["total_tokens"] == 5


def test_default_model_when_unspecified(client):
    r = client.post("/v1/embeddings", json={"input": "hello"})
    assert r.status_code == 200
    # Default model was the only registered one
    assert r.json()["model"] == "fake-embed"


def test_unknown_model_returns_openai_shape_404(client):
    r = client.post("/v1/embeddings", json={
        "input": "hi", "model": "no-such-model",
    })
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert "detail" not in body
    assert body["error"]["code"] == "model_not_found"


def test_encoding_format_base64_returns_base64_strings(client):
    r = client.post("/v1/embeddings", json={
        "input": "hello",
        "model": "fake-embed",
        "encoding_format": "base64",
    })
    assert r.status_code == 200
    entry = r.json()["data"][0]
    assert isinstance(entry["embedding"], str)
    # Decode and verify round-trip
    raw = base64.b64decode(entry["embedding"])
    assert len(raw) == 16  # 4 floats * 4 bytes
    vals = [struct.unpack("<f", raw[i:i+4])[0] for i in range(0, 16, 4)]
    for v, expected in zip(vals, [0.0, 0.0, 0.0, 0.0]):
        assert abs(v - expected) < 1e-6


def test_dimensions_truncation(client):
    r = client.post("/v1/embeddings", json={
        "input": "hello",
        "model": "fake-embed",
        "dimensions": 2,
    })
    assert r.status_code == 200
    emb = r.json()["data"][0]["embedding"]
    assert len(emb) == 2


def test_empty_input_string_rejected(client):
    r = client.post("/v1/embeddings", json={"input": "", "model": "fake-embed"})
    assert r.status_code in (400, 422)


def test_empty_input_list_rejected(client):
    r = client.post("/v1/embeddings", json={"input": [], "model": "fake-embed"})
    assert r.status_code in (400, 422)


def test_dimensions_out_of_range_rejected(client):
    r = client.post("/v1/embeddings", json={
        "input": "hi", "model": "fake-embed",
        "dimensions": 99999,
    })
    assert r.status_code in (400, 422)


def test_invalid_encoding_format_rejected(client):
    r = client.post("/v1/embeddings", json={
        "input": "hi", "model": "fake-embed",
        "encoding_format": "bogus",
    })
    assert r.status_code in (400, 422)


def test_user_field_accepted_and_ignored(client):
    """OpenAI allows optional `user` for abuse monitoring; we accept + ignore."""
    r = client.post("/v1/embeddings", json={
        "input": "hi", "model": "fake-embed",
        "user": "user-42",
    })
    assert r.status_code == 200
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/embeddings/test_routes.py -v
```

Expected: `ModuleNotFoundError` on collection.

- [ ] **Step 3: Implement `src/muse/embeddings/routes.py`**

```python
"""FastAPI router for /v1/embeddings.

Follows OpenAI's /v1/embeddings contract:
  - `input`: str | list[str] (required, at least one non-empty)
  - `model`: str (optional; uses modality default if absent)
  - `encoding_format`: "float" (default) | "base64"
  - `dimensions`: int (optional; backend-dependent truncation)
  - `user`: str (optional; accepted for compat, ignored)

Response shape:
  {
    "object": "list",
    "data": [
      {"object": "embedding", "embedding": [...] | "base64...", "index": 0},
      ...
    ],
    "model": "...",
    "usage": {"prompt_tokens": N, "total_tokens": N}
  }
"""
from __future__ import annotations

import asyncio
import logging
from threading import Lock
from typing import Union

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from muse.core.errors import ModelNotFoundError
from muse.core.registry import ModalityRegistry
from muse.embeddings.codec import embedding_to_base64

logger = logging.getLogger(__name__)

MODALITY = "embeddings"
_inference_lock = Lock()


class EmbeddingsRequest(BaseModel):
    # Union types in pydantic v2 accept str OR list[str].
    # We validate non-emptiness ourselves below.
    input: Union[str, list[str]]
    model: str | None = None
    encoding_format: str = Field(default="float", pattern="^(float|base64)$")
    dimensions: int | None = Field(default=None, ge=1, le=8192)
    user: str | None = None  # OpenAI compat; ignored

    @field_validator("input")
    @classmethod
    def _input_nonempty(cls, v):
        if isinstance(v, str):
            if not v:
                raise ValueError("input string cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("input list cannot be empty")
            if any(not isinstance(s, str) or not s for s in v):
                raise ValueError("input list must contain non-empty strings")
        return v


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["embeddings"])

    @router.post("/embeddings")
    async def embeddings(req: EmbeddingsRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        def _call():
            with _inference_lock:
                return model.embed(req.input, dimensions=req.dimensions)

        result = await asyncio.to_thread(_call)

        data = []
        for i, vec in enumerate(result.embeddings):
            if req.encoding_format == "base64":
                embedding_field = embedding_to_base64(vec)
            else:
                embedding_field = vec
            data.append({
                "object": "embedding",
                "embedding": embedding_field,
                "index": i,
            })

        return {
            "object": "list",
            "data": data,
            "model": result.model_id,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "total_tokens": result.prompt_tokens,
            },
        }

    return router
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/embeddings/test_routes.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Regression check**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: prior tests still pass + 12 new = previous_count + 12.

- [ ] **Step 6: Commit**

```bash
git add src/muse/embeddings/routes.py tests/embeddings/test_routes.py
git commit -m "feat(embeddings): add /v1/embeddings FastAPI router

Matches OpenAI's request/response shape exactly so openai-python SDK
clients work against muse with base_url override. Supports str or
list[str] input, float or base64 encoding_format, optional dimensions
truncation, optional user field (accepted + ignored). Unknown model
returns OpenAI-shape 404 via ModelNotFoundError."
```

---

## Part D — Backend

### Task D1: `MiniLMBackend` (sentence-transformers)

**Files:**
- Create: `src/muse/embeddings/backends/minilm.py`
- Create: `tests/embeddings/test_minilm.py`

The real backend. Uses `sentence-transformers` library to run `all-MiniLM-L6-v2`. Tests mock the library entirely so they run without the dep installed.

- [ ] **Step 1: Write failing tests**

File: `tests/embeddings/test_minilm.py`

```python
"""Tests for MiniLMBackend (fully mocked; sentence-transformers not loaded)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from muse.embeddings.protocol import EmbeddingResult


def _mock_model(encode_return=None, dim=384):
    """Build a fake SentenceTransformer instance with a predictable shape."""
    m = MagicMock()
    if encode_return is None:
        m.encode.return_value = np.zeros((1, dim), dtype=np.float32)
    else:
        m.encode.return_value = np.asarray(encode_return, dtype=np.float32)
    m.get_sentence_embedding_dimension.return_value = dim
    # tokenize returns a dict with attention_mask we can sum
    def _tok(texts):
        n = len(texts)
        # pretend each text is 5 tokens
        return {
            "input_ids": np.ones((n, 5), dtype=np.int64),
            "attention_mask": np.ones((n, 5), dtype=np.int64),
        }
    m.tokenize.side_effect = _tok
    return m


def test_minilm_model_id_and_dimensions():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="sentence-transformers/all-MiniLM-L6-v2",
                          local_dir="/fake")
        assert m.model_id == "all-minilm-l6-v2"
        assert m.dimensions == 384


def test_minilm_embed_single_string():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        fake = _mock_model(encode_return=[[0.1] * 384])
        mock_cls.return_value = fake
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="sentence-transformers/all-MiniLM-L6-v2",
                          local_dir="/fake")
        result = m.embed("hello")
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 384
        assert result.dimensions == 384
        assert result.model_id == "all-minilm-l6-v2"
        # Tokenize was called with ["hello"]
        call = fake.tokenize.call_args.args[0]
        assert call == ["hello"]


def test_minilm_embed_list_of_strings():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        fake = _mock_model(encode_return=np.zeros((3, 384), dtype=np.float32))
        mock_cls.return_value = fake
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="sentence-transformers/all-MiniLM-L6-v2",
                          local_dir="/fake")
        result = m.embed(["a", "b", "c"])
        assert len(result.embeddings) == 3
        call = fake.tokenize.call_args.args[0]
        assert call == ["a", "b", "c"]


def test_minilm_counts_tokens_from_attention_mask():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        fake = _mock_model(encode_return=np.zeros((2, 384), dtype=np.float32))
        mock_cls.return_value = fake
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="...", local_dir="/fake")
        result = m.embed(["a", "b"])
        # _mock_model returns attention_mask of shape (2, 5) of all 1s
        assert result.prompt_tokens == 10


def test_minilm_dimensions_truncation_l2_normalizes():
    """Truncating + renormalizing produces unit vectors at the smaller dim."""
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        # Encode returns a known vector so we can check the math
        raw = np.array([[3.0, 4.0, 0.0, 0.0] + [0.0] * 380], dtype=np.float32)
        fake = _mock_model(encode_return=raw)
        mock_cls.return_value = fake
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="...", local_dir="/fake")
        result = m.embed("hi", dimensions=2)
        assert result.dimensions == 2
        truncated = result.embeddings[0]
        assert len(truncated) == 2
        # L2 norm of (3, 4) / sqrt(25) = (0.6, 0.8)
        assert abs(truncated[0] - 0.6) < 1e-5
        assert abs(truncated[1] - 0.8) < 1e-5


def test_minilm_full_dimensions_skips_truncation():
    """If dimensions >= model dim, no truncation applied."""
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        raw = np.ones((1, 384), dtype=np.float32)
        fake = _mock_model(encode_return=raw)
        mock_cls.return_value = fake
        from muse.embeddings.backends.minilm import MiniLMBackend
        m = MiniLMBackend(hf_repo="...", local_dir="/fake")
        result = m.embed("hi", dimensions=384)
        assert result.dimensions == 384


def test_minilm_prefers_local_dir_over_hf_repo():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.minilm import MiniLMBackend
        MiniLMBackend(hf_repo="sentence-transformers/all-MiniLM-L6-v2",
                      local_dir="/real/local/path")
        # First positional arg should be local_dir
        args = mock_cls.call_args.args
        assert args[0] == "/real/local/path"


def test_minilm_falls_back_to_hf_repo_when_local_dir_none():
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.minilm import MiniLMBackend
        MiniLMBackend(hf_repo="sentence-transformers/all-MiniLM-L6-v2",
                      local_dir=None)
        args = mock_cls.call_args.args
        assert args[0] == "sentence-transformers/all-MiniLM-L6-v2"


def test_minilm_accepts_unknown_kwargs():
    """Future catalog kwargs (**_) should be absorbed."""
    with patch("muse.embeddings.backends.minilm.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_model()
        from muse.embeddings.backends.minilm import MiniLMBackend
        # Should not TypeError
        MiniLMBackend(
            hf_repo="x", local_dir="/fake",
            device="cpu", future_param="ignored",
        )


def test_minilm_module_imports_without_sentence_transformers_installed():
    """Module-level try/except leaves SentenceTransformer=None when missing."""
    # Rather than uninstall sentence-transformers in the test env,
    # we verify the module has a SentenceTransformer symbol. When not
    # installed, it's None and instantiation raises; otherwise it's
    # the real class. Either way, importing the module succeeds.
    import muse.embeddings.backends.minilm as mod
    assert hasattr(mod, "SentenceTransformer")
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/embeddings/test_minilm.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/muse/embeddings/backends/minilm.py`**

```python
"""MiniLM embeddings backend via sentence-transformers.

Uses sentence-transformers/all-MiniLM-L6-v2 — 22MB, 384 dims, CPU-fast.
Tests mock the sentence-transformers library entirely; real weights
only download when `muse pull all-minilm-l6-v2` is invoked.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from muse.embeddings.protocol import EmbeddingResult

logger = logging.getLogger(__name__)

# Heavy imports are deferred so `muse --help` and the CLI work without
# sentence-transformers installed. `muse pull all-minilm-l6-v2` installs it.
try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    SentenceTransformer = None  # type: ignore


class MiniLMBackend:
    model_id = "all-minilm-l6-v2"
    dimensions = 384

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        **_: Any,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed; "
                "run `muse pull all-minilm-l6-v2`"
            )
        self._device = _select_device(device)
        src = local_dir or hf_repo
        logger.info("loading MiniLM from %s (device=%s)", src, self._device)
        self._model = SentenceTransformer(src, device=self._device)

    def embed(
        self,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        **_: Any,
    ) -> EmbeddingResult:
        texts = [input] if isinstance(input, str) else list(input)

        # sentence-transformers encode returns np.ndarray (N, dim)
        raw = self._model.encode(texts, convert_to_numpy=True)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Count tokens via the tokenizer's attention_mask (matches OpenAI
        # spirit of "tokens used" without needing their exact tokenizer)
        prompt_tokens = _count_tokens(self._model, texts)

        # Dimensions truncation. MiniLM isn't matryoshka-trained, so
        # naively truncating + re-normalizing is lossy but OpenAI-SDK-
        # compatible. Skip work if dimensions is None or >= full size.
        if dimensions is not None and dimensions < arr.shape[1]:
            arr = arr[:, :dimensions]
            # L2 re-normalize so the vectors remain unit-length
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms
            out_dim = dimensions
        else:
            out_dim = arr.shape[1]

        return EmbeddingResult(
            embeddings=arr.tolist(),
            dimensions=out_dim,
            model_id=self.model_id,
            prompt_tokens=prompt_tokens,
            metadata={"source": "sentence-transformers"},
        )


def _count_tokens(model: Any, texts: list[str]) -> int:
    """Sum attention_mask entries to get actual (non-padding) token count."""
    try:
        tok = model.tokenize(texts)
        attn = tok.get("attention_mask")
        if attn is None:
            return 0
        if hasattr(attn, "sum"):
            total = attn.sum()
            if hasattr(total, "item"):
                return int(total.item())
            return int(total)
        # list-of-lists fallback
        return sum(sum(row) for row in attn)
    except Exception as e:
        logger.debug("token counting failed: %s", e)
        return 0


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/embeddings/test_minilm.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/muse/embeddings/backends/minilm.py tests/embeddings/test_minilm.py
git commit -m "feat(embeddings): add MiniLMBackend via sentence-transformers

First embeddings backend. 384 dims, 22MB, CPU-fast. Lazy-imports
torch + sentence-transformers so the module loads without them.
Tests fully mock the library. Naive truncation + L2 renormalization
for the `dimensions` parameter (MiniLM isn't matryoshka-trained;
quality degrades, but matches OpenAI SDK contract)."
```

---

## Part E — Client

### Task E1: `EmbeddingsClient`

**Files:**
- Create: `src/muse/embeddings/client.py`
- Create: `tests/embeddings/test_client.py`

- [ ] **Step 1: Write failing tests**

File: `tests/embeddings/test_client.py`

```python
"""Tests for EmbeddingsClient HTTP client."""
import base64
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from muse.embeddings.client import EmbeddingsClient


def _make_response(embeddings, encoding_format="float", model="fake-embed", tokens=3):
    """Build a fake OpenAI-shape response body."""
    data = []
    for i, vec in enumerate(embeddings):
        if encoding_format == "base64":
            arr = np.asarray(vec, dtype="<f4")
            embedding_field = base64.b64encode(arr.tobytes()).decode()
        else:
            embedding_field = vec
        data.append({"object": "embedding", "embedding": embedding_field, "index": i})
    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


def test_default_server_url():
    c = EmbeddingsClient()
    assert c.server_url == "http://localhost:8000"


def test_trailing_slash_stripped():
    c = EmbeddingsClient(server_url="http://lan:8000/")
    assert c.server_url == "http://lan:8000"


def test_muse_server_env_fallback(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://custom:9999")
    c = EmbeddingsClient()
    assert c.server_url == "http://custom:9999"


def test_embed_single_string_returns_list_of_vectors():
    fake_body = _make_response([[0.1, 0.2, 0.3]])
    with patch("muse.embeddings.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: fake_body)
        c = EmbeddingsClient()
        vectors = c.embed("hello")
        assert len(vectors) == 1
        assert vectors[0] == [0.1, 0.2, 0.3]
        body = mock_post.call_args.kwargs["json"]
        assert body["input"] == "hello"
        assert body["encoding_format"] == "float"


def test_embed_list_input_returns_list_of_vectors():
    fake_body = _make_response([[0.1], [0.2], [0.3]])
    with patch("muse.embeddings.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: fake_body)
        c = EmbeddingsClient()
        vectors = c.embed(["a", "b", "c"])
        assert len(vectors) == 3
        body = mock_post.call_args.kwargs["json"]
        assert body["input"] == ["a", "b", "c"]


def test_embed_base64_decodes_to_floats():
    original = [1.0, 2.0, 3.0, 4.0]
    fake_body = _make_response([original], encoding_format="base64")
    with patch("muse.embeddings.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: fake_body)
        c = EmbeddingsClient()
        # Client requests base64 from the server, decodes to floats for caller
        vectors = c.embed("hi", encoding_format="base64")
        assert len(vectors) == 1
        for a, b in zip(vectors[0], original):
            assert abs(a - b) < 1e-6


def test_embed_sends_optional_fields_when_provided():
    fake_body = _make_response([[0.1]])
    with patch("muse.embeddings.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: fake_body)
        c = EmbeddingsClient()
        c.embed("hi", model="all-minilm-l6-v2", dimensions=128)
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "all-minilm-l6-v2"
        assert body["dimensions"] == 128


def test_embed_omits_none_optional_fields():
    fake_body = _make_response([[0.1]])
    with patch("muse.embeddings.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: fake_body)
        c = EmbeddingsClient()
        c.embed("hi")
        body = mock_post.call_args.kwargs["json"]
        for field in ("model", "dimensions"):
            assert field not in body


def test_embed_raises_on_http_error():
    with patch("muse.embeddings.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=500, text="boom")
        c = EmbeddingsClient()
        with pytest.raises(RuntimeError, match="500"):
            c.embed("x")


def test_embed_rejects_invalid_encoding_format():
    c = EmbeddingsClient()
    with pytest.raises(ValueError, match="encoding_format"):
        c.embed("hi", encoding_format="bogus")
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/embeddings/test_client.py -v
```

- [ ] **Step 3: Implement `src/muse/embeddings/client.py`**

```python
"""HTTP client for /v1/embeddings.

Mirrors the shape of SpeechClient and GenerationsClient: server_url
constructor param with MUSE_SERVER env fallback, synchronous POST,
returns the essential payload (list[list[float]] of vectors).

Consumers who want the full OpenAI-shape response (with usage, model,
etc.) can POST directly or use the openai-python SDK against muse.
"""
from __future__ import annotations

import os
from typing import Any, Union

import requests

from muse.embeddings.codec import base64_to_embedding


class EmbeddingsClient:
    """Thin HTTP client against muse's /v1/embeddings endpoint."""

    def __init__(self, server_url: str | None = None, timeout: float = 60.0) -> None:
        server_url = server_url or os.environ.get(
            "MUSE_SERVER", "http://localhost:8000",
        )
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def embed(
        self,
        input: Union[str, list[str]],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: str = "float",
    ) -> list[list[float]]:
        """Return embeddings as list[list[float]] regardless of wire format.

        If encoding_format='base64' is requested (to save bandwidth on
        the wire), the server returns base64 strings which the client
        decodes back to float lists before returning to the caller.
        """
        if encoding_format not in ("float", "base64"):
            raise ValueError(
                f"encoding_format must be 'float' or 'base64', got {encoding_format!r}"
            )

        body: dict[str, Any] = {
            "input": input,
            "encoding_format": encoding_format,
        }
        if model is not None:
            body["model"] = model
        if dimensions is not None:
            body["dimensions"] = dimensions

        r = requests.post(
            f"{self.server_url}/v1/embeddings",
            json=body, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")

        payload = r.json()
        entries = payload["data"]
        if encoding_format == "base64":
            return [base64_to_embedding(e["embedding"]) for e in entries]
        return [e["embedding"] for e in entries]
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/embeddings/test_client.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/muse/embeddings/client.py tests/embeddings/test_client.py
git commit -m "feat(embeddings): add EmbeddingsClient HTTP client

Mirrors SpeechClient / GenerationsClient shape. server_url param
with MUSE_SERVER env fallback. Accepts str or list[str] input,
optional model / dimensions / encoding_format. base64 wire format
is decoded client-side so callers always receive list[list[float]]."
```

---

## Part F — Integration

### Task F1: pyproject extras, catalog entry, subpackage exports

**Files:**
- Modify: `pyproject.toml` (add `embeddings` extras)
- Modify: `src/muse/core/catalog.py` (add KNOWN_MODELS entry)
- Modify: `src/muse/embeddings/__init__.py` (export public API)
- Modify: `tests/core/test_catalog.py` (add test for the new catalog entry)

- [ ] **Step 1: Add `embeddings` extras to `pyproject.toml`**

Find `[project.optional-dependencies]`. Add after the `images` block:

```toml
# Embeddings modality
embeddings = [
  "torch>=2.1.0",
  "sentence-transformers>=2.2.0",
]
```

- [ ] **Step 2: Add failing test for the new catalog entry**

Open `tests/core/test_catalog.py`. Find `test_known_models_seeded_with_required_entries` (or whatever the "seeded entries" test is named — check with `grep -n "seeded" tests/core/test_catalog.py`).

Update that test to also assert `all-minilm-l6-v2` is present:

```python
def test_known_models_seeded_with_required_entries():
    assert "soprano-80m" in KNOWN_MODELS
    assert "kokoro-82m" in KNOWN_MODELS
    assert "bark-small" in KNOWN_MODELS
    assert "sd-turbo" in KNOWN_MODELS
    assert "all-minilm-l6-v2" in KNOWN_MODELS
```

Also find `test_known_models_entries_have_valid_modality` — update the valid modalities set:

```python
def test_known_models_entries_have_valid_modality():
    valid = {"audio.speech", "images.generations", "embeddings"}
    for model_id, entry in KNOWN_MODELS.items():
        assert entry.modality in valid, \
            f"model {model_id} has invalid modality {entry.modality!r}"
```

- [ ] **Step 3: Run — these tests fail**

```bash
cd /home/spinoza/github/repos/muse
pytest tests/core/test_catalog.py::test_known_models_seeded_with_required_entries \
       tests/core/test_catalog.py::test_known_models_entries_have_valid_modality -v
```

Expected: FAIL — the entry doesn't exist yet.

- [ ] **Step 4: Add the catalog entry in `src/muse/core/catalog.py`**

Find the `KNOWN_MODELS` dict. Add this entry after the `sd-turbo` entry:

```python
    "all-minilm-l6-v2": CatalogEntry(
        model_id="all-minilm-l6-v2",
        modality="embeddings",
        backend_path="muse.embeddings.backends.minilm:MiniLMBackend",
        hf_repo="sentence-transformers/all-MiniLM-L6-v2",
        description="MiniLM sentence embeddings: 384 dims, 22MB, CPU-friendly",
        pip_extras=("torch>=2.1.0", "sentence-transformers>=2.2.0"),
        extra={"dimensions": 384},
    ),
```

- [ ] **Step 5: Run — pass**

```bash
pytest tests/core/test_catalog.py -v 2>&1 | tail -10
```

Expected: all catalog tests pass.

- [ ] **Step 6: Export public API from `src/muse/embeddings/__init__.py`**

Replace the empty `__init__.py` with:

```python
"""Muse embeddings modality: text-to-vector.

Backends implementing the EmbeddingsModel protocol return
EmbeddingResult dataclasses. The /v1/embeddings router serializes
these to OpenAI-compatible JSON (with optional base64 encoding).
"""
from muse.embeddings.client import EmbeddingsClient
from muse.embeddings.protocol import EmbeddingModel as _ReexportCheck  # noqa
from muse.embeddings.protocol import EmbeddingsModel, EmbeddingResult

__all__ = ["EmbeddingsClient", "EmbeddingsModel", "EmbeddingResult"]
```

WAIT — the `_ReexportCheck` line references `EmbeddingModel` (singular) which doesn't exist. Remove it. The correct content is:

```python
"""Muse embeddings modality: text-to-vector.

Backends implementing the EmbeddingsModel protocol return
EmbeddingResult dataclasses. The /v1/embeddings router serializes
these to OpenAI-compatible JSON (with optional base64 encoding).
"""
from muse.embeddings.client import EmbeddingsClient
from muse.embeddings.protocol import EmbeddingsModel, EmbeddingResult

__all__ = ["EmbeddingsClient", "EmbeddingsModel", "EmbeddingResult"]
```

- [ ] **Step 7: Verify imports work**

```bash
python -c "from muse.embeddings import EmbeddingsClient, EmbeddingsModel, EmbeddingResult; print('ok')"
```

Expected: `ok`.

- [ ] **Step 8: Regression — full suite**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: all prior passes + the updated catalog tests + new embeddings tests. No regressions.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml \
        src/muse/core/catalog.py \
        src/muse/embeddings/__init__.py \
        tests/core/test_catalog.py
git commit -m "feat(embeddings): wire into pyproject + catalog + subpackage exports

pyproject gains [embeddings] extras (torch + sentence-transformers).
KNOWN_MODELS gains all-minilm-l6-v2 entry pointing at MiniLMBackend.
embeddings.__init__ exports EmbeddingsClient / EmbeddingsModel /
EmbeddingResult for ergonomic import."
```

---

## Part G — Worker wiring

### Task G1: Mount the embeddings router in `worker.py`

**Files:**
- Modify: `src/muse/cli_impl/worker.py`
- Modify: `tests/cli_impl/test_worker.py` (optional assertion tighten)

The worker always mounts all modality routers unconditionally (even if the registry is empty for a modality) so 404s go through the OpenAI envelope. After this task, embeddings joins audio.speech and images.generations.

- [ ] **Step 1: Read current worker.py to see the mount pattern**

```bash
grep -n "build_router\|routers\[" src/muse/cli_impl/worker.py
```

You should see something like:

```python
from muse.audio.speech.routes import build_router as build_audio
from muse.images.generations.routes import build_router as build_images

routers["audio.speech"] = build_audio(registry)
routers["images.generations"] = build_images(registry)
```

- [ ] **Step 2: Add the embeddings mount**

Edit `src/muse/cli_impl/worker.py`. Add the import alongside the other two:

```python
from muse.embeddings.routes import build_router as build_embeddings
```

And the mount alongside the other two:

```python
routers["embeddings"] = build_embeddings(registry)
```

Final worker block should have all three:

```python
from muse.audio.speech.routes import build_router as build_audio
from muse.embeddings.routes import build_router as build_embeddings
from muse.images.generations.routes import build_router as build_images

routers["audio.speech"] = build_audio(registry)
routers["embeddings"] = build_embeddings(registry)
routers["images.generations"] = build_images(registry)
```

Sort imports alphabetically if the existing file is sorted that way; otherwise match the existing order.

- [ ] **Step 3: Run existing worker tests to confirm no regression**

```bash
pytest tests/cli_impl/test_worker.py -v
```

Expected: 4 passed (existing tests; they pass FakeTTS so the new mount doesn't break them).

- [ ] **Step 4: Add an integration assertion to `tests/cli_impl/test_worker.py`**

Append this test to the end of `tests/cli_impl/test_worker.py`:

```python
@patch("muse.cli_impl.worker.uvicorn")
def test_worker_mounts_all_three_modality_routers(mock_uvicorn):
    """Regression guard: all modality routers mounted regardless of registry content.

    Empty-registry requests must get the OpenAI 404 envelope, not FastAPI's
    default {"detail": "Not Found"}. That requires the router to exist.
    """
    from unittest.mock import ANY
    run_worker(host="127.0.0.1", port=9999, models=[], device="cpu")

    # create_app is called with the `routers` dict — we verify all three
    # expected keys are present by intercepting the create_app call.
    # Rather than patching create_app (would be noisy), assert that
    # uvicorn.run received an app whose routes include all three prefixes.
    mock_uvicorn.run.assert_called_once()
    app = mock_uvicorn.run.call_args.args[0]
    route_paths = {getattr(r, "path", "") for r in app.routes}
    # Modality endpoints are POST routes; prefixes differ per router.
    # We check by looking for these literal paths being somewhere in the
    # route tree (FastAPI compiles routes by their full path).
    paths_str = "\n".join(route_paths)
    assert "/v1/audio/speech" in paths_str
    assert "/v1/images/generations" in paths_str
    assert "/v1/embeddings" in paths_str
```

- [ ] **Step 5: Run tests — pass**

```bash
pytest tests/cli_impl/test_worker.py -v
```

Expected: 5 passed (4 + 1 new).

- [ ] **Step 6: E2E smoke still works**

```bash
pytest tests/cli_impl/test_e2e_supervisor.py -v --timeout=60
```

Expected: 1 passed. The e2e test will now spawn a real subprocess with the embeddings router mounted too. Gateway + worker + all three modalities = full integration check.

- [ ] **Step 7: Commit**

```bash
git add src/muse/cli_impl/worker.py tests/cli_impl/test_worker.py
git commit -m "feat(worker): mount /v1/embeddings router alongside audio + images

Three-line change (import + one dict assignment). Regression test
asserts all three modality prefixes exist in the app's route tree
so empty-registry requests always get the OpenAI envelope 404."
```

---

## Part H — Documentation

### Task H1: Update `README.md` + `CLAUDE.md`

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

Quick docs refresh. The soul-voice hook in the repo rejects em-dashes (U+2014); use colons, commas, periods, or parentheses instead.

- [ ] **Step 1: Update `README.md`**

In the "Architecture" bullet list, add an embeddings line. Find the block:

```markdown
## Architecture

- `muse.core`: ...
- `muse.cli_impl`: ...
- `muse.audio.speech`: text-to-speech (Soprano, Kokoro, Bark backends)
- `muse.images.generations`: text-to-image (SD-Turbo backend)
```

Add after the last bullet:

```markdown
- `muse.embeddings`: text-to-vector (MiniLM backend; OpenAI-compatible /v1/embeddings)
```

In the "HTTP endpoints" table, add a row:

```markdown
| `POST /v1/embeddings` | text embeddings (OpenAI-compatible) |
```

In the "Quick start" section, add an example right after the image example:

```bash
# Embeddings (single string or list)
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"hello world","model":"all-minilm-l6-v2"}'
```

- [ ] **Step 2: Update `CLAUDE.md`**

In the "Project overview" section's modality list, add:

```markdown
- **embeddings**: text-to-vector via `/v1/embeddings` (MiniLM; sentence-transformers)
```

In the "Development commands" example (where Python client usage appears), add an embeddings line:

```python
from muse.embeddings import EmbeddingsClient
EmbeddingsClient().embed(["alpha", "beta"])   # -> list[list[float]]
```

- [ ] **Step 3: Verify no em-dashes introduced**

```bash
grep -n "—" README.md CLAUDE.md
```

Expected: no output. (If the hook blocks the edit, replace the em-dash with a comma or colon and retry.)

- [ ] **Step 4: Regression — tests unaffected by docs**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: unchanged count, all passing.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add embeddings modality to README + CLAUDE.md

Architecture list, HTTP endpoints table, Quick start curl example,
and CLAUDE.md modality list + client usage snippet."
```

---

## Part I — Verification + merge

### Task I1: Final sweep

**Files:** none (verification only)

- [ ] **Step 1: Fresh install**

```bash
cd /home/spinoza/github/repos/muse
pip install -e ".[dev,server]"
```

- [ ] **Step 2: Full test suite excluding slow**

```bash
pytest tests/ -q -m "not slow" --cov=muse --cov-report=term-missing 2>&1 | tail -30
```

Expected: all unit tests pass (previous count + ~43 new from embeddings: 5 protocol + 6 codec + 12 routes + 10 minilm + 10 client).

- [ ] **Step 3: Run the e2e slow test**

```bash
pytest tests/cli_impl/test_e2e_supervisor.py -v --timeout=60
```

Expected: 1 passed.

- [ ] **Step 4: Import smokes**

```bash
python -c "from muse.embeddings import EmbeddingsClient, EmbeddingsModel, EmbeddingResult"
python -c "from muse.embeddings.codec import embedding_to_base64, base64_to_embedding"
python -c "from muse.embeddings.routes import build_router"
python -c "import muse.embeddings.backends.minilm"
```

All should succeed silently.

- [ ] **Step 5: CLI smoke**

```bash
muse models list
muse models info all-minilm-l6-v2
```

`muse models list` should include the new entry with modality `embeddings`. `muse models info all-minilm-l6-v2` should print the catalog details.

- [ ] **Step 6: Gateway smoke**

Start a quick server and post to `/v1/embeddings`. Since no model is pulled, expect a 404 OpenAI envelope (not a 500 and not a FastAPI default):

```bash
muse serve --host 127.0.0.1 --port 8765 &
SERVER_PID=$!
sleep 3

curl -s -X POST http://127.0.0.1:8765/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input":"hello","model":"all-minilm-l6-v2"}' \
     -w "\n--- status: %{http_code}\n"

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
```

Expected output: status 404 with `{"error":{"code":"model_not_found",...}}`. This proves the gateway + supervisor + embeddings router are all plumbed correctly.

- [ ] **Step 7: Fix anything broken**

If any step fails, diagnose and fix. Apply the minimal change that addresses the root cause. Commit any fixes:

```bash
git status
# If changes:
git add -A
git commit -m "fix: address issues found in final verification"
```

- [ ] **Step 8: Show commit history**

```bash
git log --oneline main..HEAD
```

Should show ~9 commits on the feature branch.

- [ ] **Step 9: Merge back to main with `--no-ff`**

(This step is typically handled by `superpowers:finishing-a-development-branch` at the controller level; included here for completeness.)

```bash
git checkout main
git merge --no-ff <feature-branch> -m "feat: add embeddings modality (OpenAI-compatible /v1/embeddings)

Third modality alongside audio.speech and images.generations. MiniLM
backend (sentence-transformers/all-MiniLM-L6-v2, 384 dims, 22MB).

Validates the architectural claim that new modalities are additive:
- Gateway unchanged (routes by body.model, not URL path)
- Supervisor unchanged
- Catalog gained one entry
- Worker gained one import + one mount (3 lines)
- pyproject gained one extras group
- Everything else in muse untouched

43 new tests; full suite passing plus e2e smoke.

See docs/plans/2026-04-13-embeddings-modality.md for the full plan."

git worktree remove <worktree-path>
git branch -d <feature-branch>
```

---

## Scope notes (deferred)

Explicitly NOT in this plan:

- **Additional embeddings backends.** MiniLM is the first; future backends (BGE, E5, stella, Jina) are drop-ins after this lands. Each is a new file in `muse/embeddings/backends/` + a new `KNOWN_MODELS` entry. Zero changes elsewhere.
- **Matryoshka-native dimensions truncation.** MiniLM's naive truncation works but degrades quality. A future `text-embedding-3-small`-family backend would implement the matryoshka scaling properly.
- **Batching and async backends.** `embed(input)` is synchronous per request. Multi-request batching (across concurrent clients) would need a request queue + micro-batcher in the backend. Out of scope.
- **Reranking endpoint.** Some inference servers expose `/v1/rerank` alongside embeddings. Not an OpenAI endpoint; defer unless needed.
- **Similarity computation.** Muse returns raw vectors; cosine similarity / top-k search is the caller's job. (A future "vector store" modality could be built on top, but that's a different product.)

## Self-review

**Spec coverage:**
- Protocol + Result dataclass → Task A1 ✅
- Base64 codec (OpenAI wire format) → Task B1 ✅
- /v1/embeddings FastAPI router with full OpenAI shape → Task C1 ✅
- MiniLM backend via sentence-transformers → Task D1 ✅
- HTTP client mirroring other modalities' clients → Task E1 ✅
- pyproject extras + catalog entry + subpackage exports → Task F1 ✅
- Worker integration (router mount) → Task G1 ✅
- Docs → Task H1 ✅
- Verification + merge → Task I1 ✅

**Placeholder scan:** No TBDs, "implement later", or "add error handling" placeholders. Every code block is complete. Every test specifies its expected count.

**Type consistency:**
- `EmbeddingResult` fields (`embeddings`, `dimensions`, `model_id`, `prompt_tokens`, `metadata`): consistent across protocol, backend, router.
- `EmbeddingsModel` protocol methods (`model_id`, `dimensions`, `embed`): consistent.
- `EmbeddingsRequest` Pydantic fields (`input`, `model`, `encoding_format`, `dimensions`, `user`): consistent in router + client.
- `EmbeddingsClient.__init__(server_url=...)`: matches `SpeechClient.__init__` shape (post-review-fix from the muse-restructure work).
- `embedding_to_base64` / `base64_to_embedding`: names and signatures match across codec + router + client.
- Catalog entry `model_id="all-minilm-l6-v2"` matches `MiniLMBackend.model_id`. Backend path `muse.embeddings.backends.minilm:MiniLMBackend` matches the file.

Plan complete. 9 tasks, ~43 new tests, no new external infrastructure.
