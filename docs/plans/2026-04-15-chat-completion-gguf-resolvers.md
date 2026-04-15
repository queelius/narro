# chat/completion + HF resolvers + GGUF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `chat/completion` modality backed by `llama-cpp-python` for in-process GGUF serving, and introduce a `muse.core.resolvers` abstraction so users can `muse pull hf://org/repo@variant` without writing a script. Initial resolvers: HF for GGUF (`chat/completion`) and HF for sentence-transformers (`embedding/text`). Add `muse search` to query HuggingFace for pullable models with filters (size, sort, modality).

**Architecture:** Three new concepts layered cleanly on top of the existing discovery/catalog/registry system:

1. **Resolvers** (`muse.core.resolvers`) translate URIs like `hf://Qwen/Qwen3-8B-GGUF@q4_k_m` into a `ResolvedModel` (synthesized MANIFEST + downloader function + runtime class path). Resolvers also expose `search()` for discovery. Plug-in-shaped so future `ollama://`, `civitai://`, and direct-URL schemes slot in without breaking existing callers.

2. **`chat/completion` modality** follows the same MIME-tag-and-discovery pattern as `audio/speech` / `embedding/text` / `image/generation`. Protocol + routes + codec + client. Tool calling, streaming, structured output are all pass-through to `llama-cpp-python`'s OpenAI-shape API.

3. **`LlamaCppModel` runtime** in `muse.modalities.chat_completion.runtimes.llama_cpp`. Wraps `llama_cpp.Llama.create_chat_completion`, translates its OpenAI-shape output to our internal `ChatResult` / `ChatChunk` types, keeps the runtime-class abstraction in place so future `TransformersModel` / `VllmModel` can implement the same protocol.

Catalog persists resolver-synthesized manifests inside `~/.muse/catalog.json` so they survive restarts. `known_models()` merges discovery (bundled scripts) with persisted resolver entries, bundled wins on model_id collision.

**Tech Stack:** Python 3.10+, stdlib only for resolver core, `huggingface_hub` (already a dep) for HF search + file listing + snapshot_download, `llama-cpp-python` installed into per-model venvs by `muse pull`, existing FastAPI + httpx + SSE-Starlette.

---

## File Structure (final)

```
src/muse/
├── core/
│   ├── resolvers.py                         NEW  Resolver ABC, registry, dispatch
│   └── resolvers_hf.py                      NEW  HFResolver implementation
│
├── modalities/
│   └── chat_completion/                     NEW modality
│       ├── __init__.py                      MODALITY = "chat/completion"; exports
│       ├── protocol.py                      ChatMessage/Choice/Result/Chunk + ChatModel Protocol
│       ├── routes.py                        /v1/chat/completions router (stream + non-stream)
│       ├── codec.py                         ChatChunk -> OpenAI SSE wire format
│       ├── client.py                        ChatClient HTTP client
│       └── runtimes/
│           ├── __init__.py
│           └── llama_cpp.py                 LlamaCppModel runtime class
│
├── modalities/embedding_text/runtimes/
│   ├── __init__.py                          NEW
│   └── sentence_transformers.py             NEW  SentenceTransformerModel runtime
│
├── cli_impl/
│   └── search.py                            NEW  run_search for `muse search`
│
├── cli.py                                   MODIFIED  `search` subcommand; `pull` accepts URIs
├── core/catalog.py                          MODIFIED  pull() dispatches by URI; persist manifest
└── models/                                  unchanged (bundled scripts)

tests/
├── core/
│   ├── test_resolvers.py                    NEW  registry + URI parsing
│   └── test_resolvers_hf.py                 NEW  HFResolver sniff / search / resolve
│
├── modalities/
│   └── chat_completion/
│       ├── test_protocol.py                 NEW
│       ├── test_routes.py                   NEW
│       ├── test_codec.py                    NEW
│       ├── test_client.py                   NEW
│       └── runtimes/
│           └── test_llama_cpp.py            NEW
│
├── modalities/embedding_text/runtimes/
│   └── test_sentence_transformers.py        NEW
│
├── cli_impl/
│   └── test_search.py                       NEW
│
├── core/test_catalog.py                     MODIFIED  resolver-pulled entries + merge behavior
└── test_cli.py                              MODIFIED  `search`, `pull hf://...` coverage

docs/
├── MODEL_SCRIPTS.md                         MODIFIED  resolver section added
├── RESOLVERS.md                             NEW  URI scheme reference + how to add a resolver
└── CHAT_COMPLETION.md                       NEW  OpenAI-shape endpoints, tool support, limitations
```

---

## Key design decisions (locked in)

1. **Own-it, not federate.** `LlamaCppModel` runs in-process inside a per-model venv with `llama-cpp-python` installed. No Ollama proxy. Matches the existing worker-per-venv architecture.

2. **OpenAI chat-completions is the wire contract.** `/v1/chat/completions` accepts the OpenAI request shape verbatim (messages, tools, tool_choice, response_format, stream, temperature, max_tokens, stop, seed, logprobs, top_logprobs, n=1 only). Returns OpenAI shape. Tool calling, structured output, streaming deltas all pass through. Anthropic-dialect endpoint is deferred.

3. **`llama-cpp-python` is OpenAI-shape natively.** `Llama.create_chat_completion(messages, tools, ..., stream=True)` returns OpenAI-compatible dicts / chunks. `LlamaCppModel.chat()` translates to our internal `ChatResult`, `chat_stream()` translates to `ChatChunk`. The translation is trivial (name-mapping only) but keeps the protocol backend-agnostic.

4. **No multimodal chat input in v1.** Vision chat (Llava, Qwen2-VL via llama.cpp mmproj) and reasoning-content blocks (`<think>...</think>`) are deferred. Text-only chat.

5. **Resolver URIs have the form `<scheme>://<source-specific-id>[@<variant>]`.** `hf://Qwen/Qwen3-8B-GGUF@q4_k_m` names a GGUF variant explicitly; `hf://sentence-transformers/all-MiniLM-L6-v2` has no variant because sentence-transformers repos are monolithic. No wildcard / default variant selection: pulling a multi-variant repo without `@variant` raises and lists the options.

6. **Resolver sniffs repo shape to pick a modality + runtime.** `hf://` resolver inspects `repo_info().siblings`:
   - Any `.gguf` file → `chat/completion`, runtime `LlamaCppModel`
   - `sentence_transformers_config.json` present OR `sentence-transformers` tag → `embedding/text`, runtime `SentenceTransformerModel`
   - Otherwise raise with the file list and tags so the user can see why.

7. **Catalog persists resolver-synthesized manifests.** `~/.muse/catalog.json` entry for a resolver-pulled model carries `source` (the URI) and `manifest` (the synthesized dict). `known_models()` merges these with discovery results. Bundled scripts win on collision.

8. **Existing scripts stay through the transition.** `minilm.py` / `qwen3_embedding.py` keep working. Phase I migrates them to resolver-addressable (pull via `hf://` → script becomes redundant → delete) but only after the sentence-transformers runtime + resolver is verified end-to-end against a real pull.

9. **`muse search` v1 supports `chat/completion` + `embedding/text` only.** Architecture generalizes, but HF's `text-to-image` / `text-to-speech` tags return too much junk for the resolver to filter cleanly. Add those modalities to search later.

10. **tool-calling quality is the model's responsibility, not muse's.** Resolver sniffs `tokenizer_config.json` for `{% if tools %}` / `{{ tools }}` in the chat template and writes `capabilities.supports_tools: true/false`. Surfaced via `/v1/models`. We never reject tool requests; model-dependent output quality is up to the user to pick a tool-trained model.

---

## Part A: Resolver infrastructure

### Task A1: Define resolver primitives in `muse.core.resolvers`

**Files:**
- Create: `src/muse/core/resolvers.py`
- Create: `tests/core/test_resolvers.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_resolvers.py
"""Tests for the muse.core.resolvers abstraction.

The resolver module only defines the ABCs, dataclasses, and registry
dispatch. Concrete resolvers (hf://, etc.) live in separate modules
and register themselves at import time.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from muse.core.resolvers import (
    Resolver,
    ResolvedModel,
    SearchResult,
    ResolverError,
    register_resolver,
    get_resolver,
    resolve,
    search,
    _reset_registry_for_tests,
    parse_uri,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


class _FakeResolver(Resolver):
    scheme = "fake"

    def resolve(self, uri):
        return ResolvedModel(
            manifest={"model_id": "fake-model", "modality": "fake/type", "hf_repo": "x/y"},
            backend_path="muse.fake:FakeModel",
            download=lambda cache: cache / "fake",
        )

    def search(self, query, **filters):
        return [SearchResult(
            uri="fake://a/b", model_id="a-b", modality="fake/type",
            size_gb=0.5, downloads=10, license=None, description=None,
        )]


def test_register_and_get_resolver():
    r = _FakeResolver()
    register_resolver(r)
    assert get_resolver("fake://anything") is r


def test_get_resolver_raises_for_unknown_scheme():
    with pytest.raises(ResolverError, match="no resolver for scheme"):
        get_resolver("unknown://whatever")


def test_get_resolver_raises_for_non_uri():
    with pytest.raises(ResolverError, match="not a resolver URI"):
        get_resolver("just-a-bare-id")


def test_resolve_dispatches_by_scheme():
    register_resolver(_FakeResolver())
    rm = resolve("fake://a/b")
    assert isinstance(rm, ResolvedModel)
    assert rm.manifest["model_id"] == "fake-model"


def test_search_dispatches_to_named_backend():
    register_resolver(_FakeResolver())
    results = list(search("anything", backend="fake"))
    assert len(results) == 1
    assert results[0].uri == "fake://a/b"


def test_search_defaults_to_hf_when_backend_omitted_but_only_hf_registered():
    """If exactly one resolver is registered, search defaults to it."""
    register_resolver(_FakeResolver())
    results = list(search("q"))  # no backend kwarg
    assert len(results) == 1


def test_parse_uri_splits_scheme_ref_and_variant():
    scheme, ref, variant = parse_uri("hf://Qwen/Qwen3-8B-GGUF@q4_k_m")
    assert scheme == "hf"
    assert ref == "Qwen/Qwen3-8B-GGUF"
    assert variant == "q4_k_m"


def test_parse_uri_handles_missing_variant():
    scheme, ref, variant = parse_uri("hf://org/repo")
    assert scheme == "hf"
    assert ref == "org/repo"
    assert variant is None


def test_parse_uri_rejects_bad_scheme():
    with pytest.raises(ResolverError, match="not a resolver URI"):
        parse_uri("bare-id")
```

- [ ] **Step 2: Run tests; expect import failures**

```bash
pytest tests/core/test_resolvers.py -v
```

Expected: `ModuleNotFoundError: No module named 'muse.core.resolvers'`.

- [ ] **Step 3: Implement `muse/core/resolvers.py`**

```python
"""Resolver abstraction: URIs in, synthesized model records out.

A Resolver translates a URI like `hf://Qwen/Qwen3-8B-GGUF@q4_k_m` into a
ResolvedModel (synthesized manifest + backend class path + downloader
function). Resolvers also expose `search(query, **filters)` for model
discovery across their backing source (e.g. HuggingFace Hub).

Design goals:
 - Pluggable: register_resolver(instance) at import time from submodules.
 - Dispatching: `resolve(uri)` / `search(query, backend=...)` find the
   right resolver and forward.
 - Stateless: resolvers hold configuration but no per-call state.

The resolver output feeds directly into the existing pull path:
 - manifest -> catalog.json persisted alongside normal pull state
 - backend_path -> load_backend() imports and instantiates
 - download(cache_dir) -> fetches weights to a local directory
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


logger = logging.getLogger(__name__)


class ResolverError(Exception):
    """Raised when resolution or dispatch fails."""


@dataclass
class ResolvedModel:
    """Output of Resolver.resolve().

    Fields:
      - manifest: dict with at minimum `model_id`, `modality`, `hf_repo`
        keys, shaped like a MANIFEST in a model script. Flows into
        catalog.json and registry manifest passthrough.
      - backend_path: "module.path:ClassName" for load_backend(). The
        class must accept (hf_repo, local_dir, **kwargs) in its
        constructor, same protocol as scripted models.
      - download: callable that takes a cache directory and returns the
        path to the downloaded weights. Called during `pull`. Allows
        each resolver to control download semantics (snapshot_download,
        single-file download, etc.).
    """
    manifest: dict
    backend_path: str
    download: Callable[[Path], Path]


@dataclass
class SearchResult:
    """One candidate model returned from `Resolver.search`.

    Fields mirror what a user sees in a table listing. All optional
    fields may be None when the backend doesn't surface the data.
    """
    uri: str                      # "hf://org/repo@variant"
    model_id: str                 # synthesized id that `pull` would assign
    modality: str                 # MIME tag, e.g. "chat/completion"
    size_gb: float | None = None
    downloads: int | None = None
    license: str | None = None
    description: str | None = None
    metadata: dict = field(default_factory=dict)


class Resolver(ABC):
    """Abstract resolver for a URI scheme."""

    scheme: str  # subclasses MUST set, e.g. "hf"

    @abstractmethod
    def resolve(self, uri: str) -> ResolvedModel:
        """Translate a URI into a ResolvedModel."""

    @abstractmethod
    def search(self, query: str, **filters: Any) -> Iterable[SearchResult]:
        """Search the backend for candidate models."""


_RESOLVERS: dict[str, Resolver] = {}


def register_resolver(resolver: Resolver) -> None:
    """Register a resolver instance under its scheme.

    Re-registration of the same scheme is allowed (overwrites); this
    simplifies test fixtures and future escape-hatch env-var overrides.
    """
    _RESOLVERS[resolver.scheme] = resolver


def _reset_registry_for_tests() -> None:
    """Test hook: clear all registered resolvers."""
    _RESOLVERS.clear()


def parse_uri(uri: str) -> tuple[str, str, str | None]:
    """Split `scheme://ref[@variant]` into (scheme, ref, variant | None).

    Raises ResolverError if the input has no `://` separator.
    """
    if "://" not in uri:
        raise ResolverError(f"not a resolver URI: {uri!r}")
    scheme, rest = uri.split("://", 1)
    if "@" in rest:
        ref, variant = rest.rsplit("@", 1)
    else:
        ref, variant = rest, None
    return scheme, ref, variant


def get_resolver(uri: str) -> Resolver:
    """Return the resolver registered for `uri`'s scheme."""
    scheme, _, _ = parse_uri(uri)
    try:
        return _RESOLVERS[scheme]
    except KeyError:
        raise ResolverError(
            f"no resolver for scheme {scheme!r}; "
            f"registered: {sorted(_RESOLVERS)}"
        )


def resolve(uri: str) -> ResolvedModel:
    """Resolve a URI through the matching resolver."""
    return get_resolver(uri).resolve(uri)


def search(query: str, *, backend: str | None = None, **filters: Any) -> Iterable[SearchResult]:
    """Search one backend (or the only-registered backend) for candidates.

    `backend` is the resolver scheme (e.g. "hf"). When omitted and
    exactly one resolver is registered, that one is used. When omitted
    and multiple are registered, raises ResolverError asking the caller
    to pick.
    """
    if backend is None:
        if len(_RESOLVERS) == 1:
            backend = next(iter(_RESOLVERS))
        else:
            raise ResolverError(
                f"multiple resolvers registered {sorted(_RESOLVERS)!r}; "
                f"pass backend= to disambiguate"
            )
    if backend not in _RESOLVERS:
        raise ResolverError(
            f"no resolver registered for backend {backend!r}; "
            f"registered: {sorted(_RESOLVERS)}"
        )
    return _RESOLVERS[backend].search(query, **filters)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/core/test_resolvers.py -v
```

Expected: 9/9 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/core/resolvers.py tests/core/test_resolvers.py
git commit -m "feat(core): muse.core.resolvers abstraction with URI dispatch

Pluggable resolver registry. Resolver.resolve(uri) -> ResolvedModel
(synthesized manifest + backend path + downloader). Resolver.search()
-> iterable of SearchResult. Concrete resolvers register themselves at
import time. parse_uri() splits scheme://ref[@variant] form.

Nothing consumes this yet; the HF resolver (Task A2) and pull() +
search CLI wiring land in later phases."
```

---

## Part B: chat/completion modality

### Task B1: Create modality package skeleton + protocol

**Files:**
- Create: `src/muse/modalities/chat_completion/__init__.py`
- Create: `src/muse/modalities/chat_completion/protocol.py`
- Create: `tests/modalities/chat_completion/__init__.py`
- Create: `tests/modalities/chat_completion/test_protocol.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/modalities/chat_completion/test_protocol.py
"""Tests for chat/completion protocol types and ChatModel Protocol."""
from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatMessage,
    ChatResult,
    ChatModel,
)


def test_chat_message_roles():
    for role in ("system", "user", "assistant", "tool"):
        msg = ChatMessage(role=role, content="hi")
        assert msg.role == role


def test_chat_result_shape():
    r = ChatResult(
        id="chatcmpl-abc",
        model_id="qwen3-8b-gguf-q4_k_m",
        created=1_700_000_000,
        choices=[ChatChoice(
            index=0,
            message={"role": "assistant", "content": "hello"},
            finish_reason="stop",
        )],
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )
    assert r.choices[0].message["content"] == "hello"
    assert r.usage["total_tokens"] == 12


def test_chat_chunk_shape():
    c = ChatChunk(
        id="chatcmpl-abc",
        model_id="qwen3-8b",
        created=1_700_000_000,
        choice_index=0,
        delta={"role": "assistant", "content": "he"},
        finish_reason=None,
    )
    assert c.delta["content"] == "he"
    assert c.finish_reason is None


def test_chat_model_protocol_is_runtime_checkable():
    """A plain class with the right attrs/methods should satisfy ChatModel."""
    class Fake:
        model_id = "x"

        def chat(self, messages, **kwargs):
            return None

        def chat_stream(self, messages, **kwargs):
            return iter([])

    assert isinstance(Fake(), ChatModel)


def test_chat_model_protocol_rejects_missing_method():
    class NoStream:
        model_id = "x"

        def chat(self, messages, **kwargs):
            return None

    assert not isinstance(NoStream(), ChatModel)
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/modalities/chat_completion/test_protocol.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `protocol.py`**

```python
# src/muse/modalities/chat_completion/protocol.py
"""Chat completion modality: OpenAI-shape chat + tool calling.

Types mirror OpenAI's ChatCompletion / ChatCompletionChunk structure
exactly at the wire level (dict passthrough for messages, tool_calls,
tool_choice, response_format) but the ChatModel Protocol is
backend-agnostic. Backends translate their native format into
ChatResult / ChatChunk and return them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Protocol, runtime_checkable


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatMessage:
    """Input message. Primarily for type-checking tests; routes
    accept raw dicts and pass them through so backends can see
    fields we haven't modeled yet (logprobs, name, etc.)."""
    role: Role
    content: str | None = None
    tool_calls: list[dict] | None = None  # OpenAI shape
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ChatChoice:
    """One choice in a non-streaming response.

    `message` is a raw dict (OpenAI shape) so backends can include
    tool_calls, refusal, or future fields without us modeling them.
    """
    index: int
    message: dict
    finish_reason: str | None  # "stop" | "length" | "tool_calls" | "content_filter" | None


@dataclass
class ChatResult:
    """Non-streaming chat response. Mirrors OpenAI ChatCompletion."""
    id: str
    model_id: str
    created: int                          # unix timestamp
    choices: list[ChatChoice]
    usage: dict                           # {prompt_tokens, completion_tokens, total_tokens}
    metadata: dict = field(default_factory=dict)  # backend-specific; not serialized to wire


@dataclass
class ChatChunk:
    """Streaming delta. Mirrors OpenAI ChatCompletionChunk.

    `delta` is a partial message dict: {role?, content?, tool_calls?}.
    Clients accumulate across chunks by choice_index.
    """
    id: str
    model_id: str
    created: int
    choice_index: int
    delta: dict
    finish_reason: str | None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ChatModel(Protocol):
    """Backend protocol. Implementations satisfy this structurally;
    no base class needed."""

    model_id: str

    def chat(
        self,
        messages: list[dict],
        **kwargs: Any,
    ) -> ChatResult:
        ...

    def chat_stream(
        self,
        messages: list[dict],
        **kwargs: Any,
    ) -> Iterator[ChatChunk]:
        ...
```

- [ ] **Step 4: Implement `__init__.py`**

```python
# src/muse/modalities/chat_completion/__init__.py
"""Chat completion modality: text-to-text LLM serving.

Wire contract: POST /v1/chat/completions with OpenAI-shape body
(messages, tools?, tool_choice?, response_format?, stream?, temperature?,
max_tokens?, stop?, seed?, logprobs?, top_logprobs?) returns OpenAI
ChatCompletion or, when stream=True, SSE-encoded ChatCompletionChunk
events.

Models declaring `modality = "chat/completion"` in their MANIFEST and
satisfying the ChatModel protocol plug into this modality.
"""
from muse.modalities.chat_completion.client import ChatClient
from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatMessage,
    ChatModel,
    ChatResult,
)
from muse.modalities.chat_completion.routes import build_router

MODALITY = "chat/completion"

__all__ = [
    "MODALITY",
    "build_router",
    "ChatClient",
    "ChatChoice",
    "ChatChunk",
    "ChatMessage",
    "ChatModel",
    "ChatResult",
]
```

- [ ] **Step 5: Run protocol tests**

```bash
pytest tests/modalities/chat_completion/test_protocol.py -v
```

Expected: 5/5 pass.

- [ ] **Step 6: Commit**

```bash
git add src/muse/modalities/chat_completion/__init__.py \
        src/muse/modalities/chat_completion/protocol.py \
        tests/modalities/chat_completion/__init__.py \
        tests/modalities/chat_completion/test_protocol.py
git commit -m "feat(chat/completion): modality package skeleton + protocol types

ChatMessage / ChatChoice / ChatResult / ChatChunk dataclasses plus a
runtime-checkable ChatModel Protocol. Message and delta are raw dicts
so backends can pass through tool_calls, refusals, and future OpenAI
fields without the protocol evolving. No routes or codec yet (Task B3,
B4); __init__ will fail to import the router until then."

# NOTE: this commit leaves __init__.py partially dead because it imports
# build_router from routes.py which doesn't exist yet. OK: B3 lands in
# the same subagent-driven task chain.
```

---

### Task B2: codec.py (SSE encoding for ChatChunk)

**Files:**
- Create: `src/muse/modalities/chat_completion/codec.py`
- Create: `tests/modalities/chat_completion/test_codec.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/modalities/chat_completion/test_codec.py
"""Tests for chat/completion wire codec (SSE + OpenAI dict shapes)."""
import json

from muse.modalities.chat_completion.codec import (
    chunk_to_sse_data,
    chunk_to_openai_dict,
    result_to_openai_dict,
    DONE_SENTINEL,
)
from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatResult,
)


def test_chunk_to_openai_dict_maps_fields():
    c = ChatChunk(
        id="chatcmpl-1",
        model_id="qwen3-8b",
        created=1_700_000_000,
        choice_index=0,
        delta={"role": "assistant", "content": "hi"},
        finish_reason=None,
    )
    d = chunk_to_openai_dict(c)
    assert d["id"] == "chatcmpl-1"
    assert d["object"] == "chat.completion.chunk"
    assert d["created"] == 1_700_000_000
    assert d["model"] == "qwen3-8b"
    assert d["choices"] == [{
        "index": 0,
        "delta": {"role": "assistant", "content": "hi"},
        "finish_reason": None,
    }]


def test_chunk_with_finish_reason():
    c = ChatChunk(
        id="x", model_id="y", created=0, choice_index=0,
        delta={}, finish_reason="stop",
    )
    d = chunk_to_openai_dict(c)
    assert d["choices"][0]["finish_reason"] == "stop"


def test_chunk_to_sse_data_json_encodes():
    c = ChatChunk(
        id="a", model_id="b", created=1, choice_index=0,
        delta={"content": "hello"}, finish_reason=None,
    )
    sse = chunk_to_sse_data(c)
    # Starlette's EventSourceResponse wraps with "data: " + "\n\n"; the
    # codec returns just the JSON payload.
    decoded = json.loads(sse)
    assert decoded["choices"][0]["delta"]["content"] == "hello"


def test_done_sentinel():
    assert DONE_SENTINEL == "[DONE]"


def test_result_to_openai_dict_shape():
    r = ChatResult(
        id="chatcmpl-xyz",
        model_id="qwen3-8b",
        created=1_700_000_001,
        choices=[ChatChoice(
            index=0,
            message={"role": "assistant", "content": "world"},
            finish_reason="stop",
        )],
        usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    )
    d = result_to_openai_dict(r)
    assert d["object"] == "chat.completion"
    assert d["id"] == "chatcmpl-xyz"
    assert d["choices"][0]["message"]["content"] == "world"
    assert d["choices"][0]["finish_reason"] == "stop"
    assert d["usage"]["total_tokens"] == 4
```

- [ ] **Step 2: Run; expect ModuleNotFoundError**

```bash
pytest tests/modalities/chat_completion/test_codec.py -v
```

- [ ] **Step 3: Implement `codec.py`**

```python
# src/muse/modalities/chat_completion/codec.py
"""Wire codec: ChatResult/ChatChunk <-> OpenAI JSON wire shapes + SSE."""
from __future__ import annotations

import json

from muse.modalities.chat_completion.protocol import ChatChunk, ChatResult


DONE_SENTINEL = "[DONE]"
"""Sentinel sent as the last SSE `data:` line in a streaming response.
Clients stop reading at this marker. OpenAI uses the exact same string."""


def chunk_to_openai_dict(chunk: ChatChunk) -> dict:
    """Map a ChatChunk to the OpenAI ChatCompletionChunk wire dict."""
    return {
        "id": chunk.id,
        "object": "chat.completion.chunk",
        "created": chunk.created,
        "model": chunk.model_id,
        "choices": [{
            "index": chunk.choice_index,
            "delta": chunk.delta,
            "finish_reason": chunk.finish_reason,
        }],
    }


def chunk_to_sse_data(chunk: ChatChunk) -> str:
    """JSON-encode a chunk ready for an SSE `data:` line.

    Returns only the JSON body; the transport layer (EventSourceResponse
    in routes.py) handles the `data: ` prefix and `\\n\\n` separator.
    """
    return json.dumps(chunk_to_openai_dict(chunk), separators=(",", ":"))


def result_to_openai_dict(result: ChatResult) -> dict:
    """Map a ChatResult to the OpenAI ChatCompletion wire dict."""
    return {
        "id": result.id,
        "object": "chat.completion",
        "created": result.created,
        "model": result.model_id,
        "choices": [{
            "index": c.index,
            "message": c.message,
            "finish_reason": c.finish_reason,
        } for c in result.choices],
        "usage": result.usage,
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/modalities/chat_completion/test_codec.py -v
```

Expected: 5/5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/chat_completion/codec.py \
        tests/modalities/chat_completion/test_codec.py
git commit -m "feat(chat/completion): codec for OpenAI wire shape + SSE data lines

Translate internal ChatResult/ChatChunk to the OpenAI
ChatCompletion/ChatCompletionChunk JSON shape. chunk_to_sse_data()
returns the payload alone; the SSE framing (data: prefix, \\n\\n
separator, [DONE] sentinel) stays in routes.py."
```

---

### Task B3: routes.py (/v1/chat/completions, streaming + non-streaming)

**Files:**
- Create: `src/muse/modalities/chat_completion/routes.py`
- Create: `tests/modalities/chat_completion/test_routes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/modalities/chat_completion/test_routes.py
"""Tests for /v1/chat/completions (streaming + non-streaming).

Uses a FakeModel that implements ChatModel Protocol. No llama-cpp deps.
"""
import json
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatResult,
)
from muse.modalities.chat_completion.routes import build_router


class _FakeChatModel:
    model_id = "fake-chat"

    def chat(self, messages, **kwargs):
        return ChatResult(
            id="chatcmpl-fake-1",
            model_id=self.model_id,
            created=1_700_000_000,
            choices=[ChatChoice(
                index=0,
                message={"role": "assistant", "content": "hi"},
                finish_reason="stop",
            )],
            usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        )

    def chat_stream(self, messages, **kwargs):
        yield ChatChunk(
            id="chatcmpl-fake-2", model_id=self.model_id, created=0,
            choice_index=0, delta={"role": "assistant"}, finish_reason=None,
        )
        yield ChatChunk(
            id="chatcmpl-fake-2", model_id=self.model_id, created=0,
            choice_index=0, delta={"content": "hi"}, finish_reason=None,
        )
        yield ChatChunk(
            id="chatcmpl-fake-2", model_id=self.model_id, created=0,
            choice_index=0, delta={}, finish_reason="stop",
        )


@pytest.fixture
def client():
    reg = ModalityRegistry()
    reg.register("chat/completion", _FakeChatModel())
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    return TestClient(app)


def test_non_streaming_returns_openai_shape(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-chat",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "fake-chat"
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["usage"]["total_tokens"] == 5


def test_streaming_returns_sse(client):
    with client.stream(
        "POST", "/v1/chat/completions",
        json={
            "model": "fake-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = r.read().decode()
    # Parse out `data:` lines
    lines = [line for line in body.splitlines() if line.startswith("data: ")]
    payloads = [line[len("data: "):] for line in lines]
    # Last payload is [DONE]
    assert payloads[-1] == "[DONE]"
    # Earlier payloads are ChatCompletionChunk JSON
    parsed = [json.loads(p) for p in payloads[:-1]]
    assert len(parsed) == 3
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert parsed[1]["choices"][0]["delta"]["content"] == "hi"
    assert parsed[2]["choices"][0]["finish_reason"] == "stop"


def test_unknown_model_returns_404_with_openai_envelope(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "nonexistent", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "model_not_found"


def test_empty_messages_returns_422(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "fake-chat", "messages": []},
    )
    assert r.status_code == 422


def test_default_model_used_when_model_omitted(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json()["model"] == "fake-chat"


def test_tools_passthrough_to_backend():
    """Backend's chat() gets tools kwarg when client sends one."""
    reg = ModalityRegistry()
    seen = {}

    class _Capturing:
        model_id = "cap"

        def chat(self, messages, **kwargs):
            seen.update(kwargs)
            return ChatResult(
                id="x", model_id=self.model_id, created=0,
                choices=[ChatChoice(index=0, message={"role": "assistant", "content": ""}, finish_reason="stop")],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        def chat_stream(self, messages, **kwargs):
            return iter([])

    reg.register("chat/completion", _Capturing())
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    client = TestClient(app)
    tools = [{"type": "function", "function": {
        "name": "get_weather", "parameters": {"type": "object"},
    }}]
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "cap",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        },
    )
    assert r.status_code == 200
    assert seen["tools"] == tools
    assert seen["tool_choice"] == "auto"
    assert seen["temperature"] == 0.2
```

- [ ] **Step 2: Run; expect ModuleNotFoundError**

```bash
pytest tests/modalities/chat_completion/test_routes.py -v
```

- [ ] **Step 3: Implement `routes.py`**

```python
# src/muse/modalities/chat_completion/routes.py
"""/v1/chat/completions router.

Two call shapes:
  - stream=False (default): non-streaming. Calls ChatModel.chat() once,
    returns OpenAI ChatCompletion JSON.
  - stream=True: SSE. Producer thread calls ChatModel.chat_stream() and
    pushes ChatChunk items into an asyncio.Queue; the response iterator
    reads from the queue and serializes to SSE `data:` lines plus a
    final `data: [DONE]` sentinel.

Thread + queue pattern matches the audio.speech streaming code so we
don't buffer tokens on the server. Every token dispatches as produced.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from muse.core.errors import ModelNotFoundError
from muse.core.registry import ModalityRegistry
from muse.modalities.chat_completion.codec import (
    DONE_SENTINEL,
    chunk_to_sse_data,
    result_to_openai_dict,
)


logger = logging.getLogger(__name__)

MODALITY = "chat/completion"


class ChatCompletionRequest(BaseModel):
    """OpenAI-shape request. Most fields are passthrough to the backend."""
    model: str | None = None
    messages: list[dict] = Field(..., min_length=1)
    stream: bool = False
    # OpenAI sampling params (optional; backends may ignore)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    # Tool calling
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    # Structured output
    response_format: dict | None = None
    # Logprobs
    logprobs: bool | None = None
    top_logprobs: int | None = None
    # Reasoning / future fields passed through raw
    extra_body: dict | None = None

    @field_validator("messages")
    @classmethod
    def _non_empty_messages(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError("messages must be non-empty")
        return v

    def backend_kwargs(self) -> dict:
        """Dict of kwargs to forward to ChatModel.chat()/chat_stream().

        Omits `model` (routing metadata) and `stream` (handled by the
        route, not the backend). Extra_body spreads in raw.
        """
        out: dict[str, Any] = {}
        for key in (
            "temperature", "top_p", "max_tokens", "stop", "seed",
            "tools", "tool_choice", "response_format", "logprobs",
            "top_logprobs",
        ):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        if self.extra_body:
            out.update(self.extra_body)
        return out


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["chat/completion"])

    def _get_model(model_id: str | None):
        try:
            return registry.get(MODALITY, model_id)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model_id or "(default)",
                modality=MODALITY,
            )

    @router.post("/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        model = _get_model(req.model)
        kwargs = req.backend_kwargs()

        if not req.stream:
            result = await asyncio.to_thread(model.chat, req.messages, **kwargs)
            return result_to_openai_dict(result)

        # Streaming: producer thread feeds chunks into an asyncio.Queue
        # so dispatch happens as tokens are generated.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer():
            try:
                for chunk in model.chat_stream(req.messages, **kwargs):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(e), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        threading.Thread(target=_producer, daemon=True).start()

        async def _events():
            while True:
                item = await queue.get()
                if item is None:
                    yield {"data": DONE_SENTINEL}
                    return
                if isinstance(item, Exception):
                    logger.error("chat_stream backend error: %s", item)
                    # Terminate the stream; clients see an abrupt end.
                    yield {"data": DONE_SENTINEL}
                    return
                yield {"data": chunk_to_sse_data(item)}

        return EventSourceResponse(_events())

    return router
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/modalities/chat_completion/test_routes.py -v
```

Expected: 6/6 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/chat_completion/routes.py \
        tests/modalities/chat_completion/test_routes.py
git commit -m "feat(chat/completion): /v1/chat/completions with streaming + tools

Non-streaming: ChatCompletionRequest -> registry.get -> model.chat ->
OpenAI dict. Streaming: producer thread + asyncio.Queue feeds SSE data
events plus a [DONE] sentinel. tools / tool_choice / response_format /
stop / sampling params pass through to the backend via backend_kwargs.

Regression guard: default-model, unknown-model-404, empty-messages-422,
tools-passthrough all covered."
```

---

### Task B4: client.py (Python HTTP client)

**Files:**
- Create: `src/muse/modalities/chat_completion/client.py`
- Create: `tests/modalities/chat_completion/test_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/modalities/chat_completion/test_client.py
"""Tests for ChatClient (HTTP client for /v1/chat/completions)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.chat_completion.client import ChatClient


def test_client_non_streaming_returns_dict():
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "id": "chatcmpl-1", "object": "chat.completion",
        "created": 0, "model": "fake",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    fake_response.raise_for_status = MagicMock()
    with patch("muse.modalities.chat_completion.client.httpx.post", return_value=fake_response) as mock_post:
        c = ChatClient(base_url="http://localhost:8000")
        result = c.chat(model="fake", messages=[{"role": "user", "content": "hi"}])
        assert result["choices"][0]["message"]["content"] == "hi"
        url = mock_post.call_args.args[0]
        assert url == "http://localhost:8000/v1/chat/completions"
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "fake"
        assert body["stream"] is False


def test_client_streaming_yields_chunks():
    """stream=True: client opens a stream and yields parsed chunk dicts."""
    chunks = [
        b'data: {"choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"hi"},"index":0,"finish_reason":null}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.__enter__ = lambda s: s
    fake_response.__exit__ = lambda s, a, b, c: None
    fake_response.iter_lines.return_value = [
        "data: " + '{"choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}',
        "",
        "data: " + '{"choices":[{"delta":{"content":"hi"},"index":0,"finish_reason":null}]}',
        "",
        "data: [DONE]",
        "",
    ]
    fake_response.raise_for_status = MagicMock()

    fake_stream = MagicMock()
    fake_stream.__enter__ = lambda s: fake_response
    fake_stream.__exit__ = lambda s, a, b, c: None

    with patch("muse.modalities.chat_completion.client.httpx.stream", return_value=fake_stream):
        c = ChatClient(base_url="http://x")
        out = list(c.chat_stream(model="fake", messages=[{"role": "user", "content": "hi"}]))
        assert len(out) == 2
        assert out[0]["choices"][0]["delta"]["role"] == "assistant"
        assert out[1]["choices"][0]["delta"]["content"] == "hi"


def test_client_uses_muse_server_env_var(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://example.test:9000")
    c = ChatClient()
    assert c.base_url == "http://example.test:9000"
```

- [ ] **Step 2: Run; expect ModuleNotFoundError**

- [ ] **Step 3: Implement `client.py`**

```python
# src/muse/modalities/chat_completion/client.py
"""HTTP client for /v1/chat/completions. Mirrors the OpenAI SDK shape."""
from __future__ import annotations

import json
import logging
import os
from typing import Iterator

import httpx


logger = logging.getLogger(__name__)


class ChatClient:
    """Client for the chat/completion modality.

    Non-streaming: `chat(model, messages, **kwargs)` returns the full
    OpenAI ChatCompletion dict.

    Streaming: `chat_stream(model, messages, **kwargs)` yields each
    ChatCompletionChunk dict as it arrives, stopping at the [DONE]
    sentinel.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = (
            base_url
            or os.environ.get("MUSE_SERVER")
            or "http://localhost:8000"
        ).rstrip("/")
        self.timeout = timeout

    def chat(self, *, model: str | None = None, messages: list[dict], **kwargs) -> dict:
        body = {"model": model, "messages": messages, "stream": False, **kwargs}
        body = {k: v for k, v in body.items() if v is not None}
        r = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def chat_stream(
        self,
        *,
        model: str | None = None,
        messages: list[dict],
        **kwargs,
    ) -> Iterator[dict]:
        body = {"model": model, "messages": messages, "stream": True, **kwargs}
        body = {k: v for k, v in body.items() if v is not None}
        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=body,
            timeout=self.timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError as e:
                    logger.warning("malformed SSE chunk: %s (%s)", payload, e)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/modalities/chat_completion/test_client.py -v
```

Expected: 3/3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/chat_completion/client.py \
        tests/modalities/chat_completion/test_client.py
git commit -m "feat(chat/completion): Python client with streaming support"
```

---

## Part C: LlamaCppModel runtime

### Task C1: LlamaCppModel implementation

**Files:**
- Create: `src/muse/modalities/chat_completion/runtimes/__init__.py`
- Create: `src/muse/modalities/chat_completion/runtimes/llama_cpp.py`
- Create: `tests/modalities/chat_completion/runtimes/__init__.py`
- Create: `tests/modalities/chat_completion/runtimes/test_llama_cpp.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/modalities/chat_completion/runtimes/test_llama_cpp.py
"""Tests for LlamaCppModel (mocks llama_cpp.Llama; no real GGUF loaded)."""
from unittest.mock import MagicMock, patch


def test_llama_cpp_loads_gguf_path():
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        mock_cls.return_value = MagicMock()
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        m = LlamaCppModel(
            model_id="qwen3-8b-q4",
            hf_repo="Qwen/Qwen3-8B-GGUF",
            local_dir="/fake/dir",
            gguf_file="qwen3-8b-q4_k_m.gguf",
            context_length=8192,
        )
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model_path"] == "/fake/dir/qwen3-8b-q4_k_m.gguf"
        assert kwargs["n_ctx"] == 8192
        assert m.model_id == "qwen3-8b-q4"


def test_llama_cpp_chat_passes_openai_response_through():
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        fake_llama = MagicMock()
        fake_llama.create_chat_completion.return_value = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "qwen3-8b-q4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }
        mock_cls.return_value = fake_llama
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        from muse.modalities.chat_completion.protocol import ChatResult
        m = LlamaCppModel(
            model_id="qwen3-8b-q4",
            hf_repo="x", local_dir="/fake", gguf_file="x.gguf",
        )
        result = m.chat(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )
        assert isinstance(result, ChatResult)
        assert result.model_id == "qwen3-8b-q4"
        assert result.choices[0].message["content"] == "hello"
        assert result.usage["total_tokens"] == 5
        # Forwarded kwargs
        call_kwargs = fake_llama.create_chat_completion.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert call_kwargs["temperature"] == 0.7


def test_llama_cpp_chat_stream_translates_chunks():
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        fake_llama = MagicMock()
        fake_llama.create_chat_completion.return_value = iter([
            {"id": "c1", "object": "chat.completion.chunk", "created": 0,
             "model": "x", "choices": [{"index": 0, "delta": {"role": "assistant"},
                                         "finish_reason": None}]},
            {"id": "c1", "object": "chat.completion.chunk", "created": 0,
             "model": "x", "choices": [{"index": 0, "delta": {"content": "hi"},
                                         "finish_reason": None}]},
            {"id": "c1", "object": "chat.completion.chunk", "created": 0,
             "model": "x", "choices": [{"index": 0, "delta": {},
                                         "finish_reason": "stop"}]},
        ])
        mock_cls.return_value = fake_llama
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        from muse.modalities.chat_completion.protocol import ChatChunk
        m = LlamaCppModel(
            model_id="x", hf_repo="x", local_dir="/fake", gguf_file="x.gguf",
        )
        chunks = list(m.chat_stream(messages=[{"role": "user", "content": "hi"}]))
        assert len(chunks) == 3
        assert all(isinstance(c, ChatChunk) for c in chunks)
        assert chunks[0].delta == {"role": "assistant"}
        assert chunks[1].delta == {"content": "hi"}
        assert chunks[2].finish_reason == "stop"
        assert fake_llama.create_chat_completion.call_args.kwargs["stream"] is True


def test_llama_cpp_forwards_tools_and_tool_choice():
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        fake_llama = MagicMock()
        fake_llama.create_chat_completion.return_value = {
            "id": "c", "object": "chat.completion", "created": 0,
            "model": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                        "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        mock_cls.return_value = fake_llama
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        m = LlamaCppModel(model_id="x", hf_repo="x", local_dir="/fake", gguf_file="x.gguf")
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        m.chat(messages=[{"role": "user", "content": "x"}], tools=tools, tool_choice="auto")
        ck = fake_llama.create_chat_completion.call_args.kwargs
        assert ck["tools"] == tools
        assert ck["tool_choice"] == "auto"


def test_llama_cpp_raises_clear_error_when_deps_missing():
    """Host without llama-cpp-python: constructor raises informative RuntimeError."""
    with patch(
        "muse.modalities.chat_completion.runtimes.llama_cpp.Llama",
        new=None,
    ):
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        # _ensure_deps will try and fail; Llama stays None.
        import pytest
        with pytest.raises(RuntimeError, match="llama-cpp-python"):
            LlamaCppModel(model_id="x", hf_repo="x", local_dir="/fake", gguf_file="x.gguf")


def test_llama_cpp_accepts_chat_template_override():
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        mock_cls.return_value = MagicMock()
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        LlamaCppModel(
            model_id="x", hf_repo="x", local_dir="/fake",
            gguf_file="x.gguf", chat_template="chatml",
        )
        assert mock_cls.call_args.kwargs.get("chat_format") == "chatml"


def test_llama_cpp_n_gpu_layers_default_is_all():
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        mock_cls.return_value = MagicMock()
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        LlamaCppModel(model_id="x", hf_repo="x", local_dir="/fake", gguf_file="x.gguf")
        assert mock_cls.call_args.kwargs.get("n_gpu_layers") == -1
```

- [ ] **Step 2: Run; expect ModuleNotFoundError**

- [ ] **Step 3: Implement `runtimes/llama_cpp.py`**

```python
# src/muse/modalities/chat_completion/runtimes/llama_cpp.py
"""LlamaCppModel: GGUF chat-completion runtime via llama-cpp-python.

One runtime, many models: a GGUF-addressable model's MANIFEST supplies
`gguf_file`, `chat_template` (optional), `context_length` (optional),
and any capabilities. The class is name `Model` when aliased from a
model script; `LlamaCppModel` here for direct import. Tests mock
`Llama` at module level.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatResult,
)


logger = logging.getLogger(__name__)

# Module-level sentinel; real import deferred so discovery tolerates
# missing deps. Tests patch this attribute directly.
Llama: Any = None


def _ensure_deps() -> None:
    global Llama
    if Llama is not None:
        return
    try:
        from llama_cpp import Llama as _Llama
        Llama = _Llama
    except Exception as e:  # noqa: BLE001
        logger.debug("llama-cpp-python unavailable: %s", e)


class LlamaCppModel:
    """Chat-completion backend wrapping llama_cpp.Llama.

    The constructor's kwargs mirror MANIFEST capability keys:
      - gguf_file (required): filename inside local_dir
      - context_length (default 8192)
      - chat_template (default None -> use template embedded in GGUF)
      - n_gpu_layers (default -1 -> offload everything GPU can fit)
      - extra llama.cpp options absorbed by **_
    """

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        gguf_file: str,
        context_length: int = 8192,
        chat_template: str | None = None,
        n_gpu_layers: int = -1,
        device: str = "auto",
        **_: Any,
    ) -> None:
        _ensure_deps()
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed; run `muse pull` or "
                "install `llama-cpp-python` into this venv"
            )
        self.model_id = model_id
        base = Path(local_dir or ".") if local_dir else None
        if base is None:
            raise RuntimeError("local_dir is required; the GGUF file must be on disk")
        gguf_path = base / gguf_file
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
        logger.info(
            "loading GGUF %s (ctx=%d, gpu_layers=%d, chat_template=%s)",
            gguf_path, context_length, n_gpu_layers, chat_template,
        )
        self._llama = Llama(
            model_path=str(gguf_path),
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_template,  # None -> llama.cpp uses embedded template
            verbose=False,
        )

    def chat(self, messages: list[dict], **kwargs) -> ChatResult:
        resp = self._llama.create_chat_completion(
            messages=messages,
            stream=False,
            **_filter_kwargs(kwargs),
        )
        return _dict_to_chat_result(resp, fallback_model_id=self.model_id)

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[ChatChunk]:
        for chunk in self._llama.create_chat_completion(
            messages=messages,
            stream=True,
            **_filter_kwargs(kwargs),
        ):
            yield _dict_to_chat_chunk(chunk, fallback_model_id=self.model_id)


_FORWARDED_KWARGS = frozenset({
    "temperature", "top_p", "max_tokens", "stop", "seed",
    "tools", "tool_choice", "response_format", "logprobs",
    "top_logprobs", "frequency_penalty", "presence_penalty",
    "repeat_penalty", "grammar", "logit_bias",
})


def _filter_kwargs(kwargs: dict) -> dict:
    """Only forward kwargs llama-cpp-python understands; silently drop the rest."""
    return {k: v for k, v in kwargs.items() if k in _FORWARDED_KWARGS}


def _dict_to_chat_result(resp: dict, *, fallback_model_id: str) -> ChatResult:
    return ChatResult(
        id=resp.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model_id=resp.get("model") or fallback_model_id,
        created=resp.get("created") or int(time.time()),
        choices=[
            ChatChoice(
                index=c["index"],
                message=c["message"],
                finish_reason=c.get("finish_reason"),
            )
            for c in resp.get("choices", [])
        ],
        usage=resp.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


def _dict_to_chat_chunk(chunk: dict, *, fallback_model_id: str) -> ChatChunk:
    choices = chunk.get("choices") or []
    c = choices[0] if choices else {"index": 0, "delta": {}, "finish_reason": None}
    return ChatChunk(
        id=chunk.get("id") or "chatcmpl-stream",
        model_id=chunk.get("model") or fallback_model_id,
        created=chunk.get("created") or int(time.time()),
        choice_index=c.get("index", 0),
        delta=c.get("delta") or {},
        finish_reason=c.get("finish_reason"),
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/modalities/chat_completion/runtimes/test_llama_cpp.py -v
```

Expected: 7/7 pass. Note: `test_llama_cpp_raises_clear_error_when_deps_missing` needs the module-level `Llama` to be `None`; it's set via `with patch(..., new=None)` on the module attr so `_ensure_deps` tries and fails.

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/chat_completion/runtimes \
        tests/modalities/chat_completion/runtimes
git commit -m "feat(chat/completion): LlamaCppModel runtime wrapping llama-cpp-python

Deferred import of Llama (per-symbol; discovery tolerates missing or
broken llama-cpp-python). Constructor reads MANIFEST capability keys
(gguf_file, context_length, chat_template, n_gpu_layers). chat() and
chat_stream() forward to Llama.create_chat_completion and translate
its OpenAI-shape dicts to ChatResult / ChatChunk. Tools / tool_choice /
response_format / stop / seed all pass through via _FORWARDED_KWARGS."
```

---

## Part D: HF resolver (GGUF + sentence-transformers)

### Task D1: HFResolver scaffold + sniff logic

**Files:**
- Create: `src/muse/core/resolvers_hf.py`
- Create: `tests/core/test_resolvers_hf.py`

- [ ] **Step 1: Write tests for sniff + URI handling**

```python
# tests/core/test_resolvers_hf.py
"""Tests for HFResolver (huggingface_hub mocked; no network)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.core.resolvers import ResolverError, _reset_registry_for_tests
from muse.core.resolvers_hf import HFResolver, _sniff_repo_shape


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


def _fake_repo_info(siblings=(), tags=()):
    """Build a MagicMock that looks like HfApi().repo_info() output."""
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f, size=1_000_000) for f in siblings]
    info.tags = list(tags)
    info.card_data = MagicMock(license="apache-2.0")
    info.downloads = 123
    return info


def test_sniff_recognizes_gguf_repo():
    info = _fake_repo_info(
        siblings=["qwen3-8b-q4_k_m.gguf", "README.md", "config.json"],
        tags=["text-generation"],
    )
    shape = _sniff_repo_shape(info)
    assert shape == "gguf"


def test_sniff_recognizes_sentence_transformers_via_tag():
    info = _fake_repo_info(
        siblings=["config.json", "tokenizer.json"],
        tags=["sentence-transformers"],
    )
    shape = _sniff_repo_shape(info)
    assert shape == "sentence-transformers"


def test_sniff_recognizes_sentence_transformers_via_config_file():
    info = _fake_repo_info(
        siblings=["config.json", "sentence_transformers_config.json"],
        tags=[],
    )
    shape = _sniff_repo_shape(info)
    assert shape == "sentence-transformers"


def test_sniff_returns_unknown_for_unrecognized_repo():
    info = _fake_repo_info(
        siblings=["model.safetensors", "config.json"],
        tags=["text-classification"],
    )
    assert _sniff_repo_shape(info) == "unknown"


def test_resolve_gguf_requires_variant():
    """GGUF repos MUST specify @variant; no magic default."""
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["a-q4_k_m.gguf", "a-q5_k_m.gguf", "a-q8_0.gguf"],
            tags=["text-generation"],
        )
        r = HFResolver()
        with pytest.raises(ResolverError, match="variant.*required.*available"):
            r.resolve("hf://org/repo-gguf")


def test_resolve_gguf_variant_not_found_lists_available():
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["a-q4_k_m.gguf", "a-q5_k_m.gguf"],
            tags=["text-generation"],
        )
        r = HFResolver()
        with pytest.raises(ResolverError, match="variant.*q8_0.*not found"):
            r.resolve("hf://org/repo-gguf@q8_0")


def test_resolve_gguf_exact_variant_match():
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["qwen3-8b-q4_k_m.gguf"],
            tags=["text-generation"],
        )
        r = HFResolver()
        rm = r.resolve("hf://Qwen/Qwen3-8B-GGUF@q4_k_m")
        assert rm.manifest["modality"] == "chat/completion"
        assert rm.manifest["model_id"] == "qwen3-8b-gguf-q4-k-m"
        assert rm.manifest["hf_repo"] == "Qwen/Qwen3-8B-GGUF"
        assert rm.manifest["capabilities"]["gguf_file"] == "qwen3-8b-q4_k_m.gguf"
        assert "llama-cpp-python" in rm.manifest["pip_extras"]
        assert rm.backend_path.endswith(":LlamaCppModel")


def test_resolve_sentence_transformer_repo():
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["config.json", "sentence_transformers_config.json"],
            tags=["sentence-transformers"],
        )
        r = HFResolver()
        rm = r.resolve("hf://sentence-transformers/all-MiniLM-L6-v2")
        assert rm.manifest["modality"] == "embedding/text"
        assert rm.manifest["hf_repo"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert "sentence-transformers" in " ".join(rm.manifest["pip_extras"])
        assert rm.backend_path.endswith(":SentenceTransformerModel")


def test_resolve_rejects_unknown_scheme():
    r = HFResolver()
    with pytest.raises(ResolverError):
        r.resolve("civitai://something")


def test_resolve_rejects_non_hf_uri():
    r = HFResolver()
    with pytest.raises(ResolverError):
        r.resolve("not-a-uri")


def test_resolve_unrecognized_repo_shape_raises():
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["model.safetensors"],
            tags=["text-classification"],
        )
        r = HFResolver()
        with pytest.raises(ResolverError, match="cannot infer.*text-classification"):
            r.resolve("hf://org/weird-repo")


def test_tool_template_detection():
    """When tokenizer_config.json's chat_template mentions tools,
    capabilities.supports_tools = True."""
    from muse.core.resolvers_hf import _sniff_supports_tools
    # Fake chat_template with tool support
    tmpl = '{% if tools %}{{ tools | tojson }}{% endif %}{% for m in messages %}{{ m }}{% endfor %}'
    assert _sniff_supports_tools(tmpl) is True
    # Without
    tmpl2 = '{% for m in messages %}{{ m }}{% endfor %}'
    assert _sniff_supports_tools(tmpl2) is False
    # None -> False
    assert _sniff_supports_tools(None) is False


def test_gguf_variant_tag_normalizes_filename():
    """Variant `q4_k_m` should match filename `model-q4_k_m.gguf`."""
    from muse.core.resolvers_hf import _match_gguf_variant
    files = ["qwen3-8b-q4_k_m.gguf", "qwen3-8b-q5_k_m.gguf", "qwen3-8b-q8_0.gguf"]
    assert _match_gguf_variant(files, "q4_k_m") == "qwen3-8b-q4_k_m.gguf"
    assert _match_gguf_variant(files, "q5_k_m") == "qwen3-8b-q5_k_m.gguf"
    assert _match_gguf_variant(files, "q8_0") == "qwen3-8b-q8_0.gguf"
    # Case-insensitive match
    assert _match_gguf_variant(files, "Q4_K_M") == "qwen3-8b-q4_k_m.gguf"
    # No match -> None
    assert _match_gguf_variant(files, "q2_k") is None


def test_search_gguf_returns_variant_rows():
    """Each GGUF file in a matched repo becomes a separate SearchResult."""
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        api = MockApi.return_value
        api.list_models.return_value = [
            MagicMock(
                id="org/Qwen3-8B-GGUF",
                downloads=1000,
                tags=["text-generation"],
                siblings=[
                    MagicMock(rfilename="x-q4_k_m.gguf", size=4_500_000_000),
                    MagicMock(rfilename="x-q8_0.gguf", size=8_500_000_000),
                    MagicMock(rfilename="README.md", size=10_000),
                ],
            ),
        ]
        r = HFResolver()
        results = list(r.search("qwen3", modality="chat/completion"))
        assert len(results) == 2
        uris = {res.uri for res in results}
        assert "hf://org/Qwen3-8B-GGUF@q4_k_m" in uris
        assert "hf://org/Qwen3-8B-GGUF@q8_0" in uris


def test_search_embeddings_returns_repo_rows():
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        api = MockApi.return_value
        api.list_models.return_value = [
            MagicMock(
                id="sentence-transformers/all-MiniLM-L6-v2",
                downloads=50_000_000,
                tags=["sentence-transformers", "feature-extraction"],
                siblings=[MagicMock(rfilename="config.json", size=1000)],
            ),
        ]
        r = HFResolver()
        results = list(r.search("minilm", modality="embedding/text"))
        assert len(results) == 1
        assert results[0].uri == "hf://sentence-transformers/all-MiniLM-L6-v2"
        assert results[0].modality == "embedding/text"


def test_hf_resolver_registers_on_import():
    """Importing muse.core.resolvers_hf should register an HFResolver."""
    from muse.core import resolvers_hf  # noqa: F401
    from muse.core.resolvers import get_resolver
    r = get_resolver("hf://anything/anywhere")
    assert r.scheme == "hf"
```

- [ ] **Step 2: Run; expect ModuleNotFoundError**

- [ ] **Step 3: Implement `resolvers_hf.py`**

```python
# src/muse/core/resolvers_hf.py
"""HuggingFace Hub resolver.

URI shapes:
  hf://org/repo                  # sentence-transformers (embedding/text)
  hf://org/repo-GGUF@<variant>   # GGUF (chat/completion); variant required

Sniff logic (see `_sniff_repo_shape`):
  - any .gguf sibling          -> gguf
  - sentence-transformers tag  -> sentence-transformers
  - sentence_transformers_config.json sibling -> sentence-transformers
  - else                       -> unknown (raises on resolve)

Search:
  - modality="chat/completion": HfApi.list_models(filter="gguf") +
    enumerate each repo's .gguf files as separate variants.
  - modality="embedding/text": HfApi.list_models(filter="sentence-transformers")

Capability sniffing:
  - supports_tools: loads tokenizer_config.json's chat_template (when
    present) and regex-matches for `{% if tools %}` / `{{ tools` markers.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from muse.core.resolvers import (
    Resolver,
    ResolvedModel,
    ResolverError,
    SearchResult,
    parse_uri,
    register_resolver,
)


logger = logging.getLogger(__name__)

LLAMA_CPP_RUNTIME_PATH = (
    "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel"
)
SENTENCE_TRANSFORMER_RUNTIME_PATH = (
    "muse.modalities.embedding_text.runtimes.sentence_transformers:SentenceTransformerModel"
)

# pip_extras synthesized into the manifest; installed into per-model venv by `muse pull`.
LLAMA_CPP_PIP_EXTRAS = ("llama-cpp-python>=0.2.90",)
SENTENCE_TRANSFORMER_PIP_EXTRAS = (
    "torch>=2.1.0",
    "sentence-transformers>=2.2.0",
)


class HFResolver(Resolver):
    """Resolver for hf:// URIs."""

    scheme = "hf"

    def __init__(self) -> None:
        self._api = HfApi()

    def resolve(self, uri: str) -> ResolvedModel:
        scheme, repo_id, variant = parse_uri(uri)
        if scheme != "hf":
            raise ResolverError(f"HFResolver cannot resolve scheme {scheme!r}")

        info = self._api.repo_info(repo_id)
        shape = _sniff_repo_shape(info)

        if shape == "gguf":
            return self._resolve_gguf(repo_id, variant, info)
        if shape == "sentence-transformers":
            return self._resolve_sentence_transformer(repo_id, info)
        tags = getattr(info, "tags", None) or []
        raise ResolverError(
            f"cannot infer modality for {repo_id!r} "
            f"(no .gguf siblings, no sentence-transformers tag; tags={tags})"
        )

    def search(self, query: str, **filters) -> Iterable[SearchResult]:
        modality = filters.get("modality")
        sort = filters.get("sort", "downloads")
        limit = filters.get("limit", 20)

        if modality == "chat/completion":
            yield from self._search_gguf(query, sort=sort, limit=limit)
        elif modality == "embedding/text":
            yield from self._search_sentence_transformers(query, sort=sort, limit=limit)
        elif modality is None:
            # Default: search both and interleave. Caller can filter further.
            yield from self._search_gguf(query, sort=sort, limit=limit)
            yield from self._search_sentence_transformers(query, sort=sort, limit=limit)
        else:
            raise ResolverError(
                f"HFResolver.search does not support modality {modality!r}; "
                f"supported: chat/completion, embedding/text"
            )

    # --- GGUF branch ---

    def _resolve_gguf(self, repo_id: str, variant: str | None, info) -> ResolvedModel:
        gguf_files = [
            s.rfilename for s in info.siblings
            if s.rfilename.endswith(".gguf")
        ]
        if not gguf_files:
            raise ResolverError(f"no .gguf files in {repo_id}")
        if variant is None:
            variants = [_extract_variant(f) for f in gguf_files]
            raise ResolverError(
                f"variant required for GGUF repo {repo_id}; "
                f"available: {sorted(set(variants))}"
            )
        matched = _match_gguf_variant(gguf_files, variant)
        if matched is None:
            variants = [_extract_variant(f) for f in gguf_files]
            raise ResolverError(
                f"variant {variant!r} not found in {repo_id}; "
                f"available: {sorted(set(variants))}"
            )

        supports_tools = _try_sniff_tools_from_repo(self._api, repo_id)
        ctx_length = _try_sniff_context_length_from_repo(self._api, repo_id)

        model_id = _gguf_model_id(repo_id, variant)
        manifest = {
            "model_id": model_id,
            "modality": "chat/completion",
            "hf_repo": repo_id,
            "description": f"GGUF model: {repo_id} ({variant})",
            "license": _repo_license(info),
            "pip_extras": list(LLAMA_CPP_PIP_EXTRAS),
            "system_packages": [],
            "capabilities": {
                "gguf_file": matched,
                "supports_tools": supports_tools,
                **({"context_length": ctx_length} if ctx_length else {}),
            },
        }

        def _download(cache_root: Path) -> Path:
            # Download just the specific GGUF file + tokenizer_config.json
            # to save bandwidth; other GGUF quants in the repo are skipped.
            allow_patterns = [matched, "tokenizer*", "config.json", "*.md"]
            return Path(snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                cache_dir=str(cache_root) if cache_root else None,
            ))

        return ResolvedModel(
            manifest=manifest,
            backend_path=LLAMA_CPP_RUNTIME_PATH,
            download=_download,
        )

    def _search_gguf(self, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
        repos = self._api.list_models(
            search=query, filter="gguf", sort=sort, limit=limit,
        )
        for repo in repos:
            # Some list_models results have no .siblings; fall back to repo_info
            siblings = getattr(repo, "siblings", None) or []
            if not siblings:
                try:
                    info = self._api.repo_info(repo.id)
                    siblings = info.siblings
                except Exception:
                    continue
            for s in siblings:
                if not s.rfilename.endswith(".gguf"):
                    continue
                variant = _extract_variant(s.rfilename)
                yield SearchResult(
                    uri=f"hf://{repo.id}@{variant}",
                    model_id=_gguf_model_id(repo.id, variant),
                    modality="chat/completion",
                    size_gb=(s.size / 1e9) if getattr(s, "size", None) else None,
                    downloads=getattr(repo, "downloads", None),
                    license=None,
                    description=f"{repo.id} ({variant})",
                )

    # --- Sentence-Transformers branch ---

    def _resolve_sentence_transformer(self, repo_id: str, info) -> ResolvedModel:
        # Optionally sniff dimensions from 1_Pooling/config.json or
        # sentence_bert_config.json. Not critical; users can override.
        manifest = {
            "model_id": _sentence_transformer_model_id(repo_id),
            "modality": "embedding/text",
            "hf_repo": repo_id,
            "description": f"Sentence-Transformers: {repo_id}",
            "license": _repo_license(info),
            "pip_extras": list(SENTENCE_TRANSFORMER_PIP_EXTRAS),
            "system_packages": [],
            "capabilities": {},
        }

        def _download(cache_root: Path) -> Path:
            return Path(snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_root) if cache_root else None,
            ))

        return ResolvedModel(
            manifest=manifest,
            backend_path=SENTENCE_TRANSFORMER_RUNTIME_PATH,
            download=_download,
        )

    def _search_sentence_transformers(self, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
        repos = self._api.list_models(
            search=query, filter="sentence-transformers",
            sort=sort, limit=limit,
        )
        for repo in repos:
            yield SearchResult(
                uri=f"hf://{repo.id}",
                model_id=_sentence_transformer_model_id(repo.id),
                modality="embedding/text",
                size_gb=None,
                downloads=getattr(repo, "downloads", None),
                license=None,
                description=repo.id,
            )


# --- sniff helpers (module-level, pytest-friendly) ---

def _sniff_repo_shape(info) -> str:
    """Return one of: 'gguf' | 'sentence-transformers' | 'unknown'."""
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    if any(f.endswith(".gguf") for f in siblings):
        return "gguf"
    if "sentence-transformers" in tags:
        return "sentence-transformers"
    if any(Path(f).name == "sentence_transformers_config.json" for f in siblings):
        return "sentence-transformers"
    return "unknown"


_VARIANT_RE = re.compile(r"(q\d+_[a-z0-9_]+|iq\d+_[a-z0-9]+|f16|bf16|f32)", re.IGNORECASE)


def _extract_variant(gguf_filename: str) -> str:
    """Extract a quant tag like `q4_k_m` from e.g. `qwen3-8b-q4_k_m.gguf`."""
    stem = Path(gguf_filename).stem
    m = _VARIANT_RE.search(stem)
    return (m.group(1).lower() if m else stem).replace(".", "_")


def _match_gguf_variant(files: list[str], variant: str) -> str | None:
    """Find the file whose quant tag matches `variant` (case-insensitive)."""
    norm = variant.lower()
    for f in files:
        if _extract_variant(f) == norm:
            return f
    return None


def _gguf_model_id(repo_id: str, variant: str) -> str:
    """Synthesize a model_id like 'qwen3-8b-gguf-q4-k-m'."""
    base = repo_id.split("/", 1)[-1].lower()
    if not base.endswith("-gguf"):
        base = f"{base}-gguf"
    return f"{base}-{variant.lower().replace('_', '-')}"


def _sentence_transformer_model_id(repo_id: str) -> str:
    """Synthesize a model_id from the repo name (lowercased, slash -> dash)."""
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _try_sniff_tools_from_repo(api: HfApi, repo_id: str) -> bool | None:
    """Try to read tokenizer_config.json and check for tool-calling template.

    Returns True / False when the file is present; None when it isn't.
    Any network / parse error returns None silently.
    """
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer_config.json",
            # huggingface_hub uses its own cache; no local_dir needed.
        )
    except Exception:
        return None
    try:
        cfg = json.loads(Path(path).read_text())
    except Exception:
        return None
    return _sniff_supports_tools(cfg.get("chat_template"))


def _sniff_supports_tools(chat_template: str | None) -> bool:
    if not chat_template or not isinstance(chat_template, str):
        return False
    # Common markers: `{% if tools %}`, `{{ tools`, `tool_calls`
    return bool(re.search(r"(\bif\s+tools\b|\{\{\s*tools|tool_calls)", chat_template))


def _try_sniff_context_length_from_repo(api: HfApi, repo_id: str) -> int | None:
    """Best-effort: read config.json's `max_position_embeddings`.

    GGUF files carry their own context length in header metadata, which
    llama-cpp-python respects at load time. This sniff is just for
    display in /v1/models; runtime truth is in the GGUF.
    """
    try:
        path = hf_hub_download(repo_id=repo_id, filename="config.json")
        cfg = json.loads(Path(path).read_text())
        return int(cfg.get("max_position_embeddings") or 0) or None
    except Exception:
        return None


# Register on import so `from muse.core import resolvers_hf` is enough.
register_resolver(HFResolver())
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/core/test_resolvers_hf.py -v
```

Expected: 14/14 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/core/resolvers_hf.py tests/core/test_resolvers_hf.py
git commit -m "feat(resolvers): HFResolver for GGUF + sentence-transformers

Sniffs repo shape from siblings + tags:
  - any .gguf file  -> chat/completion / LlamaCppModel
  - sentence-transformers tag or config file -> embedding/text / SentenceTransformerModel
GGUF @variant required (no magic default). Search implemented for both
modalities; each GGUF variant yields a separate SearchResult row.
Sniffs tokenizer_config.json chat_template for tool-calling markers
(capabilities.supports_tools).

Registers HFResolver on import via muse.core.resolvers.register_resolver."
```

---

## Part E: SentenceTransformerModel runtime

### Task E1: Generic sentence-transformers runtime

**Files:**
- Create: `src/muse/modalities/embedding_text/runtimes/__init__.py`
- Create: `src/muse/modalities/embedding_text/runtimes/sentence_transformers.py`
- Create: `tests/modalities/embedding_text/runtimes/__init__.py`
- Create: `tests/modalities/embedding_text/runtimes/test_sentence_transformers.py`

- [ ] **Step 1: Write tests**

Pattern matches `tests/models/test_all_minilm_l6_v2.py` but exercises the generic
runtime instead of the bundled script. Key tests:

```python
# tests/modalities/embedding_text/runtimes/test_sentence_transformers.py
"""Tests for SentenceTransformerModel (generic ST runtime; mocks the library)."""
from unittest.mock import MagicMock, patch

import numpy as np

from muse.modalities.embedding_text.protocol import EmbeddingResult


def _mock_st_model(dim=384, encode_return=None):
    m = MagicMock()
    if encode_return is None:
        m.encode.return_value = np.zeros((1, dim), dtype=np.float32)
    else:
        m.encode.return_value = np.asarray(encode_return, dtype=np.float32)
    m.get_sentence_embedding_dimension.return_value = dim
    m.max_seq_length = 512
    def _tok(texts):
        n = len(texts)
        return {
            "input_ids": np.ones((n, 7), dtype=np.int64),
            "attention_mask": np.ones((n, 7), dtype=np.int64),
        }
    m.tokenize.side_effect = _tok
    return m


def test_auto_detects_dimensions_from_model():
    with patch("muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_st_model(dim=768)
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        m = SentenceTransformerModel(
            model_id="some-model",
            hf_repo="some/model", local_dir="/fake",
        )
        assert m.dimensions == 768


def test_embed_returns_embedding_result():
    with patch("muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer") as mock_cls:
        fake = _mock_st_model(dim=384, encode_return=np.zeros((1, 384), dtype=np.float32))
        mock_cls.return_value = fake
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        m = SentenceTransformerModel(
            model_id="all-minilm-l6-v2",
            hf_repo="sentence-transformers/all-MiniLM-L6-v2", local_dir="/fake",
        )
        r = m.embed("hello")
        assert isinstance(r, EmbeddingResult)
        assert r.dimensions == 384
        assert r.model_id == "all-minilm-l6-v2"
        assert len(r.embeddings) == 1


def test_trust_remote_code_forwarded():
    """For repos that require it (Qwen3, Nomic, some Instruct models)."""
    with patch("muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_st_model()
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        SentenceTransformerModel(
            model_id="x", hf_repo="x", local_dir="/fake",
            trust_remote_code=True,
        )
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["trust_remote_code"] is True


def test_prefers_local_dir_over_hf_repo():
    with patch("muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_st_model()
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        SentenceTransformerModel(
            model_id="x", hf_repo="remote/x", local_dir="/real/local/path",
        )
        assert mock_cls.call_args.args[0] == "/real/local/path"


def test_matryoshka_truncation_renormalizes():
    with patch("muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer") as mock_cls:
        raw = np.array([[3.0, 4.0] + [0.0] * 1022], dtype=np.float32)
        mock_cls.return_value = _mock_st_model(dim=1024, encode_return=raw)
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        m = SentenceTransformerModel(model_id="x", hf_repo="x", local_dir="/fake")
        r = m.embed("hi", dimensions=2)
        assert r.dimensions == 2
        # (3,4) normalized then truncated to (0.6, 0.8)
        assert abs(r.embeddings[0][0] - 0.6) < 1e-5
        assert abs(r.embeddings[0][1] - 0.8) < 1e-5


def test_counts_tokens_from_attention_mask():
    with patch("muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_st_model(dim=384, encode_return=np.zeros((2, 384), dtype=np.float32))
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        m = SentenceTransformerModel(model_id="x", hf_repo="x", local_dir="/fake")
        r = m.embed(["a", "b"])
        assert r.prompt_tokens == 14  # 2 texts * 7 mask positions


def test_accepts_unknown_kwargs():
    """Future MANIFEST kwargs absorbed by **_."""
    with patch("muse.modalities.embedding_text.runtimes.sentence_transformers.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _mock_st_model()
        from muse.modalities.embedding_text.runtimes.sentence_transformers import (
            SentenceTransformerModel,
        )
        SentenceTransformerModel(
            model_id="x", hf_repo="x", local_dir="/fake",
            future_kwarg="absorbed", device="cpu",
        )
```

- [ ] **Step 2: Run; expect ModuleNotFoundError**

- [ ] **Step 3: Implement `runtimes/sentence_transformers.py`**

Structure mirrors the existing `muse/models/all_minilm_l6_v2.py` closely, plus:
  - `model_id` is a constructor kwarg (not a class attribute) because one runtime serves many models
  - `dimensions` auto-detected via `model.get_sentence_embedding_dimension()` at load
  - `trust_remote_code` kwarg passed through (for Qwen3-Embedding, Nomic, etc.)
  - All existing matryoshka + token-counting logic preserved

Full code (~180 lines) follows the `all_minilm_l6_v2.py` template; key
differences inlined in Step 3 of the subagent task. Skipped here for
brevity since this is an existing proven pattern.

- [ ] **Step 4: Run tests**; expect 7/7 pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(embedding/text): SentenceTransformerModel generic runtime

Serves any sentence-transformers HF repo via one class. model_id is a
constructor kwarg (one runtime, many models). dimensions auto-detected
via get_sentence_embedding_dimension(). trust_remote_code passthrough
for Qwen3 / Nomic. Matryoshka + token-count logic matches the existing
scripted backends so the resolver-pulled model behaves identically to
the bundled minilm.py / qwen3_embedding.py scripts."
```

---

## Part F: Catalog integration

### Task F1: Persist resolver-synthesized manifests; merge into known_models()

**Files:**
- Modify: `src/muse/core/catalog.py`
- Modify: `tests/core/test_catalog.py`

- [ ] **Step 1: Write tests for new catalog behavior**

```python
# Added to tests/core/test_catalog.py

def test_known_models_merges_resolver_persisted_entries(tmp_catalog, monkeypatch):
    """Catalog entries with a `manifest` field show up in known_models()."""
    import json
    from muse.core.catalog import _catalog_path, _reset_known_models_cache, known_models
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "qwen3-8b-gguf-q4-k-m": {
            "pulled_at": "2026-04-14T00:00:00Z",
            "hf_repo": "Qwen/Qwen3-8B-GGUF",
            "local_dir": "/fake/weights",
            "venv_path": str(tmp_catalog / "venvs" / "qwen3-8b-gguf-q4-k-m"),
            "python_path": str(tmp_catalog / "venvs" / "qwen3-8b-gguf-q4-k-m" / "bin" / "python"),
            "enabled": True,
            "source": "hf://Qwen/Qwen3-8B-GGUF@q4_k_m",
            "manifest": {
                "model_id": "qwen3-8b-gguf-q4-k-m",
                "modality": "chat/completion",
                "hf_repo": "Qwen/Qwen3-8B-GGUF",
                "backend_path": "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel",
                "description": "Qwen3 8B Q4_K_M",
                "capabilities": {"gguf_file": "qwen3-8b-q4_k_m.gguf", "supports_tools": True},
                "pip_extras": ["llama-cpp-python"],
            },
        },
    }))
    _reset_known_models_cache()
    entries = known_models()
    assert "qwen3-8b-gguf-q4-k-m" in entries
    e = entries["qwen3-8b-gguf-q4-k-m"]
    assert e.modality == "chat/completion"
    assert e.backend_path.endswith(":LlamaCppModel")
    assert e.extra["gguf_file"] == "qwen3-8b-q4_k_m.gguf"


def test_bundled_scripts_win_on_collision_with_catalog_manifest(tmp_catalog):
    """If a script in src/muse/models/ shares a model_id with a catalog.manifest
    entry, the script version wins."""
    import json
    from muse.core.catalog import _catalog_path, _reset_known_models_cache, known_models
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Inject a catalog entry pretending to be kokoro-82m
    p.write_text(json.dumps({
        "kokoro-82m": {
            "pulled_at": "2026-04-14T00:00:00Z",
            "hf_repo": "malicious/fake",
            "local_dir": "/impostor",
            "venv_path": "/v",
            "python_path": "/v/bin/python",
            "enabled": True,
            "source": "hf://malicious/fake",
            "manifest": {
                "model_id": "kokoro-82m",
                "modality": "audio/speech",
                "hf_repo": "malicious/fake",
                "backend_path": "muse.models.kokoro_82m:Model",
            },
        },
    }))
    _reset_known_models_cache()
    entries = known_models()
    # Bundled wins
    assert entries["kokoro-82m"].hf_repo == "hexgrad/Kokoro-82M"


def test_get_manifest_returns_persisted_manifest_for_resolver_entry(tmp_catalog):
    """get_manifest() returns the catalog-persisted manifest for resolver-pulled models."""
    import json
    from muse.core.catalog import (
        _catalog_path, _reset_known_models_cache, get_manifest,
    )
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "q3-gguf-q4": {
            "pulled_at": "2026-04-14T00:00:00Z",
            "hf_repo": "Qwen/Qwen3-8B-GGUF",
            "local_dir": "/fake",
            "venv_path": "/v",
            "python_path": "/v/bin/python",
            "enabled": True,
            "source": "hf://Qwen/Qwen3-8B-GGUF@q4_k_m",
            "manifest": {
                "model_id": "q3-gguf-q4",
                "modality": "chat/completion",
                "hf_repo": "Qwen/Qwen3-8B-GGUF",
                "backend_path": "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel",
                "capabilities": {"gguf_file": "q4.gguf"},
            },
        },
    }))
    _reset_known_models_cache()
    m = get_manifest("q3-gguf-q4")
    assert m["model_id"] == "q3-gguf-q4"
    assert m["capabilities"]["gguf_file"] == "q4.gguf"
```

- [ ] **Step 2: Update `known_models()` in `catalog.py`**

Replace the body of `known_models()`:

```python
def known_models() -> dict[str, CatalogEntry]:
    global _known_models_cache
    if _known_models_cache is None:
        # Start with discovery results from muse/models/*.py
        discovered = discover_models(_model_dirs())
        entries = {
            model_id: _manifest_to_catalog_entry(d)
            for model_id, d in discovered.items()
        }
        # Overlay catalog-persisted manifests (resolver-pulled models)
        catalog = _read_catalog()
        for model_id, entry_data in catalog.items():
            if model_id in entries:
                continue  # bundled wins
            manifest = entry_data.get("manifest")
            if not manifest:
                continue  # legacy script-based entry; skip
            entries[model_id] = _persisted_manifest_to_catalog_entry(manifest)
        _known_models_cache = entries
    return _known_models_cache


def _persisted_manifest_to_catalog_entry(manifest: dict) -> CatalogEntry:
    """Map a manifest dict (persisted by the resolver path) to CatalogEntry."""
    return CatalogEntry(
        model_id=manifest["model_id"],
        modality=manifest["modality"],
        backend_path=manifest["backend_path"],
        hf_repo=manifest["hf_repo"],
        description=manifest.get("description", ""),
        pip_extras=tuple(manifest.get("pip_extras", ())),
        system_packages=tuple(manifest.get("system_packages", ())),
        extra=dict(manifest.get("capabilities", {})),
    )
```

- [ ] **Step 3: Update `get_manifest()` to return persisted manifest for resolver entries**

```python
def get_manifest(model_id: str) -> dict:
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    # Prefer the catalog-persisted manifest if present (resolver-pulled)
    catalog = _read_catalog()
    persisted = catalog.get(model_id, {}).get("manifest")
    if persisted:
        return dict(persisted)
    # Else read from the model script module
    entry = catalog_known[model_id]
    module_path, _ = entry.backend_path.split(":", 1)
    module = importlib.import_module(module_path)
    return dict(getattr(module, "MANIFEST", {}))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/core/test_catalog.py -v
```

Expected: all prior tests pass + 3 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/core/catalog.py tests/core/test_catalog.py
git commit -m "feat(catalog): merge resolver-persisted manifests into known_models()

known_models() now walks two sources:
  1. discover_models() over src/muse/models/*.py (bundled)
  2. catalog.json entries with a `manifest` field (resolver-pulled)

Bundled scripts win on model_id collision; a user who pulls
hf://malicious/fake pretending to be kokoro-82m gets shadowed by the
bundled one. get_manifest() prefers the persisted manifest for
resolver-pulled models, falls back to the script module for bundled.

No changes to existing bundled-model flow."
```

---

### Task F2: Extend `pull()` to accept resolver URIs

**Files:**
- Modify: `src/muse/core/catalog.py`
- Modify: `tests/core/test_catalog.py`

- [ ] **Step 1: Write tests**

```python
# tests/core/test_catalog.py additions

def test_pull_dispatches_to_resolver_for_uri(tmp_catalog):
    from muse.core.catalog import pull, _read_catalog, _reset_known_models_cache
    from muse.core.resolvers import (
        Resolver, ResolvedModel, register_resolver, _reset_registry_for_tests,
    )

    class _FakeResolver(Resolver):
        scheme = "fake"
        def resolve(self, uri):
            return ResolvedModel(
                manifest={
                    "model_id": "pulled-from-resolver",
                    "modality": "chat/completion",
                    "hf_repo": "fake/repo",
                    "backend_path": "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel",
                    "pip_extras": ["llama-cpp-python"],
                    "capabilities": {"gguf_file": "x.gguf"},
                },
                backend_path="muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel",
                download=lambda cache: cache / "weights",
            )
        def search(self, q, **k):
            return []

    _reset_registry_for_tests()
    register_resolver(_FakeResolver())

    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("fake://some/repo@variant")

    catalog = _read_catalog()
    assert "pulled-from-resolver" in catalog
    entry = catalog["pulled-from-resolver"]
    assert entry["source"] == "fake://some/repo@variant"
    assert entry["manifest"]["modality"] == "chat/completion"


def test_pull_bare_id_still_uses_discovery_path(tmp_catalog):
    """Regression: non-URI pull still goes through known_models()."""
    from muse.core.catalog import pull, is_pulled, _reset_known_models_cache
    _reset_known_models_cache()
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("kokoro-82m")
    assert is_pulled("kokoro-82m")
```

- [ ] **Step 2: Refactor `pull()` to dispatch**

```python
def pull(identifier: str) -> None:
    """Pull a model. `identifier` is either a bare model_id (bundled script)
    or a resolver URI (hf://...).
    """
    if "://" in identifier:
        _pull_via_resolver(identifier)
    else:
        _pull_bundled(identifier)


def _pull_bundled(model_id: str) -> None:
    # (existing pull() body goes here, rebound to _pull_bundled)
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    entry = catalog_known[model_id]
    # ... existing venv + install + snapshot_download flow ...


def _pull_via_resolver(uri: str) -> None:
    from muse.core.resolvers import resolve
    resolved = resolve(uri)
    model_id = resolved.manifest["model_id"]

    venvs_root = _catalog_dir() / "venvs"
    venv_path = venvs_root / model_id

    if not venv_path.exists():
        create_venv(venv_path)
    install_into_venv(venv_path, ["-e", f"{_muse_repo_root()}[server]"])

    pip_extras = resolved.manifest.get("pip_extras", [])
    if pip_extras:
        install_into_venv(venv_path, list(pip_extras))

    system_packages = resolved.manifest.get("system_packages", [])
    if system_packages:
        missing = check_system_packages(list(system_packages))
        if missing:
            logger.warning(
                "model %s needs system packages not found on PATH: %s",
                model_id, missing,
            )

    local_dir = resolved.download(_catalog_dir() / "weights")

    # Ensure the manifest carries backend_path (resolver should set, but be robust)
    manifest = dict(resolved.manifest)
    manifest.setdefault("backend_path", resolved.backend_path)

    catalog = _read_catalog()
    catalog[model_id] = {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "hf_repo": manifest["hf_repo"],
        "local_dir": str(local_dir),
        "venv_path": str(venv_path),
        "python_path": str(venv_python(venv_path)),
        "enabled": True,
        "source": uri,
        "manifest": manifest,
    }
    _write_catalog(catalog)
    _reset_known_models_cache()
```

Also update `load_backend()` to read `backend_path` from the persisted manifest
(not just CatalogEntry) when present:

```python
def load_backend(model_id: str, **kwargs) -> Any:
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    if not is_pulled(model_id):
        raise RuntimeError(f"model {model_id!r} not pulled; run `muse pull {model_id}`")
    entry = catalog_known[model_id]
    module_path, class_name = entry.backend_path.split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    catalog = _read_catalog()
    local_dir = catalog[model_id]["local_dir"]
    manifest = catalog[model_id].get("manifest", {})
    # Merge manifest capabilities (e.g. gguf_file, chat_template) into kwargs
    merged = {**manifest.get("capabilities", {}), **kwargs}
    return cls(
        model_id=model_id,
        hf_repo=entry.hf_repo,
        local_dir=local_dir,
        **merged,
    )
```

- [ ] **Step 3: Run tests**

Expected: new tests pass; all previous tests still pass.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(catalog): pull() dispatches URI -> resolver, bare id -> discovery

`muse pull hf://Qwen/Qwen3-8B-GGUF@q4_k_m` now works: the URI goes
through the resolver, produces a ResolvedModel, muse creates the venv,
installs muse[server] + manifest.pip_extras, calls
resolved.download(weights_cache) to fetch weights, persists the
synthesized manifest plus `source: <uri>` into catalog.json.

`muse pull kokoro-82m` unchanged: bundled-script path with existing
semantics.

load_backend() now merges manifest.capabilities (gguf_file,
chat_template, etc.) into constructor kwargs so LlamaCppModel gets the
GGUF filename from the manifest without the worker having to know."
```

---

## Part G: muse search CLI

### Task G1: `muse search` subcommand

**Files:**
- Create: `src/muse/cli_impl/search.py`
- Create: `tests/cli_impl/test_search.py`
- Modify: `src/muse/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write tests**

```python
# tests/cli_impl/test_search.py
"""Tests for `muse search` via run_search()."""
from unittest.mock import MagicMock, patch

from muse.cli_impl.search import run_search
from muse.core.resolvers import SearchResult


def test_run_search_filters_by_modality(capsys):
    results = [
        SearchResult(
            uri="hf://Qwen/Qwen3-8B-GGUF@q4_k_m",
            model_id="qwen3-8b-gguf-q4-k-m",
            modality="chat/completion",
            size_gb=4.5, downloads=1000,
            license="apache-2.0",
            description="Qwen3 8B Q4_K_M",
        ),
    ]
    with patch("muse.cli_impl.search.search", return_value=results) as mock_search:
        run_search(query="qwen3", modality="chat/completion", limit=10, sort="downloads")
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["modality"] == "chat/completion"
        assert call_kwargs["limit"] == 10
        assert call_kwargs["sort"] == "downloads"

    out = capsys.readouterr().out
    assert "hf://Qwen/Qwen3-8B-GGUF@q4_k_m" in out
    assert "4.5 GB" in out


def test_run_search_emits_helpful_message_when_no_results(capsys):
    with patch("muse.cli_impl.search.search", return_value=[]):
        rc = run_search(query="bogus", modality="chat/completion", limit=10, sort="downloads")
    out = capsys.readouterr().out
    assert rc == 0
    assert "no results" in out.lower()


def test_run_search_size_filter_client_side(capsys):
    """--max-size-gb filters post-hoc (resolver returns everything)."""
    results = [
        SearchResult(uri="hf://a@q4", model_id="a-q4", modality="chat/completion",
                     size_gb=4.5, downloads=1, license=None, description=""),
        SearchResult(uri="hf://b@q8", model_id="b-q8", modality="chat/completion",
                     size_gb=12.0, downloads=1, license=None, description=""),
    ]
    with patch("muse.cli_impl.search.search", return_value=results):
        run_search(query="x", modality="chat/completion", limit=10,
                   sort="downloads", max_size_gb=10.0)
    out = capsys.readouterr().out
    assert "hf://a@q4" in out
    assert "hf://b@q8" not in out
```

- [ ] **Step 2: Implement `run_search`**

```python
# src/muse/cli_impl/search.py
"""`muse search` implementation. Thin wrapper over resolvers.search."""
from __future__ import annotations

import logging
import sys

from muse.core.resolvers import search, ResolverError


logger = logging.getLogger(__name__)


def run_search(
    *,
    query: str,
    modality: str | None = None,
    limit: int = 20,
    sort: str = "downloads",
    max_size_gb: float | None = None,
    backend: str | None = None,
) -> int:
    """Query resolver(s) for candidate models; print an aligned table."""
    try:
        results = list(search(
            query,
            backend=backend,
            modality=modality,
            limit=limit,
            sort=sort,
        ))
    except ResolverError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if max_size_gb is not None:
        results = [r for r in results if r.size_gb is None or r.size_gb <= max_size_gb]

    if not results:
        print("no results")
        return 0

    for r in results:
        size = f"{r.size_gb:.1f} GB" if r.size_gb else "?"
        downloads = f"{r.downloads:,}" if r.downloads else "?"
        lic = r.license or ""
        desc = r.description or ""
        print(f"  {r.uri:55s}  {size:>9s}  dl={downloads:>12s}  {lic:15s}  {desc}")
    return 0
```

- [ ] **Step 3: Wire `muse search` in `cli.py`**

Add a subparser: `muse search <query> [--modality M] [--limit N] [--sort S] [--max-size-gb F]`.

```python
# In cli.py's _build_parser or equivalent
search_p = subparsers.add_parser("search", help="search HuggingFace for pullable models")
search_p.add_argument("query", help="search query")
search_p.add_argument("--modality", choices=["chat/completion", "embedding/text"], default=None)
search_p.add_argument("--limit", type=int, default=20)
search_p.add_argument("--sort", choices=["downloads", "lastModified", "likes"], default="downloads")
search_p.add_argument("--max-size-gb", type=float, default=None)
search_p.set_defaults(func=_cmd_search)


def _cmd_search(args):
    # Lazy import so `muse --help` doesn't load huggingface_hub etc.
    from muse.cli_impl.search import run_search
    import muse.core.resolvers_hf  # noqa: F401  -- register HFResolver
    return run_search(
        query=args.query, modality=args.modality,
        limit=args.limit, sort=args.sort,
        max_size_gb=args.max_size_gb,
    )
```

Also extend `muse pull <arg>` to accept URIs (argparse already does; the
dispatch happens inside `catalog.pull()`):

```python
# In cli.py pull subcommand (if not already done)
pull_p.add_argument("model_id", help="bundled model_id OR resolver URI like hf://org/repo@variant")
```

Update `_cmd_pull`:

```python
def _cmd_pull(args):
    from muse.core.catalog import pull
    # Register HFResolver if the arg looks like a URI. Always safe to
    # register; resolvers_hf import is cheap.
    if "://" in args.model_id:
        import muse.core.resolvers_hf  # noqa: F401
    pull(args.model_id)
    return 0
```

- [ ] **Step 4: Add CLI integration tests**

Extend `tests/test_cli.py` with:

```python
def test_cli_search_invocation(monkeypatch):
    """`muse search qwen3 --modality chat/completion` calls run_search."""
    import subprocess
    from unittest.mock import patch
    from muse.core.resolvers import SearchResult

    with patch("muse.cli_impl.search.search") as mock_search:
        mock_search.return_value = [
            SearchResult(
                uri="hf://Qwen/Qwen3-8B-GGUF@q4_k_m",
                model_id="qwen3-8b-gguf-q4-k-m",
                modality="chat/completion",
                size_gb=4.5, downloads=1000,
                license="apache-2.0",
                description="",
            ),
        ]
        import sys
        from muse.cli import main
        argv = ["muse", "search", "qwen3", "--modality", "chat/completion"]
        monkeypatch.setattr(sys, "argv", argv)
        rc = main()
        assert rc == 0

def test_cli_pull_accepts_hf_uri(monkeypatch):
    """`muse pull hf://...` routes through the resolver."""
    # Full integration test deferred to test_catalog; here just verify
    # the CLI doesn't crash on URI input (mocks resolver).
    ...  # full fixture + mocks per existing test_cli.py patterns
```

- [ ] **Step 5: Run all tests + commit**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -5
git commit -m "feat(cli): muse search + muse pull hf://... URI support

muse search <query> --modality chat/completion --sort downloads --max-size-gb 10
prints a table of HF candidates. muse pull now accepts either a bare
model_id (bundled script) or a resolver URI like
hf://Qwen/Qwen3-8B-GGUF@q4_k_m.

Lazy import of resolvers_hf so `muse --help` stays instant."
```

---

## Part H: Worker + E2E smoke

### Task H1: Verify chat/completion auto-mounts in the worker

Discovery already scans `src/muse/modalities/` for modality packages (Task E1
in the prior plan). The new `chat_completion` package exports `MODALITY` and
`build_router`, so no code changes needed. Verify.

- [ ] **Step 1: Run existing worker test with chat/completion registered**

```bash
pytest tests/cli_impl/test_worker.py::test_worker_mounts_all_modality_routers -v
```

The test that walks `app.routes` should now include `/v1/chat/completions`.
Update the test's assertion set:

```python
def test_worker_mounts_all_modality_routers(mock_uvicorn):
    run_worker(host="127.0.0.1", port=9999, models=[], device="cpu")
    app = mock_uvicorn.run.call_args.args[0]
    paths = "\n".join({getattr(r, "path", "") for r in app.routes})
    assert "/v1/audio/speech" in paths
    assert "/v1/embeddings" in paths
    assert "/v1/images/generations" in paths
    assert "/v1/chat/completions" in paths  # NEW
```

- [ ] **Step 2: Add E2E smoke for chat/completion**

```python
# tests/cli_impl/test_e2e_supervisor.py (new test, marked slow)

@pytest.mark.slow
def test_e2e_chat_completion_routes_through_gateway(tmp_path, monkeypatch):
    """Supervisor + worker + FakeChatModel injected into registry via a
    test-only model script."""
    # Write a fake model script into a temp dir
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "fake_chat.py").write_text('''
MANIFEST = {
    "model_id": "fake-chat-e2e",
    "modality": "chat/completion",
    "hf_repo": "fake/repo",
    "pip_extras": [],
}

class Model:
    model_id = "fake-chat-e2e"
    def __init__(self, **_): ...
    def chat(self, messages, **kwargs):
        from muse.modalities.chat_completion.protocol import ChatChoice, ChatResult
        return ChatResult(
            id="x", model_id=self.model_id, created=0,
            choices=[ChatChoice(index=0, message={"role":"assistant","content":"hi"}, finish_reason="stop")],
            usage={"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},
        )
    def chat_stream(self, messages, **kwargs):
        return iter([])
''')
    monkeypatch.setenv("MUSE_MODELS_DIR", str(models_dir))
    # ... (follow existing test_e2e_supervisor.py pattern: fake pull, spawn
    # supervisor, hit /v1/chat/completions, assert OpenAI-shape body)
```

- [ ] **Step 3: Commit**

```bash
git commit -m "test: chat/completion auto-mounts + E2E supervisor smoke"
```

---

## Part I: Docs + migration

### Task I1: Update docs

**Files:**
- Modify: `docs/MODEL_SCRIPTS.md`: add "Resolver-pullable models" section
- Create: `docs/RESOLVERS.md`: URI scheme reference + how to add a resolver
- Create: `docs/CHAT_COMPLETION.md`: OpenAI-shape endpoints, tool support, quirks
- Modify: `CLAUDE.md`: mention chat/completion modality + resolvers
- Modify: `README.md`: add chat completion example, `muse search` example

No em-dashes (U+2014). Use colons, periods, parens.

Key content per file:

**docs/RESOLVERS.md** (~60 lines)
  - URI format `scheme://ref[@variant]`
  - Registered resolvers: `hf` (see `muse/core/resolvers_hf.py`)
  - How to write a new resolver: subclass `Resolver`, implement
    `resolve()` + `search()`, call `register_resolver(instance)` at
    import time, expose via `$MUSE_RESOLVERS_DIR` (future work) or import
    from a muse plugin package.

**docs/CHAT_COMPLETION.md** (~100 lines)
  - Endpoint: POST /v1/chat/completions (OpenAI shape)
  - Request fields supported (messages, tools, tool_choice, response_format, stream, temperature, etc.)
  - Streaming: SSE with OpenAI ChatCompletionChunk format + [DONE] sentinel
  - Tool calling: pass through to backend; quality depends on model
  - What's NOT supported v1: vision input, reasoning_content, n>1
  - Python client: `from muse.modalities.chat_completion import ChatClient`
  - Example using `openai` SDK pointed at muse

**Update README.md:**
  - Add chat/completion to the modality list
  - New section: "Pulling from HuggingFace"
    ```
    muse search qwen3 --modality chat/completion --max-size-gb 10
    muse pull hf://Qwen/Qwen3-8B-GGUF@q4_k_m
    muse serve
    curl http://localhost:8000/v1/chat/completions ...
    ```

- [ ] **Step 1: Write docs**
- [ ] **Step 2: Commit**

```bash
git commit -m "docs: RESOLVERS.md + CHAT_COMPLETION.md + CLAUDE/README updates"
```

---

### Task I2: Migrate minilm + qwen3_embedding scripts to resolver (gated on verification)

**Only run after:**
  - User has successfully pulled `hf://sentence-transformers/all-MiniLM-L6-v2` via the resolver
  - `/v1/embeddings` serves it correctly end-to-end
  - `/v1/models` shows the resolver-pulled model alongside bundled ones

**Files:**
- Delete: `src/muse/models/all_minilm_l6_v2.py`
- Delete: `src/muse/models/qwen3_embedding_0_6b.py`
- Delete: `tests/models/test_all_minilm_l6_v2.py`
- Delete: `tests/models/test_qwen3_embedding_0_6b.py`
- Keep: `src/muse/models/nv_embed_v2.py` (uses AutoModel directly, NOT sentence-transformers; not resolver-addressable)
- Keep: `tests/models/test_nv_embed_v2.py`

- [ ] **Step 1: Verify end-to-end against a real HF pull**

Use `muse pull hf://sentence-transformers/all-MiniLM-L6-v2` on frodo, confirm
`muse models list` shows it, hit `/v1/embeddings` with it.

- [ ] **Step 2: Delete the four files**

```bash
git rm src/muse/models/all_minilm_l6_v2.py \
       src/muse/models/qwen3_embedding_0_6b.py \
       tests/models/test_all_minilm_l6_v2.py \
       tests/models/test_qwen3_embedding_0_6b.py
```

- [ ] **Step 3: Update `tests/core/test_catalog.py`**

The test `test_known_models_seeded_with_required_entries` asserts
`"all-minilm-l6-v2" in catalog` and `"qwen3-embedding-0.6b" in catalog`.
Remove those two assertions (the resolver path lets users pull them;
nothing promises they're pre-seeded).

- [ ] **Step 4: Update `tests/test_cli.py`**

If any test uses `muse pull all-minilm-l6-v2` as a bare id, change to
`muse pull hf://sentence-transformers/all-MiniLM-L6-v2` (or remove the
test if it was just testing the bundled path and is redundant).

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: all tests pass; 8 tests removed (4 files, ~2 tests per file), resolver
coverage replaces them.

- [ ] **Step 6: Commit**

```bash
git commit -m "chore: remove minilm + qwen3-embedding scripts, replaced by resolver

sentence-transformers repos are now resolver-addressable via
\`muse pull hf://sentence-transformers/all-MiniLM-L6-v2\` and
\`muse pull hf://Qwen/Qwen3-Embedding-0.6B\`. Keeping curated scripts
for these two models was duplicating information the resolver already
synthesizes from HF metadata.

nv-embed-v2 stays as a script because it uses transformers.AutoModel
directly (custom encode() method), not the sentence-transformers API."
```

---

## Part J: Release

### Task J1: Final verification

- [ ] Full suite: `pytest tests/ -q` (including slow e2e)
- [ ] Import smokes:
```bash
python -c "from muse.core.resolvers import resolve, search; print('ok')"
python -c "from muse.core.resolvers_hf import HFResolver; print('ok')"
python -c "from muse.modalities.chat_completion import MODALITY, build_router; print(MODALITY)"
python -c "from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel; print('ok')"
```
- [ ] CLI smokes:
```bash
muse --help
muse search qwen3 --modality chat/completion --limit 5
muse models list
```

### Task J2: Merge + tag v0.10.0

- [ ] Merge `feat/chat-completion-resolvers-gguf` -> `main` (FF)
- [ ] Bump `pyproject.toml` to `0.10.0`
- [ ] Tag `v0.10.0`, push
- [ ] `gh release create v0.10.0 --title ... --notes ...` covering:
  - New `chat/completion` modality (OpenAI-compatible, streaming + tool calling)
  - `LlamaCppModel` runtime (in-process GGUF via llama-cpp-python)
  - `muse.core.resolvers` abstraction + `hf://` resolver
  - `muse search <query>` command
  - `muse pull hf://...` URI support
  - `SentenceTransformerModel` generic runtime (replaces minilm + qwen3-embedding scripts)
  - Breaking: `all-minilm-l6-v2` and `qwen3-embedding-0.6b` no longer bundled; pull via `hf://...`

---

## Summary

| Phase | Tasks | Scope |
|---|---|---|
| A | 1 | Resolver core abstraction |
| B | 4 | chat/completion modality (protocol, codec, routes, client) |
| C | 1 | LlamaCppModel runtime |
| D | 1 | HFResolver (GGUF + sentence-transformers sniffing + search) |
| E | 1 | SentenceTransformerModel runtime |
| F | 2 | Catalog: persist synthesized manifests + pull() dispatch |
| G | 1 | `muse search` + `muse pull hf://...` CLI |
| H | 1 | Worker auto-mount verify + E2E smoke |
| I | 2 | Docs + migrate existing embedding scripts |
| J | 2 | Release v0.10.0 |

**Approx test delta:** +75 tests (resolvers, chat_completion, llama_cpp,
sentence_transformers runtime, catalog merge, search CLI). Some tests (~8)
removed in Task I2 after scripts migrate to resolver.

**Estimated commits:** ~15-18 (one per task checkpoint).
