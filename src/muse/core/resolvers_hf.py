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

        # Look up curated chat-format hints (chat_formats.yaml). The
        # lookup table is the source of truth for "this model family
        # works with this llama.cpp chat handler" and lets users get
        # working tool calls without hand-editing manifests. The hints
        # also override the supports_tools sniff result when they
        # disagree (the YAML is curated; the sniff is heuristic).
        from muse.core.chat_formats import lookup_chat_format
        hints = lookup_chat_format(repo_id) or {}

        model_id = _gguf_model_id(repo_id, variant)
        capabilities: dict[str, Any] = {
            "gguf_file": matched,
            "supports_tools": hints.get("supports_tools", supports_tools),
        }
        if "chat_format" in hints:
            capabilities["chat_format"] = hints["chat_format"]
        if ctx_length:
            capabilities["context_length"] = ctx_length

        manifest = {
            "model_id": model_id,
            "modality": "chat/completion",
            "hf_repo": repo_id,
            "description": f"GGUF model: {repo_id} ({variant})",
            "license": _repo_license(info),
            "pip_extras": list(LLAMA_CPP_PIP_EXTRAS),
            "system_packages": [],
            "capabilities": capabilities,
        }

        def _download(cache_root: Path) -> Path:
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
            siblings = getattr(repo, "siblings", None) or []
            if not siblings:
                try:
                    # `files_metadata=True` is what makes RepoSibling.size populated.
                    # Without it, .size is always None and --max-size-gb is meaningless.
                    info = self._api.repo_info(repo.id, files_metadata=True)
                    siblings = info.siblings
                except Exception:
                    continue
            # Per-repo deduplication by variant: sharded GGUFs (model-q4_k_m-00001-of-00003.gguf)
            # and repos that publish duplicate quants emit the same @variant tag for multiple
            # files. We sum sizes across files sharing a variant (so a sharded model reports
            # its true total size) and emit one row per variant per repo.
            variant_to_size: dict[str, float] = {}
            variant_to_first_file: dict[str, str] = {}
            for s in siblings:
                if not s.rfilename.endswith(".gguf"):
                    continue
                variant = _extract_variant(s.rfilename)
                size_bytes = getattr(s, "size", None) or 0
                variant_to_size[variant] = variant_to_size.get(variant, 0) + size_bytes
                variant_to_first_file.setdefault(variant, s.rfilename)
            for variant, total_bytes in variant_to_size.items():
                yield SearchResult(
                    uri=f"hf://{repo.id}@{variant}",
                    model_id=_gguf_model_id(repo.id, variant),
                    modality="chat/completion",
                    size_gb=(total_bytes / 1e9) if total_bytes else None,
                    downloads=getattr(repo, "downloads", None),
                    license=None,
                    description=f"{repo.id} ({variant})",
                )

    # --- Sentence-Transformers branch ---

    def _resolve_sentence_transformer(self, repo_id: str, info) -> ResolvedModel:
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
    """Synthesize a model_id from the repo name (lowercased)."""
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
