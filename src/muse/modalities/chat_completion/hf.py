"""HF resolver plugin for GGUF and VLM chat/completion models.

Sniffs HuggingFace repos for `.gguf` siblings (-> LlamaCppModel) or for
the `image-text-to-text` tag / VLM model_type (-> HFVisionLanguageModel).
Variant (quant tag) is required for GGUFs: a single GGUF repo often
publishes 5+ quants and there is no defensible default.
`muse search foo --modality chat/completion` enumerates each variant as a
separate row.

This plugin is loaded by `discover_hf_plugins` via single-file import,
so it must NOT use relative imports or import from sibling modality
modules. See docs/HF_PLUGINS.md for the authoring rules.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from muse.core.chat_formats import lookup_chat_format
from muse.core.resolvers import (
    ResolvedModel,
    ResolverError,
    SearchResult,
    hf_commit_revision,
)


_VARIANT_RE = re.compile(
    r"(q\d+_[a-z0-9_]+|iq\d+_[a-z0-9]+|f16|bf16|f32)", re.IGNORECASE,
)

_RUNTIME_PATH = "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel"
_PIP_EXTRAS = ("llama-cpp-python>=0.2.90",)

# VLM detection constants
_VLM_MULTI_IMAGE_MODEL_TYPES = frozenset({
    "idefics3", "smolvlm", "qwen2_vl", "pixtral", "llava_next", "minicpmv",
})
_VLM_SINGLE_IMAGE_MODEL_TYPES = frozenset({"llava"})
_VLM_RUNTIME_PATH = (
    "muse.modalities.chat_completion.runtimes.transformers_vlm:HFVisionLanguageModel"
)
_VLM_PIP_EXTRAS = ("torch>=2.1.0", "transformers>=4.46.0", "accelerate", "Pillow")


def _extract_variant(gguf_filename: str) -> str:
    stem = Path(gguf_filename).stem
    m = _VARIANT_RE.search(stem)
    return (m.group(1).lower() if m else stem).replace(".", "_")


def _match_gguf_variant(files: list[str], variant: str) -> str | None:
    norm = variant.lower()
    for f in files:
        if _extract_variant(f) == norm:
            return f
    return None


def _gguf_model_id(repo_id: str, variant: str) -> str:
    base = repo_id.split("/", 1)[-1].lower()
    if not base.endswith("-gguf"):
        base = f"{base}-gguf"
    return f"{base}-{variant.lower().replace('_', '-')}"


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff_supports_tools(chat_template: str | None) -> bool:
    if not chat_template or not isinstance(chat_template, str):
        return False
    return bool(re.search(r"(\bif\s+tools\b|\{\{\s*tools|tool_calls)", chat_template))


def _try_sniff_tools_from_repo(
    repo_id: str,
    revision: str | None = None,
) -> bool | None:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer_config.json",
            revision=revision,
        )
    except Exception:
        return None
    try:
        cfg = json.loads(Path(path).read_text())
    except Exception:
        return None
    return _sniff_supports_tools(cfg.get("chat_template"))


def _try_sniff_context_length_from_repo(
    repo_id: str,
    revision: str | None = None,
) -> int | None:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
        )
        cfg = json.loads(Path(path).read_text())
        return int(cfg.get("max_position_embeddings") or 0) or None
    except Exception:
        return None


def _is_vlm(info) -> tuple[bool, bool]:
    """Detect VLM repos. Returns (is_vlm, supports_multi_image)."""
    tags = set(getattr(info, "tags", []) or [])
    cfg = getattr(info, "config", None) or {}
    model_type = (cfg.get("model_type") or "").lower()
    if "image-text-to-text" in tags:
        # Tag-only signal: assume multi-image (most VLMs of this generation
        # support it). model_type can refine.
        if model_type in _VLM_SINGLE_IMAGE_MODEL_TYPES:
            return True, False
        return True, True
    if model_type in _VLM_MULTI_IMAGE_MODEL_TYPES:
        return True, True
    if model_type in _VLM_SINGLE_IMAGE_MODEL_TYPES:
        return True, False
    return False, False


def _sniff(info) -> bool:
    if _is_vlm(info)[0]:
        return True
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    return any(f.endswith(".gguf") for f in siblings)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    revision = hf_commit_revision(info)
    is_vlm, multi_image = _is_vlm(info)
    if is_vlm:
        license_str = _repo_license(info)
        manifest = {
            "model_id": repo_id.split("/", 1)[-1].lower(),
            "modality": "chat/completion",
            "hf_repo": repo_id,
            "description": f"VLM via HF resolver: {repo_id}",
            "license": license_str,
            "pip_extras": list(_VLM_PIP_EXTRAS),
            "capabilities": {
                "supports_vision": True,
                "supports_multi_image": multi_image,
                "supports_tools": False,
            },
        }
        if revision is not None:
            manifest["revision"] = revision

        def _download(cache_dir: Path) -> Path:
            return Path(snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir) if cache_dir else None,
                revision=revision,
            ))

        return ResolvedModel(
            manifest=manifest,
            backend_path=_VLM_RUNTIME_PATH,
            download=_download,
        )

    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    gguf_files = [f for f in siblings if f.endswith(".gguf")]
    if not gguf_files:
        raise ResolverError(f"no .gguf files in {repo_id}")
    if variant is None:
        variants = sorted({_extract_variant(f) for f in gguf_files})
        raise ResolverError(
            f"variant required for GGUF repo {repo_id}; available: {variants}"
        )
    matched = _match_gguf_variant(gguf_files, variant)
    if matched is None:
        variants = sorted({_extract_variant(f) for f in gguf_files})
        raise ResolverError(
            f"variant {variant!r} not found in {repo_id}; available: {variants}"
        )

    supports_tools = _try_sniff_tools_from_repo(repo_id, revision=revision)
    ctx_length = _try_sniff_context_length_from_repo(repo_id, revision=revision)

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
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        allow_patterns = [matched, "tokenizer*", "config.json", "*.md"]
        return Path(snapshot_download(
            repo_id=repo_id, allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
            revision=revision,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(search=query, filter="gguf", sort=sort, limit=limit)
    for repo in repos:
        siblings = getattr(repo, "siblings", None) or []
        if not siblings:
            try:
                info = api.repo_info(repo.id, files_metadata=True)
                siblings = info.siblings
            except Exception:
                continue
        variant_to_size: dict[str, float] = {}
        for s in siblings:
            if not s.rfilename.endswith(".gguf"):
                continue
            variant = _extract_variant(s.rfilename)
            size_bytes = getattr(s, "size", None) or 0
            variant_to_size[variant] = variant_to_size.get(variant, 0) + size_bytes
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


HF_PLUGIN = {
    "modality": "chat/completion",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 100,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
