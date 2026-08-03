"""Hugging Face resolver for Qwen3 forced-alignment checkpoints."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import (
    ResolvedModel,
    ResolverError,
    SearchResult,
    hf_commit_revision,
)


_RUNTIME = (
    "muse.modalities.audio_alignment.runtimes.qwen3_forced_aligner"
    ":Qwen3ForcedAlignerRuntime"
)
_PIP_EXTRAS = (
    "torch>=2.11",
    "torchcodec>=0.12",
    "transformers>=5.13.0",
    "nagisa>=0.2.11",
    "soynlp>=0.0.493",
)
_REPO_NAME = "qwen/qwen3-forcedaligner-0.6b-hf"
_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_ALLOW_PATTERNS = (
    *_REQUIRED_FILES,
    "chat_template.jinja",
    "README.md",
)
_SUPPORTED_LANGUAGES = (
    "English",
    "Chinese",
    "Cantonese",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    return getattr(card, "license", None) if card is not None else None


def _is_supported_name(repo_id: str) -> bool:
    return repo_id.lower() == _REPO_NAME


def _has_checkpoint_shape(info) -> bool:
    siblings = {
        sibling.rfilename
        for sibling in (getattr(info, "siblings", None) or [])
    }
    return all(name in siblings for name in _REQUIRED_FILES)


def _sniff(info) -> bool:
    repo_id = getattr(info, "id", "") or ""
    return _is_supported_name(repo_id) and _has_checkpoint_shape(info)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    if not _is_supported_name(repo_id) or not _has_checkpoint_shape(info):
        raise ResolverError(
            f"unsupported audio-alignment checkpoint shape for {repo_id!r}"
        )
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "audio/alignment",
        "hf_repo": repo_id,
        "description": (
            "Qwen3 ForcedAligner 0.6B reference-text word alignment: "
            f"{repo_id}"
        ),
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": ["ffmpeg"],
        "capabilities": {
            "device": "auto",
            "memory_gb": 4.0,
            "sample_rate": 16000,
            "max_duration_seconds": 300.0,
            "max_input_tokens": 8192,
            "max_reference_words": 2048,
            "word_timestamps": True,
            "confidence": True,
            "supported_languages": list(_SUPPORTED_LANGUAGES),
        },
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=list(_ALLOW_PATTERNS),
            cache_dir=str(cache_root) if cache_root else None,
            revision=revision,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME,
        download=_download,
    )


def _matches_query(query: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    haystack = (
        "qwen qwen3 forced aligner alignment audio speech timestamps "
        "word audiobook reference transcript"
    )
    return all(token in haystack for token in tokens)


def _search(
    api: HfApi,
    query: str,
    *,
    sort: str,
    limit: int,
) -> Iterable[SearchResult]:
    if limit <= 0 or not _matches_query(query):
        return
    for repo in api.list_models(
        search="Qwen3-ForcedAligner",
        sort=sort,
        limit=None,
        full=True,
    ):
        if not _is_supported_name(repo.id):
            continue
        siblings = getattr(repo, "siblings", None)
        info = repo if siblings else api.model_info(repo.id)
        if not _has_checkpoint_shape(info):
            continue
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="audio/alignment",
            size_gb=1.84,
            downloads=getattr(info, "downloads", None),
            license=_repo_license(info),
            description="Qwen3 ForcedAligner 0.6B word-level timestamps",
            metadata={
                "likes": getattr(info, "likes", None),
                "last_modified": getattr(info, "last_modified", None),
            },
        )
        return


HF_PLUGIN = {
    "modality": "audio/alignment",
    "runtime_path": _RUNTIME,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": ("ffmpeg",),
    "priority": 90,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
