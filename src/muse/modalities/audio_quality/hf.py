"""HF resolver plugin for ``audio/quality`` model families.

Only explicitly supported custom checkpoints are claimed. This plugin runs
at priority 100 so ``facebook/audiobox-aesthetics`` reaches its quality
runtime before the broad ``audio/classification`` tag resolver.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import (
    ResolvedModel,
    ResolverError,
    SearchResult,
    hf_commit_revision,
)


_UTMOS_RUNTIME = "muse.modalities.audio_quality.runtimes.utmos:UTMOSRuntime"
_AUDIOBOX_RUNTIME = (
    "muse.modalities.audio_quality.runtimes.audiobox_aesthetics"
    ":AudioboxAestheticsRuntime"
)
_UTMOS_EXTRAS = ("torch>=2.11", "torchcodec>=0.12")
_AUDIOBOX_EXTRAS = (
    "torch>=2.11",
    "audiobox_aesthetics>=0.0.4",
    "torchcodec>=0.12",
)


@dataclass(frozen=True)
class _Family:
    name_hints: tuple[str, ...]
    required_files: tuple[str, ...]
    runtime_path: str
    pip_extras: tuple[str, ...]
    system_packages: tuple[str, ...]
    allow_patterns: tuple[str, ...]
    description: str
    search_terms: tuple[str, ...]
    quality_axes: tuple[str, ...]
    primary_score: str


_FAMILIES = (
    _Family(
        name_hints=("utmos",),
        required_files=("utmos_scripted.pt",),
        runtime_path=_UTMOS_RUNTIME,
        pip_extras=_UTMOS_EXTRAS,
        system_packages=("ffmpeg",),
        allow_patterns=("utmos_scripted.pt", "README.md"),
        description="UTMOS speech-naturalness MOS predictor",
        search_terms=(
            "audio", "speech", "quality", "naturalness", "mos", "tts",
        ),
        quality_axes=("naturalness",),
        primary_score="naturalness",
    ),
    _Family(
        name_hints=("audiobox-aesthetics",),
        required_files=("model.safetensors", "config.json"),
        runtime_path=_AUDIOBOX_RUNTIME,
        pip_extras=_AUDIOBOX_EXTRAS,
        system_packages=("ffmpeg",),
        allow_patterns=("model.safetensors", "config.json", "README.md"),
        description="Audiobox Aesthetics four-axis audio quality assessor",
        search_terms=(
            "audio", "speech", "quality", "production", "aesthetics",
            "music", "sound",
        ),
        quality_axes=(
            "content_enjoyment",
            "content_usefulness",
            "production_complexity",
            "production_quality",
        ),
        primary_score="production_quality",
    ),
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    return getattr(card, "license", None) if card is not None else None


def _family_for_name(repo_id: str) -> _Family | None:
    name = repo_id.lower()
    return next(
        (family for family in _FAMILIES
         if any(hint in name for hint in family.name_hints)),
        None,
    )


def _family_for_info(info) -> _Family | None:
    repo_id = getattr(info, "id", "") or ""
    family = _family_for_name(repo_id)
    if family is None:
        return None
    siblings = {
        sibling.rfilename
        for sibling in (getattr(info, "siblings", None) or [])
    }
    return family if all(name in siblings for name in family.required_files) else None


def _sniff(info) -> bool:
    return _family_for_info(info) is not None


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    family = _family_for_info(info)
    if family is None:
        raise ResolverError(
            f"unsupported audio-quality checkpoint shape for {repo_id!r}"
        )
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "audio/quality",
        "hf_repo": repo_id,
        "description": f"{family.description}: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(family.pip_extras),
        "system_packages": list(family.system_packages),
        "capabilities": {
            "device": "auto",
            "window_seconds": 10.0,
            "max_duration_seconds": 600.0,
            "quality_axes": list(family.quality_axes),
            "primary_score": family.primary_score,
            "supports_reference_text": False,
        },
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=list(family.allow_patterns),
            cache_dir=str(cache_root) if cache_root else None,
            revision=revision,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=family.runtime_path,
        download=_download,
    )


def _matches_query(repo_id: str, family: _Family, query: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    if not tokens:
        return True
    haystack = " ".join((
        repo_id.lower(),
        family.description.lower(),
        *family.name_hints,
        *family.search_terms,
    ))
    return all(token in haystack for token in tokens)


def _complete_info(api: HfApi, repo):
    siblings = getattr(repo, "siblings", None)
    if isinstance(siblings, (list, tuple)) and siblings:
        return repo
    return api.model_info(repo.id)


def _search(
    api: HfApi,
    query: str,
    *,
    sort: str,
    limit: int,
) -> Iterable[SearchResult]:
    if limit <= 0:
        return
    # Generic queries such as "speech quality" do not necessarily occur in
    # repository ids. Search each supported family, then apply relevance and
    # shape validation locally before enforcing the caller's output limit.
    candidates: dict[str, SearchResult] = {}
    hints = dict.fromkeys(
        hint for family in _FAMILIES for hint in family.name_hints
    )
    for hint in hints:
        for repo in api.list_models(
            search=hint,
            sort=sort,
            limit=None,
            full=True,
        ):
            if repo.id in candidates:
                continue
            hinted_family = _family_for_name(repo.id)
            if hinted_family is None or not _matches_query(
                repo.id, hinted_family, query,
            ):
                continue
            info = _complete_info(api, repo)
            family = _family_for_info(info)
            if family is None:
                continue
            candidates[repo.id] = SearchResult(
                uri=f"hf://{repo.id}",
                model_id=_model_id(repo.id),
                modality="audio/quality",
                size_gb=None,
                downloads=getattr(info, "downloads", None),
                license=_repo_license(info),
                description=family.description,
                metadata={
                    "likes": getattr(info, "likes", None),
                    "last_modified": getattr(info, "last_modified", None),
                },
            )

    rows = list(candidates.values())
    if sort == "downloads":
        rows.sort(key=lambda row: row.downloads or 0, reverse=True)
    elif sort == "likes":
        rows.sort(
            key=lambda row: row.metadata.get("likes") or 0,
            reverse=True,
        )
    elif sort == "lastModified":
        rows.sort(
            key=lambda row: str(row.metadata.get("last_modified") or ""),
            reverse=True,
        )
    yield from rows[:limit]


HF_PLUGIN = {
    "modality": "audio/quality",
    "runtime_path": _UTMOS_RUNTIME,
    "pip_extras": _UTMOS_EXTRAS,
    "system_packages": ("ffmpeg",),
    "priority": 100,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
