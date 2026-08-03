"""HF resolver plugin for CT2 faster-whisper audio/transcription models.

Sniffs HF repos for the CT2 file shape (model.bin + config.json +
tokenizer.json or vocabulary.txt) plus the ASR tag. Synthesizes a
manifest that targets FasterWhisperModel. ffmpeg is declared as a
system package so the catalog warns when it's missing on PATH.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision


_RUNTIME_PATH = "muse.modalities.audio_transcription.runtimes.faster_whisper:FasterWhisperModel"
_PIP_EXTRAS = ("faster-whisper>=1.0.0",)
_SYSTEM_PACKAGES = ("ffmpeg",)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _looks_like_ct2(siblings: list[str], tags: list[str]) -> bool:
    names = {Path(f).name for f in siblings}
    has_ct2_shape = (
        "model.bin" in names
        and "config.json" in names
        and ("vocabulary.txt" in names or "tokenizer.json" in names)
    )
    has_asr_tag = "automatic-speech-recognition" in tags
    return has_ct2_shape and has_asr_tag


def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    return _looks_like_ct2(siblings, tags)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "audio/transcription",
        "hf_repo": repo_id,
        "description": f"Faster-Whisper: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": list(_SYSTEM_PACKAGES),
        "capabilities": {},
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
            revision=revision,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(
        search=query, filter="automatic-speech-recognition",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="audio/transcription",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "audio/transcription",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": _SYSTEM_PACKAGES,
    "priority": 100,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
