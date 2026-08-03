"""HF resolver plugin for Stable Audio Open shaped repos.

Sniff: HF repo with the `text-to-audio` tag AND `model_index.json`
sibling AND `stable-audio` substring in the repo id (case-insensitive).
The triple guard pins this plugin to repos that share the
`diffusers.StableAudioPipeline` shape so the synthesized manifest
plus generic StableAudioRuntime are guaranteed to work.

Priority 105: more specific than embedding/text (110) since the sniff
requires three conditions; less specific than file-pattern plugins at
100 (GGUF, model_index.json + text-to-image, faster-whisper). Other
text-to-audio architectures (MusicGen, AudioGen, AudioLDM2) need
their own runtimes; for v1 those are deferred and arrive as bundled
scripts when implemented.

Loaded via single-file import; no relative imports. See
docs/HF_PLUGINS.md for authoring rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision


_RUNTIME_PATH = (
    "muse.modalities.audio_generation.runtimes.stable_audio:StableAudioRuntime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "diffusers>=0.27.0",
    # transformers required by StableAudioPipeline's T5 text encoder.
    "transformers>=4.36.0",
    "accelerate",
    "soundfile",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff(info) -> bool:
    """Match Stable Audio Open shaped repos (and only those).

    Three conditions must hold:
      1. tag includes `text-to-audio`
      2. `model_index.json` is a sibling (diffusers pipeline shape)
      3. repo name contains `stable-audio` (case-insensitive)

    The third guard is what excludes MusicGen, AudioGen, AudioLDM2,
    Bark, etc. Those have different pipeline classes and need their
    own runtimes; this plugin's runtime is StableAudioPipeline-specific.
    """
    tags = getattr(info, "tags", None) or []
    if "text-to-audio" not in tags:
        return False
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    if not any(Path(f).name == "model_index.json" for f in siblings):
        return False
    repo_id = (getattr(info, "id", "") or "").lower()
    return "stable-audio" in repo_id


def _stable_audio_capabilities() -> dict:
    """Capability defaults for the Stable Audio Open lineage.

    Hardcoded per the official Stable Audio Open 1.0 model card:
    50 inference steps, guidance 7.0, 44.1kHz stereo, 47s native max,
    10s default duration, 1s minimum.

    Music + SFX both supported (the model is one of the few audio-gen
    models that handles both well via prompt differentiation).
    """
    return {
        "device": "auto",
        "supports_music": True,
        "supports_sfx": True,
        "default_duration": 10.0,
        "min_duration": 1.0,
        "max_duration": 47.0,
        "default_sample_rate": 44100,
        "default_steps": 50,
        "default_guidance": 7.0,
        "memory_gb": 6.0,
    }


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "audio/generation",
        "hf_repo": repo_id,
        "description": f"Stable Audio Open: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        # ffmpeg optional for mp3/opus codec; declaring so muse pull
        # can install when available. Not required for wav/flac.
        "system_packages": ["ffmpeg"],
        "capabilities": _stable_audio_capabilities(),
    }
    if revision is not None:
        manifest["revision"] = revision

    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    has_fp16 = any(".fp16." in f for f in siblings)

    def _download(cache_root: Path) -> Path:
        # Restrict to the diffusers fp16 subfolder layout so the cache
        # stays lean (~3.4GB instead of ~7GB unrestricted). Stable Audio
        # Open 1.0 ships transformer/, vae/, text_encoder/, tokenizer/,
        # scheduler/. Top-level model_index.json is required.
        if has_fp16:
            allow_patterns = [
                "model_index.json",
                "*/*.json",
                "*/*.txt",
                "*/*.fp16.safetensors",
                "tokenizer/*",
            ]
        else:
            allow_patterns = [
                "model_index.json",
                "*/*.json",
                "*/*.txt",
                "*/*.safetensors",
                "tokenizer/*",
            ]
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
            allow_patterns=allow_patterns,
            revision=revision,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    """Search HuggingFace for stable-audio repos.

    Filter: text-to-audio tag. Post-filter requires `stable-audio`
    in the repo name to keep the result set narrow.
    """
    repos = api.list_models(
        search=query, filter="text-to-audio",
        sort=sort, limit=limit,
    )
    for repo in repos:
        if "stable-audio" not in (repo.id or "").lower():
            continue
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="audio/generation",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "audio/generation",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": ("ffmpeg",),
    # 105: more specific than embedding/text (110); less specific
    # than file-pattern plugins at 100. Triple-guarded sniff (tag +
    # file shape + name pattern) makes this plugin reliable on the
    # Stable Audio Open lineage and inert elsewhere.
    "priority": 105,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
