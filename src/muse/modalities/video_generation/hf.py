"""HF resolver plugin for text-to-video repos.

Sniffs HF repos with the `text-to-video` tag whose name matches one
of the supported architecture patterns: wan, cogvideox, ltx, mochi,
hunyuan. Priority 105.

Per-architecture dispatch:
  - *wan*           -> WanRuntime (production-ready)
  - *cogvideox*     -> CogVideoXRuntime (production-ready)
  - *ltx-video*     -> WanRuntime fallback (manifest synthesized but
                       pipeline class won't match; v1.next adds
                       LTXVideoRuntime)
  - *mochi*         -> WanRuntime fallback (v1.next: MochiRuntime)
  - *hunyuan*       -> WanRuntime fallback (v1.next: HunyuanRuntime)
  - generic         -> WanRuntime (conservative defaults)

Loaded via single-file import; no relative imports. Stdlib +
huggingface_hub + muse.core only at module top.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision


_WAN_RUNTIME_PATH = (
    "muse.modalities.video_generation.runtimes.wan_runtime:WanRuntime"
)
_COGVIDEOX_RUNTIME_PATH = (
    "muse.modalities.video_generation.runtimes.cogvideox_runtime"
    ":CogVideoXRuntime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "diffusers>=0.32.0",
    "transformers>=4.36.0",
    "accelerate",
    "Pillow>=9.1.0",
    "imageio[ffmpeg]>=2.31.0",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _infer_defaults(repo_id: str) -> tuple[str, dict[str, Any]]:
    """Return (runtime_path, capabilities) per architecture pattern."""
    rid = repo_id.lower()
    if "cogvideox" in rid:
        caps = {
            "device": "cuda",
            "default_duration_seconds": 6.0,
            "default_fps": 8,
            "default_size": [720, 480],
            "min_duration_seconds": 1.0,
            "max_duration_seconds": 10.0,
            "default_steps": 50,
            "default_guidance": 6.0,
            "supports_image_to_video": False,
            "memory_gb": 9.0,
        }
        return _COGVIDEOX_RUNTIME_PATH, caps
    if "wan" in rid:
        caps = {
            "device": "cuda",
            "default_duration_seconds": 5.0,
            "default_fps": 5,
            "default_size": [832, 480],
            "min_duration_seconds": 1.0,
            "max_duration_seconds": 10.0,
            "default_steps": 30,
            "default_guidance": 5.0,
            "supports_image_to_video": False,
            "memory_gb": 6.0,
        }
        return _WAN_RUNTIME_PATH, caps
    if (
        "ltx-video" in rid
        or "ltx_video" in rid
        or "ltxvideo" in rid
    ):
        caps = {
            "device": "cuda",
            "default_duration_seconds": 5.0,
            "default_fps": 30,
            "default_size": [1216, 704],
            "default_steps": 20,
            "default_guidance": 3.0,
            "supports_image_to_video": False,
            "memory_gb": 16.0,
        }
        return _WAN_RUNTIME_PATH, caps
    if "mochi" in rid:
        caps = {
            "device": "cuda",
            "default_duration_seconds": 5.0,
            "default_fps": 30,
            "default_size": [848, 480],
            "default_steps": 64,
            "default_guidance": 4.5,
            "supports_image_to_video": False,
            "memory_gb": 24.0,
        }
        return _WAN_RUNTIME_PATH, caps
    if "hunyuan" in rid:
        caps = {
            "device": "cuda",
            "default_duration_seconds": 5.0,
            "default_fps": 24,
            "default_size": [1280, 720],
            "default_steps": 50,
            "default_guidance": 6.0,
            "supports_image_to_video": False,
            "memory_gb": 60.0,
        }
        return _WAN_RUNTIME_PATH, caps
    # Generic fallback
    caps = {
        "device": "cuda",
        "default_duration_seconds": 5.0,
        "default_fps": 8,
        "default_size": [768, 432],
        "default_steps": 30,
        "default_guidance": 5.0,
        "supports_image_to_video": False,
        "memory_gb": 8.0,
    }
    return _WAN_RUNTIME_PATH, caps


_KNOWN_PATTERNS = (
    "wan", "cogvideox",
    "ltx-video", "ltx_video", "ltxvideo",
    "mochi", "hunyuan",
)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "text-to-video" not in tags:
        return False
    repo_id = (getattr(info, "id", "") or "").lower()
    return any(p in repo_id for p in _KNOWN_PATTERNS)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    runtime_path, capabilities = _infer_defaults(repo_id)
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "video/generation",
        "hf_repo": repo_id,
        "description": f"text-to-video model: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
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
        backend_path=runtime_path,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(
        search=query, filter="text-to-video",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="video/generation",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "video/generation",
    # Nominal runtime; resolve() dispatches to the correct path per arch.
    "runtime_path": _WAN_RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 105,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
