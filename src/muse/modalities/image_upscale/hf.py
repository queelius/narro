"""HF resolver plugin for diffusers-shape image upscalers.

Sniffs HF repos for `model_index.json` (the diffusers pipeline config)
plus the `image-to-image` tag plus an upscaler-name allowlist
(upscaler / super-resolution / esrgan / upscale / x4-upscaler /
ldm-super). Synthesizes a manifest with capabilities inferred from
repo name (x4 -> [4]; fallback -> [4]).

Priority 105: between image/animation (110) and image/generation (100),
so upscaler repos get correctly classified even though they share the
diffusers `model_index.json` shape and the `image-to-image` tag with
regular i2i checkpoints.

GAN-based upscalers (AuraSR, Real-ESRGAN) have non-diffusers shapes
and are NOT claimed by this plugin. They need their own runtime
classes; deferred to v1.next.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision


_RUNTIME_PATH = (
    "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
    ":DiffusersUpscaleRuntime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "diffusers>=0.27.0",
    "transformers>=4.36.0",
    "accelerate",
    "Pillow",
    "safetensors",
)
_UPSCALER_NAMES = (
    "x4-upscaler",
    "upscaler",
    "super-resolution",
    "upscale",
    "ldm-super",
    "esrgan",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _is_upscaler_name(repo_id: str) -> bool:
    rid = (repo_id or "").lower()
    return any(s in rid for s in _UPSCALER_NAMES)


def _infer_defaults(repo_id: str) -> dict[str, Any]:
    """Sensible per-pattern defaults so each upscaler model lands with
    reasonable scale / steps / guidance defaults out of the box."""
    rid = repo_id.lower()
    if "x4-upscaler" in rid or "x4" in rid:
        return {
            "default_scale": 4,
            "supported_scales": [4],
            "default_steps": 20,
            "default_guidance": 9.0,
        }
    return {
        "default_scale": 4,
        "supported_scales": [4],
        "default_steps": 20,
        "default_guidance": 9.0,
    }


def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    has_pipeline_config = any(
        Path(f).name == "model_index.json" for f in siblings
    )
    has_i2i_tag = "image-to-image" in tags
    return has_pipeline_config and has_i2i_tag and _is_upscaler_name(
        getattr(info, "id", "") or ""
    )


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    defaults = _infer_defaults(repo_id)
    capabilities = {
        **defaults,
        "device": "cuda",
        "memory_gb": 6.0,
    }
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/upscale",
        "hf_repo": repo_id,
        "description": f"Diffusers super-resolution upscaler: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    has_fp16_variants = any(".fp16." in name for name in siblings)

    def _download(cache_root: Path) -> Path:
        # Mirror image_generation's download pattern: include the top-level
        # model_index.json plus only the per-component subfolder weights.
        allow_patterns = [
            "model_index.json",
            "*/*.json",
            "*/*.txt",
        ]
        if has_fp16_variants:
            allow_patterns.extend([
                "*/*.fp16.safetensors",
                "*/*.fp16.bin",
            ])
        else:
            allow_patterns.extend([
                "*/*.safetensors",
                "*/*.bin",
            ])
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
    repos = api.list_models(
        search=query, filter="image-to-image",
        sort=sort, limit=limit,
    )
    for repo in repos:
        if not _is_upscaler_name(getattr(repo, "id", "") or ""):
            continue
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="image/upscale",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "image/upscale",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 105,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
