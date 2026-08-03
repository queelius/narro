"""HF resolver plugin for fused-checkpoint AnimateDiff variants.

Sniffs HF repos that ship a complete diffusers pipeline (model_index.json
sibling), advertise the `text-to-video` tag, and have `animate` or
`motion` in the repo name. AnimateLCM is the canonical match.

The plugin pairs every match with one immutable SD 1.5 base
(`emilianJR/epiCRealism`) and pulls both repositories as a local bundle.
A different base requires a model definition whose artifact bundle declares
that base explicitly; changing only a capability cannot replace downloaded
assets.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.artifacts import download_hf_artifact_bundle
from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision


_RUNTIME_PATH = (
    "muse.modalities.image_animation.runtimes.animatediff"
    ":AnimateDiffRuntime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "diffusers>=0.27.0",
    "transformers>=4.36.0",
    "accelerate",
    "Pillow>=9.1.0",
    "safetensors",
)
_BASE_MODEL = "emilianJR/epiCRealism"
_BASE_REVISION = "6522cf856b8c8e14638a0aaa7bd89b1b098aed17"
_ADAPTER_SUBDIR = "motion_adapter"
_BASE_SUBDIR = "base_model"
_ADAPTER_ALLOW_PATTERNS = (
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.md",
)
_ADAPTER_REQUIRED_PATTERNS = (
    "model_index.json",
    "*.safetensors",
)
_BASE_ALLOW_PATTERNS = (
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.model",
    "*.md",
    "feature_extractor/*",
    "scheduler/*",
    "safety_checker/*.safetensors",
    "safety_checker/*.json",
    "text_encoder/*.safetensors",
    "text_encoder/*.json",
    "tokenizer/*",
    "unet/*.safetensors",
    "unet/*.json",
    "vae/*.safetensors",
    "vae/*.json",
)
_BASE_REQUIRED_PATTERNS = (
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "safety_checker/config.json",
    "safety_checker/*.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/*.safetensors",
    "tokenizer/tokenizer_config.json",
    "unet/config.json",
    "unet/*.safetensors",
    "vae/config.json",
    "vae/*.safetensors",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _infer_defaults(repo_id: str) -> dict[str, Any]:
    """Sensible per-pattern defaults for fused AnimateDiff variants.

    Resolver-pulled models advertise `supports_text_to_animation: True`
    and `supports_image_to_animation: False`. The base-model fields describe
    the fixed second member of the immutable bundle used by Muse-managed
    loads. Direct runtime construction without ``local_dir`` may still pass a
    different compatible base repository.
    """
    rid = repo_id.lower()
    base = {
        "default_size": [512, 512],
        "default_frames": 16,
        "default_fps": 8,
        "base_model": _BASE_MODEL,
        "base_model_revision": _BASE_REVISION,
        "adapter_model_subdir": _ADAPTER_SUBDIR,
        "base_model_subdir": _BASE_SUBDIR,
        "supports_text_to_animation": True,
        "supports_image_to_animation": False,
        "min_frames": 8,
        "max_frames": 24,
    }
    if "animatelcm" in rid:
        return {**base, "default_steps": 4, "default_guidance": 1.0}
    return {**base, "default_steps": 25, "default_guidance": 7.5}


def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    repo_id = getattr(info, "id", "") or ""
    has_pipeline_config = any(
        Path(f).name == "model_index.json" for f in siblings
    )
    has_t2v_tag = "text-to-video" in tags
    rid = repo_id.lower()
    name_matches = ("animate" in rid) or ("motion" in rid)
    return has_pipeline_config and has_t2v_tag and name_matches


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    capabilities = _infer_defaults(repo_id)
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/animation",
        "hf_repo": repo_id,
        "description": f"AnimateDiff fused checkpoint: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    artifacts = (
        {
            "repo_id": repo_id,
            "revision": revision,
            "subdir": _ADAPTER_SUBDIR,
            "allow_patterns": _ADAPTER_ALLOW_PATTERNS,
            "required_patterns": _ADAPTER_REQUIRED_PATTERNS,
        },
        {
            "repo_id": _BASE_MODEL,
            "revision": _BASE_REVISION,
            "subdir": _BASE_SUBDIR,
            "allow_patterns": _BASE_ALLOW_PATTERNS,
            "required_patterns": _BASE_REQUIRED_PATTERNS,
        },
    )

    def _download(cache_root: Path) -> Path:
        return download_hf_artifact_bundle(
            cache_root,
            bundle_name=_model_id(repo_id),
            artifacts=artifacts,
            snapshot_download_fn=snapshot_download,
        )

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
        artifact_provenance=tuple(
            {
                "repo_id": artifact["repo_id"],
                "revision": artifact["revision"],
                "subdir": artifact["subdir"],
                "allow_patterns": list(artifact["allow_patterns"]),
                "required_patterns": list(artifact["required_patterns"]),
            }
            for artifact in artifacts
        ),
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
            modality="image/animation",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "image/animation",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
