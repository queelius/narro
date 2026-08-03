"""HF resolver plugin for promptable segmenters (SAM / SAM-2 family).

Sniffs HF repos whose tags include ``mask-generation`` or
``image-segmentation``. Synthesizes a manifest with capabilities
inferred from repo name patterns:

  - ``*sam2-hiera-tiny*``: small (40MB), points + boxes + auto.
  - ``*sam2-hiera-base*``: balanced (150MB).
  - ``*sam2-hiera-large*``: best quality (225MB).
  - ``*sam-*``: original SAM family.
  - ``*clipseg*``: text-prompted segmentation; flips supports_*
    flags (text on, points/boxes/auto off).
  - fallback: conservative defaults preferring point + box prompts.

Priority 110: equal to image/animation, but the
``mask-generation`` / ``image-segmentation`` tag-set is unique to
segmenters so collisions with i2i upscalers (priority 105) do not
arise in practice.

CLIPSeg has a different runtime backbone (CLIP encoder + ViT decoder
plus text conditioning) but presents the same wire shape; the
``supports_text_prompts`` capability gate steers requests away from
SAM-style models without a runtime change.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision


_RUNTIME_PATH = (
    "muse.modalities.image_segmentation.runtimes.sam2_runtime"
    ":SAM2Runtime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "transformers>=4.43.0",
    "Pillow>=9.1.0",
    "numpy",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _infer_capabilities(repo_id: str) -> dict[str, Any]:
    """Sensible per-pattern defaults by repo-id substring match."""
    rid = (repo_id or "").lower()
    if "sam2-hiera-tiny" in rid:
        return {
            "max_masks": 64,
            "memory_gb": 0.8,
            "supports_text_prompts": False,
            "supports_point_prompts": True,
            "supports_box_prompts": True,
            "supports_automatic": True,
        }
    if "sam2-hiera-base" in rid:
        return {
            "max_masks": 64,
            "memory_gb": 1.5,
            "supports_text_prompts": False,
            "supports_point_prompts": True,
            "supports_box_prompts": True,
            "supports_automatic": True,
        }
    if "sam2-hiera-large" in rid:
        return {
            "max_masks": 64,
            "memory_gb": 2.5,
            "supports_text_prompts": False,
            "supports_point_prompts": True,
            "supports_box_prompts": True,
            "supports_automatic": True,
        }
    if "clipseg" in rid:
        return {
            "max_masks": 16,
            "memory_gb": 0.6,
            "supports_text_prompts": True,
            "supports_point_prompts": False,
            "supports_box_prompts": False,
            "supports_automatic": False,
        }
    if "sam-" in rid or rid.endswith("/sam"):
        return {
            "max_masks": 64,
            "memory_gb": 1.5,
            "supports_text_prompts": False,
            "supports_point_prompts": True,
            "supports_box_prompts": True,
            "supports_automatic": True,
        }
    return {
        "max_masks": 16,
        "memory_gb": 1.0,
        "supports_text_prompts": False,
        "supports_point_prompts": True,
        "supports_box_prompts": True,
        "supports_automatic": True,
    }


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    return "mask-generation" in tags or "image-segmentation" in tags


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    capabilities = {
        "device": "auto",
        **_infer_capabilities(repo_id),
    }
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/segmentation",
        "hf_repo": repo_id,
        "description": f"Promptable segmenter: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        # SAM-2 repos ship a small set of files: weights, config,
        # processor config. snapshot_download with a generous allowlist
        # picks them all up without dragging in unnecessary extras.
        allow_patterns = [
            "*.safetensors",
            "*.json",
            "*.txt",
            "preprocessor_config.json",
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
    repos = api.list_models(
        search=query, filter="mask-generation",
        sort=sort, limit=limit,
    )
    for repo in repos:
        repo_id = getattr(repo, "id", "") or ""
        yield SearchResult(
            uri=f"hf://{repo_id}",
            model_id=_model_id(repo_id),
            modality="image/segmentation",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo_id,
        )


HF_PLUGIN = {
    "modality": "image/segmentation",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
