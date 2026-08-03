"""HF resolver plugin for diffusers text-to-image models.

Sniffs HF repos for `model_index.json` (the diffusers pipeline config)
plus the `text-to-image` tag. Synthesizes a manifest with capabilities
inferred from repo name (turbo/flux/sdxl/sd3 patterns set sensible
default steps/guidance/size).

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

import logging

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import (
    ResolvedModel,
    ResolverError,
    SearchResult,
    hf_commit_revision,
)

logger = logging.getLogger(__name__)


_RUNTIME_PATH = (
    "muse.modalities.image_generation.runtimes.diffusers"
    ":DiffusersText2ImageModel"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "diffusers>=0.27.0",
    "transformers>=4.36.0",
    "accelerate",
    "Pillow",
    "safetensors",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _infer_defaults(repo_id: str) -> dict[str, Any]:
    """Sensible per-pattern defaults so each model lands with reasonable
    steps/guidance/size without users having to tweak. Override per-call
    via request fields or per-model via curated capabilities overlay."""
    rid = repo_id.lower()
    if "flux" in rid and "schnell" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 4,
            "default_guidance": 0.0,
        }
    if "flux" in rid and "dev" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 28,
            "default_guidance": 3.5,
        }
    if "turbo" in rid:
        return {
            "default_size": [512, 512],
            "default_steps": 1,
            "default_guidance": 0.0,
        }
    if "sdxl" in rid or "stable-diffusion-xl" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 25,
            "default_guidance": 7.5,
        }
    if "stable-diffusion-3" in rid or "sd3" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 28,
            "default_guidance": 4.5,
        }
    return {
        "default_size": [512, 512],
        "default_steps": 25,
        "default_guidance": 7.5,
    }


def _is_lora_adapter(siblings: list[str], tags: list[str]) -> bool:
    """Adapter-only repo shape: top-level safetensors weights plus an
    explicit LoRA signal in the tags. Callers have already established
    the repo is text-to-image and has NO model_index.json."""
    has_weights = any(
        f.endswith(".safetensors") and "/" not in f for f in siblings
    )
    has_lora_tag = "lora" in tags or any(
        t.startswith("base_model:adapter:") for t in tags
    )
    return has_weights and has_lora_tag


_LORA_PIP_EXTRAS = _PIP_EXTRAS + ("peft",)

# base_model:<qualifier>:<repo> forms that do NOT name a usable base.
_NON_BASE_QUALIFIERS = (
    "base_model:finetune:", "base_model:quantized:", "base_model:merge:",
)


def _lora_base_from_tags(tags: list[str]) -> str | None:
    """Extract the base repo from HF's base_model tag convention.

    `base_model:adapter:<repo>` is the explicit adapter-relationship tag;
    prefer it. Fall back to a plain `base_model:<repo>` tag that is not a
    qualified non-base form. Returns None when nothing matches (the
    post-overlay validation in catalog.pull handles that case).
    """
    for t in tags:
        if t.startswith("base_model:adapter:"):
            return t[len("base_model:adapter:"):]
    for t in tags:
        if (
            t.startswith("base_model:")
            and not t.startswith("base_model:adapter:")
            and not t.startswith(_NON_BASE_QUALIFIERS)
        ):
            return t[len("base_model:"):]
    return None


def _estimate_repo_weights_gb(repo_id: str) -> float | None:
    """Sum an HF repo's weight-file sizes (GB) for a memory estimate.

    Used to size a LoRA entry from its BASE repo, because the adapter's
    own on-disk footprint (tens of MB) would grossly undersize the load.
    Mirrors the fp16-preference of the downloader: when fp16 variants
    exist, only they are fetched, so only they should be summed.
    Returns None on any failure; the post-pull probe measures the real
    peak and self-heals sizing regardless.
    """
    if "/" not in repo_id:
        return None  # muse catalog id: sizing derives from that entry
    try:
        api = HfApi()
        info = api.model_info(repo_id, files_metadata=True)
        files = [
            (s.rfilename, s.size or 0)
            for s in getattr(info, "siblings", [])
            if s.rfilename.endswith((".safetensors", ".bin"))
        ]
        if not files:
            return None
        fp16 = [(n, sz) for n, sz in files if ".fp16." in n]
        total = sum(sz for _, sz in (fp16 or files))
        return total / 1e9 if total else None
    except Exception as e:  # noqa: BLE001
        logger.debug("base weight-size estimate failed for %s: %s", repo_id, e)
        return None


def _resolve_lora(
    repo_id: str, info, *, base_override: str | None = None,
) -> ResolvedModel:
    """Synthesize a manifest for an adapter-only repo.

    The manifest reuses the standard diffusers runtime; the runtime's
    lora_adapter branch loads the BASE pipeline and layers the adapter
    on top (unfused). base_model may be absent here (tagless repo with
    no --base override); catalog.pull validates the merged result.

    `base_override` (the operator's `--base` pin, fix I2) wins over the
    tag-declared base when set. Generation defaults (steps/guidance/size)
    and the memory estimate then derive from this EFFECTIVE base, so a
    `--base sdxl-turbo` pairing re-derives turbo defaults instead of
    keeping whatever the adapter's own tags declared.
    """
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []

    weights = [
        f for f in siblings if f.endswith(".safetensors") and "/" not in f
    ]
    if len(weights) != 1:
        raise ResolverError(
            f"LoRA repo {repo_id!r} has {len(weights)} top-level .safetensors "
            f"files ({sorted(weights)}); muse supports exactly one adapter "
            f"weight file per entry"
        )

    base = base_override or _lora_base_from_tags(tags)
    capabilities: dict[str, Any] = {
        # Generation defaults follow the EFFECTIVE base (override wins
        # over tag-declared); turbo bases get 1-step/no-guidance
        # automatically, whether declared by the repo or by --base.
        **_infer_defaults(base if base else repo_id),
        "lora_adapter": True,
        "lora_scale": 1.0,
        "supports_negative_prompt": True,
        "supports_seeded_generation": True,
        "supports_img2img": True,
        "supports_inpainting": True,
        "supports_variations": True,
    }
    if base:
        capabilities["base_model"] = base
        est = _estimate_repo_weights_gb(base)
        if est:
            # Adapter + runtime overhead margin on top of base weights.
            capabilities["memory_gb"] = round(est + 0.3, 1)

    revision = hf_commit_revision(info)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/generation",
        "hf_repo": repo_id,
        "description": f"Diffusers LoRA adapter: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_LORA_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        # Adapter repos are flat: weights + configs at the top level,
        # tens of MB total. No subfolder pipeline tree to filter.
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
            allow_patterns=["*.safetensors", "*.json", "*.txt"],
            revision=revision,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    # `pipeline_tag` is HF's canonical single-value task field and is the
    # authoritative text-to-image signal. Many community SD checkpoints set
    # it but do NOT mirror "text-to-image" into the loose `tags` bag (which
    # may only carry "diffusers:StableDiffusionPipeline"). Read the structured
    # field first, falling back to the tags mirror for repos that only tag.
    is_text_to_image = (
        getattr(info, "pipeline_tag", None) == "text-to-image"
        or "text-to-image" in tags
    )
    if not is_text_to_image:
        return False
    has_pipeline_config = any(
        Path(f).name == "model_index.json" for f in siblings
    )
    if has_pipeline_config:
        return True
    # Second accepted shape: a LoRA adapter repo (weights only, no
    # pipeline config). Resolved via _resolve_lora and served by pairing
    # with a base pipeline at load time.
    return _is_lora_adapter(siblings, tags)


def _resolve(
    repo_id: str,
    variant: str | None,
    info,
    *,
    base_override: str | None = None,
) -> ResolvedModel:
    revision = hf_commit_revision(info)
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    has_pipeline_config = any(
        Path(f).name == "model_index.json" for f in siblings
    )
    if not has_pipeline_config and _is_lora_adapter(siblings, tags):
        return _resolve_lora(repo_id, info, base_override=base_override)
    # Non-LoRA t2i repos have no base to override; base_override is a
    # silent no-op below (falls through to the standard pipeline path).

    defaults = _infer_defaults(repo_id)
    capabilities = {
        **defaults,
        "supports_negative_prompt": True,
        "supports_seeded_generation": True,
        "supports_img2img": True,
        # Most diffusers text-to-image checkpoints support inpainting and
        # variations via from_pipe(...) on the AutoPipelineForInpainting /
        # AutoPipelineForImage2Image factories. Default both to True; rare
        # exceptions can be overridden per-curated-entry.
        "supports_inpainting": True,
        "supports_variations": True,
    }
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/generation",
        "hf_repo": repo_id,
        "description": f"Diffusers text-to-image: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    # SDXL-Turbo and similar repos ship fp32 + fp16 + standalone single-file
    # checkpoints simultaneously (~47GB total). The diffusers runtime only
    # needs the fp16 subfolder weights (~7GB). Detect fp16 variants here
    # and filter the snapshot_download manifest accordingly.
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    has_fp16_variants = any(".fp16." in name for name in siblings)

    def _download(cache_root: Path) -> Path:
        # The "*/" prefix excludes top-level standalone checkpoints
        # (A1111/ComfyUI single-file format). Diffusers files always
        # live in subfolders (unet/, vae/, text_encoder/, etc.).
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
        search=query, filter="text-to-image",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="image/generation",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "image/generation",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 100,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
