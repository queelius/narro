"""HF resolver plugin for 3d/generation.

Sniffs HF repos for image-to-3d / text-to-3d shape and synthesizes a
manifest pointing at the right runtime. Priority 110, same slot as the
other modality-specific plugins (audio_classification, audio_embedding,
image_segmentation, image_ocr, image_cv).

Per-family dispatch via _family_for(): Shap-E repos route to
ShapERuntime; TRELLIS repos route to TRELLISRuntime; Hunyuan3D-2
repos route to Hunyuan3DRuntime (both added via v0.44.0+ _Family
entries); unknown repos fall through to TripoSRRuntime via
_DEFAULT_FAMILY. Wonder3D is deferred indefinitely. Adding a new
family means appending one _Family entry to _FAMILIES plus shipping a
runtime file; no new dispatch functions and no new conditional branches
in _resolve.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

import re
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.artifacts import download_hf_artifact_bundle
from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision


_TRIPOSR_RUNTIME_PATH = (
    "muse.modalities.model_3d_generation.runtimes.triposr:TripoSRRuntime"
)
_SHAPE_E_RUNTIME_PATH = (
    "muse.modalities.model_3d_generation.runtimes.shape_e:ShapERuntime"
)

# pip_extras audit philosophy: declare every direct + transitive import
# the runtime can hit at load time. tsr is the canonical TripoSR pip
# package; trimesh handles GLB serialization; omegaconf + einops are
# tsr's transitive deps that AutoModel.from_pretrained triggers.
_TRIPOSR_PIP_EXTRAS: tuple[str, ...] = (
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.40.0",
    "trimesh>=4.0",
    "tsr",
    "Pillow",
    "numpy",
    "omegaconf",
    "einops",
    "huggingface_hub",
)
_SHAPE_E_PIP_EXTRAS: tuple[str, ...] = (
    "torch>=2.1.0",
    "diffusers>=0.27.0",
    "transformers",
    "trimesh",
)
_TRELLIS_RUNTIME_PATH = (
    "muse.modalities.model_3d_generation.runtimes.trellis:TRELLISRuntime"
)
# TRELLIS uses Microsoft's standalone SDK (NOT a transformers/diffusers
# AutoPipeline). Verified at v0.44.0 implementation time against the
# real downloaded SDK. The TRELLIS GitHub repo has native-build deps
# (kaolin, xformers, flash-attn, nvdiffrast) that pip cannot install
# cleanly on all systems; Microsoft's setup.sh script is the official
# install reference. Muse installs its portable ``--basic`` dependencies and
# a sparse, immutable SDK checkout; users still need compatible native CUDA
# packages for their local PyTorch/CUDA stack. Upstream has no tags/releases;
# the SDK API was reviewed at the immutable commit below.
# See: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/setup.sh
_TRELLIS_REVISION = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
_TRELLIS_FLEXICUBES_REVISION = "815e075a2a400d06c48d94c347674344ed6ae5c5"
_TRELLIS_PIP_EXTRAS: tuple[str, ...] = (
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.46.0",
    "diffusers>=0.27.0",
    "safetensors",
    "huggingface_hub",
    "trimesh",
    "accelerate",
    "Pillow",
    "numpy",
    # Microsoft's official setup.sh --basic dependency inventory. Native
    # CUDA/PyTorch-specific flags remain an explicit operator prerequisite.
    "imageio",
    "imageio-ffmpeg",
    "tqdm",
    "easydict",
    "opencv-python-headless",
    "scipy",
    "ninja",
    "rembg",
    "onnxruntime",
    "open3d",
    "xatlas",
    "pyvista",
    "pymeshfix",
    "igraph",
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
)
_TRELLIS_PYTHON_SOURCES: tuple[dict, ...] = (
    {
        "type": "git",
        "name": "trellis",
        "url": "https://github.com/microsoft/TRELLIS.git",
        "revision": _TRELLIS_REVISION,
        "sparse_paths": ("trellis",),
        "required_paths": (
            "trellis/__init__.py",
            "trellis/pipelines/trellis_image_to_3d.py",
            "trellis/representations/mesh/flexicubes/flexicubes.py",
        ),
        "pth_path": ".",
        "submodules": (
            {
                "path": "trellis/representations/mesh/flexicubes",
                "url": "https://github.com/MaxtirError/FlexiCubes.git",
                "revision": _TRELLIS_FLEXICUBES_REVISION,
            },
        ),
    },
)
_HUNYUAN3D_RUNTIME_PATH = (
    "muse.modalities.model_3d_generation.runtimes.hunyuan3d:Hunyuan3DRuntime"
)
_HUNYUAN3D_T2I_REPO = (
    "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
)
_HUNYUAN3D_T2I_REVISION = "527cf2ecce7c04021975938f8b0e44e35d2b1ed9"
_HUNYUAN3D_SHAPE_SUBDIR = "shape"
_HUNYUAN3D_T2I_SUBDIR = "text_to_image"
# Hunyuan3D-2 uses Tencent's standalone SDK (NOT a transformers/diffusers
# AutoPipeline). Verified at v0.45.0 implementation time. Tencent's
# GitHub repo has native-build deps (kaolin, xformers, custom CUDA
# extensions) that pip cannot install cleanly on all systems; Tencent's
# setup script is the official install path. Users may need to follow that
# script manually after `muse pull hunyuan3d-2` if the git pip-install below
# fails. Upstream has no tags/releases; the SDK API was reviewed at the
# immutable commit below. See: https://github.com/Tencent-Hunyuan/Hunyuan3D-2/tree/f8db63096c8282cb27354314d896feba5ba6ff8a
_HUNYUAN3D_PIP_EXTRAS: tuple[str, ...] = (
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.46.0",
    "diffusers>=0.27.0",
    "trimesh",
    "accelerate",
    "Pillow",
    "numpy",
    # Tencent Hunyuan3D-2 SDK from GitHub. May fail on hosts without
    # CUDA toolchain or compatible NVIDIA driver. Fallback: clone +
    # follow setup.sh inside the per-model venv at ~/.muse/venvs/hunyuan3d-2/.
    "hy3dgen @ git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git@f8db63096c8282cb27354314d896feba5ba6ff8a",
)


@dataclass(frozen=True)
class _Family:
    """One 3D-generation model family: how to detect it and how to load it.

    Adding a new family in v0.44.0+ means appending one _Family entry to
    _FAMILIES below, plus shipping the named runtime file. No new
    dispatch functions, no new conditional branches in _resolve.

    Fields:
      name_hints: repo-name substrings (lowercased) that identify this family.
      runtime_path: backend_path written into the catalog manifest.
      pip_extras: per-model pip dependencies installed into the model venv.
      python_sources: reviewed non-packaged source trees installed into the venv.
      capability_overrides: dict merged over the default capabilities block;
        values here win. Shap-E uses this to flip image/text support flags.
      system_packages: OS-level packages (e.g. libGL) needed by the runtime.
        Empty for most families; Wonder3D / TRELLIS may declare libGL.
    """

    name_hints: tuple[str, ...]
    runtime_path: str
    pip_extras: tuple[str, ...]
    python_sources: tuple[dict, ...] = ()
    capability_overrides: dict = field(default_factory=dict)
    system_packages: tuple[str, ...] = ()


_FAMILIES: tuple[_Family, ...] = (
    _Family(  # Shap-E (unchanged from v0.43.x)
        name_hints=("shap-e", "shape-e"),
        runtime_path=_SHAPE_E_RUNTIME_PATH,
        pip_extras=_SHAPE_E_PIP_EXTRAS,
        capability_overrides={
            "supports_image_to_3d": False,
            "supports_text_to_3d": True,
        },
    ),
    _Family(  # TRELLIS (unchanged from v0.44.0)
        name_hints=("trellis",),
        runtime_path=_TRELLIS_RUNTIME_PATH,
        pip_extras=_TRELLIS_PIP_EXTRAS,
        python_sources=_TRELLIS_PYTHON_SOURCES,
        capability_overrides={
            "supports_image_to_3d": True,
            "supports_text_to_3d": False,
        },
        system_packages=("git", "nvcc"),
    ),
    _Family(  # NEW v0.45.0: Hunyuan3D-2, dual-direction
        name_hints=("hunyuan3d",),
        runtime_path=_HUNYUAN3D_RUNTIME_PATH,
        pip_extras=_HUNYUAN3D_PIP_EXTRAS,
        capability_overrides={
            "supports_image_to_3d": True,
            "supports_text_to_3d": True,
        },
    ),
    # Wonder3D: deferred indefinitely (v0.44.0 decision).
)

_DEFAULT_FAMILY = _Family(
    name_hints=(),
    runtime_path=_TRIPOSR_RUNTIME_PATH,
    pip_extras=_TRIPOSR_PIP_EXTRAS,
)


def _matches_hint(name: str, hint: str) -> bool:
    """Word-boundary substring match.

    The hint matches `name` only when it appears with non-alphanumeric
    chars on both sides (or at string boundaries). Prevents false positives
    like `my-reshape-enhancer` matching `shape-e`.
    """
    pattern = re.compile(
        rf"(?<![a-z0-9]){re.escape(hint)}(?![a-z0-9])",
        re.IGNORECASE,
    )
    return bool(pattern.search(name))


def _family_for(repo_id: str) -> _Family:
    """Pick the family by name-hint match; fall back to TripoSR."""
    name = repo_id.lower()
    return next(
        (f for f in _FAMILIES if any(_matches_hint(name, h) for h in f.name_hints)),
        _DEFAULT_FAMILY,
    )



# Repo-name allowlist: the canonical 3D generation repos. These match
# regardless of HF tagging (some repos are sloppy with tags). Highest
# precedence.
_NAME_HINTS = (
    "triposr",
    "trellis",
    "wonder3d",
    "hunyuan3d",
    "shap-e",
    "instantmesh",
    "stable-3d",
)
# Tag-based fallback. The canonical 3D generation tags on HF.
_TAG_HINTS = ("image-to-3d", "text-to-3d")
def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _download_hunyuan_bundle(
    cache_root: Path,
    *,
    repo_id: str,
    revision: str,
) -> Path:
    """Download both immutable Hunyuan checkpoints and publish atomically."""
    return download_hf_artifact_bundle(
        cache_root,
        bundle_name="hunyuan3d",
        artifacts=(
            {
                "repo_id": repo_id,
                "revision": revision,
                "subdir": _HUNYUAN3D_SHAPE_SUBDIR,
            },
            {
                "repo_id": _HUNYUAN3D_T2I_REPO,
                "revision": _HUNYUAN3D_T2I_REVISION,
                "subdir": _HUNYUAN3D_T2I_SUBDIR,
            },
        ),
        snapshot_download_fn=snapshot_download,
    )


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff(info) -> bool:
    repo_id = (getattr(info, "id", "") or "").lower()
    # Repo-name allowlist dominates: well-known 3D generators always
    # claim, even when their HF tags are sloppy or absent.
    if any(s in repo_id for s in _NAME_HINTS):
        return True
    # Tag-based fallback.
    tags = getattr(info, "tags", None) or []
    return any(t in tags for t in _TAG_HINTS)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    """Synthesize a ResolvedModel by dispatching to the matching _Family.

    The per-family registry (_FAMILIES) selects the runtime by repo-name
    substring match. Current families: Shap-E (text-to-3D),
    TRELLIS (image-to-3D, installed SDK), Hunyuan3D-2 (both
    directions, installed SDK). Unknown repos fall through to
    TripoSR (image-to-3D) via _DEFAULT_FAMILY.
    """
    family = _family_for(repo_id)
    revision = hf_commit_revision(info)
    is_hunyuan = family.runtime_path == _HUNYUAN3D_RUNTIME_PATH

    capabilities: dict = {
        "device": "cuda",
        "supports_image_to_3d": True,
        "supports_text_to_3d": False,
        "output_format": "glb",
    }
    capabilities.update(family.capability_overrides)
    if is_hunyuan:
        capabilities.update({
            "shape_model_subdir": _HUNYUAN3D_SHAPE_SUBDIR,
            "t2i_model_subdir": _HUNYUAN3D_T2I_SUBDIR,
            "t2i_model_id": _HUNYUAN3D_T2I_REPO,
            "t2i_revision": _HUNYUAN3D_T2I_REVISION,
        })

    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "3d/generation",
        "hf_repo": repo_id,
        "description": f"3D generation: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(family.pip_extras),
        "system_packages": list(family.system_packages),
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision
    if family.python_sources:
        manifest["python_sources"] = copy.deepcopy(list(family.python_sources))

    def _download(cache_root: Path) -> Path:
        # snapshot_download returns the local cache directory. Without
        # explicit allow_patterns the full repo lands; 3d generators
        # tend to ship config.json + model.safetensors plus optional
        # decoder/triplane files, so taking everything is safer than
        # guessing patterns per family.
        if revision is None:
            raise RuntimeError(
                f"refusing mutable Hugging Face download for {repo_id!r}: "
                "repository metadata did not include an immutable commit"
            )
        if is_hunyuan:
            return _download_hunyuan_bundle(
                cache_root,
                repo_id=repo_id,
                revision=revision,
            )
        return Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    provenance: tuple[dict, ...] = ()
    if revision is not None:
        items = [{
            "repo_id": repo_id,
            "revision": revision,
            "subdir": _HUNYUAN3D_SHAPE_SUBDIR if is_hunyuan else ".",
        }]
        if is_hunyuan:
            items.append({
                "repo_id": _HUNYUAN3D_T2I_REPO,
                "revision": _HUNYUAN3D_T2I_REVISION,
                "subdir": _HUNYUAN3D_T2I_SUBDIR,
            })
        provenance = tuple(items)

    return ResolvedModel(
        manifest=manifest,
        backend_path=family.runtime_path,
        download=_download,
        artifact_provenance=provenance,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    """Search HF for 3d-tagged repos.

    Iterates both `image-to-3d` and `text-to-3d` tags and dedupes by
    repo id so a multi-tagged repo (TRELLIS, Hunyuan3D-2) yields one
    SearchResult, not two.
    """
    seen: set[str] = set()
    for tag in _TAG_HINTS:
        repos = api.list_models(
            search=query, filter=tag,
            sort=sort, limit=limit,
        )
        for repo in repos:
            repo_id = getattr(repo, "id", None)
            if not repo_id or repo_id in seen:
                continue
            seen.add(repo_id)
            yield SearchResult(
                uri=f"hf://{repo_id}",
                model_id=_model_id(repo_id),
                modality="3d/generation",
                size_gb=None,
                downloads=getattr(repo, "downloads", None),
                license=None,
                description=repo_id,
            )


HF_PLUGIN = {
    "scheme": "hf",
    "modality": "3d/generation",
    # Framework-required top-level keys. The values here are TripoSR-specific
    # placeholders that satisfy the plugin contract (REQUIRED_HF_PLUGIN_KEYS in
    # core/discovery.py); the actual per-resolve runtime_path and pip_extras
    # come from _family_for(repo_id) inside _resolve below.
    "runtime_path": _TRIPOSR_RUNTIME_PATH,
    "pip_extras": _TRIPOSR_PIP_EXTRAS,
    "system_packages": (),
    # 110: tag-based, more specific than text-classification (200) but
    # loses to file-pattern plugins (100). Same slot as
    # image_segmentation, audio_embedding, image_ocr, audio_classification,
    # image_cv.
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
