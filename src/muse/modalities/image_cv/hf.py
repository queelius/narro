"""HF resolver plugin for image/cv (depth + keypoints + object detection).

Three-way tag-based dispatch in a single plugin file. Each branch
sets a different runtime_path and a different capability flag on the
synthesized manifest:

  depth-estimation      -> HFDepthRuntime, supports_depth=True
  keypoint-detection    -> HFKeypointRuntime, supports_keypoints=True
  object-detection      -> HFObjectDetectionRuntime, supports_detection=True

Repo-name fallbacks catch checkpoints without the canonical tag:

  depth: trocr is excluded; substrings 'depth', 'dpt', 'zoedepth',
         'midas' all suggest depth.
  keypoint: 'pose', 'keypoint', 'vitpose', 'rtmpose', 'rtmo' all
            suggest keypoint detection.
  detection: 'detr', 'yolos', 'rtdetr', 'rt-detr', 'owlvit',
             'conditional-detr' suggest object detection.

`metric_depth` is auto-set to True when the resolved depth model's
name contains 'zoedepth' or 'metric'; relative-depth models default
to False.

Priority 110: tag-based, same slot as image/ocr, image/segmentation.
Loses to file-pattern plugins (100); wins over text-classification
catch-all (200).

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision


_DEPTH_RUNTIME = (
    "muse.modalities.image_cv.runtimes.hf_depth:HFDepthRuntime"
)
_KEYPOINT_RUNTIME = (
    "muse.modalities.image_cv.runtimes.hf_keypoint:HFKeypointRuntime"
)
_DETECTION_RUNTIME = (
    "muse.modalities.image_cv.runtimes.hf_object_detection"
    ":HFObjectDetectionRuntime"
)
_PIP_EXTRAS_BASE = (
    "torch>=2.1.0",
    # Transformers 5 image processors use torchvision-backed operations.
    "torchvision>=0.16.0",
    "transformers>=4.46.0",
    "Pillow",
    "numpy",
)
# Detection adds timm for backbones (DETR, RT-DETR-ResNet).
_PIP_EXTRAS_DETECTION = (*_PIP_EXTRAS_BASE, "timm")


_DEPTH_NAME_HINTS = ("depth", "dpt", "zoedepth", "midas")
_KEYPOINT_NAME_HINTS = ("vitpose", "rtmpose", "rtmo", "pose-", "keypoint")
_DETECTION_NAME_HINTS = (
    "detr", "yolos", "rtdetr", "rt-detr", "owlvit", "owl-vit",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _is_depth(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "depth-estimation" in tags:
        return True
    repo_id = (getattr(info, "id", "") or "").lower()
    return any(h in repo_id for h in _DEPTH_NAME_HINTS)


def _is_keypoint(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "keypoint-detection" in tags:
        return True
    repo_id = (getattr(info, "id", "") or "").lower()
    return any(h in repo_id for h in _KEYPOINT_NAME_HINTS)


def _is_object_detection(info) -> bool:
    tags = getattr(info, "tags", None) or []
    if "object-detection" in tags:
        return True
    repo_id = (getattr(info, "id", "") or "").lower()
    return any(h in repo_id for h in _DETECTION_NAME_HINTS)


def _is_metric_depth(repo_id: str) -> bool:
    name = repo_id.lower()
    return "zoedepth" in name or "metric" in name


def _sniff(info) -> bool:
    return (
        _is_depth(info)
        or _is_keypoint(info)
        or _is_object_detection(info)
    )


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    revision = hf_commit_revision(info)
    # Per-primitive dispatch. _is_depth / _is_keypoint /
    # _is_object_detection each OR a canonical tag check with a
    # repo-name-hint fallback: a matching tag short-circuits to True;
    # if there's no tag, the name hints catch checkpoints whose
    # uploaders forgot the canonical tag.
    if _is_depth(info):
        runtime_path = _DEPTH_RUNTIME
        pip_extras = list(_PIP_EXTRAS_BASE)
        capabilities = {
            "device": "auto",
            "supports_depth": True,
            "supports_keypoints": False,
            "supports_detection": False,
            "metric_depth": _is_metric_depth(repo_id),
        }
        kind = "Depth"
    elif _is_keypoint(info):
        runtime_path = _KEYPOINT_RUNTIME
        pip_extras = list(_PIP_EXTRAS_BASE)
        capabilities = {
            "device": "auto",
            "supports_depth": False,
            "supports_keypoints": True,
            "supports_detection": False,
        }
        kind = "Keypoint detection"
    else:
        # _is_object_detection branch (sniff guarantees one of the three).
        runtime_path = _DETECTION_RUNTIME
        pip_extras = list(_PIP_EXTRAS_DETECTION)
        capabilities = {
            "device": "auto",
            "supports_depth": False,
            "supports_keypoints": False,
            "supports_detection": True,
        }
        kind = "Object detection"

    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/cv",
        "hf_repo": repo_id,
        "description": f"{kind} model: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": pip_extras,
        "system_packages": [],
        "capabilities": capabilities,
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        siblings = [s.rfilename for s in getattr(info, "siblings", [])]
        has_fp16 = any(".fp16." in f for f in siblings)
        if has_fp16:
            allow_patterns = ["*.fp16.safetensors", "*.json", "*.txt", "*.md"]
        else:
            allow_patterns = ["*.safetensors", "*.json", "*.txt", "*.md"]
        # Older repos still ship pytorch_model.bin (no safetensors).
        allow_patterns.append("pytorch_model.bin")
        # Processor / config files. AutoImageProcessor reads
        # preprocessor_config.json to dispatch to the right processor.
        allow_patterns.extend([
            "preprocessor_config.json", "image_processor_config.json",
        ])
        return Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
            revision=revision,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=runtime_path,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    """Search across all three primitives by issuing one HF query
    per task tag and merging.

    HF's list_models filter accepts only one tag; we issue three
    queries and dedupe by repo id.
    """
    seen: set[str] = set()
    per_task_limit = max(1, limit // 3)
    for task in ("depth-estimation", "keypoint-detection", "object-detection"):
        repos = api.list_models(
            search=query, filter=task,
            sort=sort, limit=per_task_limit,
        )
        for repo in repos:
            if repo.id in seen:
                continue
            seen.add(repo.id)
            yield SearchResult(
                uri=f"hf://{repo.id}",
                model_id=_model_id(repo.id),
                modality="image/cv",
                size_gb=None,
                downloads=getattr(repo, "downloads", None),
                license=None,
                description=f"{task}: {repo.id}",
            )


HF_PLUGIN = {
    "modality": "image/cv",
    # Top-level runtime_path is the depth runtime; _resolve picks per
    # primitive. The metadata field is informational for the discovery
    # registry, not the actual dispatch path.
    "runtime_path": _DEPTH_RUNTIME,
    "pip_extras": _PIP_EXTRAS_BASE,
    "system_packages": (),
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
