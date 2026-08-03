"""Bundled muse model: depth-anything-v2-small.

Relative monocular depth estimation (Depth-Anything V2 Small,
~25M params, ~100MB on disk, Apache 2.0). Strong cross-domain
generalization for relative depth; the bundled default for the
image/cv modality's depth primitive.

Wraps HFDepthRuntime; `metric_depth: False` because Depth-Anything
emits relative inverse depth (not metric meters). For metric depth,
use the curated `zoedepth-nyu-kitti` entry instead.
"""
from __future__ import annotations

from muse.modalities.image_cv.runtimes import HFDepthRuntime


MANIFEST = {
    "model_id": "depth-anything-v2-small",
    "modality": "image/cv",
    "hf_repo": "depth-anything/Depth-Anything-V2-Small-hf",
    "revision": "5426e4f0f36572d16453bbda7a8389317b1bef99",
    "description": (
        "Depth-Anything V2 Small: 25M, relative inverse depth, "
        "Apache 2.0. Strong cross-domain generalization; CPU-friendly "
        "default for the image/cv depth primitive."
    ),
    "license": "Apache 2.0",
    "pip_extras": [
        # torchvision: transformers>=5 builds the DPT/DepthAnything image
        # processor via torchvision-backed ops; without it AutoImageProcessor
        # raises "requires the Torchvision library". (detr-resnet-50 gets it
        # transitively through timm; this model declares neither, so list it.)
        "torch>=2.1.0", "torchvision", "transformers>=4.46.0", "Pillow", "numpy",
    ],
    "system_packages": [],
    "capabilities": {
        "supports_depth": True,
        "supports_keypoints": False,
        "supports_detection": False,
        "metric_depth": False,
        "device": "auto",
        "memory_gb": 0.5,
    },
}


class Model(HFDepthRuntime):
    """The runtime is the model.

    Discovery requires every bundled script to expose a class named
    `Model`; this trivial subclass satisfies that.
    """
    pass
