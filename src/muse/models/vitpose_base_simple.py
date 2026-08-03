"""Bundled muse model: vitpose-base-simple.

ViTPose base (simple variant) keypoint / pose detection. ~85M params,
~330MB on disk, Apache 2.0. COCO 17-keypoint output (nose, eyes, ears,
shoulders, elbows, wrists, hips, knees, ankles).

Wraps HFKeypointRuntime; v1 receives a single full-image bbox per
call (the whole image is the entity to extract pose from).
Multi-person extraction needs an upstream person detector.
"""
from __future__ import annotations

from muse.modalities.image_cv.runtimes import HFKeypointRuntime


MANIFEST = {
    "model_id": "vitpose-base-simple",
    "modality": "image/cv",
    "hf_repo": "usyd-community/vitpose-base-simple",
    "revision": "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45",
    "description": (
        "ViTPose Base Simple: ~85M, COCO 17-keypoint pose, "
        "Apache 2.0. v1 takes one full-image bbox per call; "
        "multi-person pose needs a person detector upstream."
    ),
    "license": "Apache 2.0",
    "pip_extras": [
        "torch>=2.1.0", "torchvision>=0.16.0",
        "transformers>=4.48.0", "scipy", "Pillow", "numpy",
    ],
    "system_packages": [],
    "capabilities": {
        "supports_depth": False,
        "supports_keypoints": True,
        "supports_detection": False,
        "device": "auto",
        "memory_gb": 0.6,
    },
}


class Model(HFKeypointRuntime):
    pass
