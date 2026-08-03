"""Bundled muse model: detr-resnet-50.

DETR with ResNet-50 backbone (Facebook, ~41M params, ~167MB, Apache 2.0).
COCO 80-class object detection. The classic transformer-based detector;
robust baseline for the image/cv detection primitive.

Wraps HFObjectDetectionRuntime.
"""
from __future__ import annotations

from muse.modalities.image_cv.runtimes import HFObjectDetectionRuntime


MANIFEST = {
    "model_id": "detr-resnet-50",
    "modality": "image/cv",
    "hf_repo": "facebook/detr-resnet-50",
    "revision": "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b",
    "description": (
        "DETR ResNet-50: ~41M, COCO 80-class object detection, "
        "Apache 2.0. CPU-friendly default for the image/cv "
        "detection primitive."
    ),
    "license": "Apache 2.0",
    "pip_extras": [
        "torch>=2.1.0", "transformers>=4.46.0", "Pillow",
        "numpy", "timm",  # DETR's ResNet backbone uses timm
    ],
    "system_packages": [],
    "capabilities": {
        "supports_depth": False,
        "supports_keypoints": False,
        "supports_detection": True,
        "device": "auto",
        "memory_gb": 0.5,
    },
}


class Model(HFObjectDetectionRuntime):
    pass
