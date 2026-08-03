"""Generic runtimes for image/cv.

Three sibling runtimes, one per primitive:

  HFDepthRuntime
    AutoModelForDepthEstimation + AutoImageProcessor. Depth-Anything,
    DPT, ZoeDepth, etc. all share the encoder-decoder shape.

  HFKeypointRuntime
    Config-dispatched keypoint loading + AutoImageProcessor. ViTPose uses
    VitPoseForPoseEstimation (transformers 4.48+) and receives a single
    full-image bbox per call in v1; other supported families retain the
    generic auto-model path.

  HFObjectDetectionRuntime
    AutoModelForObjectDetection + AutoImageProcessor. DETR, YOLOS,
    RT-DETR, etc.

The HF plugin's _resolve dispatches per-model by tag
(depth-estimation / keypoint-detection / object-detection). The route
layer's capability gate (supports_depth / supports_keypoints /
supports_detection) decides which model each route accepts.
"""
from muse.modalities.image_cv.runtimes.hf_depth import HFDepthRuntime
from muse.modalities.image_cv.runtimes.hf_keypoint import HFKeypointRuntime
from muse.modalities.image_cv.runtimes.hf_object_detection import (
    HFObjectDetectionRuntime,
)


__all__ = [
    "HFDepthRuntime",
    "HFKeypointRuntime",
    "HFObjectDetectionRuntime",
]
