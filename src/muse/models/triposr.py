"""Bundled muse model: triposr (image-to-3d).

Stability AI's TripoSR: single-image to 3D mesh, ~120MB safetensors,
MIT license. Bundled default for the 3d/generation modality. GLB
output via trimesh; image-to-3d only (the route layer's
`supports_text_to_3d` capability gate keeps the unrelated text route
from invoking this runtime).

Wraps TripoSRRuntime; this script declares the manifest with the right
capability flags (`supports_image_to_3d=True`, `supports_text_to_3d=False`)
and joins the smoke-test matrix.

Foreground isolation: the runtime uses TripoSR's no-remove-bg path,
so the input image should already have its foreground isolated against
a near-uniform (transparent or white) background. Operators wanting
automatic background removal should preprocess client-side before
calling.
"""
from __future__ import annotations

from muse.modalities.model_3d_generation.runtimes.triposr import (
    TripoSRRuntime as Model,
)


MANIFEST = {
    "model_id": "triposr",
    "modality": "3d/generation",
    "hf_repo": "stabilityai/TripoSR",
    "revision": "5b521936b01fbe1890f6f9baed0254ab6351c04a",
    "description": (
        "TripoSR: single-image to 3D mesh, ~120MB, MIT. Image input "
        "should be foreground-isolated (transparent or white "
        "background works best). GLB output."
    ),
    "license": "MIT",
    "pip_extras": [
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
    ],
    "system_packages": [],
    "capabilities": {
        "memory_gb": 1.5,
        "device": "cuda",
        "supports_image_to_3d": True,
        "supports_text_to_3d": False,
        "output_format": "glb",
    },
}


__all__ = ["MANIFEST", "Model"]
