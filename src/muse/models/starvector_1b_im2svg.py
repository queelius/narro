"""Bundled StarVector-1B image-to-SVG model.

The official 5.14GB mixed-precision checkpoint is pinned to a reviewed
Hugging Face revision and loaded in fp16. Muse downloads only data
files and uses its local inference-only architecture adapter; no Hub
Python code is executed.
"""
from muse.modalities.image_vectorization.runtimes import StarVectorRuntime


_REVISION = "380ab95d25a8e9ab1dc825debe238b4953ae13b9"

MANIFEST = {
    "model_id": "starvector-1b-im2svg",
    "modality": "image/vectorization",
    "hf_repo": "starvector/starvector-1b-im2svg",
    "description": (
        "StarVector-1B im2svg: raster icons, logos, and diagrams to "
        "editable static SVG; Apache 2.0; 5.14GB checkpoint, fp16 load"
    ),
    "license": "Apache 2.0",
    "pip_extras": [
        "torch>=2.1.0",
        "transformers==4.49.0",
        "accelerate>=0.26.0",
        "safetensors",
        "Pillow>=9.1.0",
        "numpy<2",
    ],
    "system_packages": [],
    "capabilities": {
        "device": "auto",
        "dtype": "fp16",
        "memory_gb": 7.0,
        "max_new_tokens": 4096,
        "supports_image_to_svg": True,
        "output_mime_type": "image/svg+xml",
        "static_svg_only": True,
        "revision": _REVISION,
        "allow_patterns": [
            "config.json",
            "*.safetensors",
            "*.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        ],
    },
}


class Model(StarVectorRuntime):
    """Discovery wrapper around the exact StarVector runtime."""
