"""Bundled muse model: trocr-base-printed.

English printed-text OCR (Microsoft, 334M, MIT). CPU-friendly default
for the image/ocr modality. Returns plain extracted text per image.

Wraps HFVision2SeqRuntime; this script declares the manifest with the
right capability flags (supports_handwritten=False, supports_math=False)
and joins the smoke-test matrix.
"""
from __future__ import annotations

from muse.modalities.image_ocr.runtimes import HFVision2SeqRuntime


MANIFEST = {
    "model_id": "trocr-base-printed",
    "modality": "image/ocr",
    "hf_repo": "microsoft/trocr-base-printed",
    "revision": "93450be3f1ed40a930690d951ef3932687cc1892",
    "description": (
        "TrOCR base printed: 334M, English printed text, MIT. "
        "CPU-friendly default OCR for line-level extraction."
    ),
    "license": "MIT",
    "pip_extras": [
        "torch>=2.1.0", "torchvision>=0.16.0",
        "transformers>=4.40.0", "Pillow",
    ],
    "system_packages": [],
    "capabilities": {
        "device": "auto",
        "memory_gb": 0.7,
        "max_new_tokens": 256,
        "supports_handwritten": False,
        "supports_math": False,
    },
}


class Model(HFVision2SeqRuntime):
    """The runtime IS the model.

    All loading + inference logic lives in HFVision2SeqRuntime. This
    subclass exists to satisfy discover_models, which expects every
    bundled script to expose a class named exactly `Model`.
    """
    pass
