"""SmolVLM-256M-Instruct: tiny CPU-runnable VLM, ~500MB.

Default vision-language model bundled with muse. Image captioning,
basic VQA, multi-image inference. Apache 2.0 license. Runs on CPU
in ~5-15 seconds per inference.

Architecture: Idefics3 (HuggingFaceTB), the same family as the larger
SmolVLM-Instruct (2.2B) and SmolVLM2 video model.
"""
from muse.modalities.chat_completion.runtimes.transformers_vlm import (
    HFVisionLanguageModel as Model,
)


MANIFEST = {
    "model_id": "smolvlm-256m-instruct",
    "modality": "chat/completion",
    "hf_repo": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "revision": "7e3e67edbbed1bf9888184d9df282b700a323964",
    "description": (
        "SmolVLM-256M-Instruct: tiny CPU-runnable VLM, ~500MB. "
        "Image captioning, basic VQA, multi-image."
    ),
    "license": "apache-2.0",
    "pip_extras": [
        "torch>=2.1.0",
        # torchvision: transformers>=5 builds the SmolVLM image processor via
        # torchvision-backed ops; without it AutoProcessor fails to load the
        # model ("Unrecognized image processor").
        "torchvision",
        "transformers>=4.46.0",
        "accelerate",
        "Pillow",
    ],
    "capabilities": {
        "memory_gb": 1.0,
        "device": "auto",
        "supports_vision": True,
        "supports_multi_image": True,
        "supports_tools": False,
        "chat_format": None,
    },
}
