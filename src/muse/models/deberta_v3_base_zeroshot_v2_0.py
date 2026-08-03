"""Bundled muse model: deberta-v3-base-zeroshot-v2.0.

English zero-shot NLI classifier (184M, MIT). Strong cross-domain
transfer; commonly competitive with proprietary LLMs on novel-label
classification benchmarks while running CPU-friendly.

Wraps the HFZeroShotPipeline runtime. Pass `candidate_labels` at
request time; the model scores each label as an NLI hypothesis
against the input text.
"""
from __future__ import annotations

from muse.modalities.text_classification.runtimes import HFZeroShotPipeline


MANIFEST = {
    "model_id": "deberta-v3-base-zeroshot-v2.0",
    "modality": "text/classification",
    "hf_repo": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "revision": "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8",
    "description": (
        "DeBERTa v3 base zero-shot NLI (184M, MIT). English. "
        "Pass candidate_labels at request time; the model scores each "
        "label as an NLI hypothesis against the input. Strong "
        "performance on novel-label classification."
    ),
    "license": "MIT",
    "pip_extras": ["torch>=2.1.0", "transformers>=4.40.0"],
    "system_packages": [],
    "capabilities": {
        "supports_classification": False,
        "supports_zero_shot": True,
        "device": "auto",
        "memory_gb": 1.2,
    },
}


class Model(HFZeroShotPipeline):
    """The runtime IS the model.

    All loading + inference logic lives in HFZeroShotPipeline. This
    subclass exists to satisfy discover_models, which expects every
    bundled script to expose a class named exactly `Model`.
    """
    pass
