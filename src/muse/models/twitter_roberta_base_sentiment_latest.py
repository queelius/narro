"""Bundled muse model: twitter-roberta-base-sentiment-latest.

3-label English sentiment classifier (positive/neutral/negative).
125M params, ~500MB on disk, MIT license. CPU-friendly default for
sentiment analysis.

Wraps the existing HFTextClassifier runtime. The script exists to
declare the manifest with the right capability flags
(supports_classification=True), join the smoke-test matrix, and
provide a curated-alias entry point.
"""
from __future__ import annotations

from muse.modalities.text_classification.runtimes import HFTextClassifier


MANIFEST = {
    "model_id": "twitter-roberta-base-sentiment-latest",
    "modality": "text/classification",
    "hf_repo": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "revision": "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
    "description": (
        "3-label English sentiment classifier (125M, MIT). "
        "Returns positive/neutral/negative with softmax probabilities. "
        "CPU-friendly; common baseline for sentiment analysis."
    ),
    "license": "MIT",
    "pip_extras": ["torch>=2.1.0", "transformers>=4.40.0"],
    "system_packages": [],
    "capabilities": {
        "supports_classification": True,
        "supports_zero_shot": False,
        "device": "auto",
        "memory_gb": 0.6,
    },
}


class Model(HFTextClassifier):
    """The runtime IS the model.

    All actual loading + inference logic lives in HFTextClassifier
    (lazy import discipline, _ensure_deps, classify method). This
    subclass exists to satisfy discover_models, which expects every
    bundled script to expose a class named exactly `Model`.
    """
    pass
