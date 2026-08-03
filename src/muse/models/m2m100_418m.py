"""facebook/m2m100_418M: many-to-many multilingual translation, 100 languages.

Bundled default for `text/translation`. ~2GB on disk, MIT license.
Backs the LibreTranslate-compatible `/v1/translate` route (and
`/v1/languages`) via `muse.modalities.text_translation`.

`Model` aliases `TranslationRuntime` directly (the VLM-bundled pattern,
see `smolvlm_256m_instruct.py`): one generic runtime already serves
m2m100/nllb/opus-mt/madlad via per-family dispatch on the HF repo id
(see `muse.modalities.text_translation.runtimes.hf_translation`), so
this script must not duplicate any construction logic.
"""
from muse.modalities.text_translation.runtimes.hf_translation import (
    TranslationRuntime as Model,
)


MANIFEST = {
    "model_id": "m2m100-418m",
    "modality": "text/translation",
    "hf_repo": "facebook/m2m100_418M",
    "revision": "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636",
    "description": (
        "M2M-100 418M: many-to-many translation across 100 languages, "
        "~2GB, MIT. Backs the LibreTranslate-compatible /v1/translate route."
    ),
    "license": "MIT",
    "pip_extras": [
        "transformers",
        "torch",
        "sentencepiece",
    ],
    "capabilities": {
        "device": "auto",
        # Conservative estimate; `muse models probe` self-heals it to
        # the measured peak on first load.
        "memory_gb": 2.5,
        "num_beams": 4,
    },
}
