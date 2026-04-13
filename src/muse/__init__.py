"""Muse: model-agnostic multi-modality generation server.

Modalities:
  - audio.speech  — /v1/audio/speech (text-to-speech)
  - images.generations — /v1/images/generations (text-to-image)

Heavy backends (transformers, diffusers) are imported lazily inside
individual backend modules to keep CLI startup instant.
"""

__version__ = "0.9.0"
