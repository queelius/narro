"""TTS model protocol and audio types for the Narro framework.

Narro is a model-agnostic TTS server. Any model that implements the
TTSModel protocol can be registered and served via the HTTP API.

The protocol is deliberately minimal: text in, audio out. Model-specific
features (alignment, hidden-state caching, etc.) are communicated through
the ``metadata`` dict on AudioResult, not through protocol extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@dataclass
class AudioResult:
    """Output from a TTS model synthesis.

    Attributes:
        audio: Float32 waveform in [-1, 1], shape ``(samples,)``.
        sample_rate: Audio sample rate in Hz.
        metadata: Model-specific extras (e.g. alignment data, entropy).
    """
    audio: np.ndarray
    sample_rate: int
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Audio duration in seconds."""
        return len(self.audio) / self.sample_rate


@dataclass
class AudioChunk:
    """A chunk of streaming audio from a TTS model.

    Attributes:
        audio: Float32 waveform in [-1, 1], shape ``(samples,)``.
        sample_rate: Audio sample rate in Hz.
    """
    audio: np.ndarray
    sample_rate: int


@runtime_checkable
class TTSModel(Protocol):
    """Contract that any Narro TTS backend must satisfy.

    Models are loaded once and registered with the server.  The server
    routes requests to models by ``model_id``.  Each model owns its
    own weights, tokeniser, and decoder -- the server never touches
    model internals.

    Implementations must be safe for sequential reuse (the server
    serialises calls behind a lock).
    """

    @property
    def model_id(self) -> str:
        """Unique identifier for this model (e.g. ``"soprano-80m"``)."""
        ...

    @property
    def sample_rate(self) -> int:
        """Output audio sample rate in Hz."""
        ...

    def synthesize(self, text: str, **kwargs) -> AudioResult:
        """Convert *text* to audio.

        Keyword arguments are model-specific (temperature, top_p, etc.).
        Unknown kwargs should be silently ignored so that the server can
        forward request params without filtering per-model.
        """
        ...

    def synthesize_stream(self, text: str, **kwargs) -> Iterator[AudioChunk]:
        """Streaming variant of :meth:`synthesize`.

        Yields audio chunks as they become available.  The server
        converts these to SSE events.
        """
        ...
