"""Tests for muse.models.supertonic_3: Supertonic ONNX TTS adapter (SDK mocked)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from muse.modalities.audio_speech.protocol import AudioChunk, AudioResult, TTSModel


def _make_adapter():
    """Build a Model bypassing __init__, with a mocked supertonic TTS."""
    from muse.models.supertonic_3 import Model, SUPERTONIC_SAMPLE_RATE

    tts = MagicMock()
    tts.get_voice_style = MagicMock(return_value="STYLE_OBJ")
    # synthesize returns (wav, duration); wav is shape (1, N) float32 in [-1, 1].
    # This matches the real SDK: wav has shape (1, N), not (N,).
    N = SUPERTONIC_SAMPLE_RATE
    wav = (np.random.rand(1, N).astype(np.float32) * 2) - 1
    tts.synthesize = MagicMock(return_value=(wav, np.array([1.0])))

    adapter = object.__new__(Model)
    adapter._tts = tts
    adapter._device = "cpu"
    return adapter


def test_protocol_conformance():
    assert isinstance(_make_adapter(), TTSModel)


def test_model_id():
    assert _make_adapter().model_id == "supertonic-3"


def test_sample_rate_matches_constant():
    from muse.models.supertonic_3 import SUPERTONIC_SAMPLE_RATE
    assert _make_adapter().sample_rate == SUPERTONIC_SAMPLE_RATE


def test_synthesize_returns_float32_audio_result():
    from muse.models.supertonic_3 import SUPERTONIC_SAMPLE_RATE
    r = _make_adapter().synthesize("Hello world")
    assert isinstance(r, AudioResult)
    assert r.sample_rate == SUPERTONIC_SAMPLE_RATE
    assert r.audio.dtype == np.float32
    # The (1, N) wav must be flattened to 1-D
    assert r.audio.ndim == 1
    assert float(r.audio.max()) <= 1.0 and float(r.audio.min()) >= -1.0


def test_synthesize_uses_voice_style_and_lang():
    a = _make_adapter()
    a.synthesize("Hi", voice="M2", lang="ko")
    a._tts.get_voice_style.assert_called_once_with(voice_name="M2")
    # synthesize called with the resolved style + lang
    _, kwargs = a._tts.synthesize.call_args
    assert kwargs.get("voice_style") == "STYLE_OBJ"
    assert kwargs.get("lang") == "ko"


def test_synthesize_defaults_voice_and_lang():
    from muse.models.supertonic_3 import DEFAULT_VOICE
    a = _make_adapter()
    a.synthesize("Hi")
    a._tts.get_voice_style.assert_called_once_with(voice_name=DEFAULT_VOICE)
    assert a._tts.synthesize.call_args.kwargs.get("lang") == "en"


def test_synthesize_defaults_voice_when_none():
    # The /v1/audio/speech route declares `voice: str | None = None` and always
    # forwards `voice=req.voice`, so an omitted voice arrives as explicit None.
    # The default must still apply (mirrors the kokoro fix in d28c82a; a dict
    # default only fires when the key is absent, not present-but-None).
    from muse.models.supertonic_3 import DEFAULT_VOICE
    a = _make_adapter()
    a.synthesize("Hi", voice=None)
    a._tts.get_voice_style.assert_called_once_with(voice_name=DEFAULT_VOICE)


def test_stream_defaults_voice_when_none():
    from muse.models.supertonic_3 import DEFAULT_VOICE
    a = _make_adapter()
    list(a.synthesize_stream("Hi", voice=None))
    a._tts.get_voice_style.assert_called_once_with(voice_name=DEFAULT_VOICE)


def test_synthesize_ignores_unknown_kwargs():
    # protocol: unknown kwargs silently ignored, i.e. NOT forwarded to the SDK.
    a = _make_adapter()
    r = a.synthesize("Hi", temperature=0.9, nonsense=True)
    assert isinstance(r, AudioResult)
    call = a._tts.synthesize.call_args
    assert "temperature" not in call.kwargs and "nonsense" not in call.kwargs


def test_stream_yields_one_chunk():
    chunks = list(_make_adapter().synthesize_stream("Hello"))
    assert len(chunks) == 1
    assert isinstance(chunks[0], AudioChunk)


def test_stream_chunk_is_1d_float32():
    chunks = list(_make_adapter().synthesize_stream("Hello"))
    chunk = chunks[0]
    assert chunk.audio.ndim == 1
    assert chunk.audio.dtype == np.float32


def test_voices_property_returns_list():
    from muse.models.supertonic_3 import Model, SUPERTONIC_VOICES
    a = object.__new__(Model)
    assert a.voices is Model.VOICES
    assert a.voices == SUPERTONIC_VOICES
    assert len(a.voices) > 0


def test_manifest_required_fields():
    from muse.models.supertonic_3 import MANIFEST
    assert MANIFEST["model_id"] == "supertonic-3"
    assert MANIFEST["modality"] == "audio/speech"
    assert MANIFEST["hf_repo"] == "Supertone/supertonic-3"
    assert "supertonic" in MANIFEST["pip_extras"]
    assert MANIFEST["capabilities"]["device"] == "cpu"
    assert len(MANIFEST["capabilities"]["languages"]) == 31


def test_muse_managed_local_assets_disable_sdk_auto_download():
    from muse.models.supertonic_3 import Model

    factory = MagicMock()
    with patch.dict("sys.modules", {
        "supertonic": SimpleNamespace(TTS=factory),
    }):
        Model(local_dir="/muse/pulled/supertonic", device="cpu")

    factory.assert_called_once_with(
        model_dir="/muse/pulled/supertonic",
        auto_download=False,
    )


def test_direct_construction_retains_explicit_sdk_download_fallback():
    from muse.models.supertonic_3 import Model

    factory = MagicMock()
    with patch.dict("sys.modules", {
        "supertonic": SimpleNamespace(TTS=factory),
    }):
        Model(local_dir=None, device="cpu")

    factory.assert_called_once_with(model_dir=None, auto_download=True)
