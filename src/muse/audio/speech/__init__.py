"""Muse audio.speech modality: text-to-speech."""
from muse.audio.speech.client import SpeechClient
from muse.audio.speech.protocol import AudioChunk, AudioResult, TTSModel

__all__ = ["SpeechClient", "AudioChunk", "AudioResult", "TTSModel"]
