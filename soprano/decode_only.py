"""Lightweight decode-only module for Soprano TTS.

Loads only the Vocos decoder (~5MB) without the LLM (~374MB).
Useful for decoding pre-encoded .soprano files on resource-constrained devices.

Usage:
    from soprano.decode_only import decode, decode_to_wav
    from soprano.encoded import load

    encoded = load("blog-post.soprano")
    audio = decode(encoded)              # list[torch.Tensor]
    decode_to_wav(encoded, "output.wav") # convenience
"""

import logging

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.io import wavfile

from .encoded import EncodedSpeech
from .tts import SAMPLE_RATE, INT16_MAX, TOKEN_SIZE, HIDDEN_DIM, MODEL_ID
from .vocos.decoder import SopranoDecoder
from .vocos.migrate_weights import load_with_migration

logger = logging.getLogger(__name__)


def load_decoder(model_path=None, compile=False):
    """Load and return a standalone Vocos decoder.

    Args:
        model_path: Path to local model directory. If None, downloads from HuggingFace.
        compile: Enable torch.compile (default False for lightweight usage).

    Returns:
        SopranoDecoder ready for inference.
    """
    import os
    decoder = SopranoDecoder()
    if model_path:
        decoder_path = os.path.join(model_path, 'decoder.pth')
    else:
        decoder_path = hf_hub_download(repo_id=MODEL_ID, filename='decoder.pth')
    state_dict = torch.load(decoder_path, map_location='cpu', weights_only=True)
    state_dict = load_with_migration(state_dict)
    decoder.load_state_dict(state_dict)
    decoder.train(False)

    if compile:
        try:
            decoder = torch.compile(decoder, mode="reduce-overhead")
        except Exception as e:
            import warnings
            warnings.warn(f"torch.compile failed for decoder: {e}")

    return decoder


def decode(encoded, decoder=None, decoder_batch_size=4):
    """Decode an EncodedSpeech into audio tensors.

    Args:
        encoded: EncodedSpeech from encode_batch() or load().
        decoder: Pre-loaded SopranoDecoder. If None, loads from HuggingFace.
        decoder_batch_size: Batch size for decoder.

    Returns:
        List of 1D torch.Tensor audio waveforms, one per input text.
    """
    if decoder is None:
        decoder = load_decoder()

    items = []
    for s in encoded.sentences:
        hs = torch.from_numpy(s.hidden_states)
        items.append((hs, s.text_index, s.sentence_index))

    items.sort(key=lambda x: -x[0].size(0))

    num_texts = encoded.num_texts
    audio_concat = [[] for _ in range(num_texts)]
    for s in encoded.sentences:
        audio_concat[s.text_index].append(None)

    hidden_states = [x[0] for x in items]
    meta = [(x[1], x[2]) for x in items]

    for idx in range(0, len(hidden_states), decoder_batch_size):
        batch = hidden_states[idx:idx+decoder_batch_size]
        lengths = [x.size(0) for x in batch]
        N = len(lengths)
        max_len = lengths[0]

        batch_hidden_states = torch.zeros((N, HIDDEN_DIM, max_len))
        for i in range(N):
            seq_len = lengths[i]
            batch_hidden_states[i, :, max_len-seq_len:] = batch[i].T

        with torch.inference_mode():
            audio = decoder(batch_hidden_states)

        for i in range(N):
            text_id, sentence_id = meta[idx+i]
            trim = lengths[i] * TOKEN_SIZE - TOKEN_SIZE
            audio_concat[text_id][sentence_id] = audio[i].squeeze()[-trim:] if trim > 0 else audio[i].squeeze()[:0]

    return [torch.cat(x) for x in audio_concat]


def decode_to_wav(encoded, out_path, decoder=None, decoder_batch_size=4):
    """Decode an EncodedSpeech and write to a WAV file.

    All texts are concatenated into a single audio stream.

    Args:
        encoded: EncodedSpeech from encode_batch() or load().
        out_path: Output WAV file path.
        decoder: Pre-loaded SopranoDecoder. If None, loads from HuggingFace.
        decoder_batch_size: Batch size for decoder.
    """
    audio_list = decode(encoded, decoder=decoder, decoder_batch_size=decoder_batch_size)
    audio = torch.cat(audio_list)
    pcm = (np.clip(audio.numpy(), -1.0, 1.0) * INT16_MAX).astype(np.int16)
    wavfile.write(out_path, SAMPLE_RATE, pcm)
    logger.info("Wrote %s (%.1f seconds)", out_path, len(pcm) / SAMPLE_RATE)
