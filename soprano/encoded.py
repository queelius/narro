"""Intermediate representation for Soprano TTS encoder/decoder pipeline.

The IR decouples the LLM encoder from the Vocos decoder, enabling:
- Offline encoding (encode once, decode many times)
- Cross-platform decoding (ship .soprano files to browser/JS)
- Quality estimation via per-token entropy
- Text-audio alignment via attention weights
"""

import io
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


@dataclass
class SentenceEncoding:
    """One sentence's encoder output.

    Attributes:
        hidden_states: Decoder input, shape (T, 512) float32.
        token_ids: Generated token IDs, shape (T,) int32.
        token_entropy: Per-token entropy in nats, shape (T,) float32.
        finish_reason: 'stop' (EOS reached) or 'length' (max tokens).
        text: Original sentence text before wrapping.
        text_index: Index within the input texts batch.
        sentence_index: Index within sentences of that text.
        attention_weights: Last-layer attention averaged over heads, shape (T, input_len). Opt-in.
        input_token_offsets: Char offsets for input tokens, for word-level highlighting.
    """
    hidden_states: np.ndarray
    token_ids: np.ndarray
    token_entropy: np.ndarray
    finish_reason: str
    text: str
    text_index: int
    sentence_index: int
    attention_weights: Optional[np.ndarray] = None
    input_token_offsets: Optional[list] = None


@dataclass
class EncodedSpeech:
    """Full encoding result from the Soprano encoder.

    Attributes:
        sentences: All sentence encodings, ordered by (text_index, sentence_index).
        model_id: HuggingFace model identifier.
        format_version: Wire format version for forward compatibility.
        sample_rate: Output audio sample rate (Hz).
        token_audio_samples: Audio samples per decoder token.
        hidden_dim: LLM hidden state dimension.
        top_p: Nucleus sampling threshold used during generation.
        temperature: Sampling temperature used during generation.
        repetition_penalty: Repetition penalty used during generation.
    """
    sentences: list
    model_id: str
    format_version: int = FORMAT_VERSION
    sample_rate: int = 32000
    token_audio_samples: int = 2048
    hidden_dim: int = 512
    top_p: float = 0.95
    temperature: float = 0.0
    repetition_penalty: float = 1.2

    @property
    def total_tokens(self) -> int:
        return sum(len(s.hidden_states) for s in self.sentences)

    @property
    def mean_entropy(self) -> float:
        all_entropy = np.concatenate([s.token_entropy for s in self.sentences])
        if len(all_entropy) == 0:
            return 0.0
        return float(np.mean(all_entropy))

    @property
    def estimated_duration(self) -> float:
        """Estimated audio duration in seconds."""
        return self.total_tokens * self.token_audio_samples / self.sample_rate

    @property
    def num_texts(self) -> int:
        if not self.sentences:
            return 0
        return max(s.text_index for s in self.sentences) + 1


def save(encoded: EncodedSpeech, path: str, compress: bool = True) -> None:
    """Save EncodedSpeech to a .soprano file (npz container).

    Wire format uses float16 for hidden states and entropy to halve file size.
    Token IDs are stored as uint16. No pickle is used — all data is stored as
    numpy arrays with JSON metadata (safe serialization).

    Args:
        encoded: The encoded speech to save.
        path: Output file path.
        compress: Use np.savez_compressed (default True).
    """
    arrays = {}
    sentence_meta = []

    for i, s in enumerate(encoded.sentences):
        arrays[f'hidden_{i}'] = s.hidden_states.astype(np.float16)
        arrays[f'token_ids_{i}'] = s.token_ids.astype(np.uint16)
        arrays[f'entropy_{i}'] = s.token_entropy.astype(np.float16)
        if s.attention_weights is not None:
            arrays[f'attention_{i}'] = s.attention_weights.astype(np.float16)

        sentence_meta.append({
            'finish_reason': s.finish_reason,
            'text': s.text,
            'text_index': s.text_index,
            'sentence_index': s.sentence_index,
            'has_attention': s.attention_weights is not None,
            'input_token_offsets': s.input_token_offsets,
        })

    meta = {
        'format_version': encoded.format_version,
        'model_id': encoded.model_id,
        'sample_rate': encoded.sample_rate,
        'token_audio_samples': encoded.token_audio_samples,
        'hidden_dim': encoded.hidden_dim,
        'top_p': encoded.top_p,
        'temperature': encoded.temperature,
        'repetition_penalty': encoded.repetition_penalty,
        'num_sentences': len(encoded.sentences),
        'sentences': sentence_meta,
    }

    meta_json = json.dumps(meta).encode('utf-8')
    arrays['meta'] = np.frombuffer(meta_json, dtype=np.uint8)

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path, **arrays)

    logger.info("Saved %d sentences (%d tokens) to %s",
                len(encoded.sentences), encoded.total_tokens, path)


def load(path: str) -> EncodedSpeech:
    """Load EncodedSpeech from a .soprano file.

    Converts float16 wire format back to float32 for computation.
    Uses allow_pickle=False for safe deserialization.

    Args:
        path: Path to the .soprano file.

    Returns:
        EncodedSpeech with all sentences restored.
    """
    data = np.load(path, allow_pickle=False)
    meta_bytes = data['meta'].tobytes()
    meta = json.loads(meta_bytes.decode('utf-8'))

    sentences = []
    for i in range(meta['num_sentences']):
        smeta = meta['sentences'][i]
        attention = None
        if smeta['has_attention']:
            attention = data[f'attention_{i}'].astype(np.float32)

        sentences.append(SentenceEncoding(
            hidden_states=data[f'hidden_{i}'].astype(np.float32),
            token_ids=data[f'token_ids_{i}'].astype(np.int32),
            token_entropy=data[f'entropy_{i}'].astype(np.float32),
            finish_reason=smeta['finish_reason'],
            text=smeta['text'],
            text_index=smeta['text_index'],
            sentence_index=smeta['sentence_index'],
            attention_weights=attention,
            input_token_offsets=smeta.get('input_token_offsets'),
        ))

    return EncodedSpeech(
        sentences=sentences,
        model_id=meta['model_id'],
        format_version=meta['format_version'],
        sample_rate=meta['sample_rate'],
        token_audio_samples=meta['token_audio_samples'],
        hidden_dim=meta['hidden_dim'],
        top_p=meta['top_p'],
        temperature=meta['temperature'],
        repetition_penalty=meta['repetition_penalty'],
    )
