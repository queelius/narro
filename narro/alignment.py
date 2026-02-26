"""Word-level alignment extraction.

Provides two alignment strategies:
1. Proportional timing: distributes sentence duration across words by
   character length. Sentence boundaries are exact (derived from known
   token counts). This is the default and most reliable method.
2. Attention-based: uses center-of-mass over generated tokens weighted
   by attention. Requires accurate token-to-word mapping and a model
   whose attention patterns correlate with speech timing.
"""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_alignment(attention, token_to_word, token_duration):
    """Extract word-level timestamps from attention weights.

    Uses center-of-mass over the generated token timeline, weighted by
    each word's total attention. This works well when the model's attention
    correlates with speech timing (e.g., encoder-decoder cross-attention),
    but may produce overlapping ranges for causal LM self-attention.

    Args:
        attention: np.ndarray of shape (T, input_len) where T is the number
            of generated audio tokens and input_len is the number of input
            text tokens.
        token_to_word: dict mapping input token index -> word string.
            Only word tokens should be included (not special tokens).
        token_duration: float, seconds per generated token (typically 0.064).

    Returns:
        List of {"word": str, "start": float, "end": float} ordered by
        start time. Timestamps are rounded to 3 decimal places.
    """
    T = attention.shape[0]
    if T == 0 or not token_to_word:
        return []

    # Collect word instances and their input token indices.
    word_instances = []  # list of (word_str, [input_token_indices])
    seen_positions = set()
    for token_idx, word in sorted(token_to_word.items()):
        if token_idx in seen_positions:
            continue
        seen_positions.add(token_idx)
        if word_instances and word_instances[-1][0] == word:
            prev_indices = word_instances[-1][1]
            if token_idx == prev_indices[-1] + 1:
                prev_indices.append(token_idx)
                continue
        word_instances.append((word, [token_idx]))

    timestamps = np.arange(T) * token_duration

    alignment = []
    for word, indices in word_instances:
        word_attention = np.zeros(T, dtype=np.float32)
        for idx in indices:
            word_attention += attention[:, idx]

        total = word_attention.sum()
        if total < 1e-10:
            alignment.append({'word': word, 'start': 0.0, 'end': 0.0})
            continue

        weights = word_attention / total
        center = float(np.dot(weights, timestamps))
        variance = float(np.dot(weights, (timestamps - center) ** 2))
        spread = max(float(np.sqrt(variance)), token_duration / 2)

        start = max(0.0, center - spread)
        end = min(T * token_duration, center + spread)

        alignment.append({
            'word': word,
            'start': round(start, 3),
            'end': round(end, 3),
        })

    alignment.sort(key=lambda x: x['start'])
    return alignment


def save_alignment(alignment, path):
    """Write alignment list to a JSON file."""
    with open(path, 'w') as f:
        json.dump(alignment, f, indent=2)
    logger.info("Saved alignment (%d words) to %s", len(alignment), path)


def extract_alignment_from_encoded(encoded):
    """Extract word-level alignment from an EncodedSpeech.

    Uses proportional timing within each sentence: sentence duration is
    exact (from token count), and words are distributed proportionally
    by character length. This produces clean, non-overlapping timestamps
    suitable for karaoke-style word highlighting.

    Args:
        encoded: EncodedSpeech instance.

    Returns:
        List of {"word": str, "start": float, "end": float} with cumulative
        timestamps across all sentences, ordered chronologically.
    """
    from .tts import TOKEN_SIZE, SAMPLE_RATE

    token_duration = TOKEN_SIZE / SAMPLE_RATE
    cumulative_offset = 0.0
    all_alignment = []

    for sentence in encoded.sentences:
        T = len(sentence.hidden_states)
        sentence_duration = T * token_duration

        words = sentence.text.split()
        if not words:
            cumulative_offset += sentence_duration
            continue

        # Proportional timing: distribute sentence duration by character length
        char_lengths = [len(w) for w in words]
        total_chars = sum(char_lengths)
        if total_chars == 0:
            cumulative_offset += sentence_duration
            continue

        char_pos = 0
        for word, char_len in zip(words, char_lengths):
            start = cumulative_offset + sentence_duration * char_pos / total_chars
            char_pos += char_len
            end = cumulative_offset + sentence_duration * char_pos / total_chars

            all_alignment.append({
                'word': word,
                'start': round(start, 3),
                'end': round(end, 3),
            })

        cumulative_offset += sentence_duration

    return all_alignment
