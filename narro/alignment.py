"""Word-level alignment from attention weights.

Extracts word timestamps by analyzing which input tokens each generated
audio token was attending to. Uses center-of-mass over the generated token
timeline weighted by attention, with spread estimated via weighted std dev.
"""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_alignment(attention, token_to_word, token_duration):
    """Extract word-level timestamps from attention weights.

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

    # Collect unique words and their input token indices
    # A word may span multiple input tokens
    word_positions = {}  # word -> list of input token indices
    word_order = []  # preserve insertion order for determinism
    for token_idx, word in sorted(token_to_word.items()):
        if word not in word_positions:
            word_positions[word] = []
            word_order.append(word)
        word_positions[word].append(token_idx)

    # Generated token timestamps (center of each token's time interval)
    timestamps = np.arange(T) * token_duration

    alignment = []
    for word in word_order:
        indices = word_positions[word]

        # Sum attention across all input tokens belonging to this word
        word_attention = np.zeros(T, dtype=np.float32)
        for idx in indices:
            word_attention += attention[:, idx]

        total = word_attention.sum()
        if total < 1e-10:
            # No attention to this word; place at 0 with zero duration
            alignment.append({
                'word': word,
                'start': 0.0,
                'end': 0.0,
            })
            continue

        # Normalize to get a probability distribution over generated tokens
        weights = word_attention / total

        # Center of mass (expected timestamp)
        center = float(np.dot(weights, timestamps))

        # Weighted standard deviation as spread
        variance = float(np.dot(weights, (timestamps - center) ** 2))
        spread = float(np.sqrt(variance))

        start = max(0.0, center - spread)
        end = min(T * token_duration, center + spread)

        alignment.append({
            'word': word,
            'start': round(start, 3),
            'end': round(end, 3),
        })

    # Sort by start time
    alignment.sort(key=lambda x: x['start'])

    return alignment


def save_alignment(alignment, path):
    """Write alignment list to a JSON file.

    Args:
        alignment: list of {"word": str, "start": float, "end": float}.
        path: output file path.
    """
    with open(path, 'w') as f:
        json.dump(alignment, f, indent=2)
    logger.info("Saved alignment (%d words) to %s", len(alignment), path)


def extract_alignment_from_encoded(encoded):
    """Extract word-level alignment from an EncodedSpeech.

    Convenience function that processes all sentences with attention weights,
    accumulating timestamps across sentences.

    Args:
        encoded: EncodedSpeech instance.

    Returns:
        List of {"word": str, "start": float, "end": float} with cumulative
        timestamps across all sentences.
    """
    from .tts import TOKEN_SIZE, SAMPLE_RATE

    token_duration = TOKEN_SIZE / SAMPLE_RATE
    cumulative_offset = 0.0
    all_alignment = []

    for sentence in encoded.sentences:
        T = len(sentence.hidden_states)
        sentence_duration = T * token_duration

        if sentence.attention_weights is None:
            cumulative_offset += sentence_duration
            continue

        attention = sentence.attention_weights
        input_len = attention.shape[1]

        # Build token_to_word mapping
        # Input format: [STOP][TEXT] word1 word2 ... [START]
        # Skip first 2 tokens ([STOP][TEXT]) and last 1 ([START])
        words = sentence.text.split()
        num_special_start = 2  # [STOP] and [TEXT]
        num_special_end = 1    # [START]
        usable_positions = input_len - num_special_start - num_special_end

        if usable_positions <= 0 or len(words) == 0:
            cumulative_offset += sentence_duration
            continue

        # Divide usable input positions evenly among words
        token_to_word = {}
        positions_per_word = usable_positions / len(words)
        for word_idx, word in enumerate(words):
            start_pos = int(round(num_special_start + word_idx * positions_per_word))
            end_pos = int(round(num_special_start + (word_idx + 1) * positions_per_word))
            for pos in range(start_pos, min(end_pos, input_len - num_special_end)):
                token_to_word[pos] = word

        sentence_alignment = extract_alignment(attention, token_to_word, token_duration)

        # Offset timestamps by cumulative duration of previous sentences
        for entry in sentence_alignment:
            entry['start'] = round(entry['start'] + cumulative_offset, 3)
            entry['end'] = round(entry['end'] + cumulative_offset, 3)
            all_alignment.append(entry)

        cumulative_offset += sentence_duration

    return all_alignment
