import logging
import os
import time
from collections import deque

import numpy as np
import torch

from .backends.transformers import TransformersModel
from .encoded import EncodedSpeech, SentenceEncoding
from .utils.text_normalizer import clean_text
from .utils.text_splitter import split_and_recombine_text

logger = logging.getLogger(__name__)

MODEL_ID = 'ekwek/Soprano-1.1-80M'

# Audio constants
SAMPLE_RATE = 32000
INT16_MAX = 32767
RECEPTIVE_FIELD = 4
TOKEN_SIZE = 2048
HIDDEN_DIM = 512

# Hallucination detection
DIFF_THRESHOLD = 300
MAX_RUNLENGTH = 16


class SopranoTTS:
    """
    Soprano Text-to-Speech model (CPU-optimized).

    Pipeline: text -> Qwen3-80M LLM hidden states -> Vocos decoder -> audio
    Based on ekwek/Soprano-1.1-80M (LLM) and Vocos vocoder architecture.

    Args:
        model_path: Path to local model directory (uses HuggingFace if None)
        compile: Enable torch.compile for faster inference (default: True)
        quantize: Enable INT8 quantization for faster CPU inference (default: False, degrades quality)
        decoder_batch_size: Batch size for decoder (default: 4)
        num_threads: Number of CPU threads for inference (None = auto-detect)
    """
    def __init__(self,
            model_path=None,
            compile=True,
            quantize=False,
            decoder_batch_size=4,
            num_threads=None):
        # Configure threading before model load
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass  # Can only be set once per process

        torch.set_float32_matmul_precision('high')

        self.model_id = model_path if model_path else MODEL_ID
        self.pipeline = TransformersModel(model_path=model_path, compile=compile, quantize=quantize)

        from .decode_only import load_decoder
        self.decoder = load_decoder(model_path=model_path, compile=compile)
        self.decoder_batch_size = decoder_batch_size

        # Warmup decoder directly with synthetic tensors (no LLM needed).
        if compile:
            for warmup_len in [10, 20, 50]:
                with torch.inference_mode():
                    self.decoder(torch.zeros(1, HIDDEN_DIM, warmup_len))

    def _preprocess_text(self, texts, min_length=30):
        res = []
        for text_idx, text in enumerate(texts):
            cleaned_text = clean_text(text.strip())
            sentences = split_and_recombine_text(cleaned_text)

            if min_length > 0 and len(sentences) > 1:
                merged = []
                i = 0
                while i < len(sentences):
                    cur = sentences[i]
                    if len(cur) < min_length:
                        if merged:
                            merged[-1] = (merged[-1] + " " + cur).strip()
                        elif i + 1 < len(sentences):
                            sentences[i + 1] = (cur + " " + sentences[i + 1]).strip()
                    else:
                        merged.append(cur)
                    i += 1
                sentences = merged

            for sentence_idx, sentence in enumerate(sentences):
                res.append((f'[STOP][TEXT]{sentence}[START]', text_idx, sentence_idx, sentence))
        return res

    def hallucination_detector(self, hidden_state):
        if len(hidden_state) <= MAX_RUNLENGTH:
            return False
        stacked = torch.stack(list(hidden_state))
        total_diffs = torch.diff(stacked, dim=0).abs().sum(dim=1)
        is_similar = (total_diffs < DIFF_THRESHOLD).numpy()
        aah_runlength = 0
        for similar in is_similar:
            if similar:
                aah_runlength += 1
            elif aah_runlength > 0:
                aah_runlength -= 1
            if aah_runlength > MAX_RUNLENGTH:
                return True
        return False

    # ------------------------------------------------------------------
    # Encode: text -> EncodedSpeech (intermediate representation)
    # ------------------------------------------------------------------

    def encode(self, text, **kwargs):
        """Encode a single text into an EncodedSpeech."""
        return self.encode_batch([text], **kwargs)

    def encode_batch(self, texts, top_p=0.95, temperature=0.0, repetition_penalty=1.2,
                     retries=0, include_attention=False):
        """Encode multiple texts into an EncodedSpeech.

        Runs text preprocessing, LLM inference with enriched output (token IDs,
        per-token entropy, optional attention), and hallucination detection.

        Args:
            texts: List of input text strings.
            top_p: Nucleus sampling threshold.
            temperature: Sampling temperature (0.0 clamped to 0.001 in backend).
            repetition_penalty: Repetition penalty factor.
            retries: Number of hallucination retries per sentence.
            include_attention: Extract attention weights (doubles memory).

        Returns:
            EncodedSpeech with all sentence encodings.
        """
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        sentence_data = self._preprocess_text(texts)
        logger.debug("Preprocessing: %.1f ms (%d sentences)",
                      (time.perf_counter() - t0) * 1000, len(sentence_data))

        prompts = [x[0] for x in sentence_data]
        responses = [None] * len(prompts)
        pending_indices = list(range(len(prompts)))
        tries_left = 1 + max(0, retries)
        t_llm = time.perf_counter()
        while tries_left > 0 and pending_indices:
            current_prompts = [prompts[i] for i in pending_indices]
            batch_responses = self.pipeline.infer(
                current_prompts,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                include_attention=include_attention,
            )
            bad_indices = []
            for idx, response in enumerate(batch_responses):
                hidden_state = response['hidden_state']
                responses[pending_indices[idx]] = response
                if response['finish_reason'] != 'stop':
                    logger.warning("A sentence did not complete generation, likely due to hallucination.")
                if retries > 0 and self.hallucination_detector(hidden_state):
                    logger.warning("A sentence contained a hallucination.")
                    bad_indices.append(pending_indices[idx])
            if not bad_indices:
                break
            pending_indices = bad_indices
            tries_left -= 1
            if tries_left > 0:
                logger.warning("%d sentence(s) will be regenerated.", len(pending_indices))

        logger.info("LLM inference: %.1f ms", (time.perf_counter() - t_llm) * 1000)

        # Build SentenceEncoding objects (convert to numpy at the boundary)
        sentence_encodings = []
        for i, response in enumerate(responses):
            _, text_idx, sentence_idx, original_text = sentence_data[i]

            attention = None
            if response['attention'] is not None:
                attention = response['attention'].numpy()

            sentence_encodings.append(SentenceEncoding(
                hidden_states=response['hidden_state'].numpy(),
                token_ids=response['token_ids'].numpy(),
                token_entropy=response['token_entropy'].numpy(),
                finish_reason=response['finish_reason'],
                text=original_text,
                text_index=text_idx,
                sentence_index=sentence_idx,
                attention_weights=attention,
            ))

        encoded = EncodedSpeech(
            sentences=sentence_encodings,
            model_id=self.model_id,
            sample_rate=SAMPLE_RATE,
            token_audio_samples=TOKEN_SIZE,
            hidden_dim=HIDDEN_DIM,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        logger.info("encode_batch: %.1f ms (%d texts, %d sentences, %d tokens)",
                      (time.perf_counter() - t_start) * 1000,
                      len(texts), len(sentence_encodings), encoded.total_tokens)
        return encoded

    # ------------------------------------------------------------------
    # Decode: EncodedSpeech -> audio tensors
    # ------------------------------------------------------------------

    def decode(self, encoded):
        """Decode an EncodedSpeech into audio tensors.

        Delegates to decode_only.decode() — the canonical decode implementation.

        Args:
            encoded: EncodedSpeech from encode_batch() or load().

        Returns:
            List of 1D torch.Tensor audio waveforms, one per input text.
        """
        from .decode_only import decode as _decode
        t_start = time.perf_counter()
        result = _decode(encoded, decoder=self.decoder,
                         decoder_batch_size=self.decoder_batch_size)
        logger.info("decode: %.1f ms (%d texts)",
                     (time.perf_counter() - t_start) * 1000, encoded.num_texts)
        return result

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def infer(self, text, out_path=None, top_p=0.95, temperature=0.0, repetition_penalty=1.2, retries=0):
        """Encode + decode a single text. Optionally write WAV."""
        encoded = self.encode(text, top_p=top_p, temperature=temperature,
                              repetition_penalty=repetition_penalty, retries=retries)
        results = self.decode(encoded)[0]
        if out_path:
            from .decode_only import write_wav
            write_wav(results, out_path)
        return results

    def infer_batch(self, texts, out_dir=None, top_p=0.95, temperature=0.0,
                    repetition_penalty=1.2, retries=0):
        """Encode + decode multiple texts. Optionally write WAVs to out_dir."""
        encoded = self.encode_batch(texts, top_p=top_p, temperature=temperature,
                                    repetition_penalty=repetition_penalty, retries=retries)
        audio_concat = self.decode(encoded)

        if out_dir:
            from .decode_only import write_wav
            os.makedirs(out_dir, exist_ok=True)
            for i, audio in enumerate(audio_concat):
                write_wav(audio, f"{out_dir}/{i}.wav")

        return audio_concat

    def decode_to_wav(self, encoded, out_path):
        """Decode an EncodedSpeech and write concatenated audio to a WAV file."""
        from .decode_only import write_wav
        audio_list = self.decode(encoded)
        write_wav(torch.cat(audio_list), out_path)

    def infer_stream(self, text, chunk_size=1, top_p=0.95, temperature=0.0, repetition_penalty=1.2):
        start_time = time.time()
        sentence_data = self._preprocess_text([text])

        first_chunk = True
        for sentence, _, _, _ in sentence_data:
            responses = self.pipeline.stream_infer(
                sentence,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
            max_buffer = 2 * RECEPTIVE_FIELD + chunk_size
            hidden_states_buffer = deque(maxlen=max_buffer)
            chunk_counter = chunk_size
            for token in responses:
                finished = token['finish_reason'] is not None
                if not finished:
                    hidden_states_buffer.append(token['hidden_state'][-1])
                if finished or len(hidden_states_buffer) >= RECEPTIVE_FIELD + chunk_size:
                    if finished or chunk_counter == chunk_size:
                        batch_hidden_states = torch.stack(list(hidden_states_buffer))
                        inp = batch_hidden_states.unsqueeze(0).transpose(1, 2)
                        with torch.inference_mode():
                            audio = self.decoder(inp)[0]
                        if finished:
                            audio_chunk = audio[-((RECEPTIVE_FIELD+chunk_counter-1)*TOKEN_SIZE-TOKEN_SIZE):]
                        else:
                            audio_chunk = audio[-((RECEPTIVE_FIELD+chunk_size)*TOKEN_SIZE-TOKEN_SIZE):-(RECEPTIVE_FIELD*TOKEN_SIZE-TOKEN_SIZE)]
                        chunk_counter = 0
                        if first_chunk:
                            logger.info("Streaming latency: %.2f ms", 1000 * (time.time() - start_time))
                            first_chunk = False
                        yield audio_chunk
                    chunk_counter += 1
