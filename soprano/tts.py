import logging
import os
import time

import torch
from huggingface_hub import hf_hub_download
from scipy.io import wavfile

from .backends.transformers import TransformersModel
from .utils.text_normalizer import clean_text
from .utils.text_splitter import split_and_recombine_text
from .vocos.decoder import SopranoDecoder

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 32000
INT16_MAX = 32767
RECEPTIVE_FIELD = 4
TOKEN_SIZE = 2048
HIDDEN_DIM = 512

# Hallucination detection
DIFF_THRESHOLD = 300
MAX_RUNLENGTH = 16

# Text processing
MIN_SENTENCE_LENGTH = 30


class SopranoTTS:
    """
    Soprano Text-to-Speech model (CPU-optimized).

    Args:
        model_path: Path to local model directory (uses HuggingFace if None)
        compile: Enable torch.compile for faster inference (default: True)
        quantize: Enable INT8 quantization for faster CPU inference (default: False)
        decoder_batch_size: Batch size for decoder
    """
    def __init__(self,
            model_path=None,
            compile=True,
            quantize=False,
            decoder_batch_size=1):
        self.device = 'cpu'
        self.pipeline = TransformersModel(
            model_path=model_path,
            compile=compile,
            quantize=quantize
        )

        self.decoder = SopranoDecoder()
        if model_path:
            decoder_path = os.path.join(model_path, 'decoder.pth')
        else:
            decoder_path = hf_hub_download(repo_id='ekwek/Soprano-1.1-80M', filename='decoder.pth')
        self.decoder.load_state_dict(torch.load(decoder_path, map_location='cpu', weights_only=True))
        self.decoder.eval()

        if compile:
            try:
                self.decoder = torch.compile(self.decoder, mode="reduce-overhead")
            except Exception as e:
                import warnings
                warnings.warn(f"torch.compile failed for decoder: {e}")

        self.decoder_batch_size = decoder_batch_size

        self.infer("Hello world!")  # warmup

    def _preprocess_text(self, texts, min_length=30):
        res = []
        for text_idx, text in enumerate(texts):
            text = text.strip()
            cleaned_text = clean_text(text)
            sentences = split_and_recombine_text(cleaned_text)
            processed = []
            for sentence in sentences:
                processed.append({"text": sentence, "text_idx": text_idx})

            if min_length > 0 and len(processed) > 1:
                merged = []
                i = 0
                while i < len(processed):
                    cur = processed[i]
                    if len(cur["text"]) < min_length:
                        if merged:
                            merged[-1]["text"] = (merged[-1]["text"] + " " + cur["text"]).strip()
                        else:
                            if i + 1 < len(processed):
                                processed[i + 1]["text"] = (cur["text"] + " " + processed[i + 1]["text"]).strip()
                            else:
                                merged.append(cur)
                    else:
                        merged.append(cur)
                    i += 1
                processed = merged
            sentence_idxes = {}
            for item in processed:
                if item['text_idx'] not in sentence_idxes:
                    sentence_idxes[item['text_idx']] = 0
                res.append((f'[STOP][TEXT]{item["text"]}[START]', item["text_idx"], sentence_idxes[item['text_idx']]))
                sentence_idxes[item['text_idx']] += 1
        return res

    def hallucination_detector(self, hidden_state):
        if len(hidden_state) <= MAX_RUNLENGTH:
            return False
        aah_runlength = 0
        for i in range(len(hidden_state) - 1):
            current_sequences = hidden_state[i]
            next_sequences = hidden_state[i + 1]
            diffs = torch.abs(current_sequences - next_sequences)
            total_diff = diffs.sum(dim=0)
            if total_diff < DIFF_THRESHOLD:
                aah_runlength += 1
            elif aah_runlength > 0:
                aah_runlength -= 1
            if aah_runlength > MAX_RUNLENGTH:
                return True
        return False

    def infer(self, text, out_path=None, top_p=0.95, temperature=0.0, repetition_penalty=1.2, retries=0):
        results = self.infer_batch(
            [text],
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            out_dir=None,
            retries=retries
        )[0]
        if out_path:
            wavfile.write(out_path, SAMPLE_RATE, results.numpy())
        return results

    def infer_batch(self, texts, out_dir=None, top_p=0.95, temperature=0.0, repetition_penalty=1.2, retries=0):
        sentence_data = self._preprocess_text(texts)
        prompts = [x[0] for x in sentence_data]
        hidden_states = [None] * len(prompts)
        pending_indices = list(range(len(prompts)))
        tries_left = 1 + max(0, retries)
        while tries_left > 0 and pending_indices:
            current_prompts = [prompts[i] for i in pending_indices]
            responses = self.pipeline.infer(
                current_prompts,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
            bad_indices = []
            for idx, response in enumerate(responses):
                hidden_state = response['hidden_state']
                hidden_states[pending_indices[idx]] = hidden_state
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

        combined = list(zip(hidden_states, sentence_data))
        combined.sort(key=lambda x: -x[0].size(0))
        hidden_states, sentence_data = zip(*combined)

        num_texts = len(texts)
        audio_concat = [[] for _ in range(num_texts)]
        for sentence in sentence_data:
            audio_concat[sentence[1]].append(None)

        for idx in range(0, len(hidden_states), self.decoder_batch_size):
            lengths = [x.size(0) for x in hidden_states[idx:idx+self.decoder_batch_size]]
            N = len(lengths)
            max_len = lengths[0]  # Already sorted descending

            # Pre-allocate with zeros (single allocation, left-padded)
            batch_hidden_states = torch.zeros((N, HIDDEN_DIM, max_len))
            for i in range(N):
                seq_len = lengths[i]
                batch_hidden_states[i, :, max_len-seq_len:] = hidden_states[idx+i].T

            with torch.no_grad():
                audio = self.decoder(batch_hidden_states)

            for i in range(N):
                text_id = sentence_data[idx+i][1]
                sentence_id = sentence_data[idx+i][2]
                audio_concat[text_id][sentence_id] = audio[i].squeeze()[-(lengths[i]*TOKEN_SIZE-TOKEN_SIZE):]

        audio_concat = [torch.cat(x) for x in audio_concat]

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            for i in range(len(audio_concat)):
                wavfile.write(f"{out_dir}/{i}.wav", SAMPLE_RATE, audio_concat[i].numpy())
        return audio_concat

    def infer_stream(self, text, chunk_size=1, top_p=0.95, temperature=0.0, repetition_penalty=1.2):
        start_time = time.time()
        sentence_data = self._preprocess_text([text])

        first_chunk = True
        for sentence, _, _ in sentence_data:
            responses = self.pipeline.stream_infer(
                sentence,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
            hidden_states_buffer = []
            chunk_counter = chunk_size
            for token in responses:
                finished = token['finish_reason'] is not None
                if not finished:
                    hidden_states_buffer.append(token['hidden_state'][-1])
                hidden_states_buffer = hidden_states_buffer[-(2*RECEPTIVE_FIELD+chunk_size):]
                if finished or len(hidden_states_buffer) >= RECEPTIVE_FIELD + chunk_size:
                    if finished or chunk_counter == chunk_size:
                        batch_hidden_states = torch.stack(hidden_states_buffer)
                        inp = batch_hidden_states.unsqueeze(0).transpose(1, 2)
                        with torch.no_grad():
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
