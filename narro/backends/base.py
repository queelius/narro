import logging
import time

import torch
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopPLogitsWarper

logger = logging.getLogger(__name__)


def _token_entropy(logits):
    """Compute entropy in nats from a logits vector.

    Args:
        logits: Raw logits tensor of shape (vocab_size,).

    Returns:
        Scalar tensor with entropy value.
    """
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-10)).sum()


class BaseModel:

    def _build_token_word_map(self, prompt, sentence_text):
        """Build token-to-word mapping using tokenizer offset_mapping.

        Maps input token positions to the words they encode, so alignment
        extraction knows which attention columns correspond to which words.

        Returns:
            dict[int, str] mapping token position -> word string,
            or None if offset_mapping is unavailable.
        """
        try:
            encoding = self.tokenizer(
                prompt,
                return_offsets_mapping=True,
            )
        except Exception:
            return None

        offsets = encoding.get('offset_mapping')
        if offsets is None:
            return None

        # Character range of the sentence text within the prompt
        prefix = '[STOP][TEXT]'
        suffix = '[START]'
        text_start = prompt.find(prefix) + len(prefix)
        text_end = prompt.rfind(suffix)
        if text_start < 0 or text_end < 0 or text_end <= text_start:
            return None

        # Build character ranges for each word
        words = sentence_text.split()
        word_ranges = []
        search_pos = text_start
        for word in words:
            idx = prompt.find(word, search_pos)
            if idx == -1:
                continue
            word_ranges.append((idx, idx + len(word), word))
            search_pos = idx + len(word)

        # Map tokens to words by checking which word each token overlaps
        token_to_word = {}
        for tok_idx, (char_start, char_end) in enumerate(offsets):
            if char_start >= char_end:
                continue
            if char_start < text_start or char_end > text_end:
                continue
            tok_mid = (char_start + char_end) / 2
            for word_start, word_end, word in word_ranges:
                if tok_mid >= word_start and tok_mid < word_end:
                    token_to_word[tok_idx] = word
                    break

        return token_to_word if token_to_word else None

    def infer(self, prompts, top_p=0.95, temperature=0.3, repetition_penalty=1.2,
              include_attention=False):
        temperature = max(temperature, 0.001)
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=512,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                output_attentions=include_attention,
            )
        gen_time = time.perf_counter() - t0
        total_tokens = len(outputs.hidden_states)
        logger.debug("Generation: %.1f ms, %d tokens (%.0f tok/s)",
                      gen_time * 1000, total_tokens,
                      total_tokens / gen_time if gen_time > 0 else 0)

        input_len = inputs['input_ids'].shape[1]

        # Build token-to-word maps (needs individual tokenization per prompt)
        token_word_maps = [None] * len(prompts)
        if include_attention:
            for idx, prompt in enumerate(prompts):
                # Extract sentence text from wrapped prompt
                pfx, sfx = '[STOP][TEXT]', '[START]'
                pfx_pos = prompt.find(pfx)
                sfx_pos = prompt.rfind(sfx)
                if pfx_pos >= 0 and sfx_pos > pfx_pos:
                    sentence_text = prompt[pfx_pos + len(pfx):sfx_pos]
                    twm = self._build_token_word_map(prompt, sentence_text)
                    if twm is not None:
                        # Adjust for left-padding: shift positions by pad offset
                        attn_mask = inputs['attention_mask'][idx]
                        pad_offset = int((attn_mask == 0).sum().item())
                        if pad_offset > 0:
                            twm = {k + pad_offset: v for k, v in twm.items()}
                        token_word_maps[idx] = twm

        res = []
        eos_token_id = self.model.config.eos_token_id
        for i in range(len(prompts)):
            seq = outputs.sequences[i]
            hidden_states = []
            token_ids = []
            token_entropies = []
            attention_weights = []
            num_output_tokens = len(outputs.hidden_states)
            for j in range(num_output_tokens):
                token = seq[j + seq.size(0) - num_output_tokens]
                if token != eos_token_id:
                    hidden_states.append(outputs.hidden_states[j][-1][i, -1, :])
                    token_ids.append(token.item())
                    token_entropies.append(_token_entropy(outputs.scores[j][i]))

                    if include_attention and hasattr(outputs, 'attentions') and outputs.attentions:
                        # Last layer attention, average over heads, slice to input positions
                        attn = outputs.attentions[j][-1][i]  # (num_heads, seq_len, seq_len)
                        attn_avg = attn.mean(dim=0)[-1, :input_len]  # (input_len,)
                        attention_weights.append(attn_avg)

            if not hidden_states:
                last_hidden_state = torch.zeros(0, self.model.config.hidden_size)
                token_ids_tensor = torch.zeros(0, dtype=torch.int32)
                entropy_tensor = torch.zeros(0)
            else:
                last_hidden_state = torch.stack(hidden_states)
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.int32)
                entropy_tensor = torch.stack(token_entropies)

            finish_reason = 'stop' if seq[-1].item() == eos_token_id else 'length'

            result = {
                'finish_reason': finish_reason,
                'hidden_state': last_hidden_state,
                'token_ids': token_ids_tensor,
                'token_entropy': entropy_tensor,
            }

            if include_attention and attention_weights:
                result['attention'] = torch.stack(attention_weights)
            else:
                result['attention'] = None

            result['input_token_offsets'] = token_word_maps[i]

            res.append(result)
        return res

    def stream_infer(self, prompt, top_p=0.95, temperature=0.3, repetition_penalty=1.2):
        temperature = max(temperature, 0.001)

        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids']

        logits_processor = LogitsProcessorList()
        if repetition_penalty != 1.0:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))

        logits_warper = LogitsProcessorList()
        if temperature != 1.0:
            logits_warper.append(TemperatureLogitsWarper(temperature=temperature))
        if top_p < 1.0:
            logits_warper.append(TopPLogitsWarper(top_p=top_p))

        def get_next_token(logits, input_seq):
            scores = logits_processor(input_seq, logits)
            scores = logits_warper(input_seq, scores)
            probs = torch.nn.functional.softmax(scores, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        with torch.inference_mode():
            outputs = self.model(input_ids, use_cache=True, output_hidden_states=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            generated_ids = input_ids
            next_token = get_next_token(next_token_logits, generated_ids)

            max_new_tokens = 512
            eos_token_id = self.model.config.eos_token_id
            stream_t0 = time.perf_counter()

            for i in range(max_new_tokens):
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                outputs = self.model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_key_values = outputs.past_key_values
                current_hidden_state = outputs.hidden_states[-1][:, -1, :]

                finish_reason = None
                if next_token.item() == eos_token_id:
                    finish_reason = 'stop'
                elif i == max_new_tokens - 1:
                    finish_reason = 'length'

                # Compute token entropy from the logits that produced this token
                entropy = _token_entropy(next_token_logits[0])

                if (i + 1) % 50 == 0:
                    elapsed = time.perf_counter() - stream_t0
                    logger.debug("Stream: %d tokens, %.0f tok/s",
                                  i + 1, (i + 1) / elapsed if elapsed > 0 else 0)

                yield {
                    'finish_reason': finish_reason,
                    'hidden_state': current_hidden_state,
                    'token_id': next_token.item(),
                    'token_entropy': entropy,
                }

                if finish_reason:
                    break

                next_token_logits = outputs.logits[:, -1, :]
                next_token = get_next_token(next_token_logits, generated_ids)
