import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopPLogitsWarper
from .base import BaseModel


class TransformersModel(BaseModel):
    def __init__(self,
            model_path=None,
            compile=True,
            quantize=False):
        self.device = 'cpu'
        model_name_or_path = model_path if model_path else 'ekwek/Soprano-1.1-80M'

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.eval()

        # Apply INT8 dynamic quantization for CPU speedup
        if quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        # Apply torch.compile for kernel fusion
        if compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                import warnings
                warnings.warn(f"torch.compile failed: {e}")

    def infer(self, prompts, top_p=0.95, temperature=0.3, repetition_penalty=1.2):
        temperature = max(temperature, 0.001)  # Ensure positive temperature for sampling
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
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
            )
        res = []
        eos_token_id = self.model.config.eos_token_id
        for i in range(len(prompts)):
            seq = outputs.sequences[i]
            hidden_states = []
            num_output_tokens = len(outputs.hidden_states)
            for j in range(num_output_tokens):
                token = seq[j + seq.size(0) - num_output_tokens]
                if token != eos_token_id:
                    hidden_states.append(outputs.hidden_states[j][-1][i, -1, :])
            last_hidden_state = torch.stack(hidden_states).squeeze()
            finish_reason = 'stop' if seq[-1].item() == eos_token_id else 'length'
            res.append({
                'finish_reason': finish_reason,
                'hidden_state': last_hidden_state
            })
        return res

    def stream_infer(self, prompt, top_p=0.95, temperature=0.3, repetition_penalty=1.2):
        temperature = max(temperature, 0.001)  # Ensure positive temperature for sampling

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

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True, output_hidden_states=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            generated_ids = input_ids
            next_token = get_next_token(next_token_logits, generated_ids)

            max_new_tokens = 512
            eos_token_id = self.model.config.eos_token_id

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

                yield {
                    'finish_reason': finish_reason,
                    'hidden_state': current_hidden_state
                }

                if finish_reason:
                    break

                next_token_logits = outputs.logits[:, -1, :]
                next_token = get_next_token(next_token_logits, generated_ids)
