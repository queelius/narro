import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel


class TransformersModel(BaseModel):
    def __init__(self, model_path=None, compile=True, quantize=True, device='cpu'):  # noqa: ARG002 (compile kept for API compat)
        model_name_or_path = model_path if model_path else 'ekwek/Soprano-1.1-80M'
        self.device = device

        # float16 on CUDA for ~2x throughput; float32 on CPU where fp16 is slow.
        dtype = torch.float16 if device not in ('cpu',) else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.eval()
        self.model = self.model.to(device)

        if quantize and device == 'cpu':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        # torch.compile is not used on the LLM: streaming inference
        # alternates between full-sequence and single-token-with-KV-cache
        # call patterns, which causes torch.compile to hang or crash on
        # recompilation.  The Vocos decoder (fixed input shape) is still
        # compiled in tts.py / decode_only.py.
