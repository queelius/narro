import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel


class TransformersModel(BaseModel):
    def __init__(self, model_path=None, compile=True, quantize=True, device='cpu'):
        model_name_or_path = model_path if model_path else 'ekwek/Soprano-1.1-80M'
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.eval()
        self.model = self.model.to(device)

        if quantize and device == 'cpu':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        if compile:
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                import warnings
                warnings.warn(f"torch.compile failed: {e}")
