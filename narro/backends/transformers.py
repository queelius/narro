import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel


class TransformersModel(BaseModel):
    def __init__(self, model_path=None, compile=True, quantize=True):
        model_name_or_path = model_path if model_path else 'ekwek/Soprano-1.1-80M'

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.eval()

        if quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        if compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                import warnings
                warnings.warn(f"torch.compile failed: {e}")
