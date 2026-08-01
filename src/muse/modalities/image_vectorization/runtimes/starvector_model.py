"""Inference-only StarVector-1B architecture.

This is a small, dependency-trimmed adaptation of the Apache-2.0
StarVector implementation at:

https://github.com/joanrod/star-vector/tree/0e083c1911760aa31bc576ca7f337a7f8ee605ec

Only modules needed to instantiate the official im2svg checkpoint are
retained. Training, serving, metrics, remote-code registration, and the
upstream dependency on a separately downloaded StarCoder base model are
deliberately omitted. Module nesting matches the checkpoint's state-dict
keys exactly.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn
from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM
from transformers import PretrainedConfig, PreTrainedModel


class StarVectorConfig(PretrainedConfig):
    """Configuration shape stored by the official StarVector checkpoint."""

    model_type = "starvector"

    def __init__(
        self,
        starcoder_model_name: str = "bigcode/starcoderbase-1b",
        image_encoder_type: str = "clip",
        adapter_norm: str = "layer_norm",
        image_size: int = 224,
        max_length: int = 8192,
        max_length_train: int = 8192,
        use_flash_attn: bool = True,
        use_cache: bool = True,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 24,
        vocab_size: int = 49156,
        hidden_size: int = 2048,
        num_kv_heads: int = 4,
        multi_query: bool = True,
        dropout: float = 0.1,
        torch_dtype: str = "float16",
        **kwargs: Any,
    ) -> None:
        kwargs["torch_dtype"] = torch_dtype
        self.starcoder_model_name = starcoder_model_name
        self.image_encoder_type = image_encoder_type
        self.adapter_norm = adapter_norm
        self.image_size = image_size
        self.max_length = max_length
        self.max_length_train = max_length_train
        self.use_flash_attn = use_flash_attn
        self.use_cache = use_cache
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        self.multi_query = multi_query
        self.dropout = dropout
        super().__init__(**kwargs)
        # PretrainedConfig owns legacy generation fields and resets
        # ``max_length`` while initializing. Restore StarVector's model
        # context value after the parent constructor.
        self.max_length = max_length


class _LayerNorm(nn.LayerNorm):
    """LayerNorm that preserves the input dtype."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original_dtype = value.dtype
        result = super().forward(value.to(self.weight.dtype))
        return result.to(original_dtype)


class _QuickGELU(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(1.702 * value)


class _ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = _LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", _QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width)),
        ]))
        self.ln_2 = _LayerNorm(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_1(value)
        attended = self.attn(
            normalized, normalized, normalized, need_weights=False,
        )[0]
        value = value + attended
        return value + self.mlp(self.ln_2(value))


class _VisionTransformer(nn.Module):
    """CLIP-style ViT-L/14 encoder used by StarVector-1B."""

    def __init__(
        self,
        *,
        input_resolution: int,
        patch_size: int = 14,
        width: int = 1024,
        layers: int = 23,
        heads: int = 16,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.num_features = width
        self.num_heads = heads
        patches = (input_resolution // patch_size) ** 2
        self.conv1 = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False,
        )
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(patches + 1, width)
        )
        self.ln_pre = _LayerNorm(width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.Sequential(*[
            _ResidualAttentionBlock(width, heads) for _ in range(layers)
        ])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = self.conv1(image)
        value = value.reshape(value.shape[0], value.shape[1], -1)
        value = value.permute(0, 2, 1)
        class_token = self.class_embedding.to(value.dtype)
        class_token = class_token + torch.zeros(
            value.shape[0], 1, value.shape[-1],
            dtype=value.dtype, device=value.device,
        )
        value = torch.cat([class_token, value], dim=1)
        value = value + self.positional_embedding.to(value.dtype)
        value = self.ln_pre(value).permute(1, 0, 2)
        value = self.transformer.resblocks(value)
        return value.permute(1, 0, 2)


class _ImageEncoder(nn.Module):
    def __init__(self, config: StarVectorConfig) -> None:
        super().__init__()
        if config.image_encoder_type != "clip":
            raise ValueError(
                "the audited StarVector-1B runtime only supports its CLIP encoder"
            )
        self.visual_encoder = _VisionTransformer(
            input_resolution=config.image_size,
        )
        self.ln_vision = _LayerNorm(self.visual_encoder.num_features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.ln_vision(self.visual_encoder(image))


class _Swish(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(value)


class _Adapter(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        adapter_norm: str,
        query_length: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.c_fc = nn.Linear(input_size, input_size * 2)
        self.act = _Swish()
        self.c_proj = nn.Linear(input_size * 2, output_size)
        if adapter_norm == "layer_norm":
            self.norm = nn.LayerNorm([query_length, output_size])
        elif adapter_norm == "batch_norm":
            self.norm = nn.BatchNorm1d(query_length)
        else:
            raise ValueError(f"unsupported adapter norm: {adapter_norm!r}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.norm(hidden_states)


class _StarCoderModel(nn.Module):
    """Wrapper name retained so checkpoint paths remain unchanged."""

    def __init__(self, config: StarVectorConfig) -> None:
        super().__init__()
        transformer_config = GPTBigCodeConfig(
            vocab_size=config.vocab_size,
            n_positions=config.max_length,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_inner=config.hidden_size * 4,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            multi_query=config.multi_query,
            use_cache=config.use_cache,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=49152,
            torch_dtype=config.torch_dtype,
        )
        self.transformer = GPTBigCodeForCausalLM(transformer_config)


class _StarVectorInner(nn.Module):
    """The ``model.*`` namespace present in official checkpoint keys."""

    def __init__(self, config: StarVectorConfig) -> None:
        super().__init__()
        self.svg_transformer = _StarCoderModel(config)
        self.image_encoder = _ImageEncoder(config)
        # CLIP ViT-L/14 yields 257 tokens of width 1024.
        self.image_projection = _Adapter(
            1024,
            config.hidden_size,
            adapter_norm=config.adapter_norm,
            query_length=257,
            dropout_prob=config.dropout,
        )


class StarVectorForCausalLM(PreTrainedModel):
    """Load-compatible shell for the official im2svg safetensors."""

    config_class = StarVectorConfig
    _no_split_modules: list[str] = []
    _tied_weights_keys = [
        "model.svg_transformer.transformer.lm_head.weight",
    ]

    def __init__(self, config: StarVectorConfig, **_: Any) -> None:
        super().__init__(config)
        if "starcoder2" in config.starcoder_model_name.lower():
            raise ValueError(
                "StarVector-2 / StarCoder2 checkpoints are not supported by "
                "the audited StarVector-1B adapter"
            )
        self.model = _StarVectorInner(config)

    def get_input_embeddings(self) -> nn.Module:
        """Expose the nested StarCoder embeddings for weight tying."""
        return (
            self.model.svg_transformer.transformer.get_input_embeddings()
        )

    def get_output_embeddings(self) -> nn.Module:
        """Expose the omitted, tied language-model checkpoint head."""
        return (
            self.model.svg_transformer.transformer.get_output_embeddings()
        )

    def forward(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "call the image/vectorization runtime, not the checkpoint shell"
        )
