import torch
from torch import nn
from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM

from muse.modalities.image_vectorization.runtimes.starvector_model import (
    StarVectorConfig,
    StarVectorForCausalLM,
)


def test_checkpoint_shell_ties_nested_starcoder_embeddings():
    shell = StarVectorForCausalLM.__new__(StarVectorForCausalLM)
    nn.Module.__init__(shell)
    shell.config = StarVectorConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        vocab_size=16,
    )

    transformer = GPTBigCodeForCausalLM(GPTBigCodeConfig(
        vocab_size=16,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=1,
    ))
    transformer.lm_head.weight = nn.Parameter(
        transformer.lm_head.weight.detach().clone()
    )
    shell.model = nn.Module()
    shell.model.svg_transformer = nn.Module()
    shell.model.svg_transformer.transformer = transformer

    assert shell.get_input_embeddings().weight is not (
        shell.get_output_embeddings().weight
    )
    shell.tie_weights()
    assert shell.get_input_embeddings().weight is (
        shell.get_output_embeddings().weight
    )
