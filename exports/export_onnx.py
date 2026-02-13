#!/usr/bin/env python3
"""Export Soprano decoder to ONNX (FP32 + INT8).

Converts pointwise Conv1d (kernel_size=1) to Linear/MatMul for efficient
ONNX inference, while keeping depthwise Conv1d and embed Conv1d as-is.
INT8 uses MatMul-only dynamic quantization (Conv stays FP32 for
onnxruntime-web WASM compatibility).
"""

import gzip
import os
import sys

import numpy as np
import torch
from torch import nn

# Add parent dir so we can import soprano
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from soprano.vocos.modules import ChannelsFirstLayerNorm


# ---------------------------------------------------------------------------
# LinearBackbone — replaces pointwise Conv1d with Linear (= MatMul in ONNX)
# ---------------------------------------------------------------------------

class LinearConvNeXtBlock(nn.Module):
    """ConvNeXtBlock with Linear instead of Conv1d for pointwise convs."""
    def __init__(self, src_block):
        super().__init__()
        self.dwconv = src_block.dwconv  # depthwise stays as Conv1d
        self.norm = src_block.norm
        self.gamma = src_block.gamma
        self.pwconv1 = nn.Linear(src_block.pwconv1.in_channels, src_block.pwconv1.out_channels)
        self.pwconv1.weight.data = src_block.pwconv1.weight.data.squeeze(2)
        self.pwconv1.bias.data = src_block.pwconv1.bias.data
        self.pwconv2 = nn.Linear(src_block.pwconv2.in_channels, src_block.pwconv2.out_channels)
        self.pwconv2.weight.data = src_block.pwconv2.weight.data.squeeze(2)
        self.pwconv2.bias.data = src_block.pwconv2.bias.data

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pwconv1(x)
        x = nn.functional.gelu(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)
        x = self.gamma * x
        return residual + x


class LinearBackbone(nn.Module):
    """Decoder backbone with Linear layers for ONNX export.

    Excludes ISTFTHead (uses complex FFT which ONNX can't handle).
    The head projection Conv1d(768, 2050, k=1) is also converted to Linear.
    """
    def __init__(self, decoder):
        super().__init__()
        self.upscale = decoder.upscale
        self.embed = decoder.decoder.embed
        self.norm = decoder.decoder.norm
        self.final_layer_norm = decoder.decoder.final_layer_norm
        self.convnext = nn.ModuleList([
            LinearConvNeXtBlock(block) for block in decoder.decoder.convnext
        ])
        self.head_out = nn.Linear(decoder.head.out.in_channels, decoder.head.out.out_channels)
        self.head_out.weight.data = decoder.head.out.weight.data.squeeze(2)
        self.head_out.bias.data = decoder.head.out.bias.data

    def forward(self, x):
        T = x.size(2)
        x = torch.nn.functional.interpolate(
            x, size=self.upscale * (T - 1) + 1, mode='linear', align_corners=True
        )
        x = self.embed(x)
        x = self.norm(x)
        for block in self.convnext:
            x = block(x)
        x = self.final_layer_norm(x)
        x = x.transpose(1, 2)
        x = self.head_out(x)
        x = x.transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# Main export pipeline
# ---------------------------------------------------------------------------

def main():
    export_dir = os.path.dirname(os.path.abspath(__file__))

    # Load trained decoder weights
    print("Loading trained decoder weights...")
    from soprano.decode_only import load_decoder
    decoder = load_decoder(compile=False)
    print("Building LinearBackbone from trained decoder...")
    backbone = LinearBackbone(decoder)
    backbone.eval()

    # Export FP32 ONNX
    fp32_path = os.path.join(export_dir, 'soprano_decoder_fp32.onnx')
    dummy = torch.randn(1, 512, 20)
    print("Exporting FP32 ONNX...")
    torch.onnx.export(
        backbone, dummy,
        fp32_path,
        input_names=['hidden_states'],
        output_names=['mag_phase'],
        dynamic_axes={
            'hidden_states': {2: 'seq_len'},
            'mag_phase': {2: 'freq_len'},
        },
        opset_version=17,
    )
    print(f"  FP32 saved: {fp32_path} ({os.path.getsize(fp32_path) / 1e6:.1f} MB)")

    # INT8 dynamic quantization (MatMul-only — Conv stays FP32 for onnxruntime-web compat)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path = os.path.join(export_dir, 'soprano_decoder_int8.onnx')
    print("\nBuilding INT8 (MatMul-only)...")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8,
                     op_types_to_quantize=['MatMul'])
    print(f"  INT8 saved: {int8_path} ({os.path.getsize(int8_path) / 1e6:.1f} MB)")

    # Report sizes
    print("\n--- Size report ---")
    for path in [fp32_path, int8_path]:
        with open(path, 'rb') as f:
            raw = f.read()
        gz = gzip.compress(raw, compresslevel=9)
        print(f"  {os.path.basename(path)}: {len(raw)/1e6:.1f} MB raw, {len(gz)/1e6:.1f} MB gzip")

    # Verify with onnxruntime
    print("\n--- Verification ---")
    import onnxruntime as ort
    test_input = np.random.randn(1, 512, 20).astype(np.float32)
    for path in [fp32_path, int8_path]:
        name = os.path.basename(path)
        try:
            sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            result = sess.run(None, {'hidden_states': test_input})
            print(f"  {name}: OK -- output shape {result[0].shape}, range=[{result[0].min():.2f}, {result[0].max():.2f}]")
        except Exception as e:
            print(f"  {name}: FAILED -- {e}")


if __name__ == '__main__':
    main()
