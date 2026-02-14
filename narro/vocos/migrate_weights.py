"""
Weight migration utility for decoder checkpoint optimization.

Converts checkpoint weights to the current optimized format:
- ConvNeXtBlock pwconv Linear (out, in) -> Conv1d (out, in, 1)
- ConvNeXtBlock gamma (dim,) -> (dim, 1) for broadcasting
- LayerNorm weight/bias (dim,) -> ChannelsFirstLayerNorm (1, dim, 1)
- ISTFTHead Linear (out, in) -> Conv1d (out, in, 1)
"""

import torch
from pathlib import Path


def _needs_conv1d_migration(key: str) -> bool:
    """Check if key is a 2D weight that should be migrated to Conv1d (out, in) -> (out, in, 1)."""
    return ('.pwconv1.weight' in key or '.pwconv2.weight' in key
            or key.endswith('.out.weight'))


def migrate_decoder_weights(state_dict: dict) -> dict:
    """
    Migrate decoder weights from Linear to Conv1d format.

    Args:
        state_dict: Original state dict with Linear weights

    Returns:
        New state dict with Conv1d weights
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        if _needs_conv1d_migration(key) and value.dim() == 2:
            new_state_dict[key] = value.unsqueeze(2)
        elif '.gamma' in key and value.dim() == 1:
            new_state_dict[key] = value.unsqueeze(1)
        elif _is_layernorm_key(key) and value.dim() == 1:
            new_state_dict[key] = value.unsqueeze(0).unsqueeze(2)
        else:
            new_state_dict[key] = value

    return new_state_dict


def _is_layernorm_key(key: str) -> bool:
    """Check if key belongs to a LayerNorm weight or bias (now ChannelsFirstLayerNorm)."""
    suffix = key.rsplit('.', 1)[-1]  # 'weight' or 'bias'
    if suffix not in ('weight', 'bias'):
        return False
    # Match keys like 'decoder.norm.weight', 'norm.weight', 'decoder.convnext.0.norm.bias',
    # 'decoder.final_layer_norm.weight', 'final_layer_norm.bias'
    prefix = key.rsplit('.', 1)[0]  # e.g. 'decoder.norm', 'final_layer_norm'
    return prefix.endswith('.norm') or prefix == 'norm' or prefix.endswith('.final_layer_norm') or prefix == 'final_layer_norm'


def migrate_checkpoint_file(input_path: str, output_path: str | None = None) -> str:
    """
    Migrate a decoder checkpoint file from Linear to Conv1d format.

    Args:
        input_path: Path to original checkpoint file
        output_path: Path to save migrated checkpoint (default: input_path with .migrated suffix)

    Returns:
        Path to migrated checkpoint file
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.migrated{input_path.suffix}"
    else:
        output_path = Path(output_path)

    state_dict = torch.load(input_path, map_location='cpu', weights_only=True)
    migrated_dict = migrate_decoder_weights(state_dict)
    torch.save(migrated_dict, output_path)

    return str(output_path)


def is_migrated(state_dict: dict) -> bool:
    """
    Check if a state dict has already been migrated to the optimized format.

    Checks pwconv weights (Linear→Conv1d), LayerNorm weights (1D→3D),
    and ISTFTHead weights (Linear→Conv1d).

    Args:
        state_dict: State dict to check

    Returns:
        True if already in optimized format
    """
    for key, value in state_dict.items():
        if _needs_conv1d_migration(key) or _is_layernorm_key(key):
            if value.dim() != 3:
                return False
        elif '.gamma' in key and value.dim() != 2:
            return False
    return True


def load_with_migration(state_dict: dict) -> dict:
    """
    Load a state dict, migrating if necessary.

    Args:
        state_dict: State dict to load (may be Linear or Conv1d format)

    Returns:
        State dict in Conv1d format
    """
    if is_migrated(state_dict):
        return state_dict
    return migrate_decoder_weights(state_dict)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Migrate decoder weights from Linear to Conv1d format')
    parser.add_argument('input', help='Input checkpoint file')
    parser.add_argument('--output', '-o', help='Output checkpoint file (default: input.migrated.pth)')

    args = parser.parse_args()
    output = migrate_checkpoint_file(args.input, args.output)
    print(f"Migrated checkpoint saved to: {output}")
