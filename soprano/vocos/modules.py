import torch
from torch import nn


class ChannelsFirstLayerNorm(nn.Module):
    """LayerNorm that operates on (B, C, T) input without transposing.

    Normalizes over the channel dimension (dim=1). Equivalent to
    ``x.transpose(1,2) -> LayerNorm -> transpose(1,2)`` but avoids
    the two memory copies.

    Weight shape: (1, C, 1) for broadcasting with (B, C, T).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, correction=0)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        dw_kernel_size: int = 9,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=dw_kernel_size, padding=dw_kernel_size//2, groups=dim)  # depthwise conv
        self.norm = ChannelsFirstLayerNorm(dim, eps=1e-6)
        # Use Conv1d instead of Linear to avoid transpose operations
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(intermediate_dim, dim, kernel_size=1)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x  # gamma is (dim, 1), broadcasts with (B, dim, T)

        x = residual + x
        return x
