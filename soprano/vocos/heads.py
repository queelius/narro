import torch
from torch import nn
from .spectral_ops import ISTFT


class ISTFTHead(nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "center"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Conv1d(dim, out_dim, kernel_size=1)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, C, T), where B is the batch size,
                        C is the model dimension, and T is the sequence length.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag).clip(max=1e2)
        S = mag * (torch.cos(p) + 1j * torch.sin(p))
        audio = self.istft(S)
        return audio
