import torch
from torch import nn, Tensor

class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)
        self.window: Tensor  # Type hint for mypy
        self._window_sq: Tensor | None = None  # Lazy-computed for checkpoint compatibility
        self._envelope_cache: dict[int, Tensor] = {}  # Cache window envelope by T

    @property
    def window_sq(self) -> Tensor:
        """Lazily compute and cache window squared."""
        if self._window_sq is None:
            self._window_sq = self.window.square()
        return self._window_sq

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            spec[:,0] = 0
            spec[:,-1] = 0
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)

        pad = (self.win_length - self.hop_length) // 2

        if spec.dim() != 3:
            raise ValueError("Expected a 3D tensor as input")
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope (cached by T for efficiency)
        window_envelope = self._get_window_envelope(T, output_size, pad)

        # Normalize
        y = y / window_envelope

        return y

    def _get_window_envelope(self, T: int, output_size: int, pad: int) -> Tensor:
        """Get cached window envelope for given sequence length T."""
        if T not in self._envelope_cache:
            window_sq = self.window_sq.expand(1, T, -1).transpose(1, 2)
            envelope = torch.nn.functional.fold(
                window_sq,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            ).squeeze()[pad:-pad]
            # Validate envelope (only on first computation)
            if not (envelope > 1e-11).all():
                raise ValueError("Window envelope contains values too close to zero")
            self._envelope_cache[T] = envelope
        return self._envelope_cache[T]
