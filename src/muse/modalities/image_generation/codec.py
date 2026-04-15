"""Image encoding: accept PIL / numpy, output PNG / JPEG / base64 data URL."""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np


class ImageFormatError(ValueError):
    """Raised when an image can't be normalized or encoded."""


def to_pil(image: Any):
    """Normalize supported inputs to a PIL.Image.

    Accepts:
      - PIL.Image — passthrough
      - np.ndarray uint8 HxWxC (C in {3,4}) — direct
      - np.ndarray float HxWxC — rescaled to uint8 assuming [0, 1]
      - np.ndarray HxW (uint8 or float) — grayscale
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        return image

    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            return Image.fromarray(arr)
        raise ImageFormatError(f"numpy image has unsupported shape {arr.shape}")

    try:
        import torch
        if isinstance(image, torch.Tensor):
            return to_pil(image.detach().cpu().numpy())
    except ImportError:
        pass

    raise ImageFormatError(f"cannot convert {type(image).__name__} to PIL")


def to_bytes(image: Any, *, fmt: str = "png") -> bytes:
    """Encode to raw image bytes (PNG or JPEG)."""
    fmt = fmt.lower()
    if fmt not in ("png", "jpeg", "jpg"):
        raise ImageFormatError(f"unsupported image format {fmt!r}")
    pil = to_pil(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG" if fmt in ("jpeg", "jpg") else "PNG")
    return buf.getvalue()


def to_data_url(image: Any, *, fmt: str = "png") -> str:
    """Encode as a data URL suitable for inline use."""
    payload = to_bytes(image, fmt=fmt)
    mime = "jpeg" if fmt in ("jpeg", "jpg") else "png"
    return f"data:image/{mime};base64,{base64.b64encode(payload).decode()}"
