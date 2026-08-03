"""Shared image-processor fallback ladder.

Public API:
  read_encoder_hints(src) -> dict
  DerivedImageProcessor (class)
  build_image_processor(src, *, overrides, model_id) -> Any
  ImageProcessorError (exception)

Lifted from `image_ocr/runtimes/hf_vision2seq.py` in v0.42.1 so other
modalities can adopt the same fallback ladder without re-implementing it.
The override hatch (`overrides`) lets manifests declare ground-truth
preprocessing parameters when AutoImageProcessor's sniff would be
wrong (e.g. TexTeller: 1-channel grayscale at 448x448).

Dispatch order:
  1. If overrides is non-empty: skip AutoImageProcessor; build
     DerivedImageProcessor(**overrides) directly. Operator-trust path
     (mean/std may be omitted here; DerivedImageProcessor's own 0.5/0.5
     default applies since the operator explicitly signed off).
  2. AutoImageProcessor.from_pretrained(src). Return on success.
  3. On AutoImageProcessor failure: read encoder hints from config.json.
     Build DerivedImageProcessor(**hints) ONLY when hints include
     explicit image_mean AND image_std (ground-truth stats, not a
     guess). num_channels/image_size alone are not sufficient signal.
  4. Else: raise ImageProcessorError pointing at the override hatch.

Tier 3 (ViT defaults) was dropped in v0.42.1 because it produced
silently wrong output for the very class of repos this fallback targets.
The structured error message points operators at the override hatch.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class ImageProcessorError(RuntimeError):
    """Raised when the dispatch ladder cannot produce a usable
    image processor. The error message includes the override-hatch
    hint so operators have a clear remediation path.
    """


def read_encoder_hints(src: str) -> dict:
    """Read encoder preprocessing hints from a model's config.json.

    Returns a dict with any of `num_channels`, `image_size`, `image_mean`,
    `image_std` that the encoder config exposes. Empty dict on missing
    or malformed config.json.

    Vision-encoder-decoder configs nest the encoder's hyperparams under
    either `encoder` (canonical layout) or directly at the top level
    (some older repos). We check both.
    """
    config_path = Path(src) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("could not read %s: %s", config_path, e)
        return {}
    enc = cfg.get("encoder") or cfg
    out: dict = {}
    for key in ("num_channels", "image_size", "image_mean", "image_std"):
        if key in enc:
            out[key] = enc[key]
    return out


class DerivedImageProcessor:
    """Synthesize a minimal image preprocessor from explicit hyperparams.

    Mimics the AutoImageProcessor interface for callers: callable with
    (image, return_tensors="pt"), returns a BatchFeature with
    pixel_values shape (1, num_channels, H, W) supporting .to(device).

    Conventions:
      - num_channels=1: convert PIL to grayscale ("L" mode); (1,1,H,W).
      - num_channels=3 (default): convert to RGB; (1,3,H,W).
      - image_size is the side length (int) or a (h, w) pair.
      - Pixel range normalized to [-1, 1] via mean=0.5, std=0.5
        per channel (canonical ViT default for grayscale and RGB
        encoders that don't ship preprocessor configs). Override via
        explicit image_mean / image_std when needed.

    Length-validation in __init__: if image_mean / image_std are
    provided and don't match num_channels, raise. Catches accidental
    misuse at construction time, well before inference.
    """

    def __init__(
        self,
        *,
        num_channels: int = 3,
        image_size: int | tuple[int, int] = 224,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
    ) -> None:
        self.num_channels = int(num_channels)
        if isinstance(image_size, (list, tuple)):
            self.height, self.width = int(image_size[0]), int(image_size[1])
        else:
            self.height = self.width = int(image_size)
        if image_mean is not None and len(image_mean) != self.num_channels:
            raise ValueError(
                f"image_mean has {len(image_mean)} values but "
                f"num_channels is {self.num_channels}"
            )
        if image_std is not None and len(image_std) != self.num_channels:
            raise ValueError(
                f"image_std has {len(image_std)} values but "
                f"num_channels is {self.num_channels}"
            )
        # A zero (or negative) std channel makes the (arr - mean) / std
        # normalization divide by zero -> inf/nan pixels that poison the
        # whole forward pass silently. Reject it at construction with a
        # message that points at the override hatch, since this only
        # reaches here from an explicit capabilities.image_processor_overrides.
        if image_std is not None and any(s <= 0 for s in image_std):
            raise ValueError(
                f"image_std values must all be positive; got {image_std}. "
                f"A zero/negative std divides by zero during normalization."
            )
        self.image_mean = image_mean or [0.5] * self.num_channels
        self.image_std = image_std or [0.5] * self.num_channels

    def __call__(self, image: Any, *, return_tensors: str = "pt") -> Any:
        """Preprocess a PIL image into a model-ready BatchFeature.

        Output has shape (1, num_channels, H, W) and supports
        .to(device) for the runtime's inputs.to(self._device) chain.
        """
        target_mode = "L" if self.num_channels == 1 else "RGB"
        if not hasattr(image, "convert"):
            raise TypeError(
                f"DerivedImageProcessor expected a PIL Image; "
                f"got {type(image)!r}"
            )
        from PIL import Image as _PILImage
        image = image.convert(target_mode).resize(
            (self.width, self.height), _PILImage.Resampling.BICUBIC,
        )

        import numpy as np
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if self.num_channels == 1:
            arr = arr[..., None]
        mean = np.array(self.image_mean, dtype=np.float32)
        std = np.array(self.image_std, dtype=np.float32)
        arr = ((arr - mean) / std).transpose(2, 0, 1)[None, :, :, :]

        if return_tensors != "pt":
            return {"pixel_values": arr}
        try:
            import torch as _torch
        except ImportError:
            raise RuntimeError(
                "torch is not available; DerivedImageProcessor requires "
                "torch for tensor output (return_tensors='pt')"
            )
        from transformers.feature_extraction_utils import BatchFeature
        return BatchFeature({"pixel_values": _torch.from_numpy(arr)})


def _load_auto_image_processor(src: str) -> Any:
    """Lazy-load AutoImageProcessor and call from_pretrained.

    Factored as a module function so tests can monkey-patch it
    cleanly. Raises if transformers is not installed or the call fails.
    """
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(src)


def build_image_processor(
    src: str,
    *,
    overrides: dict | None = None,
    model_id: str,
) -> Any:
    """Three-tier image-processor dispatch (override-first).

    Tier 1: If `overrides` is non-empty, build DerivedImageProcessor
    directly with those values. AutoImageProcessor is NOT consulted.
    Tier 2: AutoImageProcessor.from_pretrained(src). Return on success.
    Tier 3: On AutoImageProcessor failure, read encoder hints from
    config.json. If hints found, build DerivedImageProcessor.
    Tier 4: Else, raise ImageProcessorError pointing at the override hatch.

    `overrides` schema (when set): {num_channels, image_size,
    image_mean?, image_std?}. Empty dict is treated as None.
    """
    if overrides:
        logger.info(
            "Building DerivedImageProcessor for %s from manifest overrides "
            "(num_channels=%s image_size=%s)",
            model_id,
            overrides.get("num_channels"),
            overrides.get("image_size"),
        )
        return DerivedImageProcessor(
            num_channels=int(overrides.get("num_channels", 3)),
            image_size=overrides.get("image_size", 224),
            image_mean=overrides.get("image_mean"),
            image_std=overrides.get("image_std"),
        )

    auto_error: Exception | None = None
    try:
        return _load_auto_image_processor(src)
    except Exception as e_auto:  # noqa: BLE001
        auto_error = e_auto
        logger.debug(
            "AutoImageProcessor.from_pretrained(%s) failed: %s",
            src, e_auto,
        )

    hints = read_encoder_hints(src)
    # Only trust hints-derived normalization when the config actually
    # supplies image_mean/image_std. Falling back to a silent 0.5/0.5
    # guess here would reintroduce the exact silently-wrong-normalization
    # failure mode that the old ViT-defaults tier was removed for in
    # v0.42.1 -- just via a different tier. num_channels/image_size alone
    # are not enough signal to trust; require the actual stats.
    if hints and hints.get("image_mean") is not None and hints.get("image_std") is not None:
        logger.info(
            "Loaded %s with DerivedImageProcessor "
            "(num_channels=%s image_size=%s; derived from config.json)",
            model_id,
            hints.get("num_channels"),
            hints.get("image_size"),
        )
        return DerivedImageProcessor(
            num_channels=int(hints.get("num_channels", 3)),
            image_size=hints.get("image_size", 224),
            image_mean=hints.get("image_mean"),
            image_std=hints.get("image_std"),
        )

    if auto_error is None:  # defensive: the failed auto path must set this
        auto_error = RuntimeError("unknown AutoImageProcessor failure")
    auto_message = str(auto_error).strip()
    auto_detail = type(auto_error).__name__
    if auto_message:
        auto_detail += f": {auto_message}"
    raise ImageProcessorError(
        f"Cannot load image processor for {src!r}: AutoImageProcessor "
        f"failed ({auto_detail}) and config.json provided no encoder hints "
        f"with explicit image_mean/image_std. Add "
        f"capabilities.image_processor_overrides in the manifest with "
        f"explicit values, e.g. {{num_channels: 1, image_size: 448}}."
    ) from auto_error
