"""facebook/sam2-hiera-tiny bundled model script.

SAM-2 Hiera tiny is a ~40MB promptable segmenter from Meta (Apache
2.0). It runs comfortably on CPU for low-throughput use; for batch
work it's faster on a small GPU.

Mirrors src/muse/models/bge_reranker_v2_m3.py: heavy imports are NOT
done at module import time. Discovery must be robust to transformers
+ torch being absent OR version-mismatched on the host python (they're
installed into the per-model venv by `muse pull`, not the supervisor
env). Sentinels stay None until ``_ensure_deps()`` runs inside
``Model.__init__``.

Tests that patch ``muse.models.sam2_hiera_tiny.torch`` or
``.AutoModelForMaskGeneration`` set the module attrs directly;
``_ensure_deps`` sees the non-None mocks and skips the real import so
the mocks aren't clobbered.

The Model class delegates to the runtime's shared dispatch helper
(``_segment_with``) so the auto / points / boxes mode logic lives in
one place.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import select_device
from muse.modalities.image_segmentation.protocol import SegmentationResult
from muse.modalities.image_segmentation.runtimes.sam2_runtime import (
    _segment_with,
)


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
AutoModelForMaskGeneration: Any = None
AutoProcessor: Any = None


def _ensure_deps() -> None:
    global torch, AutoModelForMaskGeneration, AutoProcessor
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("sam2_hiera_tiny: torch unavailable: %s", e)
    if AutoModelForMaskGeneration is None:
        try:
            from transformers import (
                AutoModelForMaskGeneration as _amg,
                AutoProcessor as _ap,
            )
            AutoModelForMaskGeneration = _amg
            AutoProcessor = _ap
        except Exception as e:  # noqa: BLE001
            logger.debug("sam2_hiera_tiny: transformers unavailable: %s", e)


MANIFEST = {
    "model_id": "sam2-hiera-tiny",
    "modality": "image/segmentation",
    "hf_repo": "facebook/sam2-hiera-tiny",
    "revision": "7c218beaf0bb87874785f32b582f640134fc1c09",
    "description": (
        "SAM-2 Hiera tiny: ~40MB promptable segmentation, Apache 2.0"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "transformers>=4.43.0",
        "Pillow>=9.1.0",
        "numpy",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "auto",
        "max_masks": 64,
        "supports_text_prompts": False,
        "supports_point_prompts": True,
        "supports_box_prompts": True,
        "supports_automatic": True,
        "memory_gb": 0.8,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt",
        "preprocessor_config.json",
    ],
}


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


class Model:
    """SAM-2 Hiera tiny backend.

    Class is named ``Model`` per muse discovery convention. Tests alias
    ``from muse.models.sam2_hiera_tiny import Model as SAM2HieraTiny``
    for readability.
    """

    model_id = MANIFEST["model_id"]

    def __init__(
        self,
        *,
        hf_repo: str = MANIFEST["hf_repo"],
        local_dir: str | None = None,
        device: str = "auto",
        max_masks: int = 64,
        supports_text_prompts: bool = False,
        supports_point_prompts: bool = True,
        supports_box_prompts: bool = True,
        supports_automatic: bool = True,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoModelForMaskGeneration is None or AutoProcessor is None:
            raise RuntimeError(
                "transformers is not installed; "
                "run `muse pull sam2-hiera-tiny`"
            )
        self._device = _select_device(device)
        self.max_masks = int(max_masks)
        self._supports_text_prompts = bool(supports_text_prompts)
        self._supports_point_prompts = bool(supports_point_prompts)
        self._supports_box_prompts = bool(supports_box_prompts)
        self._supports_automatic = bool(supports_automatic)

        src = local_dir or hf_repo
        logger.info(
            "loading SAM-2 Hiera tiny from %s (device=%s)",
            src, self._device,
        )
        # Access the module-level transformers names so tests can patch them.
        import muse.models.sam2_hiera_tiny as _self_mod
        self._model = _self_mod.AutoModelForMaskGeneration.from_pretrained(src)
        self._processor = _self_mod.AutoProcessor.from_pretrained(src)
        if hasattr(self._model, "to"):
            self._model = self._model.to(self._device)

    def segment(
        self,
        image: Any,
        *,
        mode: str = "auto",
        prompt: str | None = None,
        points: list[list[int]] | None = None,
        boxes: list[list[int]] | None = None,
        max_masks: int | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> SegmentationResult:
        """Run promptable segmentation on a PIL.Image.

        Capability flags act as defense-in-depth; the route layer is
        the primary gate. Delegates to the runtime's shared dispatch
        helper so the logic stays in one place.
        """
        cap = max_masks if max_masks is not None else self.max_masks
        if mode == "text" and not self._supports_text_prompts:
            raise RuntimeError(
                f"model {self.model_id!r} does not support text-prompted segmentation"
            )
        if mode == "points" and not self._supports_point_prompts:
            raise RuntimeError(
                f"model {self.model_id!r} does not support point-prompted segmentation"
            )
        if mode == "boxes" and not self._supports_box_prompts:
            raise RuntimeError(
                f"model {self.model_id!r} does not support box-prompted segmentation"
            )
        if mode == "auto" and not self._supports_automatic:
            raise RuntimeError(
                f"model {self.model_id!r} does not support automatic segmentation"
            )

        records = _segment_with(
            self._model, self._processor, image,
            mode=mode, points=points, boxes=boxes,
            device=self._device, max_masks=cap,
        )
        return SegmentationResult(
            masks=records,
            image_size=image.size,
            mode=mode,
            seed=seed if seed is not None else -1,
            metadata={"model": self.model_id},
        )
