"""stabilityai/stable-diffusion-x4-upscaler bundled model script.

4x latent diffusion super-resolution. Apache 2.0. ~3GB on disk.
Uses diffusers.StableDiffusionUpscalePipeline. Roughly 30-60s per
512 -> 2048 image at 20 steps on a 12GB GPU.

Mirrors src/muse/models/sd_turbo.py: heavy imports are NOT done at
module import time. Discovery must be robust to diffusers + transformers
being absent OR version-mismatched on the host python (they're
installed into the per-model venv by `muse pull`, not the supervisor
env). Sentinels stay None until `_ensure_deps()` runs inside
Model.__init__.

Tests that patch `muse.models.stable_diffusion_x4_upscaler.torch` or
`.StableDiffusionUpscalePipeline` set the module attrs directly;
`_ensure_deps` sees the non-None mocks and skips the real import so
the mocks aren't clobbered.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import dtype_for_name, select_device
from muse.modalities.image_upscale.protocol import UpscaleResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
StableDiffusionUpscalePipeline: Any = None


def _ensure_deps() -> None:
    """Lazy-import torch + diffusers. Safe if deps are absent or broken."""
    global torch, StableDiffusionUpscalePipeline
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("sd-x4-upscaler: torch unavailable: %s", e)
    if StableDiffusionUpscalePipeline is None:
        try:
            from diffusers import StableDiffusionUpscalePipeline as _p
            StableDiffusionUpscalePipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "sd-x4-upscaler: StableDiffusionUpscalePipeline unavailable: %s",
                e,
            )


MANIFEST = {
    "model_id": "stable-diffusion-x4-upscaler",
    "modality": "image/upscale",
    "hf_repo": "stabilityai/stable-diffusion-x4-upscaler",
    "revision": "572c99286543a273bfd17fac263db5a77be12c4c",
    "description": (
        "SD x4 upscaler: 4x super-resolution via latent diffusion, Apache 2.0"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow",
        "safetensors",
    ),
    "system_packages": (),
    "capabilities": {
        # CPU is too slow at 20 steps for 256 -> 1024; mark cuda as the
        # advisory device. `auto` device selection still works if the
        # operator overrides it.
        "device": "cuda",
        "default_scale": 4,
        "supported_scales": [4],
        "default_steps": 20,
        "default_guidance": 9.0,
        # Conservative VRAM peak estimate (fp16, 256 -> 1024).
        "memory_gb": 6.0,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "feature_extractor/*.json",
        "scheduler/*.json",
        "text_encoder/*.fp16.safetensors", "text_encoder/*.json",
        "tokenizer/*",
        "unet/*.fp16.safetensors", "unet/*.json",
        "vae/*.fp16.safetensors", "vae/*.json",
    ],
}


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


class Model:
    """SD x4 upscaler backend.

    Class is named `Model` per muse discovery convention. Tests alias
    `from muse.models.stable_diffusion_x4_upscaler import Model as
    SDx4Upscaler` for readability.
    """

    model_id = MANIFEST["model_id"]
    supported_scales = list(MANIFEST["capabilities"]["supported_scales"])

    def __init__(
        self,
        *,
        hf_repo: str = MANIFEST["hf_repo"],
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        default_steps: int = 20,
        default_guidance: float = 9.0,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if StableDiffusionUpscalePipeline is None:
            raise RuntimeError(
                "diffusers is not installed; "
                "run `muse pull stable-diffusion-x4-upscaler`"
            )
        self._device = _select_device(device)
        # Access the module-level `torch` name so tests can patch it.
        import muse.models.stable_diffusion_x4_upscaler as _self_mod
        _torch = _self_mod.torch
        torch_dtype = dtype_for_name(dtype, _torch)
        self._src = local_dir or hf_repo
        self._dtype = dtype
        self._torch_dtype = torch_dtype
        self._default_steps = default_steps
        self._default_guidance = default_guidance

        logger.info(
            "loading SD x4 upscaler from %s (device=%s, dtype=%s)",
            self._src, self._device, dtype,
        )
        self._pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self._src,
            torch_dtype=torch_dtype,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

    def upscale(
        self,
        image: Any,
        *,
        scale: int | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> UpscaleResult:
        """4x super-resolution upscale of `image`.

        SD x4 always upscales by exactly 4x. The `scale` arg is
        informational; the route layer enforces supported_scales=[4].
        """
        ow, oh = image.size
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance
        prompt_str = prompt if prompt is not None else ""
        actual_scale = int(scale) if scale is not None else 4

        gen = None
        if seed is not None:
            import muse.models.stable_diffusion_x4_upscaler as _self_mod
            _torch = _self_mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt_str,
            "image": image,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        upscaled = out.images[0]
        uw, uh = upscaled.size
        return UpscaleResult(
            image=upscaled,
            original_width=ow,
            original_height=oh,
            upscaled_width=uw,
            upscaled_height=uh,
            scale=actual_scale,
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt_str,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
            },
        )
