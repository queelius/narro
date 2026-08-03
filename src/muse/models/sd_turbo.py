"""SD-Turbo: 1-step distilled Stable Diffusion, 512x512.

Model by Stability AI (SAI Community License, non-commercial + limited
commercial). Uses diffusers AutoPipelineForText2Image. Very fast (1
inference step by default) at modest quality; good first backend to
prove the image/generation modality end-to-end.

Kept as-is for first-found-wins continuity (curated `sd-turbo` aliases
this script). See `muse.modalities.image_generation.runtimes.diffusers`
for the canonical generic-runtime implementation that powers other
diffusers text-to-image models.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any

from muse.core.runtime_helpers import dtype_for_name, select_device
from muse.modalities.image_generation import ImageResult

logger = logging.getLogger(__name__)

# Heavy imports are NOT done at module import time. Discovery must be
# robust to diffusers + transformers being absent OR version-mismatched
# on the host python (they're installed into the per-model venv by
# `muse pull sd-turbo`, not the supervisor env). Sentinels stay None
# until `_ensure_deps()` runs inside Model.__init__. Tests that patch
# `muse.models.sd_turbo.torch` or `.AutoPipelineForText2Image` set the
# module attrs directly; `_ensure_deps` sees the non-None mocks and
# skips the real import so the mocks aren't clobbered.
torch: Any = None
AutoPipelineForText2Image: Any = None
AutoPipelineForImage2Image: Any = None
AutoPipelineForInpainting: Any = None


def _ensure_deps() -> None:
    """Lazy-import torch + diffusers. Safe if deps are absent or broken.

    Imports each symbol independently so that tests which patch only one
    of the module attrs (e.g. `AutoPipelineForText2Image` but not
    `torch`) still get the real unpatched symbol for the others.
    """
    global torch, AutoPipelineForText2Image, AutoPipelineForImage2Image
    global AutoPipelineForInpainting
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("sd-turbo torch unavailable: %s", e)
    if AutoPipelineForText2Image is None:
        try:
            # RuntimeError possible here when diffusers' _LazyModule
            # wraps a broken internal import chain (e.g. diffusers
            # pinned to a transformers version that removed MT5Tokenizer).
            from diffusers import AutoPipelineForText2Image as _p
            AutoPipelineForText2Image = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("sd-turbo diffusers unavailable: %s", e)
    if AutoPipelineForImage2Image is None:
        try:
            from diffusers import AutoPipelineForImage2Image as _p
            AutoPipelineForImage2Image = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("sd-turbo AutoPipelineForImage2Image unavailable: %s", e)
    if AutoPipelineForInpainting is None:
        try:
            from diffusers import AutoPipelineForInpainting as _p
            AutoPipelineForInpainting = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("sd-turbo AutoPipelineForInpainting unavailable: %s", e)


MANIFEST = {
    "model_id": "sd-turbo",
    "modality": "image/generation",
    "hf_repo": "stabilityai/sd-turbo",
    "revision": "b261bac6fd2cf515557d5d0707481eafa0485ec2",
    "description": "Stable Diffusion Turbo: 1-step distilled, 512x512",
    "license": "SAI Community License",
    "pip_extras": (
        # torch: hard runtime dep. Pulled transitively by accelerate,
        # but declaring it keeps the contract self-contained.
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        # transformers: required by StableDiffusionPipeline at construct
        # time (CLIPTextModel text encoder). diffusers lists it as an
        # optional extra, so we declare it here or per-model venvs end
        # up missing it and the worker exits at load.
        "transformers>=4.36.0",
        "accelerate",
        "Pillow",
        "safetensors",
    ),
    "system_packages": (),
    "capabilities": {
        # Pinned "cuda", not "auto": allow_patterns below downloads ONLY
        # *.fp16.safetensors (no fp32 weights on disk), so an "auto"
        # resolution on a CPU-only host would load an fp16 pipeline that
        # errors at inference (no CPU kernel for half-precision matmul).
        # CPU diffusion is impractical anyway. Matches the
        # stable-audio-open / animatediff / x4-upscaler precedent.
        "device": "cuda",
        "default_size": (512, 512),
        "supports_negative_prompt": True,
        "supports_seeded_generation": True,
        "supports_img2img": True,
        "supports_inpainting": True,
        "supports_variations": True,
        # Conservative VRAM/RAM peak estimate (fp16, default 512x512).
        # Annotation feeds `muse models list`; `muse models probe sd-turbo`
        # measures the real number on this hardware.
        "memory_gb": 4.0,
        # SD-Turbo ships fp32 + fp16 + a standalone single-file checkpoint
        # (~47 GB total). The diffusers runtime only loads the fp16
        # subfolder weights (~4 GB). Mirror the resolver plugin's
        # allow_patterns so bundled-pull doesn't bloat the cache either.
        "allow_patterns": [
            "*.fp16.safetensors", "*.json", "*.txt", "*.md",
            "feature_extractor/*.json",
            "scheduler/*.json",
            "text_encoder/*.fp16.safetensors", "text_encoder/*.json",
            "tokenizer/*",
            "unet/*.fp16.safetensors", "unet/*.json",
            "vae/*.fp16.safetensors", "vae/*.json",
        ],
    },
}


class Model:
    """SD-Turbo image generation backend.

    Class is named `Model` per muse discovery convention. Tests alias
    `from muse.models.sd_turbo import Model as SDTurboModel` for readability.
    """

    model_id = MANIFEST["model_id"]
    default_size = (512, 512)

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoPipelineForText2Image is None:
            raise RuntimeError("diffusers is not installed; run `muse pull sd-turbo`")
        self._device = _select_device(device)
        # Access the module-level `torch` name so tests can patch it.
        import muse.models.sd_turbo as _self_mod
        _torch = _self_mod.torch
        torch_dtype = dtype_for_name(dtype, _torch)
        # Stash for lazy img2img / inpaint pipeline loads (same checkpoint + dtype).
        self._src = local_dir or hf_repo
        self._dtype = dtype
        self._torch_dtype = torch_dtype
        self._i2i_pipe = None
        self._inp_pipe = None
        # Guard the lazy img2img/inpaint pipeline builds against a
        # double-build race: route handlers run each request on its own
        # asyncio.to_thread worker, so two concurrent first requests could
        # otherwise both see `self._i2i_pipe is None`, both call
        # `from_pipe(...)`, briefly double VRAM, and leak one instance.
        self._i2i_lock = threading.Lock()
        self._inp_lock = threading.Lock()
        logger.info(
            "loading SD-Turbo from %s (device=%s, dtype=%s)",
            self._src, self._device, dtype,
        )
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self._src,
            torch_dtype=torch_dtype,
            variant="fp16" if dtype == "float16" else None,
        )
        if self._device != "cpu":
            # Diffusers pipelines move in place.  Retain the exact object
            # created above instead of making ownership depend on a fluent
            # return value; wrappers and test doubles are not required to
            # return ``self`` from ``to()``.
            self._pipe.to(self._device)

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        init_image: Any = None,
        strength: float | None = None,
        **_: Any,
    ) -> ImageResult:
        if init_image is not None:
            return self._generate_img2img(
                prompt,
                init_image=init_image,
                strength=strength,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                seed=seed,
            )
        w = width or self.default_size[0]
        h = height or self.default_size[1]
        n_steps = steps if steps is not None else 1
        cfg = guidance if guidance is not None else 0.0

        gen = None
        if seed is not None:
            import muse.models.sd_turbo as _self_mod
            _torch = _self_mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "width": w,
            "height": h,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        img = out.images[0]
        return ImageResult(
            image=img,
            width=img.size[0],
            height=img.size[1],
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
            },
        )

    def _generate_img2img(
        self,
        prompt: str,
        *,
        init_image: Any,
        strength: float | None,
        negative_prompt: str | None,
        steps: int | None,
        guidance: float | None,
        seed: int | None,
    ) -> ImageResult:
        # Lazy-load the img2img pipeline, cached on the instance.
        # Double-checked locking: the cheap unlocked check keeps the hot
        # (already-built) path lock-free, while the lock around the build
        # itself serializes two concurrent first-access callers so only
        # one from_pipe() ever runs.
        if self._i2i_pipe is None:
            with self._i2i_lock:
                if self._i2i_pipe is None:
                    import muse.models.sd_turbo as _self_mod
                    _i2i = _self_mod.AutoPipelineForImage2Image
                    if _i2i is None:
                        raise RuntimeError(
                            "diffusers AutoPipelineForImage2Image is not available; "
                            "run `muse pull sd-turbo`"
                        )
                    logger.info(
                        "loading SD-Turbo img2img pipeline from %s (device=%s, dtype=%s)",
                        self._src, self._device, self._dtype,
                    )
                    # Share UNet/VAE/text-encoders with the already-loaded t2i pipeline so we
                    # don't double VRAM on small GPUs. AutoPipelineForImage2Image.from_pipe
                    # is the canonical diffusers idiom for this; from_pretrained allocates a
                    # fresh copy and OOMs.
                    # Reference: https://huggingface.co/docs/diffusers/api/pipelines/auto_pipeline
                    self._i2i_pipe = _i2i.from_pipe(self._pipe)

        n_steps = steps if steps is not None else 1
        cfg = guidance if guidance is not None else 0.0
        s = strength if strength is not None else 0.5

        # img2img diffusers contract: num_inference_steps * strength must be >= 1
        # (lower values round to 0 effective denoise steps and crash the VAE).
        # Bump steps to satisfy the contract while preserving the user's
        # requested strength.
        min_steps_for_strength = max(1, math.ceil(1.0 / max(s, 0.01)))
        if n_steps < min_steps_for_strength:
            logger.info(
                "img2img bumping num_inference_steps from %d to %d to satisfy "
                "strength=%.2f * steps >= 1 contract",
                n_steps, min_steps_for_strength, s,
            )
            n_steps = min_steps_for_strength

        gen = None
        if seed is not None:
            import muse.models.sd_turbo as _self_mod
            _torch = _self_mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "image": init_image,
            "strength": s,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._i2i_pipe(**call_kwargs)
        img = out.images[0]
        return ImageResult(
            image=img,
            width=img.size[0],
            height=img.size[1],
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "steps": n_steps,
                "guidance": cfg,
                "strength": s,
                "model": self.model_id,
                "mode": "img2img",
            },
        )

    def inpaint(
        self,
        prompt: str,
        *,
        init_image: Any,
        mask_image: Any,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        strength: float | None = None,
        **_: Any,
    ) -> ImageResult:
        """Inpainting: regenerate the masked region per prompt.

        Mirrors the pattern in
        muse.modalities.image_generation.runtimes.diffusers. White mask
        pixels are regenerated; black are preserved. RGBA masks are
        normalized to single-channel (mode L) before being passed to
        diffusers.

        Lazy-loads AutoPipelineForInpainting via from_pipe(self._pipe)
        to share VRAM with the loaded t2i pipeline.
        """
        # Double-checked locking: see _generate_img2img for rationale.
        if self._inp_pipe is None:
            with self._inp_lock:
                if self._inp_pipe is None:
                    import muse.models.sd_turbo as _self_mod
                    _inp = _self_mod.AutoPipelineForInpainting
                    if _inp is None:
                        raise RuntimeError(
                            "diffusers AutoPipelineForInpainting is not available; "
                            "run `muse pull sd-turbo`"
                        )
                    logger.info(
                        "loading SD-Turbo inpaint pipeline from %s (device=%s, dtype=%s)",
                        self._src, self._device, self._dtype,
                    )
                    self._inp_pipe = _inp.from_pipe(self._pipe)

        # Normalize mask to grayscale (mode L). Diffusers accepts L or
        # RGB; mode L is the documented contract. RGBA masks (alpha as
        # the regenerate region) get flattened by .convert("L").
        try:
            mask_mode = getattr(mask_image, "mode", None)
        except Exception:  # noqa: BLE001
            mask_mode = None
        if mask_mode is not None and mask_mode != "L":
            mask_image = mask_image.convert("L")

        n_steps = steps if steps is not None else 1
        cfg = guidance if guidance is not None else 0.0
        s = strength if strength is not None else 0.99

        # Same steps * strength >= 1 contract as img2img.
        min_steps_for_strength = max(1, math.ceil(1.0 / max(s, 0.01)))
        if n_steps < min_steps_for_strength:
            logger.info(
                "inpaint bumping num_inference_steps from %d to %d to satisfy "
                "strength=%.2f * steps >= 1 contract",
                n_steps, min_steps_for_strength, s,
            )
            n_steps = min_steps_for_strength

        gen = None
        if seed is not None:
            import muse.models.sd_turbo as _self_mod
            _torch = _self_mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "image": init_image,
            "mask_image": mask_image,
            "strength": s,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if width is not None:
            call_kwargs["width"] = width
        if height is not None:
            call_kwargs["height"] = height
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._inp_pipe(**call_kwargs)
        img = out.images[0]
        return ImageResult(
            image=img,
            width=img.size[0],
            height=img.size[1],
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "steps": n_steps,
                "guidance": cfg,
                "strength": s,
                "model": self.model_id,
                "mode": "inpaint",
            },
        )

    def vary(
        self,
        *,
        init_image: Any,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        strength: float | None = None,
        **_: Any,
    ) -> ImageResult:
        """Variations: img2img with empty prompt and high strength.

        Reuses the existing img2img path. OpenAI's variations route
        carries no prompt, so we pass empty string. Default strength
        0.85 hits the "recognizable but visibly different" zone.
        """
        result = self._generate_img2img(
            prompt="",
            init_image=init_image,
            strength=strength if strength is not None else 0.85,
            negative_prompt=None,
            steps=steps,
            guidance=guidance,
            seed=seed,
        )
        result.metadata["mode"] = "variations"
        return result


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)
