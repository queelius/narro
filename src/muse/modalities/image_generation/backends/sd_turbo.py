"""SD-Turbo backend: 1-step distilled Stable Diffusion.

Uses diffusers AutoPipelineForText2Image with stabilityai/sd-turbo.
Very fast (1 inference step by default) at modest quality; good first
backend to prove the images.generations modality end-to-end.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.images.generations.protocol import ImageResult

logger = logging.getLogger(__name__)

# Heavy imports are deferred so `muse --help` and the CLI work without
# diffusers installed. The `muse pull sd-turbo` command installs them.
try:
    import torch
    from diffusers import AutoPipelineForText2Image
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    AutoPipelineForText2Image = None  # type: ignore


class SDTurboModel:
    model_id = "sd-turbo"
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
        if AutoPipelineForText2Image is None:
            raise RuntimeError("diffusers is not installed; run `muse pull sd-turbo`")
        self._device = _select_device(device)
        # Access the module-level `torch` name so tests can patch it.
        import muse.images.generations.backends.sd_turbo as _self_mod
        _torch = _self_mod.torch
        torch_dtype = None
        if _torch is not None:
            torch_dtype = {
                "float16": _torch.float16,
                "float32": _torch.float32,
                "bfloat16": _torch.bfloat16,
            }[dtype]
        src = local_dir or hf_repo
        logger.info("loading SD-Turbo from %s (device=%s, dtype=%s)", src, self._device, dtype)
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            src,
            torch_dtype=torch_dtype,
            variant="fp16" if dtype == "float16" else None,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

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
        **_: Any,
    ) -> ImageResult:
        w = width or self.default_size[0]
        h = height or self.default_size[1]
        n_steps = steps if steps is not None else 1
        cfg = guidance if guidance is not None else 0.0

        gen = None
        if seed is not None:
            import muse.images.generations.backends.sd_turbo as _self_mod
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


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
