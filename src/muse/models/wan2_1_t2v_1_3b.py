"""Wan 2.1 T2V 1.3B: Apache 2.0, 5s clips at 832x480.

Default low-VRAM video generation bundle. The 1.3B transformer alone is
small, but the pipeline also bundles a UMT5-XXL text encoder (~11GB at
fp16); full-resident load is ~11-12GB and OOMs a 12GB card. The bundle
therefore defaults to sequential CPU offload (`capabilities.cpu_offload
= "sequential"`), which fits 8-12GB GPUs at the cost of speed. On a
bigger card, set `server.video_cpu_offload model` (whole-component
offload, faster) or `off` (full .to(cuda), fastest) to trade VRAM
headroom for speed.

Constructs robustly across diffusers versions:
  - WanPipeline.from_pretrained when WanPipeline is exported (>=0.32.0)
  - DiffusionPipeline.from_pretrained as a fallback (auto-detects from
    model_index.json)

Lazy-import sentinel pattern matches sd_turbo / animatediff_motion_v3.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import dtype_for_name, select_device
from muse.modalities.video_generation.codec import frame_dimensions
from muse.modalities.video_generation.protocol import VideoResult
from muse.modalities.video_generation.runtimes._offload import place_pipeline


logger = logging.getLogger(__name__)


# Sentinels (lazy-import pattern matches the bundled SD/AnimateDiff scripts).
torch: Any = None
WanPipeline: Any = None
DiffusionPipeline: Any = None


def _ensure_deps() -> None:
    global torch, WanPipeline, DiffusionPipeline
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("wan2_1_t2v_1_3b: torch unavailable: %s", e)
    if WanPipeline is None:
        try:
            from diffusers import WanPipeline as _p
            WanPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("wan2_1_t2v_1_3b: WanPipeline unavailable: %s", e)
    if DiffusionPipeline is None:
        try:
            from diffusers import DiffusionPipeline as _d
            DiffusionPipeline = _d
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "wan2_1_t2v_1_3b: DiffusionPipeline unavailable: %s", e,
            )


MANIFEST = {
    "model_id": "wan2-1-t2v-1-3b",
    "modality": "video/generation",
    # The diffusers-format repo (model_index.json + transformer/ vae/
    # text_encoder/ scheduler/ tokenizer/ subfolders) that WanPipeline.
    # from_pretrained requires. The original "Wan-AI/Wan2.1-T2V-1.3B" repo
    # ships the native Wan layout (loose .pth files, no model_index.json)
    # which diffusers cannot load -- the "-Diffusers" suffix is the port.
    "hf_repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
    "description": (
        "Wan 2.1 T2V 1.3B: 5s videos at 832x480, Apache 2.0. The UMT5-XXL "
        "text encoder makes full-resident load ~11-12GB, so the bundle "
        "defaults to sequential CPU offload to fit 8-12GB GPUs (slower; "
        "set server.video_cpu_offload model|off on a bigger card for speed)"
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.32.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow>=9.1.0",
        "imageio[ffmpeg]>=2.31.0",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "cuda",
        "default_duration_seconds": 5.0,
        "default_fps": 5,
        "default_size": (832, 480),
        "min_duration_seconds": 1.0,
        "max_duration_seconds": 10.0,
        "default_steps": 30,
        "default_guidance": 5.0,
        "supports_image_to_video": False,
        "memory_gb": 5.0,
        "cpu_offload": "sequential",
        "vae_tiling": True,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "scheduler/*.json",
        "transformer/*.safetensors", "transformer/*.json",
        "vae/*.safetensors", "vae/*.json",
        "text_encoder/*.safetensors", "text_encoder/*.json",
        "tokenizer/*",
    ],
}


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


class Model:
    """Wan 2.1 T2V 1.3B backend.

    Tries WanPipeline first (preferred when diffusers>=0.32.0 is
    installed); falls back to DiffusionPipeline (auto-detects from
    model_index.json) so the script keeps working when the per-model
    venv resolves an older diffusers.
    """

    model_id = MANIFEST["model_id"]

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        cpu_offload: Any = None,
        vae_tiling: bool = False,
        **_: Any,
    ) -> None:
        _ensure_deps()
        pipeline_cls = WanPipeline or DiffusionPipeline
        if pipeline_cls is None:
            raise RuntimeError(
                "diffusers is not installed; run "
                "`muse pull wan2-1-t2v-1-3b` to set up the per-model venv"
            )
        caps = MANIFEST["capabilities"]
        self._default_duration = caps["default_duration_seconds"]
        self._default_fps = caps["default_fps"]
        self._default_size = tuple(caps["default_size"])
        self._default_steps = caps["default_steps"]
        self._default_guidance = caps["default_guidance"]
        self._device = _select_device(device)

        import muse.models.wan2_1_t2v_1_3b as _mod
        _torch = _mod.torch
        torch_dtype = dtype_for_name(dtype, _torch)

        src = local_dir or hf_repo
        cls_name = getattr(pipeline_cls, "__name__", "Pipeline")
        logger.info(
            "loading %s from %s (device=%s, dtype=%s)",
            cls_name, src, self._device, dtype,
        )
        self._pipe = pipeline_cls.from_pretrained(
            src, torch_dtype=torch_dtype,
        )
        self._pipe = place_pipeline(
            self._pipe, self._device,
            cpu_offload=cpu_offload, vae_tiling=vae_tiling,
        )

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        duration_seconds: float | None = None,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> VideoResult:
        dur = (
            duration_seconds
            if duration_seconds is not None
            else self._default_duration
        )
        out_fps = fps if fps is not None else self._default_fps
        w = width or self._default_size[0]
        h = height or self._default_size[1]
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance
        n_frames = max(1, round(dur * out_fps))

        gen = None
        if seed is not None:
            import muse.models.wan2_1_t2v_1_3b as _mod
            _torch = _mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "num_frames": n_frames,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
            "width": w,
            "height": h,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        frames_list = out.frames[0]
        first = frames_list[0]
        actual_frames = len(frames_list)
        actual_duration = round(actual_frames / max(out_fps, 1), 3)
        first_w, first_h = frame_dimensions(first)
        return VideoResult(
            frames=list(frames_list),
            fps=out_fps,
            width=first_w,
            height=first_h,
            duration_seconds=actual_duration,
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "frames": actual_frames,
                "fps": out_fps,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
            },
        )
