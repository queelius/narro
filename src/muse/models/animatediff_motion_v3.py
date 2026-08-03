"""AnimateDiff motion v3 + SD 1.5 base, pulled as one pinned bundle."""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import dtype_for_name, select_device
from muse.modalities.image_animation.protocol import AnimationResult


logger = logging.getLogger(__name__)

_ADAPTER_REPO = "guoyww/animatediff-motion-adapter-v1-5-3"
_ADAPTER_REVISION = "2e8139b1d1269fd8a21deb96ad19455e187692eb"
_BASE_REPO = "emilianJR/epiCRealism"
_BASE_REVISION = "6522cf856b8c8e14638a0aaa7bd89b1b098aed17"
_ADAPTER_SUBDIR = "motion_adapter"
_BASE_SUBDIR = "base_model"
_ADAPTER_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.md"]
_ADAPTER_REQUIRED_PATTERNS = ["config.json", "*.safetensors"]
_BASE_ALLOW_PATTERNS = [
    "*.safetensors", "*.json", "*.txt", "*.model", "*.md",
    "feature_extractor/*", "scheduler/*",
    "safety_checker/*.safetensors", "safety_checker/*.json",
    "text_encoder/*.safetensors", "text_encoder/*.json",
    "tokenizer/*", "unet/*.safetensors", "unet/*.json",
    "vae/*.safetensors", "vae/*.json",
]
_BASE_REQUIRED_PATTERNS = [
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "safety_checker/config.json",
    "safety_checker/*.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/*.safetensors",
    "tokenizer/tokenizer_config.json",
    "unet/config.json",
    "unet/*.safetensors",
    "vae/config.json",
    "vae/*.safetensors",
]


# Sentinels (lazy-import pattern matches sd_turbo).
torch: Any = None
AnimateDiffPipeline: Any = None
MotionAdapter: Any = None


def _ensure_deps() -> None:
    global torch, AnimateDiffPipeline, MotionAdapter
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("animatediff_motion_v3: torch unavailable: %s", e)
    if AnimateDiffPipeline is None:
        try:
            from diffusers import AnimateDiffPipeline as _p
            AnimateDiffPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("animatediff_motion_v3: AnimateDiffPipeline unavailable: %s", e)
    if MotionAdapter is None:
        try:
            from diffusers import MotionAdapter as _m
            MotionAdapter = _m
        except Exception as e:  # noqa: BLE001
            logger.debug("animatediff_motion_v3: MotionAdapter unavailable: %s", e)


MANIFEST = {
    "model_id": "animatediff-motion-v3",
    "modality": "image/animation",
    "hf_repo": _ADAPTER_REPO,
    "revision": _ADAPTER_REVISION,
    "description": "AnimateDiff motion v3 + SD 1.5 base, 16 frames @ 8fps, 512x512",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow>=9.1.0",
        "safetensors",
    ),
    "system_packages": (),
    "capabilities": {
        "supports_text_to_animation": True,
        "supports_image_to_animation": False,
        "default_frames": 16,
        "default_fps": 8,
        "min_frames": 8,
        "max_frames": 24,
        "default_size": (512, 512),
        "default_steps": 25,
        "default_guidance": 7.5,
        "device": "cuda",
        "base_model": _BASE_REPO,
        "base_model_revision": _BASE_REVISION,
        "adapter_model_subdir": _ADAPTER_SUBDIR,
        "base_model_subdir": _BASE_SUBDIR,
        # SD 1.5 base + motion adapter at fp16, plus per-frame activations
        # for 16 frames at 512x512. Conservative peak estimate.
        "memory_gb": 10.0,
    },
    "hf_artifacts": [
        {
            "repo_id": _ADAPTER_REPO,
            "revision": _ADAPTER_REVISION,
            "subdir": _ADAPTER_SUBDIR,
            "allow_patterns": _ADAPTER_ALLOW_PATTERNS,
            "required_patterns": _ADAPTER_REQUIRED_PATTERNS,
        },
        {
            "repo_id": _BASE_REPO,
            "revision": _BASE_REVISION,
            "subdir": _BASE_SUBDIR,
            "allow_patterns": _BASE_ALLOW_PATTERNS,
            "required_patterns": _BASE_REQUIRED_PATTERNS,
        },
    ],
}


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


class Model:
    """AnimateDiff motion v3 backend.

    Muse passes ``local_dir`` as the complete pinned motion-adapter + base
    bundle. Repository identifiers are used only for explicit direct
    construction without that bundle.
    """

    model_id = MANIFEST["model_id"]

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        base_model: str = _BASE_REPO,
        adapter_model_subdir: str | None = None,
        base_model_subdir: str | None = None,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AnimateDiffPipeline is None or MotionAdapter is None:
            raise RuntimeError(
                "diffusers AnimateDiff is not installed; run "
                "`muse pull animatediff-motion-v3`"
            )
        caps = MANIFEST["capabilities"]
        self._default_frames = caps["default_frames"]
        self._default_fps = caps["default_fps"]
        self._default_size = tuple(caps["default_size"])
        self._default_steps = caps["default_steps"]
        self._default_guidance = caps["default_guidance"]
        self._device = _select_device(device)

        import muse.models.animatediff_motion_v3 as _mod
        _torch = _mod.torch
        torch_dtype = dtype_for_name(dtype, _torch)

        if local_dir is None:
            adapter_src = hf_repo
            base_src = base_model
        else:
            if adapter_model_subdir is None or base_model_subdir is None:
                raise RuntimeError(
                    "AnimateDiff local weights predate the complete artifact "
                    f"bundle; re-pull {self.model_id!r}"
                )
            from muse.core.artifacts import (
                ArtifactBundleError,
                local_artifact_directory,
            )
            try:
                adapter_src = local_artifact_directory(
                    local_dir,
                    adapter_model_subdir,
                    label="AnimateDiff motion adapter",
                )
                base_src = local_artifact_directory(
                    local_dir,
                    base_model_subdir,
                    label="AnimateDiff base model",
                )
            except ArtifactBundleError as exc:
                raise RuntimeError(
                    f"AnimateDiff artifact bundle is invalid; re-pull "
                    f"{self.model_id!r}: {exc}"
                ) from exc
        logger.info("loading MotionAdapter from %s", adapter_src)
        adapter = MotionAdapter.from_pretrained(adapter_src, torch_dtype=torch_dtype)

        logger.info(
            "loading AnimateDiffPipeline base=%s + adapter (device=%s, dtype=%s)",
            base_src, self._device, dtype,
        )
        self._pipe = AnimateDiffPipeline.from_pretrained(
            base_src,
            motion_adapter=adapter,
            torch_dtype=torch_dtype,
        )
        if self._device != "cpu":
            self._pipe.to(self._device)

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        frames: int | None = None,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        init_image: Any = None,
        strength: float | None = None,
        **_: Any,
    ) -> AnimationResult:
        if init_image is not None:
            raise NotImplementedError(
                "animatediff-motion-v3 does not support init_image; route layer "
                "should have gated this via supports_image_to_animation"
            )
        n_frames = frames if frames is not None else self._default_frames
        out_fps = fps if fps is not None else self._default_fps
        w = width or self._default_size[0]
        h = height or self._default_size[1]
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance

        gen = None
        if seed is not None:
            import muse.models.animatediff_motion_v3 as _mod
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
        return AnimationResult(
            frames=list(frames_list),
            fps=out_fps,
            width=first.size[0],
            height=first.size[1],
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "frames": n_frames,
                "fps": out_fps,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
                "base_model": MANIFEST["capabilities"]["base_model"],
            },
        )
