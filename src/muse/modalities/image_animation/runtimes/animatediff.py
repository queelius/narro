"""Generic AnimateDiff runtime via diffusers AnimateDiffPipeline.

Two-component model: a base SD 1.5 (or compatible) checkpoint provides
text encoder + UNet + VAE; a MotionAdapter provides the temporal layers.
Muse-managed loads resolve both from fixed children of ``local_dir``.
``hf_repo`` and ``base_model`` remain direct-construction fallbacks when no
local bundle is supplied.

Lazy-import sentinel pattern matches sd_turbo and runtimes/diffusers.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.artifacts import ArtifactBundleError, local_artifact_directory
from muse.core.runtime_helpers import dtype_for_name, select_device
from muse.modalities.image_animation.protocol import AnimationResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
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
            logger.debug("animatediff: torch unavailable: %s", e)
    if AnimateDiffPipeline is None:
        try:
            from diffusers import AnimateDiffPipeline as _p
            AnimateDiffPipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("animatediff: AnimateDiffPipeline unavailable: %s", e)
    if MotionAdapter is None:
        try:
            from diffusers import MotionAdapter as _m
            MotionAdapter = _m
        except Exception as e:  # noqa: BLE001
            logger.debug("animatediff: MotionAdapter unavailable: %s", e)


def _select_device(device: str) -> str:
    """Thin delegator preserved for test imports. Real logic in runtime_helpers."""
    return select_device(device, torch_module=torch)


class AnimateDiffRuntime:
    """AnimateDiff runtime backed by AnimateDiffPipeline + MotionAdapter.

    Construction kwargs:
      - hf_repo: motion adapter repo id (the manifest's hf_repo)
      - local_dir: complete adapter + base bundle from `muse pull`
      - device, dtype, model_id: standard
      - base_model: HF repo id of the base SD 1.5 (or compatible) checkpoint
      - default_frames, default_fps, default_size, default_steps, default_guidance:
        manifest-driven defaults injected via capabilities splat
      - **kwargs: absorbed (future capability flags)
    """

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        model_id: str,
        base_model: str,
        adapter_model_subdir: str | None = None,
        base_model_subdir: str | None = None,
        default_frames: int = 16,
        default_fps: int = 8,
        default_size: tuple[int, int] = (512, 512),
        default_steps: int = 25,
        default_guidance: float = 7.5,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AnimateDiffPipeline is None or MotionAdapter is None:
            raise RuntimeError(
                "diffusers AnimateDiff is not installed; ensure museq[images] "
                "extras are installed in the per-model venv"
            )
        self.model_id = model_id
        self.default_size = tuple(default_size)
        self._default_frames = default_frames
        self._default_fps = default_fps
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._device = _select_device(device)

        import muse.modalities.image_animation.runtimes.animatediff as _mod
        _torch = _mod.torch
        torch_dtype = dtype_for_name(dtype, _torch)

        if local_dir is None:
            adapter_src = hf_repo
            base_src = base_model
        else:
            if adapter_model_subdir is None or base_model_subdir is None:
                raise RuntimeError(
                    "AnimateDiff local weights predate the complete artifact "
                    f"bundle; re-pull {model_id!r}"
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
                    f"{model_id!r}: {exc}"
                ) from exc
        logger.info(
            "loading MotionAdapter from %s (model_id=%s, dtype=%s)",
            adapter_src, model_id, dtype,
        )
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
        # AnimateDiff base does not support img2vid in v1; the route layer
        # gates this via supports_image_to_animation. If init_image lands
        # here on this runtime, it's a programming error or a model with
        # mis-set capability; surface clearly.
        if init_image is not None:
            raise NotImplementedError(
                "AnimateDiffRuntime base path does not support init_image; "
                "use a model whose capability supports_image_to_animation=True"
            )

        n_frames = frames if frames is not None else self._default_frames
        out_fps = fps if fps is not None else self._default_fps
        w = width or self.default_size[0]
        h = height or self.default_size[1]
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance

        gen = None
        if seed is not None:
            import muse.modalities.image_animation.runtimes.animatediff as _mod
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
        # AnimateDiffPipeline returns out.frames as list-of-lists (one
        # video, multiple frames). Take the first video.
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
            },
        )
