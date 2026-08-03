"""Hunyuan3DRuntime: dual-direction 3D generation via Tencent's Hunyuan3D-2 SDK.

First muse 3D runtime supporting BOTH image_to_3d and text_to_3d from one model.
Text-to-3D is a two-stage pipeline: text -> HunyuanDiTPipeline -> PIL image ->
Hunyuan3DDiTFlowMatchingPipeline -> GLB mesh.

API VERIFIED (2026-05-06) against:
  - https://github.com/Tencent/Hunyuan3D-2/blob/main/hy3dgen/shapegen/pipelines.py
  - https://github.com/Tencent/Hunyuan3D-2/blob/main/gradio_app.py
  - https://github.com/Tencent/Hunyuan3D-2/blob/main/minimal_demo.py
If the SDK changes upstream, update mocks in
tests/modalities/model_3d_generation/runtimes/test_hunyuan3d.py and the
_HUNYUAN3D_PIPELINE / _HUNYUAN3D_T2I_PIPELINE sentinel logic here.

Deferred-imports pattern: sentinels populated by _ensure_deps. Tests
patch sentinels directly; _ensure_deps short-circuits on non-None.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from muse.core.artifacts import ArtifactBundleError, local_artifact_directory
from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.model_3d_generation.codec import mesh_to_glb_result
from muse.modalities.model_3d_generation.protocol import Generation3DResult


logger = logging.getLogger(__name__)


# Deferred-import sentinels. Tests pre-populate these with mocks;
# _ensure_deps short-circuits when they are non-None.
torch: Any = None
_HUNYUAN3D_PIPELINE: Any = None      # Hunyuan3DDiTFlowMatchingPipeline
_HUNYUAN3D_T2I_PIPELINE: Any = None  # HunyuanDiTPipeline (optional, text-to-3D only)
trimesh: Any = None
_LAST_IMPORT_ERROR: Exception | None = None


def _bundle_member(local_dir: str, subdir: str, *, label: str) -> str:
    """Resolve one fixed, real child directory inside a local bundle."""
    try:
        return local_artifact_directory(local_dir, subdir, label=label)
    except ArtifactBundleError as exc:
        raise RuntimeError(f"Hunyuan3D {exc}; re-pull the model") from exc


def _close_intermediate_image(image: Any) -> None:
    """Best-effort release of a text-to-3D stage-one image."""
    close = getattr(image, "close", None)
    if not callable(close):
        return
    try:
        close()
    except BaseException:  # noqa: BLE001
        # Cleanup failure must not replace a shape-pipeline failure or a
        # cancellation already propagating through the synchronous runtime.
        logger.debug("failed to close Hunyuan text-to-3D image", exc_info=True)


def _ensure_deps(load_t2i: bool = False) -> None:
    """Lazy-import torch + Hunyuan3D shape pipeline + trimesh.

    When load_t2i=True, also imports the HunyuanDiT-v1.1 text-to-image
    pipeline. The T2I import is gated so the shape-only path does not
    pay the extra import cost.
    """
    global torch, _HUNYUAN3D_PIPELINE, trimesh, _LAST_IMPORT_ERROR
    global _HUNYUAN3D_T2I_PIPELINE
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("Hunyuan3DRuntime torch unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if _HUNYUAN3D_PIPELINE is None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline as _p
            _HUNYUAN3D_PIPELINE = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("Hunyuan3DRuntime SDK unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if trimesh is None:
        try:
            import trimesh as _tm
            trimesh = _tm
        except Exception as e:  # noqa: BLE001
            logger.debug("Hunyuan3DRuntime trimesh unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if load_t2i and _HUNYUAN3D_T2I_PIPELINE is None:
        try:
            from hy3dgen.text2image import HunyuanDiTPipeline as _p
            _HUNYUAN3D_T2I_PIPELINE = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("Hunyuan3DRuntime HunyuanDiTPipeline unavailable: %s", e)
            _LAST_IMPORT_ERROR = e


class Hunyuan3DRuntime:
    """Dual-direction 3D generation runtime over Hunyuan3D-2.

    Wraps Tencent's Hunyuan3D-2 SDK for both image-to-3D and text-to-3D.
    Text-to-3D is a two-stage pipeline: text -> HunyuanDiTPipeline ->
    PIL image -> Hunyuan3DDiTFlowMatchingPipeline -> GLB mesh.

    Constructor kwargs (sourced from manifest's capabilities merged in
    by the registry at load_backend time):

      - ``model_id`` (required): catalog id; echoed in result envelope.
      - ``hf_repo``, ``local_dir``: standard weight source.
      - ``device``, ``dtype``: standard device + dtype selection.
      - ``trust_remote_code``: deprecated compatibility kwarg, accepted but
        not forwarded (Hunyuan3D-2 uses a pip-installed SDK, not Transformers
        dynamic modules).
      - ``num_inference_steps`` (default 50): denoising steps for the
        shape pipeline. The SDK default is 50; lower values are faster.
      - ``guidance_scale`` (default 5.0): classifier-free guidance for
        shape generation. SDK default.
      - ``shape_model_subdir``, ``t2i_model_subdir``: fixed members of the
        local two-checkpoint bundle created by ``muse pull``. The text-to-image
        pipeline is loaded lazily from that local bundle on first use.
      - ``t2i_model_id``: compatibility source used only when constructing the
        runtime directly without a Muse-managed ``local_dir``.
    """

    model_id: str
    supports_image_to_3d: bool = True
    supports_text_to_3d: bool = True

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "fp16",
        trust_remote_code: bool = False,
        seed: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        t2i_model_id: str = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        shape_model_subdir: str | None = None,
        t2i_model_subdir: str | None = None,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run "
                f"`muse models refresh {model_id}` or install "
                "`torch>=2.1.0` into this venv"
            )
        if _HUNYUAN3D_PIPELINE is None:
            raise RuntimeError(
                "Hunyuan3D-2 SDK not loadable; run "
                f"`muse models refresh {model_id}`. The SDK is the "
                "`hy3dgen` package from Tencent's GitHub. If the git "
                "pip-install failed during pull, follow Tencent's "
                "setup.sh manually in the per-model venv at "
                f"~/.muse/venvs/{model_id}/. "
                "See: https://github.com/Tencent/Hunyuan3D-2"
            )
        if trimesh is None:
            raise RuntimeError(
                "trimesh is not installed; needed for GLB export. "
                f"Run `muse models refresh {model_id}`."
            )
        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        self._dtype = dtype_for_name(dtype, torch_module=torch)
        self._default_seed = seed
        self._default_num_inference_steps = int(num_inference_steps)
        self._default_guidance_scale = float(guidance_scale)
        if local_dir is None:
            shape_src = hf_repo
            self._t2i_model_source: str | None = t2i_model_id
        else:
            shape_src = (
                _bundle_member(local_dir, shape_model_subdir, label="shape-model")
                if shape_model_subdir is not None
                else local_dir
            )
            self._t2i_model_source = (
                _bundle_member(local_dir, t2i_model_subdir, label="text-to-image")
                if t2i_model_subdir is not None
                else None
            )
        # Lazily loaded on first text_to_3d call; guarded by _t2i_lock.
        self._t2i_pipeline: Any = None
        self._t2i_lock = threading.Lock()

        with LoadTimer(
            f"loading Hunyuan3D-2 shape pipeline from {shape_src}", logger,
        ):
            # from_pretrained signature (verified 2026-05-06):
            #   from_pretrained(model_path, device, dtype, ...)
            # device and dtype are forwarded directly.
            self._pipeline = _HUNYUAN3D_PIPELINE.from_pretrained(
                shape_src,
                device=self._device,
                dtype=self._dtype,
            )
            # Belt-and-suspenders: chain .to() in case from_pretrained
            # did not apply device/dtype (SDK behavior may vary across versions).
            if hasattr(self._pipeline, "to"):
                moved = self._pipeline.to(self._device, self._dtype)
                # Some in-place SDK implementations return None, while
                # fluent implementations return the moved pipeline.
                if moved is not None:
                    self._pipeline = moved
        set_inference_mode(self._pipeline)

    def _load_t2i_pipeline(self) -> Any:
        """Load the text-to-image pipeline on first text_to_3d call.

        Uses _HUNYUAN3D_T2I_PIPELINE sentinel so tests can inject a mock
        without triggering a real download.
        """
        if self._t2i_model_source is None:
            raise RuntimeError(
                "text_to_3d requires the pinned text-to-image checkpoint in "
                f"the Muse-managed bundle; re-pull {self.model_id!r}"
            )
        _ensure_deps(load_t2i=True)
        if _HUNYUAN3D_T2I_PIPELINE is None:
            raise RuntimeError(
                "HunyuanDiTPipeline (text-to-image) is not loadable; the "
                "hy3dgen.text2image module is required for text_to_3d. "
                f"Run `muse models refresh {self.model_id}`. "
                "See: https://github.com/Tencent/Hunyuan3D-2"
            )
        with LoadTimer("loading Hunyuan3D-2 text-to-image pipeline", logger):
            try:
                return _HUNYUAN3D_T2I_PIPELINE(
                    self._t2i_model_source,
                    device=self._device,
                )
            except (OSError, EnvironmentError) as e:
                raise RuntimeError(
                    "text_to_3d could not load its pinned local "
                    "HunyuanDiT-v1.1 checkpoint from "
                    f"{self._t2i_model_source!r}; re-pull {self.model_id!r}. "
                    f"Original error: {e}"
                ) from e

    def _parse_call_kwargs(self, kwargs: dict) -> tuple[int, Any, int, float]:
        """Read (n, seed, num_inference_steps, guidance_scale) from per-call kwargs."""
        return (
            int(kwargs.get("n", 1)),
            kwargs.get("seed", self._default_seed),
            int(kwargs.get("num_inference_steps", self._default_num_inference_steps)),
            float(kwargs.get("guidance_scale", self._default_guidance_scale)),
        )

    def image_to_3d(
        self, image: Any, **kwargs: Any,
    ) -> list[Generation3DResult]:
        """Generate one or more 3D meshes from a single image.

        Parameters
        ----------
        image:
            A PIL.Image.Image (or path/URL string; the SDK accepts both).
            The route layer decodes the incoming multipart/form-data file
            into a PIL image before calling this method.
        kwargs:
            Forwarded to pipeline():
            - ``n`` (int, default 1): number of samples.
            - ``seed`` (int): random seed.
            - ``num_inference_steps`` (int): denoising steps.
            - ``guidance_scale`` (float): classifier-free guidance scale.

        Returns
        -------
        list[Generation3DResult]
            One item per sample (n items total). Each item carries a
            geometry-only GLB blob.
        """
        n, seed, steps, gs = self._parse_call_kwargs(kwargs)

        generator = None
        if seed is not None and torch is not None:
            generator = torch.Generator().manual_seed(int(seed))

        results: list[Generation3DResult] = []
        for _ in range(max(1, n)):
            # Verified call signature (2026-05-06):
            #   pipeline(image=..., num_inference_steps=...,
            #            guidance_scale=..., generator=None, ...)
            # Returns List[List[trimesh.Trimesh]] when output_type='trimesh'
            # (the default). result[0][0] is the first mesh.
            output = self._pipeline(
                image=image,
                num_inference_steps=steps,
                guidance_scale=gs,
                generator=generator,
            )
            results.append(mesh_to_glb_result(output[0][0], self.model_id))
        return results

    def text_to_3d(
        self, prompt: str, **kwargs: Any,
    ) -> list[Generation3DResult]:
        """Generate one or more 3D meshes from a text prompt.

        Implements the two-stage Hunyuan3D-2 text-to-3D pipeline:
          1. HunyuanDiTPipeline converts the text prompt to a PIL image.
          2. Hunyuan3DDiTFlowMatchingPipeline converts the image to a mesh.

        The T2I pipeline is loaded lazily on the first call to this
        method; subsequent calls reuse the cached pipeline instance.

        Parameters
        ----------
        prompt:
            Text description of the 3D object to generate.
        kwargs:
            - ``n`` (int, default 1): number of samples.
            - ``seed`` (int): random seed.
            - ``num_inference_steps`` (int): denoising steps for shape gen.
            - ``guidance_scale`` (float): guidance scale for shape gen.

        Returns
        -------
        list[Generation3DResult]
            One item per sample (n items total).
        """
        n, seed, steps, gs = self._parse_call_kwargs(kwargs)

        # Lazy-load the text-to-image pipeline on first use (double-checked lock).
        if self._t2i_pipeline is None:
            with self._t2i_lock:
                if self._t2i_pipeline is None:  # re-check inside the lock
                    self._t2i_pipeline = self._load_t2i_pipeline()

        generator = None
        if seed is not None and torch is not None:
            generator = torch.Generator().manual_seed(int(seed))

        results: list[Generation3DResult] = []
        for _ in range(max(1, n)):
            # Stage 1: text -> image via HunyuanDiTPipeline.
            # Verified call signature: t2i_pipeline(prompt) -> PIL.Image.Image
            image = self._t2i_pipeline(prompt)

            try:
                # Stage 2: image -> mesh via
                # Hunyuan3DDiTFlowMatchingPipeline.
                output = self._pipeline(
                    image=image,
                    num_inference_steps=steps,
                    guidance_scale=gs,
                    generator=generator,
                )
            finally:
                _close_intermediate_image(image)
            results.append(mesh_to_glb_result(output[0][0], self.model_id))
        return results
