"""TRELLISRuntime: image-to-3D via Microsoft's TRELLIS SDK.

Wraps the official TrellisImageTo3DPipeline from the TRELLIS standalone
library (https://github.com/microsoft/TRELLIS). Muse materializes the SDK
as an immutable reviewed sparse checkout rather than treating it as a
standard Hugging Face Transformers or Diffusers pipeline.

API VERIFIED (2026-05-06): The TRELLIS pipeline is loaded via
`TrellisImageTo3DPipeline.from_pretrained(repo_or_local_path)` where
`TrellisImageTo3DPipeline` is imported from `trellis.pipelines`. There
is NO `trust_remote_code` kwarg on `from_pretrained`; the class is
imported directly from the installed `trellis` package.

The pipeline is invoked via `pipeline.run(image, seed, formats=["mesh"],
sparse_structure_sampler_params={"steps": N}, slat_sampler_params={"steps": N})`.
The return value is a dict with key "mesh" -> list[MeshExtractResult].
MeshExtractResult uses `.vertices` (NOT `.verts`) and `.faces` as
attribute names for the vertex/face tensors.

Device placement: use `pipeline.cuda()` (not `.to(device)`) for GPU.
The pipeline does not inherit from nn.Module, so `.to()` is not
universally available.

Verified against TRELLIS commit
442aa1e1afb9014e80681d3bf604e8d728a86ee7 on 2026-05-06. If the SDK
pin changes, update mocks in
tests/modalities/model_3d_generation/runtimes/test_trellis.py and the
_TRELLIS_PIPELINE sentinel logic here.

Deferred-imports pattern: sentinels populated by _ensure_deps. Tests
patch sentinels directly; _ensure_deps short-circuits on non-None.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.model_3d_generation.codec import mesh_to_glb_result
from muse.modalities.model_3d_generation.protocol import Generation3DResult


logger = logging.getLogger(__name__)


# Deferred-import sentinels. Tests pre-populate these with mocks;
# _ensure_deps short-circuits when they are non-None.
torch: Any = None
_TRELLIS_PIPELINE: Any = None  # TrellisImageTo3DPipeline from trellis.pipelines
trimesh: Any = None
_LAST_IMPORT_ERROR: Exception | None = None


def _ensure_deps() -> None:
    """Lazy-import torch + TrellisImageTo3DPipeline + trimesh.

    The TRELLIS pipeline class comes from the standalone `trellis`
    package (installed via pip from GitHub, not from transformers or
    diffusers). This function verifies the library is present and
    populates the module-level sentinels so the constructor can give
    an actionable error message when a dep is missing.
    """
    global torch, _TRELLIS_PIPELINE, trimesh, _LAST_IMPORT_ERROR
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("TRELLISRuntime torch unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if _TRELLIS_PIPELINE is None:
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline as _p
            _TRELLIS_PIPELINE = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("TRELLISRuntime TrellisImageTo3DPipeline unavailable: %s", e)
            _LAST_IMPORT_ERROR = e
    if trimesh is None:
        try:
            import trimesh as _tm
            trimesh = _tm
        except Exception as e:  # noqa: BLE001
            logger.debug("TRELLISRuntime trimesh unavailable: %s", e)
            _LAST_IMPORT_ERROR = e


class TRELLISRuntime:
    """Image-to-3D runtime over Microsoft's TRELLIS SDK.

    Constructor kwargs (sourced from manifest's capabilities merged in
    by the registry at load_backend time):

      - ``model_id`` (required): catalog id; echoed in result envelope.
      - ``hf_repo``, ``local_dir``: standard weight source.
      - ``device``, ``dtype``: standard device + dtype selection.
      - ``trust_remote_code``: deprecated compatibility kwarg, accepted but
        not forwarded (TRELLIS uses a direct pip install, not Transformers
        dynamic modules).
      - ``sparse_structure_steps`` (default 12): denoising steps for the
        sparse structure sampler. The official TRELLIS default is 25;
        12 is faster and adequate for most inputs.
      - ``slat_steps`` (default 12): denoising steps for the SLAT sampler.
    """

    model_id: str
    supports_image_to_3d: bool = True
    supports_text_to_3d: bool = False
    supports_tools: bool = False

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
        sparse_structure_steps: int = 12,
        slat_steps: int = 12,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run "
                f"`muse models refresh {model_id}` or install "
                "`torch>=2.1.0` into this venv"
            )
        if _TRELLIS_PIPELINE is None:
            raise RuntimeError(
                "TRELLIS SDK source or its native CUDA dependencies are not "
                f"importable. Run `muse models refresh {model_id}` to repair "
                "the reviewed source checkout and basic dependencies. TRELLIS "
                "also requires the CUDA/PyTorch-specific native packages from "
                "Microsoft's pinned setup instructions (spconv and either "
                "flash-attn or xformers at minimum)."
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
        self._default_sparse_structure_steps = int(sparse_structure_steps)
        self._default_slat_steps = int(slat_steps)
        src = local_dir or hf_repo
        with LoadTimer(f"loading TRELLIS from {src}", logger):
            self._pipeline = _TRELLIS_PIPELINE.from_pretrained(src)
            # TRELLIS pipelines use .cuda() / .cpu() rather than .to(device)
            if self._device == "cuda":
                self._pipeline.cuda()
            elif hasattr(self._pipeline, "to"):
                self._pipeline.to(self._device)
        set_inference_mode(self._pipeline)

    def image_to_3d(
        self, image: Any, **kwargs: Any,
    ) -> list[Generation3DResult]:
        """Generate one or more 3D meshes from a single image.

        Parameters
        ----------
        image:
            A PIL.Image.Image (or any object the TRELLIS pipeline
            accepts). The route layer decodes the incoming multipart/
            form-data file into a PIL image before calling this method.
        kwargs:
            Forwarded to pipeline.run():
            - ``n`` (int, default 1): number of samples.
            - ``seed`` (int): random seed; overrides constructor default.
            - ``sparse_structure_steps`` (int): overrides constructor default.
            - ``slat_steps`` (int): overrides constructor default.

        Returns
        -------
        list[Generation3DResult]
            One item per sample (n items total). Each item carries a
            geometry-only GLB blob (no texture baking; that would
            require xatlas + rasterizer extras not declared in
            _TRELLIS_PIP_EXTRAS).
        """
        n = int(kwargs.get("n", 1))
        seed = kwargs.get("seed", self._default_seed)
        if seed is None:
            seed = 42
        seed = int(seed)
        sparse_structure_steps = int(
            kwargs.get("sparse_structure_steps", self._default_sparse_structure_steps)
        )
        slat_steps = int(
            kwargs.get("slat_steps", self._default_slat_steps)
        )

        results: list[Generation3DResult] = []
        for _ in range(max(1, n)):
            # TRELLIS run() signature (verified 2026-05-06):
            #   pipeline.run(image, seed=42, formats=["mesh"],
            #                sparse_structure_sampler_params={"steps": N},
            #                slat_sampler_params={"steps": N})
            # Output: dict {"mesh": [MeshExtractResult, ...]}
            output = self._pipeline.run(
                image,
                seed=seed,
                formats=["mesh"],
                sparse_structure_sampler_params={"steps": sparse_structure_steps},
                slat_sampler_params={"steps": slat_steps},
            )
            # MeshExtractResult uses .vertices (not .verts) and .faces
            mesh_data = output["mesh"][0]
            vertices = mesh_data.vertices.cpu().numpy()
            faces = mesh_data.faces.cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            results.append(mesh_to_glb_result(mesh, self.model_id))
        return results

    def text_to_3d(
        self, prompt: str, **kwargs: Any,
    ) -> list[Generation3DResult]:
        """Not supported: TRELLIS-image-large is image-only.

        The capability flag ``supports_text_to_3d=False`` prevents the
        route layer from calling this method; this raise is a defensive
        check for callers that bypass the route layer.
        """
        raise NotImplementedError(
            "TRELLISRuntime is image-only; TRELLIS-image-large does not "
            "support text-to-3D generation. Use TripoSR or Shap-E for "
            "text-to-3D, or pull a text-capable TRELLIS variant when "
            "available."
        )
