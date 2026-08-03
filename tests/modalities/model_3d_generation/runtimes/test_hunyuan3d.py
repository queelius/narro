"""Hunyuan3DRuntime: mocked-dep tests.

Sentinel restoration matches the sibling Shap-E / TRELLIS pattern.
Mocks reflect the REAL Hunyuan3D-2 SDK API (verified 2026-05-06 against
https://github.com/Tencent/Hunyuan3D-2, NOT inferred from the spec's
speculative shape).

Key API facts verified:
- Shape pipeline: Hunyuan3DDiTFlowMatchingPipeline from hy3dgen.shapegen
- Load: pipeline.from_pretrained(repo, device=..., dtype=...)
- Call: pipeline(image=..., num_inference_steps=N, guidance_scale=F,
                 generator=G) -> List[List[trimesh.Trimesh]]
- Mesh: output[0][0] is a standard trimesh.Trimesh (.vertices, .faces)
- Device: .to(device, dtype) method available
- Text-to-3D: two-stage pipeline:
    1. HunyuanDiTPipeline(t2i_model_id, device=device)(prompt) -> PIL.Image
    2. Shape pipeline(image=pil_image, ...) -> mesh
- T2I pipeline: loaded lazily on first text_to_3d call
- No trust_remote_code kwarg in from_pretrained (SDK is pip-installed,
  not a HuggingFace transformers repo with remote code)

If the SDK changes upstream, only the mocks in _wire_runtime need
updating; the test structure stays.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

import muse.modalities.model_3d_generation.runtimes.hunyuan3d as mod


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    """Reset module-top sentinels between tests."""
    orig = (
        mod.torch,
        mod._HUNYUAN3D_PIPELINE,
        mod._HUNYUAN3D_T2I_PIPELINE,
        mod.trimesh,
        mod._LAST_IMPORT_ERROR,
    )
    yield
    (
        mod.torch,
        mod._HUNYUAN3D_PIPELINE,
        mod._HUNYUAN3D_T2I_PIPELINE,
        mod.trimesh,
        mod._LAST_IMPORT_ERROR,
    ) = orig


# ---- missing-deps errors ----


@pytest.mark.parametrize(
    "sentinel_name,match_str",
    [
        ("torch", "torch is not installed"),
        ("_HUNYUAN3D_PIPELINE", "Hunyuan3D-2 SDK"),
        ("trimesh", "trimesh"),
    ],
)
def test_raises_when_dep_not_installed(monkeypatch, sentinel_name, match_str):
    """RuntimeError names the missing dep so the operator can act."""
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    # Wire all deps to valid mocks, then null out the one under test.
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = False
    mod.torch.backends = MagicMock(mps=None)
    mod._HUNYUAN3D_PIPELINE = MagicMock()
    mod.trimesh = MagicMock()
    setattr(mod, sentinel_name, None)
    with pytest.raises(RuntimeError, match=match_str):
        mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")


# ---- helper: wire a fully-mocked runtime ----


def _wire_runtime():
    """Install fake torch + Hunyuan3D shape pipeline + T2I pipeline + trimesh.

    Mocks reflect the REAL Hunyuan3D-2 SDK API (verified 2026-05-06):
      - Shape pipeline: from_pretrained(repo, device=..., dtype=...)
      - Shape pipeline call: pipeline(image=..., num_inference_steps=N,
            guidance_scale=F, generator=G) -> List[List[trimesh.Trimesh]]
      - Mesh: output[0][0] is a standard trimesh.Trimesh
      - T2I pipeline: HunyuanDiTPipeline(model_id, device=device)(prompt)
            -> PIL.Image
    """
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    # Generator mock for seed handling
    fake_generator = MagicMock()
    fake_torch.Generator.return_value.manual_seed.return_value = fake_generator
    mod.torch = fake_torch

    # Shape pipeline: __call__ returns List[List[trimesh.Trimesh]].
    # The real API: output[0][0] is the mesh.
    pipeline = MagicMock()
    pipeline.to = MagicMock(return_value=pipeline)

    # Build the nested list return: [[trimesh.Trimesh]]
    # The real trimesh.Trimesh has .vertices and .faces (standard interface).
    fake_mesh = MagicMock()
    fake_mesh.export = MagicMock(return_value=b"GLB_BYTES")
    # output[0][0] = fake_mesh
    pipeline.return_value = [[fake_mesh]]

    pipe_factory = MagicMock()
    pipe_factory.from_pretrained = MagicMock(return_value=pipeline)
    mod._HUNYUAN3D_PIPELINE = pipe_factory

    # T2I pipeline mock: constructor returns a callable that takes (prompt)
    # and returns a PIL image mock.
    fake_pil_image = MagicMock(name="pil_image_from_t2i")
    t2i_instance = MagicMock()
    t2i_instance.return_value = fake_pil_image
    t2i_factory = MagicMock()
    t2i_factory.return_value = t2i_instance
    mod._HUNYUAN3D_T2I_PIPELINE = t2i_factory

    # Fake trimesh module.
    fake_trimesh_mesh = MagicMock()
    fake_trimesh_mesh.export = MagicMock(return_value=b"GLB_BYTES")
    fake_trimesh = MagicMock()
    fake_trimesh.Trimesh = MagicMock(return_value=fake_trimesh_mesh)
    mod.trimesh = fake_trimesh

    return pipeline, t2i_factory, fake_trimesh, fake_mesh


# ---- happy path: image_to_3d ----


def test_image_to_3d_returns_glb_in_result():
    """image_to_3d returns a list with one Generation3DResult per sample."""
    pipeline, _, _, fake_mesh = _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(
        model_id="hunyuan3d-2", hf_repo="tencent/Hunyuan3D-2", device="cpu",
    )
    image = MagicMock(name="pil_image")
    results = runtime.image_to_3d(image)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].model_id == "hunyuan3d-2"
    assert results[0].glb_bytes == b"GLB_BYTES"
    fake_mesh.export.assert_called_once_with(file_type="glb")


def test_in_place_device_move_returning_none_keeps_shape_pipeline():
    pipeline, _, _, _ = _wire_runtime()
    pipeline.to.return_value = None

    runtime = mod.Hunyuan3DRuntime(
        model_id="hunyuan3d-2",
        hf_repo="tencent/Hunyuan3D-2",
        device="cpu",
    )

    assert runtime._pipeline is pipeline
    pipeline.to.assert_called_once()


def test_image_to_3d_calls_pipeline_with_image_kwarg():
    """Regression guard: image is passed as image= kwarg to pipeline(), not positional."""
    pipeline, _, _, _ = _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")
    image = MagicMock(name="pil_image")
    runtime.image_to_3d(image)
    call_kwargs = pipeline.call_args.kwargs
    assert "image" in call_kwargs
    assert call_kwargs["image"] is image


def test_image_to_3d_forwards_num_inference_steps_and_guidance_scale():
    """num_inference_steps and guidance_scale are forwarded to pipeline()."""
    pipeline, _, _, _ = _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")
    runtime.image_to_3d(
        MagicMock(),
        num_inference_steps=20,
        guidance_scale=3.0,
    )
    call_kwargs = pipeline.call_args.kwargs
    assert call_kwargs.get("num_inference_steps") == 20
    assert call_kwargs.get("guidance_scale") == 3.0


def test_image_to_3d_loops_over_n():
    """For n>1, pipeline() is called n times and n results are returned."""
    pipeline, _, _, _ = _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")
    results = runtime.image_to_3d(MagicMock(), n=2)
    assert len(results) == 2
    assert pipeline.call_count == 2


# ---- happy path: text_to_3d ----


def test_text_to_3d_returns_glb_in_result():
    """text_to_3d returns a list with one Generation3DResult per sample."""
    pipeline, t2i_factory, _, fake_mesh = _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(
        model_id="hunyuan3d-2", hf_repo="tencent/Hunyuan3D-2", device="cpu",
    )
    results = runtime.text_to_3d("a chair shaped like an avocado")
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].model_id == "hunyuan3d-2"
    assert results[0].glb_bytes == b"GLB_BYTES"


def test_text_to_3d_calls_t2i_pipeline_with_prompt():
    """Stage 1: HunyuanDiTPipeline is called with the text prompt."""
    pipeline, t2i_factory, _, _ = _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")
    runtime.text_to_3d("test prompt")
    # t2i_factory() creates the instance; instance("test prompt") generates image.
    t2i_instance = t2i_factory.return_value
    t2i_instance.assert_called_once_with("test prompt")


def test_text_to_3d_passes_t2i_output_image_to_shape_pipeline():
    """Stage 2: the PIL image from T2I is passed as image= to shape pipeline."""
    pipeline, t2i_factory, _, _ = _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")
    runtime.text_to_3d("test prompt")
    shape_call_kwargs = pipeline.call_args.kwargs
    # The image passed to the shape pipeline should be the T2I output.
    t2i_instance = t2i_factory.return_value
    fake_pil_image = t2i_instance.return_value
    assert shape_call_kwargs.get("image") is fake_pil_image


def test_text_to_3d_closes_t2i_image_after_shape_inference():
    """The request-local stage-one image closes after shape inference."""
    _, t2i_factory, _, _ = _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")

    runtime.text_to_3d("test prompt")

    t2i_factory.return_value.return_value.close.assert_called_once_with()


def test_text_to_3d_closes_t2i_image_when_shape_pipeline_raises():
    pipeline, t2i_factory, _, _ = _wire_runtime()
    failure = RuntimeError("shape inference failed")
    pipeline.side_effect = failure
    runtime = mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")

    with pytest.raises(RuntimeError) as caught:
        runtime.text_to_3d("test prompt")

    assert caught.value is failure
    t2i_factory.return_value.return_value.close.assert_called_once_with()


def test_text_to_3d_closes_t2i_image_when_shape_pipeline_is_cancelled():
    pipeline, t2i_factory, _, _ = _wire_runtime()
    cancellation = asyncio.CancelledError("request cancelled")
    pipeline.side_effect = cancellation
    runtime = mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")

    with pytest.raises(asyncio.CancelledError) as caught:
        runtime.text_to_3d("test prompt")

    assert caught.value is cancellation
    t2i_factory.return_value.return_value.close.assert_called_once_with()


def test_text_to_3d_forwards_num_inference_steps():
    """num_inference_steps is forwarded to the shape pipeline in text_to_3d."""
    pipeline, _, _, _ = _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")
    runtime.text_to_3d("test", num_inference_steps=20, guidance_scale=3.0)
    call_kwargs = pipeline.call_args.kwargs
    assert call_kwargs.get("num_inference_steps") == 20
    assert call_kwargs.get("guidance_scale") == 3.0


def test_text_to_3d_loops_over_n():
    """For n>1, both T2I and shape pipelines are called n times."""
    pipeline, t2i_factory, _, _ = _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")
    results = runtime.text_to_3d("test", n=2)
    assert len(results) == 2
    assert pipeline.call_count == 2
    t2i_instance = t2i_factory.return_value
    assert t2i_instance.call_count == 2


def test_text_to_3d_reuses_t2i_pipeline_across_calls():
    """T2I pipeline is loaded once and reused for subsequent text_to_3d calls."""
    _, t2i_factory, _, _ = _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(model_id="m", hf_repo="x", device="cpu")
    runtime.text_to_3d("first call")
    runtime.text_to_3d("second call")
    # T2I factory (constructor) called once, not twice.
    assert t2i_factory.call_count == 1


# ---- capability attrs ----


def test_supports_capability_attrs():
    """Class-level capability attributes: BOTH directions True for Hunyuan3D-2."""
    assert mod.Hunyuan3DRuntime.supports_image_to_3d is True
    assert mod.Hunyuan3DRuntime.supports_text_to_3d is True


# ---- other constructor / loading behaviors ----


def test_local_dir_preferred_over_hf_repo():
    """local_dir is passed to from_pretrained instead of hf_repo."""
    _wire_runtime()
    mod.Hunyuan3DRuntime(
        model_id="m",
        hf_repo="tencent/Hunyuan3D-2",
        local_dir="/tmp/local-hunyuan",
        device="cpu",
    )
    src_arg = mod._HUNYUAN3D_PIPELINE.from_pretrained.call_args.args[0]
    assert src_arg == "/tmp/local-hunyuan"


def test_managed_bundle_routes_both_pipelines_to_local_members(tmp_path):
    _wire_runtime()
    shape_dir = tmp_path / "shape"
    t2i_dir = tmp_path / "text_to_image"
    shape_dir.mkdir()
    t2i_dir.mkdir()
    runtime = mod.Hunyuan3DRuntime(
        model_id="m",
        hf_repo="tencent/Hunyuan3D-2",
        local_dir=str(tmp_path),
        shape_model_subdir="shape",
        t2i_model_subdir="text_to_image",
        device="cpu",
    )

    assert mod._HUNYUAN3D_PIPELINE.from_pretrained.call_args.args[0] == str(shape_dir)
    runtime.text_to_3d("a chair")
    assert mod._HUNYUAN3D_T2I_PIPELINE.call_args.args[0] == str(t2i_dir)


def test_legacy_local_entry_never_falls_back_to_hidden_t2i_download():
    _wire_runtime()
    runtime = mod.Hunyuan3DRuntime(
        model_id="m",
        hf_repo="tencent/Hunyuan3D-2",
        local_dir="/tmp/legacy-local-hunyuan",
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="re-pull"):
        runtime.text_to_3d("a chair")

    mod._HUNYUAN3D_T2I_PIPELINE.assert_not_called()


@pytest.mark.parametrize("subdir", ("../escape", "/absolute", ".", ""))
def test_managed_bundle_rejects_unsafe_subdirectory(tmp_path, subdir):
    _wire_runtime()
    with pytest.raises(RuntimeError, match="Hunyuan3D invalid"):
        mod.Hunyuan3DRuntime(
            model_id="m",
            hf_repo="tencent/Hunyuan3D-2",
            local_dir=str(tmp_path),
            shape_model_subdir=subdir,
            t2i_model_subdir="text_to_image",
            device="cpu",
        )


def test_managed_bundle_rejects_symlink_member(tmp_path):
    _wire_runtime()
    outside = tmp_path.parent / "outside-shape"
    outside.mkdir(exist_ok=True)
    (tmp_path / "shape").symlink_to(outside, target_is_directory=True)
    (tmp_path / "text_to_image").mkdir()

    with pytest.raises(RuntimeError, match="re-pull"):
        mod.Hunyuan3DRuntime(
            model_id="m",
            hf_repo="tencent/Hunyuan3D-2",
            local_dir=str(tmp_path),
            shape_model_subdir="shape",
            t2i_model_subdir="text_to_image",
            device="cpu",
        )


def test_constructor_does_not_pass_trust_remote_code_to_from_pretrained():
    """Hunyuan3D-2 uses a pip-installed SDK; from_pretrained must NOT receive
    trust_remote_code (real SDK has no such kwarg in from_pretrained)."""
    _wire_runtime()
    mod.Hunyuan3DRuntime(
        model_id="hunyuan3d-2",
        hf_repo="tencent/Hunyuan3D-2",
        device="cpu",
        trust_remote_code=True,
    )
    call_kwargs = mod._HUNYUAN3D_PIPELINE.from_pretrained.call_args.kwargs
    assert "trust_remote_code" not in call_kwargs
