"""TRELLISRuntime: mocked-dep tests.

Sentinel restoration matches the sibling Shap-E / TripoSR pattern.
Mocks reflect the REAL TRELLIS SDK API (verified 2026-05-06 against
https://github.com/microsoft/TRELLIS, NOT inferred from the spec's
speculative shape).

Key API facts verified:
- Load: TrellisImageTo3DPipeline.from_pretrained(repo_or_path) from
  the `trellis` package (standalone library, not transformers/diffusers).
- Call: pipeline.run(image, seed=N, formats=["mesh"],
                     sparse_structure_sampler_params={"steps": N},
                     slat_sampler_params={"steps": N})
- Output: dict {"mesh": [MeshExtractResult, ...]}
- MeshExtractResult.vertices (NOT .verts), MeshExtractResult.faces
- Device: pipeline.cuda() for GPU; no .to() method guaranteed.

If the SDK changes upstream, only the mocks in _wire_runtime need
updating; the test structure stays.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import muse.modalities.model_3d_generation.runtimes.trellis as mod


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    """Reset module-top sentinels between tests."""
    orig = (mod.torch, mod._TRELLIS_PIPELINE, mod.trimesh, mod._LAST_IMPORT_ERROR)
    yield
    (mod.torch, mod._TRELLIS_PIPELINE, mod.trimesh, mod._LAST_IMPORT_ERROR) = orig


# ---- missing-deps errors ----


@pytest.mark.parametrize(
    "sentinel_name,match_str",
    [
        ("torch", "torch is not installed"),
        ("_TRELLIS_PIPELINE", "TRELLIS SDK"),
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
    mod._TRELLIS_PIPELINE = MagicMock()
    mod.trimesh = MagicMock()
    setattr(mod, sentinel_name, None)
    with pytest.raises(RuntimeError, match=match_str):
        mod.TRELLISRuntime(model_id="m", hf_repo="x", device="cpu")


# ---- helper: wire a fully-mocked runtime ----


def _wire_runtime():
    """Install fake torch + TrellisImageTo3DPipeline + trimesh.

    The pipeline mock reflects the REAL TRELLIS SDK API (verified 2026-05-06):
      - from_pretrained(repo_or_path) -> pipeline (no trust_remote_code kwarg)
      - pipeline.run(image, seed, formats, sparse_structure_sampler_params,
                     slat_sampler_params) -> {"mesh": [MeshExtractResult]}
      - MeshExtractResult.vertices and .faces (NOT .verts)
      - No guaranteed .to() method; .cuda() is the device-placement call
    """
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    mod.torch = fake_torch

    # Fake pipeline: run() method returning a dict with "mesh" key.
    pipeline = MagicMock()
    pipeline.cuda = MagicMock(return_value=pipeline)

    # MeshExtractResult uses .vertices (not .verts) and .faces.
    fake_mesh_data = MagicMock()
    fake_mesh_data.vertices = MagicMock()
    fake_mesh_data.vertices.cpu = MagicMock(return_value=fake_mesh_data.vertices)
    fake_mesh_data.vertices.numpy = MagicMock(return_value="vertices_array")
    fake_mesh_data.faces = MagicMock()
    fake_mesh_data.faces.cpu = MagicMock(return_value=fake_mesh_data.faces)
    fake_mesh_data.faces.numpy = MagicMock(return_value="faces_array")

    # Output dict from pipeline.run() - real API shape.
    pipeline.run = MagicMock(return_value={"mesh": [fake_mesh_data]})

    pipe_factory = MagicMock()
    pipe_factory.from_pretrained = MagicMock(return_value=pipeline)
    mod._TRELLIS_PIPELINE = pipe_factory

    # Fake trimesh.
    fake_mesh = MagicMock()
    fake_mesh.export = MagicMock(return_value=b"GLB_BYTES")
    fake_trimesh = MagicMock()
    fake_trimesh.Trimesh = MagicMock(return_value=fake_mesh)
    mod.trimesh = fake_trimesh

    return pipeline, fake_trimesh, fake_mesh


# ---- happy path ----


def test_image_to_3d_returns_glb_in_result():
    """image_to_3d returns a list with one Generation3DResult per sample."""
    pipeline, fake_trimesh, fake_mesh = _wire_runtime()
    runtime = mod.TRELLISRuntime(
        model_id="trellis-image", hf_repo="JeffreyXiang/TRELLIS-image-large",
        device="cpu",
    )
    image = MagicMock(name="pil_image")
    results = runtime.image_to_3d(image)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].model_id == "trellis-image"
    assert results[0].glb_bytes == b"GLB_BYTES"
    fake_trimesh.Trimesh.assert_called_once_with(
        vertices="vertices_array", faces="faces_array",
    )
    fake_mesh.export.assert_called_once_with(file_type="glb")


def test_image_to_3d_calls_pipeline_run_not_call(monkeypatch):
    """Regression guard: pipeline.run() is used, not pipeline()."""
    pipeline, _, _ = _wire_runtime()
    runtime = mod.TRELLISRuntime(model_id="m", hf_repo="x", device="cpu")
    runtime.image_to_3d(MagicMock())
    # pipeline.run must have been called; pipeline.__call__ must not.
    assert pipeline.run.call_count == 1
    assert pipeline.call_count == 0


def test_image_to_3d_uses_mesh_only_format():
    """pipeline.run() must request only 'mesh' to skip heavy texture baking."""
    pipeline, _, _ = _wire_runtime()
    runtime = mod.TRELLISRuntime(model_id="m", hf_repo="x", device="cpu")
    runtime.image_to_3d(MagicMock())
    call_kwargs = pipeline.run.call_args.kwargs
    assert call_kwargs.get("formats") == ["mesh"]


def test_image_to_3d_passes_sampler_params_as_nested_dicts():
    """Sampler kwargs are nested dicts, not flat kwargs (real TRELLIS API)."""
    pipeline, _, _ = _wire_runtime()
    runtime = mod.TRELLISRuntime(
        model_id="m", hf_repo="x", device="cpu",
        sparse_structure_steps=8, slat_steps=8,
    )
    runtime.image_to_3d(MagicMock())
    call_kwargs = pipeline.run.call_args.kwargs
    assert call_kwargs["sparse_structure_sampler_params"] == {"steps": 8}
    assert call_kwargs["slat_sampler_params"] == {"steps": 8}


def test_image_to_3d_allows_per_call_step_overrides():
    """Per-call sparse_structure_steps / slat_steps override constructor defaults."""
    pipeline, _, _ = _wire_runtime()
    runtime = mod.TRELLISRuntime(
        model_id="m", hf_repo="x", device="cpu",
        sparse_structure_steps=12, slat_steps=12,
    )
    runtime.image_to_3d(MagicMock(), sparse_structure_steps=5, slat_steps=5)
    call_kwargs = pipeline.run.call_args.kwargs
    assert call_kwargs["sparse_structure_sampler_params"] == {"steps": 5}
    assert call_kwargs["slat_sampler_params"] == {"steps": 5}


def test_image_to_3d_forwards_seed():
    """seed kwarg is forwarded to pipeline.run()."""
    pipeline, _, _ = _wire_runtime()
    runtime = mod.TRELLISRuntime(model_id="m", hf_repo="x", device="cpu")
    runtime.image_to_3d(MagicMock(), seed=99)
    call_kwargs = pipeline.run.call_args.kwargs
    assert call_kwargs.get("seed") == 99


def test_image_to_3d_loops_over_n():
    """For n>1, pipeline.run() is called n times and n results are returned."""
    pipeline, _, _ = _wire_runtime()
    runtime = mod.TRELLISRuntime(model_id="m", hf_repo="x", device="cpu")
    results = runtime.image_to_3d(MagicMock(), n=2)
    assert len(results) == 2
    assert pipeline.run.call_count == 2


def test_text_to_3d_raises_not_implemented():
    """text_to_3d raises NotImplementedError mentioning image-only."""
    _wire_runtime()
    runtime = mod.TRELLISRuntime(model_id="m", hf_repo="x", device="cpu")
    with pytest.raises(NotImplementedError, match="image-only"):
        runtime.text_to_3d("a chair")


def test_supports_capability_attrs():
    """Class-level capability attributes match the manifest declaration."""
    assert mod.TRELLISRuntime.supports_image_to_3d is True
    assert mod.TRELLISRuntime.supports_text_to_3d is False


def test_local_dir_preferred_over_hf_repo():
    """local_dir is passed to from_pretrained instead of hf_repo."""
    _wire_runtime()
    mod.TRELLISRuntime(
        model_id="m",
        hf_repo="JeffreyXiang/TRELLIS-image-large",
        local_dir="/tmp/local-trellis",
        device="cpu",
    )
    src_arg = mod._TRELLIS_PIPELINE.from_pretrained.call_args.args[0]
    assert src_arg == "/tmp/local-trellis"


def test_constructor_does_not_pass_trust_remote_code_to_from_pretrained():
    """TRELLIS uses a reviewed sparse checkout (not trust_remote_code);
    from_pretrained omits the kwarg because the real SDK has no such param."""
    _wire_runtime()
    mod.TRELLISRuntime(
        model_id="trellis-image",
        hf_repo="JeffreyXiang/TRELLIS-image-large",
        device="cpu",
        trust_remote_code=True,
    )
    call_kwargs = mod._TRELLIS_PIPELINE.from_pretrained.call_args.kwargs
    assert "trust_remote_code" not in call_kwargs
