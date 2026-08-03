"""HF plugin tests for 3d/generation."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel
from muse.modalities.model_3d_generation.hf import HF_PLUGIN


def _fake_info(siblings=None, tags=None, repo_id="org/repo", sha="a" * 40):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    info.id = repo_id
    info.sha = sha
    return info


def test_required_keys():
    for k in REQUIRED_HF_PLUGIN_KEYS:
        assert k in HF_PLUGIN


def test_metadata():
    assert HF_PLUGIN["modality"] == "3d/generation"
    assert HF_PLUGIN["priority"] == 110
    assert HF_PLUGIN["runtime_path"].endswith(":TripoSRRuntime")


def test_pip_extras_includes_core_deps():
    """torch / tsr / trimesh / Pillow are required for any 3d/generation
    runtime. Missing one is a load-time ImportError on first inference."""
    extras = HF_PLUGIN["pip_extras"]
    assert any("torch" in e for e in extras)
    assert any(e == "tsr" or e.startswith("tsr ") or e.startswith("tsr=") for e in extras)
    assert any("trimesh" in e for e in extras)
    assert any("Pillow" in e for e in extras)


def test_sniff_image_to_3d_tag():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["image-to-3d"])) is True


def test_sniff_text_to_3d_tag():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["text-to-3d"])) is True


def test_sniff_repo_name_triposr():
    info = _fake_info(tags=[], repo_id="stabilityai/TripoSR")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_repo_name_trellis():
    info = _fake_info(tags=[], repo_id="JeffreyXiang/TRELLIS-image-large")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_repo_name_hunyuan3d():
    info = _fake_info(tags=[], repo_id="tencent/Hunyuan3D-2")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_repo_name_wonder3d():
    info = _fake_info(tags=[], repo_id="flamehaze1115/Wonder3D-v1.0")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_repo_name_shap_e():
    info = _fake_info(tags=[], repo_id="openai/shap-e")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_repo_name_instantmesh():
    info = _fake_info(tags=[], repo_id="TencentARC/InstantMesh")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_unrelated_repo():
    info = _fake_info(tags=["text-generation"], repo_id="org/llm")
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_classifier_repo():
    info = _fake_info(tags=["audio-classification"], repo_id="org/some-classifier")
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_returns_runtime_path():
    """All v0.41.0 matches route through TripoSRRuntime; future curated
    entries get dedicated runtime_path overrides via curated.yaml."""
    info = _fake_info(tags=["image-to-3d"], repo_id="stabilityai/TripoSR")
    result = HF_PLUGIN["resolve"]("stabilityai/TripoSR", None, info)
    assert isinstance(result, ResolvedModel)
    assert result.backend_path.endswith(":TripoSRRuntime")
    assert result.manifest["modality"] == "3d/generation"


def test_resolve_kebab_case_model_id():
    info = _fake_info(tags=["image-to-3d"], repo_id="stabilityai/TripoSR")
    result = HF_PLUGIN["resolve"]("stabilityai/TripoSR", None, info)
    assert result.manifest["model_id"] == "triposr"


def test_resolve_supports_image_to_3d_for_non_shape_e():
    """Non-Shap-E repos declare supports_image_to_3d=True (the dominant
    direction). Shap-E base is text-only and is tested separately."""
    for repo_id in (
        "stabilityai/TripoSR",
        "JeffreyXiang/TRELLIS-image-large",
        "tencent/Hunyuan3D-2",
        "flamehaze1115/Wonder3D-v1.0",
    ):
        info = _fake_info(tags=["image-to-3d"], repo_id=repo_id)
        result = HF_PLUGIN["resolve"](repo_id, None, info)
        caps = result.manifest["capabilities"]
        assert caps["supports_image_to_3d"] is True, repo_id


def test_resolve_shap_e_is_text_only():
    """Shap-E base is text-to-3D only: supports_image_to_3d=False,
    supports_text_to_3d=True."""
    info = _fake_info(tags=["text-to-3d"], repo_id="openai/shap-e")
    result = HF_PLUGIN["resolve"]("openai/shap-e", None, info)
    caps = result.manifest["capabilities"]
    assert caps["supports_image_to_3d"] is False
    assert caps["supports_text_to_3d"] is True


def test_resolve_supports_text_to_3d_for_trellis():
    """TRELLIS-image-large is image-only: the v0.44.0 family registry entry
    sets supports_text_to_3d=False via capability_overrides (fixing the
    capability lie present in prior releases)."""
    info = _fake_info(tags=["image-to-3d"], repo_id="JeffreyXiang/TRELLIS-image-large")
    result = HF_PLUGIN["resolve"]("JeffreyXiang/TRELLIS-image-large", None, info)
    assert result.manifest["capabilities"]["supports_text_to_3d"] is False
    assert result.manifest["capabilities"]["supports_image_to_3d"] is True


def test_resolve_supports_text_to_3d_for_hunyuan3d():
    info = _fake_info(tags=["image-to-3d"], repo_id="tencent/Hunyuan3D-2")
    result = HF_PLUGIN["resolve"]("tencent/Hunyuan3D-2", None, info)
    assert result.manifest["capabilities"]["supports_text_to_3d"] is True


def test_hunyuan_manifest_pins_both_bundle_members():
    info = _fake_info(repo_id="tencent/Hunyuan3D-2", sha="b" * 40)
    result = HF_PLUGIN["resolve"]("tencent/Hunyuan3D-2", None, info)

    assert result.manifest["revision"] == "b" * 40
    caps = result.manifest["capabilities"]
    assert caps["shape_model_subdir"] == "shape"
    assert caps["t2i_model_subdir"] == "text_to_image"
    assert caps["t2i_revision"] == "527cf2ecce7c04021975938f8b0e44e35d2b1ed9"
    assert result.artifact_provenance == (
        {
            "repo_id": "tencent/Hunyuan3D-2",
            "revision": "b" * 40,
            "subdir": "shape",
        },
        {
            "repo_id": "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
            "revision": "527cf2ecce7c04021975938f8b0e44e35d2b1ed9",
            "subdir": "text_to_image",
        },
    )


def test_hunyuan_download_builds_two_pinned_snapshot_bundle(monkeypatch, tmp_path):
    from muse.modalities.model_3d_generation import hf as module

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        (local_dir / "weights.safetensors").write_bytes(b"weights")
        return str(local_dir)

    monkeypatch.setattr(module, "snapshot_download", fake_snapshot_download)
    info = _fake_info(repo_id="tencent/Hunyuan3D-2", sha="c" * 40)
    result = module.HF_PLUGIN["resolve"]("tencent/Hunyuan3D-2", None, info)

    bundle = result.download(tmp_path)

    assert bundle.parent == tmp_path
    assert (bundle / "shape" / "weights.safetensors").is_file()
    assert (bundle / "text_to_image" / "weights.safetensors").is_file()
    assert [(call["repo_id"], call["revision"]) for call in calls] == [
        ("tencent/Hunyuan3D-2", "c" * 40),
        (
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
            "527cf2ecce7c04021975938f8b0e44e35d2b1ed9",
        ),
    ]
    assert not list(tmp_path.glob(".*.staging-*"))


def test_hunyuan_download_failure_removes_private_staging(monkeypatch, tmp_path):
    from muse.modalities.model_3d_generation import hf as module

    def fail_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        raise RuntimeError("download failed")

    monkeypatch.setattr(module, "snapshot_download", fail_snapshot_download)
    result = module.HF_PLUGIN["resolve"](
        "tencent/Hunyuan3D-2",
        None,
        _fake_info(repo_id="tencent/Hunyuan3D-2", sha="d" * 40),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        result.download(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_download_refuses_missing_immutable_revision(tmp_path):
    result = HF_PLUGIN["resolve"](
        "stabilityai/TripoSR",
        None,
        _fake_info(repo_id="stabilityai/TripoSR", sha=None),
    )

    with pytest.raises(RuntimeError, match="refusing mutable"):
        result.download(tmp_path)


def test_installed_sdk_families_do_not_claim_transformers_remote_code():
    for repo_id in (
        "JeffreyXiang/TRELLIS-image-large",
        "tencent/Hunyuan3D-2",
    ):
        result = HF_PLUGIN["resolve"](
            repo_id,
            None,
            _fake_info(tags=["image-to-3d"], repo_id=repo_id),
        )
        assert "trust_remote_code" not in result.manifest["capabilities"]


def test_resolve_supports_text_to_3d_for_shap_e():
    info = _fake_info(tags=["text-to-3d"], repo_id="openai/shap-e")
    result = HF_PLUGIN["resolve"]("openai/shap-e", None, info)
    assert result.manifest["capabilities"]["supports_text_to_3d"] is True


def test_resolve_supports_text_to_3d_false_for_triposr():
    info = _fake_info(tags=["image-to-3d"], repo_id="stabilityai/TripoSR")
    result = HF_PLUGIN["resolve"]("stabilityai/TripoSR", None, info)
    assert result.manifest["capabilities"]["supports_text_to_3d"] is False


def test_resolve_supports_text_to_3d_false_for_wonder3d():
    info = _fake_info(tags=["image-to-3d"], repo_id="flamehaze1115/Wonder3D-v1.0")
    result = HF_PLUGIN["resolve"]("flamehaze1115/Wonder3D-v1.0", None, info)
    assert result.manifest["capabilities"]["supports_text_to_3d"] is False


def test_resolve_capabilities_include_device_and_format():
    info = _fake_info(tags=["image-to-3d"], repo_id="stabilityai/TripoSR")
    result = HF_PLUGIN["resolve"]("stabilityai/TripoSR", None, info)
    caps = result.manifest["capabilities"]
    assert caps["device"] == "cuda"
    assert caps["output_format"] == "glb"


def test_resolve_pip_extras_propagated():
    info = _fake_info(tags=["image-to-3d"], repo_id="stabilityai/TripoSR")
    result = HF_PLUGIN["resolve"]("stabilityai/TripoSR", None, info)
    extras = result.manifest["pip_extras"]
    assert any("torch" in e for e in extras)
    assert any(e == "tsr" for e in extras)


def test_search_yields_results():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="stabilityai/TripoSR", downloads=42)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](
        fake_api, "triposr", sort="downloads", limit=20,
    ))
    # search iterates both image-to-3d and text-to-3d tags but dedupes by repo id.
    assert len(rows) == 1
    assert rows[0].uri == "hf://stabilityai/TripoSR"
    assert rows[0].modality == "3d/generation"
    assert rows[0].downloads == 42


def test_search_dedupes_repos_across_tags():
    """A repo tagged both image-to-3d and text-to-3d (TRELLIS, Hunyuan3D)
    must be yielded once, not once per tag."""
    fake_api = MagicMock()
    fake_repo = MagicMock(id="JeffreyXiang/TRELLIS-image-large", downloads=10)
    # Both tag iterations return the same repo.
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](
        fake_api, "trellis", sort="downloads", limit=20,
    ))
    assert len(rows) == 1
