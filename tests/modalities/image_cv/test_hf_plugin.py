"""Tests for the image_cv HF plugin (3-way dispatch)."""
from unittest.mock import MagicMock

import pytest

from muse.modalities.image_cv.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel, ResolverError


def _fake_info(siblings=None, tags=None, repo_id="org/repo", config=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    info.id = repo_id
    info.config = config or {}
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "image/cv"
    assert HF_PLUGIN["priority"] == 110


# ---------- Sniff ----------


def test_sniff_true_on_depth_estimation_tag():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["depth-estimation"])) is True


def test_sniff_true_on_keypoint_detection_tag():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["keypoint-detection"])) is True


def test_sniff_true_on_object_detection_tag():
    assert HF_PLUGIN["sniff"](_fake_info(tags=["object-detection"])) is True


def test_sniff_true_on_depth_repo_name_fallback():
    """A repo without depth-estimation tag but 'depth' in the name."""
    info = _fake_info(tags=[], repo_id="custom-org/custom-depth-model")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_keypoint_repo_name_fallback():
    info = _fake_info(tags=[], repo_id="custom-org/vitpose-finetune")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_vitpose_config_without_name_or_tag():
    info = _fake_info(
        tags=[], repo_id="custom-org/custom-pose-model",
        config={"model_type": "vitpose"},
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_vitpose_architecture_without_name_or_tag():
    info = _fake_info(
        tags=[], repo_id="custom-org/custom-pose-model",
        config={"architectures": ["VitPoseForPoseEstimation"]},
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_rejects_explicitly_unsupported_keypoint_family():
    info = _fake_info(
        tags=["keypoint-detection"], repo_id="org/rtmpose-model",
        config={"model_type": "rtmpose"},
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_rejects_unsupported_keypoint_name_without_tag():
    info = _fake_info(tags=[], repo_id="org/rtmpose-model")
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_tolerates_malformed_architecture_metadata():
    info = _fake_info(
        tags=["keypoint-detection"], config={"architectures": 123},
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_detection_repo_name_fallback():
    info = _fake_info(tags=[], repo_id="custom-org/detr-finetune")
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_on_random_repo():
    info = _fake_info(tags=["text-generation"], repo_id="org/llm")
    assert HF_PLUGIN["sniff"](info) is False


# ---------- Resolve: dispatch by primitive ----------


def test_resolve_depth_dispatches_to_depth_runtime():
    info = _fake_info(
        tags=["depth-estimation"],
        repo_id="depth-anything/Depth-Anything-V2-Small-hf",
    )
    result = HF_PLUGIN["resolve"](
        "depth-anything/Depth-Anything-V2-Small-hf", None, info,
    )
    assert isinstance(result, ResolvedModel)
    assert "HFDepthRuntime" in result.backend_path
    caps = result.manifest["capabilities"]
    assert caps["supports_depth"] is True
    assert caps["supports_keypoints"] is False
    assert caps["supports_detection"] is False
    assert caps["metric_depth"] is False
    assert "Depth" in result.manifest["description"]


def test_resolve_keypoint_dispatches_to_keypoint_runtime():
    info = _fake_info(
        tags=["keypoint-detection"],
        repo_id="usyd-community/vitpose-base-simple",
    )
    result = HF_PLUGIN["resolve"](
        "usyd-community/vitpose-base-simple", None, info,
    )
    assert "HFKeypointRuntime" in result.backend_path
    caps = result.manifest["capabilities"]
    assert caps["supports_keypoints"] is True
    assert caps["supports_depth"] is False
    assert caps["supports_detection"] is False
    assert "torch>=2.1.0" in result.manifest["pip_extras"]
    assert any(
        extra.startswith("torchvision")
        for extra in result.manifest["pip_extras"]
    )
    assert "transformers>=4.48.0" in result.manifest["pip_extras"]
    assert "scipy" in result.manifest["pip_extras"]


def test_resolve_generic_keypoint_keeps_generic_dependencies():
    info = _fake_info(
        tags=["keypoint-detection"],
        repo_id="magic-leap-community/superpoint",
        config={"model_type": "superpoint"},
    )
    result = HF_PLUGIN["resolve"](
        "magic-leap-community/superpoint", None, info,
    )

    assert "HFKeypointRuntime" in result.backend_path
    assert "transformers>=4.46.0" in result.manifest["pip_extras"]
    assert "scipy" not in result.manifest["pip_extras"]


def test_forced_resolve_rejects_unsupported_keypoint_family():
    info = _fake_info(
        tags=["keypoint-detection"], repo_id="org/rtmpose-model",
        config={"model_type": "rtmpose"},
    )

    with pytest.raises(ResolverError, match="unsupported image/cv"):
        HF_PLUGIN["resolve"]("org/rtmpose-model", None, info)


def test_resolve_opaque_keypoint_uses_safe_dependency_superset():
    info = _fake_info(tags=["keypoint-detection"], repo_id="org/opaque-model")
    result = HF_PLUGIN["resolve"]("org/opaque-model", None, info)

    assert "transformers>=4.48.0" in result.manifest["pip_extras"]
    assert "scipy" in result.manifest["pip_extras"]


def test_resolve_object_detection_dispatches_to_detection_runtime():
    info = _fake_info(
        tags=["object-detection"],
        repo_id="facebook/detr-resnet-50",
    )
    result = HF_PLUGIN["resolve"](
        "facebook/detr-resnet-50", None, info,
    )
    assert "HFObjectDetectionRuntime" in result.backend_path
    caps = result.manifest["capabilities"]
    assert caps["supports_detection"] is True
    assert caps["supports_depth"] is False
    assert caps["supports_keypoints"] is False


def test_resolve_zoedepth_sets_metric_depth_true():
    info = _fake_info(
        tags=["depth-estimation"],
        repo_id="Intel/zoedepth-nyu-kitti",
    )
    result = HF_PLUGIN["resolve"](
        "Intel/zoedepth-nyu-kitti", None, info,
    )
    assert result.manifest["capabilities"]["metric_depth"] is True


def test_resolve_metric_substring_sets_metric_depth_true():
    info = _fake_info(
        tags=["depth-estimation"],
        repo_id="custom-org/depth-metric-fine",
    )
    result = HF_PLUGIN["resolve"](
        "custom-org/depth-metric-fine", None, info,
    )
    assert result.manifest["capabilities"]["metric_depth"] is True


def test_resolve_relative_depth_metric_false():
    info = _fake_info(
        tags=["depth-estimation"],
        repo_id="depth-anything/Depth-Anything-V2-Base-hf",
    )
    result = HF_PLUGIN["resolve"](
        "depth-anything/Depth-Anything-V2-Base-hf", None, info,
    )
    assert result.manifest["capabilities"]["metric_depth"] is False


def test_resolve_detection_includes_timm_in_pip_extras():
    """DETR's ResNet backbone requires timm; the resolver-pulled
    manifest must list it so a fresh-venv pull installs it."""
    info = _fake_info(
        tags=["object-detection"],
        repo_id="facebook/detr-resnet-50",
    )
    result = HF_PLUGIN["resolve"](
        "facebook/detr-resnet-50", None, info,
    )
    assert "timm" in result.manifest["pip_extras"]


def test_resolve_depth_omits_timm_in_pip_extras():
    """Depth runtimes don't need timm; keep pip_extras lean."""
    info = _fake_info(
        tags=["depth-estimation"],
        repo_id="x/depth-model",
    )
    result = HF_PLUGIN["resolve"]("x/depth-model", None, info)
    assert "timm" not in result.manifest["pip_extras"]


def test_resolve_model_id_kebab_case():
    info = _fake_info(
        tags=["depth-estimation"], repo_id="Intel/DPT-Large",
    )
    result = HF_PLUGIN["resolve"]("Intel/DPT-Large", None, info)
    assert result.manifest["model_id"] == "dpt-large"


# ---------- Search ----------


def test_search_yields_results():
    fake_api = MagicMock()
    # Three queries (one per task tag); each returns one fake repo.
    fake_api.list_models.side_effect = [
        [MagicMock(id="org/depth-x", downloads=10)],
        [MagicMock(id="org/vitpose-x", downloads=20)],
        [MagicMock(id="org/detr-x", downloads=30)],
    ]
    rows = list(HF_PLUGIN["search"](
        fake_api, "x", sort="downloads", limit=30,
    ))
    assert len(rows) == 3
    assert {r.model_id for r in rows} == {
        "depth-x", "vitpose-x", "detr-x",
    }
    assert all(r.modality == "image/cv" for r in rows)
    assert [call.kwargs["filter"] for call in fake_api.list_models.call_args_list] == [
        "depth-estimation", "keypoint-detection", "object-detection",
    ]


def test_search_omits_unsupported_keypoint_family():
    fake_api = MagicMock()
    fake_api.list_models.side_effect = [
        [],
        [MagicMock(id="org/rtmpose-x", downloads=20)],
        [],
    ]

    rows = list(HF_PLUGIN["search"](
        fake_api, "x", sort="downloads", limit=30,
    ))

    assert rows == []


def test_search_dedupes_overlapping_repos():
    """If a repo appears under multiple task filters (rare but
    possible), we only emit it once."""
    fake_api = MagicMock()
    overlap = MagicMock(id="org/some-model", downloads=10)
    fake_api.list_models.side_effect = [
        [overlap],
        [overlap],
        [overlap],
    ]
    rows = list(HF_PLUGIN["search"](
        fake_api, "x", sort="downloads", limit=30,
    ))
    assert len(rows) == 1
