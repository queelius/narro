"""Tests for the image_animation HF plugin (fused-checkpoint AnimateDiff variants)."""
from pathlib import Path
from unittest.mock import MagicMock

from muse.modalities.image_animation.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(repo_id="org/repo", siblings=None, tags=None, sha="a" * 40):
    info = MagicMock()
    info.id = repo_id
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    info.sha = sha
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "image/animation"
    assert HF_PLUGIN["runtime_path"].endswith(":AnimateDiffRuntime")
    # priority 110: tag + repo-name pattern, more specific than the generic
    # text-classification catch-all (200) but less specific than file-pattern
    # plugins (100).
    assert HF_PLUGIN["priority"] == 110


def test_sniff_true_on_animatediff_repo():
    """Repo with model_index.json + text-to-video tag + 'animate' in name."""
    info = _fake_info(
        repo_id="guoyww/animatediff-motion-adapter-v1-5-3",
        siblings=["model_index.json", "unet/diffusion_pytorch_model.safetensors"],
        tags=["text-to-video", "diffusers"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_on_motion_in_name():
    """Repo with model_index.json + text-to-video tag + 'motion' in name."""
    info = _fake_info(
        repo_id="someorg/motion-adapter-v3",
        siblings=["model_index.json"],
        tags=["text-to-video"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_without_text_to_video_tag():
    """Has model_index.json + animate in name but no text-to-video tag."""
    info = _fake_info(
        repo_id="someorg/animatediff-something",
        siblings=["model_index.json"],
        tags=["text-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_without_animate_or_motion_in_name():
    """Has model_index.json + text-to-video tag but generic repo name."""
    info = _fake_info(
        repo_id="someorg/generic-video",
        siblings=["model_index.json"],
        tags=["text-to-video"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_animatelcm_uses_lcm_defaults():
    """AnimateLCM repos get steps=4 guidance=1.0 + base_model + supports_text_to_animation."""
    info = _fake_info(
        repo_id="wangfuyun/AnimateLCM",
        siblings=["model_index.json"],
        tags=["text-to-video"],
    )
    result = HF_PLUGIN["resolve"]("wangfuyun/AnimateLCM", None, info)
    assert isinstance(result, ResolvedModel)
    caps = result.manifest["capabilities"]
    assert caps["default_steps"] == 4
    assert caps["default_guidance"] == 1.0
    assert "base_model" in caps
    assert caps["supports_text_to_animation"] is True
    assert caps["adapter_model_subdir"] == "motion_adapter"
    assert caps["base_model_subdir"] == "base_model"


def test_download_materializes_pinned_adapter_and_base(monkeypatch, tmp_path):
    from muse.modalities.image_animation import hf as module

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True)
        if target.name == "motion_adapter":
            files = ("model_index.json", "model.safetensors")
        else:
            files = (
                "model_index.json",
                "feature_extractor/preprocessor_config.json",
                "safety_checker/config.json",
                "safety_checker/model.safetensors",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "tokenizer/tokenizer_config.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
            )
        for relative in files:
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"payload")
        return str(target)

    monkeypatch.setattr(module, "snapshot_download", fake_download)
    resolved = module.HF_PLUGIN["resolve"](
        "wangfuyun/AnimateLCM",
        None,
        _fake_info(repo_id="wangfuyun/AnimateLCM", sha="c" * 40),
    )

    bundle = resolved.download(tmp_path)

    assert (bundle / "motion_adapter" / "model.safetensors").is_file()
    assert (bundle / "base_model" / "unet" / "config.json").is_file()
    assert [(call["repo_id"], call["revision"]) for call in calls] == [
        ("wangfuyun/AnimateLCM", "c" * 40),
        (
            "emilianJR/epiCRealism",
            "6522cf856b8c8e14638a0aaa7bd89b1b098aed17",
        ),
    ]
    assert "*.ckpt" not in calls[0]["allow_patterns"]
    assert "*.safetensors" in calls[0]["allow_patterns"]
    assert resolved.artifact_provenance[1]["revision"] == (
        "6522cf856b8c8e14638a0aaa7bd89b1b098aed17"
    )
    assert "unet/*.safetensors" in (
        resolved.artifact_provenance[1]["required_patterns"]
    )


def test_search_yields_results_with_modality_tag():
    """Search filters by text-to-video and yields image/animation rows."""
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/animatediff-thing", downloads=42)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "animate", sort="downloads", limit=20))
    assert len(rows) == 1
    assert rows[0].modality == "image/animation"
    # Confirm the search filter targets text-to-video
    fake_api.list_models.assert_called_once()
    call_kwargs = fake_api.list_models.call_args.kwargs
    assert call_kwargs["filter"] == "text-to-video"
