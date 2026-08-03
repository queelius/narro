"""Cross-plugin immutable Hugging Face revision contract tests.

All Hub functions are mocked: this module never downloads model data.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


_REVISION = "a" * 40


def _case(
    module: str,
    repo_id: str,
    *,
    variant: str | None = None,
    files: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    config: dict | None = None,
    case_id: str | None = None,
):
    return pytest.param(
        module,
        repo_id,
        variant,
        files,
        tags,
        config or {},
        id=case_id or module.rsplit(".", 2)[-2],
    )


_PLUGIN_CASES = (
    _case(
        "muse.modalities.audio_alignment.hf",
        "Qwen/Qwen3-ForcedAligner-0.6B-hf",
        files=(
            "config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    ),
    _case(
        "muse.modalities.audio_classification.hf",
        "MIT/ast-finetuned-audioset",
        tags=("audio-classification",),
    ),
    _case(
        "muse.modalities.audio_embedding.hf",
        "facebook/wav2vec2-base",
        tags=("feature-extraction",),
    ),
    _case(
        "muse.modalities.audio_generation.hf",
        "stabilityai/stable-audio-open-1.0",
        files=("model_index.json",),
        tags=("text-to-audio",),
    ),
    _case(
        "muse.modalities.audio_quality.hf",
        "Blinorot/UTMOS-PyTorch",
        files=("utmos_scripted.pt",),
    ),
    _case(
        "muse.modalities.audio_transcription.hf",
        "Systran/faster-whisper-tiny",
        files=("model.bin", "config.json", "tokenizer.json"),
        tags=("automatic-speech-recognition",),
    ),
    _case(
        "muse.modalities.chat_completion.hf",
        "org/Model-GGUF",
        variant="q4_k_m",
        files=("model-q4_k_m.gguf",),
        case_id="chat-gguf",
    ),
    _case(
        "muse.modalities.chat_completion.hf",
        "HuggingFaceTB/SmolVLM-Test",
        files=("config.json", "model.safetensors"),
        tags=("image-text-to-text",),
        config={"model_type": "idefics3"},
        case_id="chat-vlm",
    ),
    _case(
        "muse.modalities.embedding_text.hf",
        "sentence-transformers/all-MiniLM-L6-v2",
        tags=("sentence-transformers",),
    ),
    _case(
        "muse.modalities.image_animation.hf",
        "wangfuyun/AnimateLCM",
        files=("model_index.json",),
        tags=("text-to-video",),
    ),
    _case(
        "muse.modalities.image_cv.hf",
        "facebook/detr-resnet-50",
        tags=("object-detection",),
    ),
    _case(
        "muse.modalities.image_embedding.hf",
        "openai/clip-vit-base-patch32",
        tags=("zero-shot-image-classification",),
    ),
    _case(
        "muse.modalities.image_generation.hf",
        "stabilityai/sdxl-turbo",
        files=("model_index.json", "unet/model.safetensors"),
        tags=("text-to-image", "diffusers"),
        case_id="image-generation-pipeline",
    ),
    _case(
        "muse.modalities.image_generation.hf",
        "org/tagless-lora",
        files=("adapter_model.safetensors",),
        tags=("text-to-image", "lora"),
        case_id="image-generation-lora",
    ),
    _case(
        "muse.modalities.image_ocr.hf",
        "microsoft/trocr-base-printed",
        tags=("image-to-text",),
    ),
    _case(
        "muse.modalities.image_segmentation.hf",
        "facebook/sam2-hiera-tiny",
        tags=("mask-generation",),
    ),
    _case(
        "muse.modalities.image_upscale.hf",
        "stabilityai/stable-diffusion-x4-upscaler",
        files=("model_index.json",),
        tags=("image-to-image",),
    ),
    _case(
        "muse.modalities.model_3d_generation.hf",
        "stabilityai/TripoSR",
        tags=("image-to-3d",),
    ),
    _case(
        "muse.modalities.text_classification.hf",
        "KoalaAI/Text-Moderation",
        tags=("text-classification",),
    ),
    _case(
        "muse.modalities.text_rerank.hf",
        "BAAI/bge-reranker-v2-m3",
        tags=("cross-encoder",),
    ),
    _case(
        "muse.modalities.text_summarization.hf",
        "facebook/bart-large-cnn",
        tags=("summarization",),
    ),
    _case(
        "muse.modalities.text_translation.hf",
        "Helsinki-NLP/opus-mt-en-fr",
        tags=("translation",),
    ),
    _case(
        "muse.modalities.video_generation.hf",
        "Wan-AI/Wan2.1-T2V-1.3B",
        tags=("text-to-video",),
    ),
)


def test_revision_cases_cover_every_modality_hf_plugin():
    modalities_root = Path(__file__).resolve().parents[2] / "src" / "muse" / "modalities"
    discovered_modules = {
        f"muse.modalities.{path.parent.name}.hf"
        for path in modalities_root.glob("*/hf.py")
    }
    covered_modules = {case.values[0] for case in _PLUGIN_CASES}

    assert covered_modules == discovered_modules


@pytest.mark.parametrize(
    "module_name,repo_id,variant,files,tags,config",
    _PLUGIN_CASES,
)
def test_plugin_manifest_and_snapshot_use_same_immutable_revision(
    monkeypatch,
    tmp_path: Path,
    module_name: str,
    repo_id: str,
    variant: str | None,
    files: tuple[str, ...],
    tags: tuple[str, ...],
    config: dict,
):
    module = importlib.import_module(module_name)
    download_calls: list[dict] = []

    def _snapshot_download(**kwargs):
        download_calls.append(kwargs)
        local_dir = kwargs.get("local_dir")
        if local_dir is not None:
            target = Path(local_dir)
            target.mkdir(parents=True, exist_ok=True)
            required_patterns: tuple[str, ...] = ()
            if module_name.endswith("image_animation.hf"):
                required_patterns = (
                    module._ADAPTER_REQUIRED_PATTERNS
                    if kwargs.get("repo_id") == repo_id
                    else module._BASE_REQUIRED_PATTERNS
                )
            for pattern in required_patterns or ("model.safetensors",):
                payload = target / pattern.replace("*", "model")
                payload.parent.mkdir(parents=True, exist_ok=True)
                payload.write_bytes(b"payload")
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(module, "snapshot_download", _snapshot_download)
    if module_name.endswith("chat_completion.hf"):
        # The metadata helpers have their own pinned hf_hub_download test below.
        monkeypatch.setattr(module, "_try_sniff_tools_from_repo", lambda *a, **k: False)
        monkeypatch.setattr(
            module,
            "_try_sniff_context_length_from_repo",
            lambda *a, **k: None,
        )

    info = SimpleNamespace(
        id=repo_id,
        sha=_REVISION,
        siblings=[SimpleNamespace(rfilename=name) for name in files],
        tags=list(tags),
        config=dict(config),
        card_data=SimpleNamespace(license=None),
    )
    resolved = module.HF_PLUGIN["resolve"](repo_id, variant, info)
    resolved.download(tmp_path)

    assert resolved.manifest["revision"] == _REVISION
    primary_calls = [
        call for call in download_calls if call.get("repo_id") == repo_id
    ]
    assert len(primary_calls) == 1
    assert primary_calls[0]["revision"] == _REVISION
    assert all(
        isinstance(call.get("revision"), str)
        and len(call["revision"]) == 40
        for call in download_calls
    )


def test_chat_metadata_downloads_use_resolved_commit(monkeypatch, tmp_path: Path):
    module = importlib.import_module("muse.modalities.chat_completion.hf")
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text('{"chat_template": "{% if tools %}x{% endif %}"}')
    model_config = tmp_path / "config.json"
    model_config.write_text('{"max_position_embeddings": 8192}')
    calls: list[dict] = []

    def _hf_hub_download(**kwargs):
        calls.append(kwargs)
        if kwargs["filename"] == "tokenizer_config.json":
            return str(tokenizer_config)
        return str(model_config)

    monkeypatch.setattr(module, "hf_hub_download", _hf_hub_download)

    assert module._try_sniff_tools_from_repo(
        "org/repo", revision=_REVISION,
    ) is True
    assert module._try_sniff_context_length_from_repo(
        "org/repo", revision=_REVISION,
    ) == 8192
    assert [call["revision"] for call in calls] == [_REVISION, _REVISION]


def test_curated_revision_reaches_metadata_manifest_and_snapshot(
    monkeypatch,
    tmp_path: Path,
):
    from unittest.mock import MagicMock

    from muse.core.resolvers_hf import HFResolver

    module = importlib.import_module("muse.modalities.audio_transcription.hf")
    info = SimpleNamespace(
        id="Systran/faster-whisper-tiny",
        sha=_REVISION,
        siblings=[
            SimpleNamespace(rfilename=name)
            for name in ("model.bin", "config.json", "tokenizer.json")
        ],
        tags=["automatic-speech-recognition"],
        card_data=SimpleNamespace(license="mit"),
    )
    resolver = HFResolver(plugins=[module.HF_PLUGIN])
    repo_info = MagicMock(return_value=info)
    monkeypatch.setattr(resolver._api, "repo_info", repo_info)
    download_calls: list[dict] = []
    monkeypatch.setattr(
        module,
        "snapshot_download",
        lambda **kwargs: download_calls.append(kwargs) or str(tmp_path / "snapshot"),
    )

    resolved = resolver.resolve_via_modality(
        "hf://Systran/faster-whisper-tiny",
        "audio/transcription",
        revision=_REVISION,
    )
    resolved.download(tmp_path)

    repo_info.assert_called_once_with(
        "Systran/faster-whisper-tiny",
        revision=_REVISION,
    )
    assert resolved.manifest["revision"] == _REVISION
    assert download_calls[0]["revision"] == _REVISION
