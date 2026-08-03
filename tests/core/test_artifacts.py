"""Immutable multi-repository artifact bundle tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from muse.core.artifacts import (
    ArtifactBundleError,
    HFSnapshotArtifact,
    download_hf_artifact_bundle,
    local_artifact_directory,
    normalize_hf_artifacts,
)
from muse.core.catalog import _manifest_to_catalog_entry


ARTIFACTS = (
    {
        "repo_id": "org/adapter",
        "revision": "a" * 40,
        "subdir": "adapter",
        "required_patterns": ["weights.safetensors"],
    },
    {
        "repo_id": "org/base",
        "revision": "b" * 40,
        "subdir": "base",
        "allow_patterns": ["*.json", "*.safetensors"],
        "required_patterns": ["weights.safetensors"],
    },
)


def _materializing_downloader(calls: list[dict]):
    def download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True)
        (target / "weights.safetensors").write_bytes(b"weights")
        return str(target)

    return download


def test_download_hf_artifact_bundle_pins_and_publishes_atomically(tmp_path):
    calls: list[dict] = []

    bundle = download_hf_artifact_bundle(
        tmp_path,
        bundle_name="animation",
        artifacts=ARTIFACTS,
        snapshot_download_fn=_materializing_downloader(calls),
    )

    assert bundle.parent == tmp_path
    assert (bundle / "adapter" / "weights.safetensors").is_file()
    assert (bundle / "base" / "weights.safetensors").is_file()
    assert [call["revision"] for call in calls] == ["a" * 40, "b" * 40]
    assert calls[1]["allow_patterns"] == ["*.json", "*.safetensors"]
    assert not list(tmp_path.glob(".*.staging-*"))


def test_download_failure_removes_only_private_staging(tmp_path):
    def fail(**kwargs):
        Path(kwargs["local_dir"]).mkdir(parents=True)
        raise RuntimeError("fetch failed")

    with pytest.raises(RuntimeError, match="fetch failed"):
        download_hf_artifact_bundle(
            tmp_path,
            bundle_name="animation",
            artifacts=ARTIFACTS,
            snapshot_download_fn=fail,
        )

    assert list(tmp_path.iterdir()) == []


def test_existing_complete_bundle_is_reused_without_download(tmp_path):
    calls: list[dict] = []
    first = download_hf_artifact_bundle(
        tmp_path,
        bundle_name="animation",
        artifacts=ARTIFACTS,
        snapshot_download_fn=_materializing_downloader(calls),
    )

    second = download_hf_artifact_bundle(
        tmp_path,
        bundle_name="animation",
        artifacts=ARTIFACTS,
        snapshot_download_fn=lambda **_: pytest.fail("unexpected download"),
    )

    assert second == first


def test_empty_member_directories_are_not_published_or_reused(tmp_path):
    def empty_download(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True)
        # Hub local-dir bookkeeping is not model payload.
        metadata = target / ".cache" / "huggingface"
        metadata.mkdir(parents=True)
        (metadata / "download.json").write_text("{}")
        return str(target)

    with pytest.raises(ArtifactBundleError, match="no payload files"):
        download_hf_artifact_bundle(
            tmp_path,
            bundle_name="animation",
            artifacts=ARTIFACTS,
            snapshot_download_fn=empty_download,
        )

    assert list(tmp_path.iterdir()) == []


def test_metadata_file_cannot_satisfy_required_payload_topology(tmp_path):
    def partial_download(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True)
        (target / "README.md").write_text("partial")
        return str(target)

    with pytest.raises(ArtifactBundleError, match="required payload pattern"):
        download_hf_artifact_bundle(
            tmp_path,
            bundle_name="animation",
            artifacts=ARTIFACTS,
            snapshot_download_fn=partial_download,
        )

    assert list(tmp_path.iterdir()) == []


def test_existing_bundle_missing_required_payload_is_not_reused(tmp_path):
    calls: list[dict] = []
    bundle = download_hf_artifact_bundle(
        tmp_path,
        bundle_name="animation",
        artifacts=ARTIFACTS,
        snapshot_download_fn=_materializing_downloader(calls),
    )
    (bundle / "adapter" / "weights.safetensors").unlink()
    (bundle / "adapter" / "README.md").write_text("not enough")

    with pytest.raises(ArtifactBundleError, match="required payload pattern"):
        download_hf_artifact_bundle(
            tmp_path,
            bundle_name="animation",
            artifacts=ARTIFACTS,
            snapshot_download_fn=lambda **_: pytest.fail("unexpected download"),
        )


@pytest.mark.parametrize(
    "mutation,match",
    (
        ({"revision": "main"}, "40-character commit"),
        ({"subdir": "../escape"}, "safe directory"),
        ({"repo_id": "no-namespace"}, "repo_id"),
        ({"required_patterns": ["../weights"]}, "unsafe relative pattern"),
    ),
)
def test_normalize_hf_artifacts_rejects_unsafe_descriptor(mutation, match):
    artifacts = [dict(item) for item in ARTIFACTS]
    artifacts[0].update(mutation)
    with pytest.raises(ArtifactBundleError, match=match):
        normalize_hf_artifacts(artifacts)


def test_dataclass_artifact_patterns_are_validated_and_canonicalized():
    artifacts = (
        HFSnapshotArtifact(
            repo_id="org/adapter",
            revision="a" * 40,
            subdir="adapter",
            allow_patterns=["*.json"],
            required_patterns=["config.json"],
        ),
        HFSnapshotArtifact(
            repo_id="org/base",
            revision="b" * 40,
            subdir="base",
        ),
    )

    normalized = normalize_hf_artifacts(artifacts)
    assert normalized[0].allow_patterns == ("*.json",)
    assert normalized[0].required_patterns == ("config.json",)

    invalid = (
        HFSnapshotArtifact(
            repo_id="org/adapter",
            revision="a" * 40,
            subdir="adapter",
            allow_patterns="*.json",
        ),
        artifacts[1],
    )
    with pytest.raises(ArtifactBundleError, match="allow_patterns"):
        normalize_hf_artifacts(invalid)


def test_local_artifact_directory_rejects_symlink(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "adapter").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactBundleError, match="not a real directory"):
        local_artifact_directory(str(tmp_path), "adapter", label="adapter")


def test_manifest_projection_preserves_top_level_download_metadata():
    model_class = type("Model", (), {})
    model_class.__module__ = "example.model"
    manifest = {
        "model_id": "example",
        "modality": "image/animation",
        "hf_repo": "org/adapter",
        "revision": "a" * 40,
        "allow_patterns": ["*.safetensors"],
        "hf_artifacts": list(ARTIFACTS),
        "capabilities": {"device": "cuda"},
    }

    entry = _manifest_to_catalog_entry(SimpleNamespace(
        manifest=manifest,
        model_class=model_class,
    ))

    assert entry.extra["device"] == "cuda"
    assert entry.extra["revision"] == "a" * 40
    assert entry.extra["allow_patterns"] == ["*.safetensors"]
    assert entry.extra["hf_artifacts"] == list(ARTIFACTS)
    manifest["hf_artifacts"][0]["revision"] = "c" * 40
    assert entry.extra["hf_artifacts"][0]["revision"] == "a" * 40
