"""Supply-chain invariants for bundled Hugging Face model artifacts."""
from __future__ import annotations

import re
from pathlib import Path

import muse.models
from muse.core.discovery import discover_models


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _bundled_manifests() -> dict[str, dict]:
    models_dir = Path(muse.models.__file__).parent
    source_paths = {
        path.resolve()
        for path in models_dir.glob("*.py")
        if not path.name.startswith("_")
    }
    discovered = discover_models([models_dir])
    discovered_paths = {
        item.source_path.resolve()
        for item in discovered.values()
    }

    assert discovered_paths == source_paths, (
        "bundled model discovery omitted source files: "
        f"{sorted(str(path) for path in source_paths - discovered_paths)}"
    )
    return {
        model_id: item.manifest
        for model_id, item in discovered.items()
    }


def test_bundled_hugging_face_artifacts_use_immutable_revisions():
    manifests = _bundled_manifests()
    assert manifests

    for model_id, manifest in manifests.items():
        repo_id = manifest.get("hf_repo")
        revision = manifest.get("revision")
        assert isinstance(repo_id, str) and repo_id, model_id
        assert isinstance(revision, str), model_id
        assert _COMMIT_RE.fullmatch(revision), model_id

        artifacts = manifest.get("hf_artifacts")
        if artifacts is None:
            continue

        assert isinstance(artifacts, (list, tuple)) and artifacts, model_id
        primary = artifacts[0]
        assert isinstance(primary, dict), f"{model_id}: primary artifact"
        assert primary.get("repo_id") == repo_id, model_id
        assert primary.get("revision") == revision, model_id

        for index, artifact in enumerate(artifacts):
            assert isinstance(artifact, dict), f"{model_id}: artifact {index}"
            artifact_repo = artifact.get("repo_id")
            artifact_revision = artifact.get("revision")
            assert isinstance(artifact_repo, str) and artifact_repo, (
                f"{model_id}: artifact {index}"
            )
            assert isinstance(artifact_revision, str), (
                f"{model_id}: artifact {index}"
            )
            assert _COMMIT_RE.fullmatch(artifact_revision), (
                f"{model_id}: artifact {index}"
            )
