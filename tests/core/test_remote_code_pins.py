"""Supply-chain invariants for code executed from Hugging Face repos."""
from __future__ import annotations

import re
from pathlib import Path

import muse.models
from muse.core.curated import load_curated
from muse.core.discovery import discover_models


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def test_bundled_remote_code_models_have_reviewed_commit_pins():
    models_dir = Path(muse.models.__file__).parent
    discovered = discover_models([models_dir])
    remote_code = {
        model_id: item.manifest
        for model_id, item in discovered.items()
        if (item.manifest.get("capabilities") or {}).get("trust_remote_code")
    }

    assert set(remote_code) == {"mert-v1-95m", "nv-embed-v2"}
    for model_id, manifest in remote_code.items():
        assert _COMMIT_RE.fullmatch(manifest.get("revision", "")), model_id


def test_curated_remote_code_models_have_reviewed_commit_pins():
    remote_code = {
        entry.id: entry
        for entry in load_curated()
        if entry.capabilities.get("trust_remote_code")
    }

    assert set(remote_code) == {
        "nomic-embed-text-v1.5",
        "jina-reranker-v2-base-multilingual",
    }
    for model_id, entry in remote_code.items():
        assert _COMMIT_RE.fullmatch(entry.revision or ""), model_id
        if entry.code_revision is not None:
            assert _COMMIT_RE.fullmatch(entry.code_revision), model_id


def test_sdk_backed_3d_entries_do_not_claim_transformers_remote_code():
    entries = {entry.id: entry for entry in load_curated()}
    for model_id in ("trellis-image", "hunyuan3d-2"):
        assert not entries[model_id].capabilities.get("trust_remote_code")
