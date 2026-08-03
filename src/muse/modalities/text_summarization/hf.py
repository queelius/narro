"""HF resolver plugin for seq2seq summarizers.

Sniffs HF repos with the `summarization` task tag. The plugin only
covers BART/PEGASUS/T5-shape summarizers via AutoModelForSeq2SeqLM.
Chat-style summarizers (Qwen-summarize, Llama-summarize) belong in the
chat/completion modality.

Priority 110: matches embedding/text. The `summarization` tag is
specific enough that no other plugin should claim it; we don't need
extra specificity. Wins over text/classification (200, catch-all),
loses to file-pattern plugins at 100 (GGUF, faster-whisper, diffusers).

Loaded via single-file import; no relative imports. See
docs/HF_PLUGINS.md for authoring rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision


_RUNTIME_PATH = (
    "muse.modalities.text_summarization.runtimes.bart_seq2seq:BartSeq2SeqRuntime"
)
_PIP_EXTRAS = ("torch>=2.1.0", "transformers>=4.36.0")


# Repo-name substrings that suggest the repo was fine-tuned on dialog
# data (meeting/chat transcripts). Used for the
# supports_dialog_summarization capability hint.
_DIALOG_HINTS = ("samsum", "dialog", "chat", "meeting")


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    return "summarization" in tags


def _supports_dialog(repo_id: str) -> bool:
    """True if the repo name suggests dialog-tuned summarization."""
    name = repo_id.lower()
    return any(hint in name for hint in _DIALOG_HINTS)


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    revision = hf_commit_revision(info)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "text/summarization",
        "hf_repo": repo_id,
        "description": f"Seq2seq summarizer: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": {
            "device": "auto",
            "default_length": "medium",
            "default_format": "paragraph",
            "supports_dialog_summarization": _supports_dialog(repo_id),
            "memory_gb": 1.5,
            "max_input_tokens": 1024,
        },
    }
    if revision is not None:
        manifest["revision"] = revision

    def _download(cache_root: Path) -> Path:
        # Keep weights light: prefer safetensors, drop tf/flax/onnx.
        siblings = [s.rfilename for s in getattr(info, "siblings", [])]
        has_fp16 = any(".fp16." in f for f in siblings)
        if has_fp16:
            allow_patterns = ["*.fp16.safetensors", "*.json", "*.txt", "*.md"]
        else:
            allow_patterns = ["*.safetensors", "*.json", "*.txt", "*.md"]
        # Some summarizers ship pytorch_model.bin (no safetensors).
        allow_patterns.append("pytorch_model.bin")
        # BART tokenizer needs vocab.json + merges.txt; PEGASUS/T5 need
        # spiece.model. Pull them all defensively so the tokenizer load
        # never blows up on a missing vocab artifact.
        allow_patterns.extend([
            "tokenizer*", "spiece.model", "merges.txt", "vocab.json",
        ])
        return Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
            revision=revision,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    """Search HuggingFace for summarization-tagged repos.

    Filter: summarization task tag. Returns one row per matching repo.
    """
    repos = api.list_models(
        search=query, filter="summarization",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="text/summarization",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "text/summarization",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    # 110: matches embedding/text's tier. Wins over text/classification
    # (200, catch-all). Loses to file-pattern plugins at 100 (GGUF,
    # faster-whisper, diffusers).
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
