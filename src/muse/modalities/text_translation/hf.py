"""HF resolver plugin for text/translation.

Sniffs the `translation` task tag, plus a repo-name fallback for the
four families TranslationRuntime supports: m2m100, nllb, opus-mt,
madlad. The name fallback exists because HF repos in this space are
inconsistently tagged -- some carry `text2text-generation` instead of
(or in addition to) `translation`, and the name pattern is a reliable
tell regardless of tagging.

Disambiguation vs text/summarization's plugin (spec-normative, see
docs/superpowers/specs/2026-07-09-text-translation-design.md):

  (a) tagged BOTH `summarization` and `translation` -> claimed here iff
      the repo name matches one of the four translation families.
  (b) tagged `summarization` WITHOUT `translation` -> NEVER claimed
      here, even on a family-name match. An "nllb-meeting-summarizer"
      fine-tune is a summarizer, not a translator; a family-name hit
      alone cannot override an explicit, unambiguous summarization tag
      with no competing translation tag.
  (c) otherwise -> claimed iff the `translation` tag is present or the
      name matches a translation family.

AS-BUILT NOTE (priority 110 -> 109): text/summarization's plugin also
sat at priority 110 when this plugin first shipped. HFResolver.resolve()
iterates plugins in (priority, modality) order (lower priority checked
first; see docs/HF_PLUGINS.md), first sniff wins; the tie-break at equal
priority is alphabetical by modality string, and
"text/summarization" < "text/translation" alphabetically. That meant
summarization's plugin was *always* consulted before this one in real
dispatch -- its own sniff (`"summarization" in tags`) claims ANY
summarization-tagged repo unconditionally, with no chance for this
plugin's disambiguation rule (a)/(b)/(c) above to run at all on a
both-tagged repo. The rule was implemented and unit-tested in isolation,
but unreachable through the real resolver.

The fix adjudicated for this task: move this plugin to priority 109, one
tier below the 110 group (audio_classification, embedding_text,
image_animation, image_cv, image_ocr, image_segmentation,
model_3d_generation, text_summarization). Being checked first means
translation's own sniff runs before ANY of those, decides per rules
(a)/(b)/(c) above, and defers (returns False) for every repo it doesn't
recognize as a translation family -- including plain summarization-only
repos and repos belonging to the other seven 110-tier modalities (see
the collateral-shadowing regression tests in
tests/modalities/text_translation/test_hf_plugin.py). Net effect: the
both-tagged disambiguation is now genuinely reachable, and nothing else
in the 110 tier loses any repo it previously claimed.

Loaded via single-file import; no relative imports. See
docs/HF_PLUGINS.md for authoring rules.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult, hf_commit_revision


_RUNTIME_PATH = (
    "muse.modalities.text_translation.runtimes.hf_translation"
    ":TranslationRuntime"
)
_PIP_EXTRAS = ("torch>=2.1.0", "transformers>=4.36.0", "sentencepiece")

# Case-insensitive substring markers identifying the four families
# TranslationRuntime dispatches on. Mirrors
# muse.modalities.text_translation.runtimes.hf_translation._FAMILY_MARKERS
# (not imported -- hf.py may not import modality siblings; see
# docs/HF_PLUGINS.md authoring rules).
_FAMILY_PATTERNS = ("m2m100", "nllb", "opus-mt", "madlad")

# opus-mt-{src}-{tgt} where src/tgt are exactly two lowercase letters
# (plain ISO-639-1 codes). Anything else -- a language-group name like
# "ROMANCE", a 3-letter macro-language code, or a multi-part variant
# name like "opus-mt-tc-big-en-pt" -- does not match, and pair
# extraction is skipped rather than guessed.
_OPUS_PAIR_RE = re.compile(r"^opus-mt-([a-z]{2})-([a-z]{2})$")


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _matches_family(repo_id: str) -> bool:
    # Scoped to the stripped model-name segment (after the org's
    # trailing slash), matching _opus_pair's existing discipline. An
    # org merely NAMED after a translation family (e.g. a hobbyist org
    # "nllb-fan-org") hosting an unrelated model must not match just
    # because the family string appears in the ORG segment.
    name = _model_id(repo_id)
    return any(pattern in name for pattern in _FAMILY_PATTERNS)


def _opus_pair(repo_id: str) -> tuple[str, str] | None:
    """Parse a `src`/`tgt` ISO-639-1 pair from an opus-mt-shaped model
    name. Returns None (no guessing) when the name doesn't match the
    strict two-letter-pair shape.
    """
    name = _model_id(repo_id)
    m = _OPUS_PAIR_RE.fullmatch(name)
    if not m:
        return None
    return m.group(1), m.group(2)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    repo_id = getattr(info, "id", "") or ""
    family_match = _matches_family(repo_id)
    has_summarization = "summarization" in tags
    has_translation = "translation" in tags

    if has_summarization and has_translation:
        # (a) Both-tagged: only claim here when the name unambiguously
        # identifies a translation family; otherwise defer to
        # text/summarization's plugin.
        return family_match

    if has_summarization:
        # (b) Summarization-tagged WITHOUT a translation tag: never
        # claim here, even on a family-name match. An
        # "nllb-meeting-summarizer" fine-tune is a summarizer, not a
        # translator -- an explicit, unambiguous summarization tag with
        # no competing translation tag always wins.
        return False

    # (c) Otherwise: claim iff the translation tag is present or the
    # name matches a translation family.
    return has_translation or family_match


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    capabilities: dict = {"device": "auto"}
    pair = _opus_pair(repo_id)
    if pair is not None:
        capabilities["source_language"], capabilities["target_language"] = pair

    revision = hf_commit_revision(info)
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "text/translation",
        "hf_repo": repo_id,
        "description": f"Translation model: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
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
        # Older seq2seq translation repos still ship pytorch_model.bin
        # (no safetensors).
        allow_patterns.append("pytorch_model.bin")
        # Tokenizer artifacts vary by family: m2m100/nllb use a shared
        # sentencepiece.bpe.model, opus-mt ships per-direction
        # source.spm/target.spm + vocab.json, madlad (T5-shape) uses
        # spiece.model. Pull them all defensively.
        allow_patterns.extend([
            "tokenizer*", "spiece.model", "sentencepiece.bpe.model",
            "source.spm", "target.spm", "vocab.json", "merges.txt",
            "special_tokens_map.json", "added_tokens.json",
            "generation_config.json",
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
    """Search HuggingFace for translation-tagged repos.

    Filter: translation task tag. Returns one row per matching repo.
    Mirrors text/summarization's search shape.
    """
    repos = api.list_models(
        search=query, filter="translation",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="text/translation",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "text/translation",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    # 109: tag-based, more specific than text-classification's catch-all
    # (200) but loses to file-pattern plugins (100). Deliberately ONE
    # BELOW the 110 tier (text/summarization, audio_classification,
    # embedding_text, image_animation, image_cv, image_ocr,
    # image_segmentation, model_3d_generation) so this plugin's sniff is
    # checked FIRST and its both-tagged disambiguation rule (see the
    # module docstring's AS-BUILT note) is actually reachable in real
    # dispatch, rather than being shadowed by text/summarization's
    # unconditional `"summarization" in tags` catch-all under the old
    # 110/110 alphabetical tie-break.
    "priority": 109,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
