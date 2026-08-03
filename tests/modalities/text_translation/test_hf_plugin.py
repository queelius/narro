"""Tests for the text/translation HF resolver plugin.

Mirrors tests/modalities/text_summarization/test_hf_plugin.py in shape.
Also pins the cross-plugin disambiguation behavior described in
docs/superpowers/specs/2026-07-09-text-translation-design.md: a repo
tagged BOTH `summarization` and `translation` resolves to translation
only when its name matches a known translation family (m2m100, nllb,
opus-mt, madlad); otherwise it is left to text/summarization's plugin.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from muse.modalities.text_translation.hf import HF_PLUGIN


def _fake_info(repo_id, tags=(), siblings=(), card=None):
    return SimpleNamespace(
        id=repo_id,
        tags=list(tags),
        siblings=[SimpleNamespace(rfilename=s) for s in siblings],
        card_data=card,
    )


def test_plugin_keys_present():
    keys = {"modality", "runtime_path", "pip_extras", "system_packages",
            "priority", "sniff", "resolve", "search"}
    assert keys.issubset(HF_PLUGIN.keys())


def test_plugin_modality_and_priority():
    assert HF_PLUGIN["modality"] == "text/translation"
    # 109: checked BEFORE the 110 tier (which includes
    # text/summarization) so the both-tagged disambiguation in _sniff
    # is actually reachable in real dispatch. See the module
    # docstring's AS-BUILT note.
    assert HF_PLUGIN["priority"] == 109


def test_plugin_runtime_path():
    assert HF_PLUGIN["runtime_path"] == (
        "muse.modalities.text_translation.runtimes.hf_translation"
        ":TranslationRuntime"
    )


def test_plugin_pip_extras_includes_torch_transformers_sentencepiece():
    extras = HF_PLUGIN["pip_extras"]
    assert any("torch" in e for e in extras)
    assert any("transformers" in e for e in extras)
    assert any("sentencepiece" in e for e in extras)


# ---------- sniff: translation tag ----------

def test_sniff_true_for_translation_tag():
    info = _fake_info("facebook/m2m100_418M", tags=["translation"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_translation_tag_with_others():
    info = _fake_info(
        "facebook/nllb-200-distilled-600M",
        tags=["translation", "text2text-generation", "transformers"],
    )
    assert HF_PLUGIN["sniff"](info) is True


# ---------- sniff: name-pattern fallback (each family) ----------

def test_sniff_true_for_m2m100_name_no_tags():
    info = _fake_info("facebook/m2m100_418M", tags=["text2text-generation"])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_nllb_name_no_tags():
    info = _fake_info("facebook/nllb-200-distilled-600M", tags=[])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_opus_mt_name_no_tags():
    info = _fake_info("Helsinki-NLP/opus-mt-en-es", tags=[])
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_true_for_madlad_name_no_tags():
    info = _fake_info("google/madlad400-3b-mt", tags=["text2text-generation"])
    assert HF_PLUGIN["sniff"](info) is True


# ---------- sniff: rejections ----------

def test_sniff_false_for_bare_summarization_repo():
    """A generic summarization repo (no translation tag, no family-name
    match) must not be claimed here -- it belongs to text/summarization.
    """
    info = _fake_info("facebook/bart-large-cnn", tags=["summarization"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_text_classification_only():
    info = _fake_info("KoalaAI/Text-Moderation", tags=["text-classification"])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_repo_without_tags_or_family_name():
    info = _fake_info("acme/empty", tags=[])
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_no_tags_attribute():
    info = SimpleNamespace(id="x/y", siblings=[], card_data=None)
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_when_family_name_only_in_org_segment():
    """Family-name matching must scope to the model-name segment (the
    part after the org's trailing slash), matching _opus_pair's existing
    discipline. An org merely NAMED after a translation family (e.g. a
    hobbyist org "nllb-fan-org") hosting an unrelated model must not
    sniff True just because "nllb" appears somewhere in the full
    org/repo string.
    """
    info = _fake_info("nllb-fan-org/bert-base", tags=[])
    assert HF_PLUGIN["sniff"](info) is False


# ---------- sniff: both-tagged disambiguation (spec-normative) ----------

def test_sniff_both_tagged_true_when_name_matches_translation_family():
    """Tagged both summarization and translation, name matches a
    translation family (opus-mt) -> claimed here.
    """
    info = _fake_info(
        "acme/opus-mt-en-es-hybrid",
        tags=["summarization", "translation"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_both_tagged_false_when_name_does_not_match_translation_family():
    """Tagged both summarization and translation, name does NOT match
    any translation family -> left to text/summarization's plugin.
    """
    info = _fake_info(
        "acme/generic-seq2seq-model",
        tags=["summarization", "translation"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_for_summarization_only_repo_even_with_family_name():
    """Tagged summarization WITHOUT a translation tag: NEVER claim here,
    even when the name matches a translation family. An
    "nllb-meeting-summarizer" fine-tune is a summarizer, not a
    translator -- the family-name match alone is not sufficient once
    summarization has staked a claim and translation has not.
    """
    info = _fake_info(
        "acme/nllb-meeting-summarizer",
        tags=["summarization"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_translation_plugin_checked_before_summarization_after_priority_fix():
    """text/translation moved to priority 109 (below text/summarization's
    110) specifically so HFResolver's (priority, modality) dispatch
    consults translation's own disambiguating sniff BEFORE
    summarization's unconditional `"summarization" in tags` catch-all.
    Before this fix, both plugins tied at 110 and the alphabetical
    tie-break ("text/summarization" < "text/translation") meant
    summarization always ran first, making the both-tagged
    disambiguation logic above unreachable in real dispatch. This test
    pins the corrected ordering directly (see hf.py's module docstring
    AS-BUILT note and docs/HF_PLUGINS.md's "lower checked first" rule).
    """
    from muse.modalities.text_summarization.hf import HF_PLUGIN as SUMM_PLUGIN

    assert HF_PLUGIN["priority"] == 109
    assert SUMM_PLUGIN["priority"] == 110
    plugins = sorted(
        [HF_PLUGIN, SUMM_PLUGIN], key=lambda p: (p["priority"], p["modality"]),
    )
    assert plugins[0] is HF_PLUGIN
    assert plugins[1] is SUMM_PLUGIN


# ---------- resolve: manifest shape ----------

def test_resolve_synthesizes_manifest_for_m2m100():
    info = _fake_info(
        "facebook/m2m100_418M",
        tags=["translation"],
        card=SimpleNamespace(license="mit"),
    )
    resolved = HF_PLUGIN["resolve"]("facebook/m2m100_418M", None, info)
    m = resolved.manifest
    assert m["model_id"] == "m2m100_418m"
    assert m["modality"] == "text/translation"
    assert m["hf_repo"] == "facebook/m2m100_418M"
    assert m["license"] == "mit"
    assert m["pip_extras"] == list(HF_PLUGIN["pip_extras"])
    assert m["capabilities"]["device"] == "auto"
    # No opus pair for a non-opus-mt repo.
    assert "source_language" not in m["capabilities"]
    assert "target_language" not in m["capabilities"]


def test_resolve_backend_path_points_to_translation_runtime():
    info = _fake_info("acme/x-nllb-mini", tags=["translation"])
    resolved = HF_PLUGIN["resolve"]("acme/x-nllb-mini", None, info)
    assert resolved.backend_path == HF_PLUGIN["runtime_path"]


def test_resolve_when_card_data_missing_license_is_none():
    info = _fake_info("acme/x-madlad-mini", tags=["translation"], card=None)
    resolved = HF_PLUGIN["resolve"]("acme/x-madlad-mini", None, info)
    assert resolved.manifest["license"] is None


# ---------- resolve: opus-mt pair parsing ----------

def test_resolve_parses_opus_pair_two_letter_codes():
    info = _fake_info("Helsinki-NLP/opus-mt-en-es", tags=["translation"])
    resolved = HF_PLUGIN["resolve"]("Helsinki-NLP/opus-mt-en-es", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["source_language"] == "en"
    assert caps["target_language"] == "es"


def test_resolve_parses_opus_pair_en_de():
    info = _fake_info("Helsinki-NLP/opus-mt-en-de", tags=["translation"])
    resolved = HF_PLUGIN["resolve"]("Helsinki-NLP/opus-mt-en-de", None, info)
    caps = resolved.manifest["capabilities"]
    assert caps["source_language"] == "en"
    assert caps["target_language"] == "de"


def test_resolve_skips_opus_pair_for_non_two_letter_target():
    """opus-mt-en-ROMANCE targets a language *group*, not an ISO-639-1
    code. Per design decision: skip pair extraction and synthesize
    WITHOUT the pair rather than guessing.
    """
    info = _fake_info("Helsinki-NLP/opus-mt-en-ROMANCE", tags=["translation"])
    resolved = HF_PLUGIN["resolve"](
        "Helsinki-NLP/opus-mt-en-ROMANCE", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert "source_language" not in caps
    assert "target_language" not in caps


def test_resolve_skips_opus_pair_for_multi_part_variant_name():
    """opus-mt-tc-big-en-pt has more than two dash-separated segments
    after the opus-mt- prefix; do not guess a pair from it.
    """
    info = _fake_info("Helsinki-NLP/opus-mt-tc-big-en-pt", tags=["translation"])
    resolved = HF_PLUGIN["resolve"](
        "Helsinki-NLP/opus-mt-tc-big-en-pt", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert "source_language" not in caps
    assert "target_language" not in caps


def test_resolve_skips_opus_pair_for_three_letter_code():
    """A 3-letter macro-language code (not plain ISO-639-1) should also
    be skipped rather than guessed.
    """
    info = _fake_info("Helsinki-NLP/opus-mt-en-itc", tags=["translation"])
    resolved = HF_PLUGIN["resolve"]("Helsinki-NLP/opus-mt-en-itc", None, info)
    caps = resolved.manifest["capabilities"]
    assert "source_language" not in caps
    assert "target_language" not in caps


# ---------- search ----------

def test_search_yields_results():
    api = MagicMock()
    repo1 = SimpleNamespace(id="facebook/m2m100_418M", downloads=2_000_000)
    repo2 = SimpleNamespace(id="Helsinki-NLP/opus-mt-en-es", downloads=800_000)
    api.list_models.return_value = [repo1, repo2]
    out = list(HF_PLUGIN["search"](
        api, "translat", sort="downloads", limit=10,
    ))
    assert len(out) == 2
    assert out[0].uri == "hf://facebook/m2m100_418M"
    assert out[0].modality == "text/translation"
    assert out[0].downloads == 2_000_000
    assert out[1].model_id == "opus-mt-en-es"


def test_search_calls_list_models_with_translation_filter():
    api = MagicMock()
    api.list_models.return_value = []
    list(HF_PLUGIN["search"](api, "x", sort="downloads", limit=5))
    _, kwargs = api.list_models.call_args
    assert kwargs["filter"] == "translation"
    assert kwargs["sort"] == "downloads"
    assert kwargs["limit"] == 5
    assert kwargs["search"] == "x"


# ---------- cross-plugin dispatch regression (real plugins, real resolver) ----------
#
# Direct unit tests on _sniff (above) exercise the disambiguation rule in
# isolation, but the priority-tie bug this task fixes was only visible in
# real HFResolver dispatch (sniff order derived from sorting BOTH
# plugins' actual priority values). These tests build a real HFResolver
# over the real translation + summarization HF_PLUGIN dicts, sorted with
# the real key, and drive it through resolve() end to end, mirroring
# tests/core/test_hf_resolver_dispatch.py's pattern.

def _sorted_translation_and_summarization_plugins():
    from muse.modalities.text_summarization.hf import HF_PLUGIN as SUMM_PLUGIN

    return sorted(
        [HF_PLUGIN, SUMM_PLUGIN], key=lambda p: (p["priority"], p["modality"]),
    )


def test_dispatch_resolves_both_tagged_family_named_repo_to_translation():
    """Both-tagged AND family-named -> real dispatch resolves to
    text/translation (the priority fix makes this reachable).
    """
    from muse.core.resolvers_hf import HFResolver

    resolver = HFResolver(plugins=_sorted_translation_and_summarization_plugins())
    info = _fake_info(
        "acme/opus-mt-en-es-hybrid",
        tags=["summarization", "translation"],
    )
    info.sha = "a" * 40
    with patch.object(resolver._api, "repo_info", return_value=info):
        resolved = resolver.resolve("hf://acme/opus-mt-en-es-hybrid")
    assert resolved.manifest["modality"] == "text/translation"


def test_dispatch_resolves_both_tagged_non_family_repo_to_summarization():
    """Both-tagged, NOT family-named -> real dispatch falls through to
    text/summarization (translation's own sniff declines; summarization's
    unconditional tag-match then claims it).
    """
    from muse.core.resolvers_hf import HFResolver

    resolver = HFResolver(plugins=_sorted_translation_and_summarization_plugins())
    info = _fake_info(
        "acme/generic-seq2seq-model",
        tags=["summarization", "translation"],
    )
    info.sha = "a" * 40
    with patch.object(resolver._api, "repo_info", return_value=info):
        resolved = resolver.resolve("hf://acme/generic-seq2seq-model")
    assert resolved.manifest["modality"] == "text/summarization"


# ---------- collateral-shadowing guard ----------
#
# translation now sits at priority 109, checked BEFORE every plugin that
# shares the historical 110 tier: audio_classification, embedding_text,
# image_animation, image_cv, image_ocr, image_segmentation,
# model_3d_generation, text_summarization. Guard against translation's
# broadened sniff (translation tag OR family-name match) accidentally
# claiming a repo that actually belongs to one of those siblings. Each
# fixture below is a realistic positive for the sibling's OWN sniff
# (constructed by reading that plugin's _sniff), used here only to
# confirm translation's sniff stays False on it.

def test_sniff_false_for_representative_repos_of_every_priority_110_sibling():
    siblings = [
        # audio/classification: audio-classification tag.
        _fake_info(
            "audeering/wav2vec2-emotion-recognition",
            tags=["audio-classification"],
        ),
        # embedding/text: sentence-transformers tag.
        _fake_info(
            "sentence-transformers/all-MiniLM-L6-v2",
            tags=["sentence-transformers"],
        ),
        # image/animation: model_index.json + text-to-video tag + name hint.
        _fake_info(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            tags=["text-to-video"],
            siblings=["model_index.json"],
        ),
        # image/cv (depth primitive): depth-estimation tag.
        _fake_info(
            "depth-anything/Depth-Anything-V2-Small-hf",
            tags=["depth-estimation"],
        ),
        # image/ocr: image-to-text tag.
        _fake_info(
            "microsoft/trocr-base-printed",
            tags=["image-to-text"],
        ),
        # image/segmentation: mask-generation tag.
        _fake_info(
            "facebook/sam2-hiera-tiny",
            tags=["mask-generation"],
        ),
        # 3d/generation: image-to-3d tag + triposr name-hint.
        _fake_info(
            "stabilityai/TripoSR",
            tags=["image-to-3d"],
        ),
        # text/summarization: summarization tag only, no translation tag,
        # no translation-family name.
        _fake_info(
            "facebook/bart-large-cnn",
            tags=["summarization"],
        ),
    ]
    for info in siblings:
        assert HF_PLUGIN["sniff"](info) is False, (
            f"translation sniff collaterally claimed {info.id!r}"
        )
