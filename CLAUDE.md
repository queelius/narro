# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Muse is a multi-modality generation server and client. It currently supports
twenty-three modalities:

- **audio/alignment**: trusted reference transcript to word timestamps via `/v1/audio/alignments` (Qwen3 ForcedAligner 0.6B curated; multipart audio + text; 11 languages; per-word start/end and uncalibrated timestamp confidence)
- **audio/classification**: audio event / emotion / language classification via `/v1/audio/classifications` (ast-audioset bundled, 527-class multi-label; emotion + speech-commands + language-ID via the resolver; multipart upload + per-input list of `{label, score}` pairs mirroring `/v1/text/classifications`)
- **audio/embedding**: audio-to-vector via `/v1/audio/embeddings` (mert-v1-95m bundled; CLAP, MERT, wav2vec family via the resolver; multipart upload + OpenAI-shape envelope mirroring `/v1/embeddings`)
- **audio/generation**: text-to-music + text-to-SFX via `/v1/audio/music` and `/v1/audio/sfx` (Stable Audio Open 1.0 bundled; ACE-Step v1 3.5B for full songs with optional structured lyrics + instrumentals; per-model capability gates on `supports_music` / `supports_sfx`; optional `lyrics` field on `/v1/audio/music`)
- **audio/quality**: single-ended speech naturalness and audio production-quality assessment via `/v1/audio/quality` (UTMOS 1-5 naturalness MOS + Meta Audiobox Aesthetics four-axis 1-10 scoring via curated HF models; multipart upload; named scaled scores with a model-selected `primary_score`)
- **audio/speech**: text-to-speech via `/v1/audio/speech` (Soprano, Kokoro, Bark, Supertonic-3: the last is an ONNX on-device CPU engine, 31 languages, preset voice styles)
- **audio/transcription**: speech-to-text via `/v1/audio/transcriptions` and `/v1/audio/translations` (Systran faster-whisper family; any CT2 Whisper on HF)
- **chat/completion**: text-to-text LLMs AND vision-language models via `/v1/chat/completions` (OpenAI-compatible incl. tools + streaming + multimodal `messages` content; powered by llama-cpp-python for GGUF chat; `transformers.AutoModelForImageTextToText` for VLMs; SmolVLM, Qwen2-VL, LLaVA via the resolver)
- **embedding/text**: text-to-vector via `/v1/embeddings` (MiniLM, Qwen3-Embedding, NV-Embed-v2; any sentence-transformers HF repo via the resolver)
- **image/animation**: text-to-animation via `/v1/images/animations` (AnimateDiff: 16-frame loops, animated WebP/GIF/MP4 output)
- **image/embedding**: image-to-vector via `/v1/images/embeddings` (dinov2-small bundled; CLIP, SigLIP, DINOv2 family via the resolver; OpenAI-shape wire envelope mirroring `/v1/embeddings`)
- **image/generation**: text-to-image and img2img via `/v1/images/generations` (SD-Turbo, SDXL-Turbo, FLUX.1-schnell, any diffusers HF repo; LoRA adapters via lora_adapter/base_model capabilities (pixel-art-xl curated, pre-paired with sdxl-turbo; per-request lora_scale; muse pull --base overrides the declared base))
- **image/cv**: image CV primitives (depth, keypoints, object detection) via `/v1/images/depth`, `/v1/images/keypoints`, `/v1/images/detect` (depth-anything-v2-small + vitpose-base-simple + detr-resnet-50 bundled; ZoeDepth + DPT-large + RT-DETR via the resolver; multipart upload; capability flags `supports_depth` / `supports_keypoints` / `supports_detection` gate per route; depth supports `png16` (16-bit grayscale, default) or `float32` raw bytes via `response_format`)
- **image/ocr**: image-to-text via `/v1/images/ocr` (trocr-base-printed bundled; Nougat / TexTeller / TrOCR-handwritten via the resolver; multipart upload, optional `prompt` for task hints, optional `max_new_tokens`; OpenAI-shape envelope mirroring `/v1/audio/transcriptions`)
- **image/segmentation**: promptable segmentation via `/v1/images/segment` (sam2-hiera-tiny bundled; SAM-2 family via the resolver; multipart upload, mode dispatch auto/points/boxes/text gated by capability flags; masks emitted as base64 PNG or COCO RLE)
- **image/upscale**: image-to-image super-resolution via `/v1/images/upscale` (stable-diffusion-x4-upscaler bundled; multipart upload, OpenAI-shape envelope; 4x diffusion-based upscaling; capability-gated on `supported_scales`; env-tunable input cap via `MUSE_UPSCALE_MAX_INPUT_SIDE`)
- **image/vectorization**: raster-to-static-SVG via `/v1/images/vectorize` (StarVector-1B im2svg bundled; multipart upload; JSON or raw `image/svg+xml`; pinned, local architecture with no Hub code execution; active SVG content rejected)
- **text/classification**: text moderation via `/v1/moderations` and full label distribution via `/v1/text/classifications` (any HF text-classifier or zero-shot NLI; bundled twitter-roberta sentiment + deberta-v3 zero-shot; capability-gated dispatch on `supports_classification` / `supports_zero_shot`)
- **text/rerank**: cross-encoder rerank via `/v1/rerank` (bge-reranker-v2-m3 bundled; any cross-encoder reranker on HF; Cohere-compat wire shape)
- **text/summarization**: BART/PEGASUS seq2seq summarization via `/v1/summarize` (bart-large-cnn bundled; any summarization-tagged HF repo via the resolver; Cohere-compat wire shape)
- **text/translation**: machine translation via `/v1/translate` and the bare `/translate` alias, plus `GET /languages` (LibreTranslate-compat wire shape; m2m100-418m bundled, ~2GB CPU-runnable; NLLB-200-distilled-600M and Opus-MT en-es/en-de curated via the resolver; the gateway's `MODEL_OPTIONAL_PATHS` default-model seam resolves the model when LibreTranslate clients omit `model` entirely)
- **video/generation**: text-to-video via `/v1/video/generations` (wan2-1-t2v-1-3b bundled with sequential CPU offload by default to fit 8-12GB GPUs; Wan / CogVideoX / LTX-Video families via the resolver; narrative clips up to 30s; mp4/webm/frames_b64 output; GPU-required; `cpu_offload` capability (model|sequential) + `vae_tiling`, globally overridable via `server.video_cpu_offload`)
- **3d/generation**: image-to-3d via `/v1/3d/from-image` and text-to-3d via `/v1/3d/generations` (triposr bundled; TRELLIS, Wonder3D, Hunyuan3D-2, Shap-E via the resolver; Shap-E is text-to-3D only via `diffusers.ShapEPipeline`; GLB output as `data:model/gltf-binary;base64,...` URL or b64_json; capability flags `supports_text_to_3d` / `supports_image_to_3d` gate per route; sync/blocking pattern mirroring video/generation)

Modality tags are MIME-style (`audio/speech`, not `audio.speech`). The HTTP
path hierarchy mirrors the OpenAI shape where possible (`/v1/audio/speech`,
`/v1/chat/completions`, `/v1/embeddings`, `/v1/images/animations`,
`/v1/images/generations`) for client compatibility.

In addition to the per-modality routes, muse also exposes an admin
REST API under `/v1/admin/*` (v0.28.0+) for runtime model control
without restarting `muse serve`: enable, disable, probe, pull, remove,
plus worker introspection and async-job tracking. Closed-by-default
behind `MUSE_ADMIN_TOKEN`. See "Admin REST API" below.

`audio/quality` is muse's twenty-first modality. `POST
/v1/audio/quality` accepts multipart `file` plus optional `model` and
returns `{id, object, model, primary_score, scores, metadata}`. Each
named score carries `value`, nominal `minimum` / `maximum`, and a
direction (`higher_is_better`, `lower_is_better`, or `descriptive`),
because a UTMOS 1-5 naturalness MOS is not numerically interchangeable
with Audiobox's 1-10 axes. Both runtimes use TorchCodec to decode 16kHz
mono ranges sequentially (10-second windows), return a duration-weighted
aggregate plus `metadata.segments`, the literal `metadata.worst_segment`,
and `metadata.worst_review_segment` (which ignores subsecond tail windows
when a longer review candidate exists), and reject audio over
`MUSE_AUDIO_QUALITY_MAX_DURATION_SECONDS` (600s default) with 413. The
priority-100 HF plugin claims only two
verified custom checkpoint shapes: `Blinorot/UTMOS-PyTorch` routes to
the local TorchScript runtime and `facebook/audiobox-aesthetics`
loads the package's official model class from Hub safetensors directly
onto Muse's selected device. Audiobox must
win before the broader priority-110 `audio/classification` plugin,
since its Hub card carries that generic tag. Curated aliases are
`utmos` (MIT, primary `naturalness`) and `audiobox-aesthetics`
(CC-BY-4.0, primary `production_quality`). Pairwise/text-conditioned
judges such as SpeechJudge are deferred to a separate comparison
route/capability rather than overloading this single-ended contract.

`audio/alignment` is muse's twenty-second modality. `POST
/v1/audio/alignments` accepts multipart `file` + trusted reference `text`,
plus optional `model` and `language`, and returns `{id, object, model, text,
language, duration_seconds, words, metadata}`. Each word has `word`, `start`,
`end`, and an uncalibrated confidence derived from the mean probability of its
start/end timestamp tokens. This is forced alignment, not ASR: the model
locates supplied words and does not decide what was spoken. The priority-90 HF
plugin claims only `Qwen/Qwen3-ForcedAligner-0.6B-hf`, downloads its safe
safetensors checkpoint, and installs Transformers 5.13+, TorchCodec, Nagisa,
and SoyNLP. Audio is bounded before inference, resampled to 16kHz mono, and
limited by `MUSE_AUDIO_ALIGNMENT_MAX_BYTES` (50 MB),
`MUSE_AUDIO_ALIGNMENT_MAX_DURATION_SECONDS` (300s), and
`MUSE_AUDIO_ALIGNMENT_MAX_TEXT_CHARS` (50,000). The curated alias is
`qwen3-forced-aligner-0.6b` (Apache 2.0; 11 languages). Before audio decode,
the runtime tokenizes the reference and rejects more than 2,048 alignable
words; after processor preparation it enforces the checkpoint's 8,192-token
context before moving inputs to the device. Confidence calculation selects
timestamp-marker logits before its float32 softmax, avoiding a full-sequence
probability tensor. Only the typed `AudioAlignmentDecodeError` emitted at the
TorchCodec boundary maps to HTTP 415; model alignment failures remain 500s.

`text/rerank` is muse's first Cohere-compat modality (rather than
OpenAI-compat): OpenAI has no rerank API, and Cohere's `/v1/rerank` is
the de-facto standard that downstream tooling (LangChain, LlamaIndex,
Haystack) expects. Response envelope mirrors Cohere's: `results[]`
with `index` + `relevance_score`, optional `document.text`, plus
`meta.billed_units.search_units` for SDK compatibility.

`text/summarization` is muse's second Cohere-compat modality (after
text/rerank). OpenAI has no summarization API; Cohere's `/v1/summarize`
was the de-facto reference until its 2024 deprecation, and the wire
shape is what summarization tooling expects. Request: `{text, length,
format, model}`. Response: `{id, model, summary, usage, meta}`.
`length` ("short"|"medium"|"long") deterministically maps to
`max_new_tokens` in the runtime: short=80, medium=180, long=400.
`format` ("paragraph"|"bullets") is recorded in `meta.format` and is
metadata-only for non-instruction summarizers like BART-CNN; future
instruction-tuned summarizers can consult it. The bundled
`bart-large-cnn` (Apache 2.0, ~400MB, CPU-friendly) is the default;
the curated `bart-cnn-samsum` (`supports_dialog_summarization=true`)
is dialog-tuned. The HF resolver sniffs any `summarization`-tagged
repo at priority 110 and serves it via `BartSeq2SeqRuntime` over
`transformers.AutoModelForSeq2SeqLM` (BART, PEGASUS, T5).

`text/translation` (v0.58.0) is muse's twentieth modality and its
first LibreTranslate-compat one -- self-hosted translation clients
(LibreTranslate SDKs, the `libretranslate` CLI) work against muse
unmodified. Wire shape: `POST /v1/translate` and the bare `POST
/translate` alias (LT clients hardcode that path) take `{q, source,
target, format?, model?}` (`q` is a string or list[str]; scalar in ->
scalar out) and return `{translatedText}`; `GET /languages` returns
`[{code, name, targets}, ...]` derived live from the resolved model's
tokenizer, not a hand-maintained table. `model` is a muse extension
LT clients never send -- see the default-model seam below. Single
runtime `TranslationRuntime` over `transformers.AutoModelForSeq2SeqLM`
+ `AutoTokenizer`, dispatching per family (`_family_for(hf_repo)`,
mirroring the 3D modality's precedent): m2m100 and NLLB use
`forced_bos_token_id` language-token injection (NLLB via a structured
ISO-639-1 -> FLORES-200 table); Opus-MT checkpoints are fixed-pair
(`capabilities.source_language`/`target_language`, any other pair
400s `invalid_language`); MADLAD-400 (curated-only, GPU-oriented)
prepends a T5-style `<2xx>` target prefix. The bundled `m2m100-418m`
(facebook/m2m100_418M, MIT, ~2GB, `device: auto`, covers all pairs
across 100 languages) is the default; curated additions are
`nllb-200-distilled-600m` (**CC-BY-NC-4.0**, 200 languages, non-
commercial license called out in its description) and the permissive
per-pair `opus-mt-en-es` / `opus-mt-en-de` (Helsinki-NLP, ~300MB
each, best per-pair quality). The HF resolver plugin sniffs
`translation`-tagged repos (plus `text2text-generation` + name-
pattern fallback) at **priority 109** -- one tier below the 110 group
that includes text/summarization's plugin -- so translation's both-
tagged disambiguation rule (a repo tagged both `summarization` and
`translation` goes to translation only on a translation-family name
match) is actually exercised by real `HFResolver` dispatch instead of
being shadowed by summarization's unconditional `"summarization" in
tags` sniff (an as-built fix; the design originally shipped this
plugin at the same 110 tier, which made the rule correct in isolated
unit tests but unreachable in practice since `"text/summarization" <
"text/translation"` alphabetically). Because LibreTranslate clients
never send `model`, the modality exports `MODEL_OPTIONAL_PATHS =
("/v1/translate", "/translate", "/languages")`; `discover_modalities`
aggregates every modality's declared paths into a `{path: modality}`
map, and the gateway's catch-all proxy resolves `model` from that map
(first ENABLED catalog model of the matching modality, by discovery
order -- bundled m2m100 wins on a fresh install) instead of 400ing
`model_required` when a declared path's request omits `model`; no
enabled model of that modality -> 503 `no_default_model`. Any other
path with a missing `model` keeps today's 400 exactly. This seam is
generic: a future modality can opt in by exporting its own
`MODEL_OPTIONAL_PATHS`, no gateway changes needed. Input size is
capped by `limits.translate_max_chars` (env `MUSE_TRANSLATE_MAX_CHARS`,
default 20000, read per-request) -> 400 `input_too_long`.
`source: "auto"` (language auto-detection) and `POST /detect` are
explicitly deferred; v1 returns a structured 400
`source_detection_not_supported`.

`image/embedding` is muse's first image-to-vector modality. The wire
envelope at `POST /v1/images/embeddings` mirrors `/v1/embeddings`
exactly (`{object: "list", data, model, usage}`) so OpenAI SDK clients
that already consume embeddings can reuse helper code. Each `input`
entry is a `data:image/...;base64,...` URL or `http(s)://...` URL
pointing at PNG/JPEG/WEBP; image decoding goes through the shared
`decode_image_input` helper from the image_generation modality.
The bundled `dinov2-small` (Apache 2.0, 88MB, 384-dim, CPU-friendly)
is the default; curated additions cover SigLIP2 and CLIP. The HF
resolver sniffs any repo with an image-feature-extraction-class tag
plus a `preprocessor_config.json` sibling at priority 105 (between
embedding/text and image-generation file-pattern) and serves it via
`ImageEmbeddingRuntime` over `transformers.AutoModel` +
`AutoProcessor`. The runtime's `_extract_embeddings` dispatch picks
the right pooling per architecture: CLIP `image_embeds` >
SigLIP/DINOv2 `pooler_output` > DINOv2 base `last_hidden_state[:, 0]`
(CLS token).

`audio/embedding` is muse's first audio-to-vector modality. Wire shape
is multipart-in (one or more `file` parts, mirroring
`/v1/audio/transcriptions`) and `/v1/embeddings`-shaped JSON out
(`{object: "list", data, model, usage}`). Audio decoding goes through
`librosa` (already installed for Whisper) inside the runtime/script,
which resamples on the way in to each model's preferred rate (CLAP
48kHz, MERT 24kHz). The bundled `mert-v1-95m` (MIT, 95MB, 768-dim
music understanding via mean-pool over time, `trust_remote_code=True`
for the custom feature extractor) is the default; the curated
`clap-htsat-fused` adds 512-dim audio + text-aligned embeddings
(BSD-3, supports_text_embeddings_too=True). The HF resolver sniffs
any repo with `feature-extraction` tag plus a name pattern matching
`clap`, `mert`, `audio-encoder`, `wav2vec`, or `audio-embedding` at
priority 105, and serves it via `AudioEmbeddingRuntime` over
`transformers.AutoModel` + `AutoFeatureExtractor` (with
`AutoProcessor` preferred for newer repos). The runtime's
`_extract_embeddings` dispatch picks the right pooling per
architecture: CLAP `audio_embeds` > pooler_output > MERT/wav2vec
`last_hidden_state.mean(dim=1)` (mean-pool over time). Per-file size
cap via `MUSE_AUDIO_EMBEDDINGS_MAX_BYTES` (default 50MB); duration
cap via `MUSE_AUDIO_EMBEDDINGS_MAX_SECONDS` (default 60s; runtime
truncates after decode).

`audio/generation` is muse's first modality with TWO URL routes mounted
on ONE MIME tag. `/v1/audio/music` and `/v1/audio/sfx` share the same
request body, codec, registry surface, and runtime. The only per-route
difference is the manifest capability key consulted: `supports_music`
or `supports_sfx`. When a flag is False (or a future MusicGen-only
model lacks `supports_sfx`), the unsupported route returns 400. The
two-URL split is for legibility: a "footsteps on gravel" prompt sent
to `/v1/audio/music` would silently produce a 30-second loop of
footsteps treated as music; routing the same prompt to `/v1/audio/sfx`
makes the user's intent explicit to operators reading logs and to the
model itself.

ACE-Step v1 3.5B (v0.49.0) is the second audio/generation backend and
muse's first *song* model: a genre/style `prompt` plus optional
structured `lyrics` (`[verse]`/`[chorus]` tags) produce a sung song;
empty/None lyrics produce an instrumental (the runtime substitutes
ACE-Step's literal `[instrumental]` tag). It is a bundled script
(`ace-step-v1-3.5b`) that aliases the shared `AceStepRuntime` as `Model`
(the VLM-bundled pattern), declares `supports_music: true` /
`supports_sfx: false` (so `/v1/audio/sfx` returns 400 for it), and pins
`device: cuda` (3.5B is impractical on CPU). The optional `lyrics` field
(max 8000 chars) was added to `AudioGenerationRequest` and the
`generate` protocol; models that ignore lyrics (Stable Audio) simply
drop it. The route `duration` ceiling was raised 120s -> 240s for
ACE-Step's long clips; the per-model runtime still clamps to its own
`max_duration`. Unlike StableAudio (which returns in-memory numpy
arrays), ACE-Step *writes WAV file(s) to disk and returns the path(s)*:
`AceStepRuntime` hands the pipeline a temp `save_path`, resolves the
on-disk output defensively (returned path if present, else the
`save_path`), reads it back via `soundfile`, and deletes the temp dir in
a `finally` (leak-safe, per the #200 temp-WAV lesson). `trust_remote_code`
is NOT needed (`from acestep.pipeline_ace_step import ACEStepPipeline` is
a direct package import). Install pins the git source (`ace-step @
git+https://github.com/ace-step/ACE-Step.git` -- the distribution name
is `ace-step` with a hyphen even though the import is `acestep`; the
PyPI `ace-step` is a stale v0.1.0). `pip_extras` also pins `torchcodec`:
ACE-Step's `save_wav_file` calls `torchaudio.save`, which on modern
torchaudio (>=2.8) delegates encoding to torchcodec (backed by ffmpeg's
libav, already in `system_packages`); without it generation fails at
save time. Both the dist-name and torchcodec gaps were caught by
real-API verification on the GPU box (Step B1), not by the fully-mocked
unit tests. The `cpu_offload` / `overlapped_decode` / `torch_compile`
low-VRAM knobs are exposed as optional manifest capabilities (default
off) and forwarded to the pipeline constructor.

The `/v1/images/generations` route also accepts optional `image` (data URL or http(s):// URL) + `strength` (0.0 to 1.0, default 0.5) fields for img2img since v0.17.0. OpenAI SDK clients pass them via `extra_body`:

    client.images.generate(prompt="oil painting", model="sdxl-turbo",
                           extra_body={"image": "data:image/png;base64,...", "strength": 0.6})

Models advertise support via `capabilities.supports_img2img`. Requests for non-supporting models return 400.

The `image/generation` modality also exposes `/v1/images/edits` (inpainting) and `/v1/images/variations` (alternates of one image, no prompt) since v0.21.0. Both are multipart/form-data routes that mount on the same modality. Inpainting takes `image` + `mask` + `prompt` and routes to `backend.inpaint(...)`, which lazy-loads `AutoPipelineForInpainting.from_pipe(self._pipe)` to share VRAM with the loaded t2i pipeline. Variations takes `image` only and routes to `backend.vary(...)`, which delegates to the existing img2img path with empty prompt and high strength (default 0.85). Capability flags `supports_inpainting` and `supports_variations` gate the routes; OpenAI SDK clients use `client.images.edit(image=..., mask=..., prompt=..., model=...)` and `client.images.create_variation(image=..., model=...)` natively.

`image/upscale` (v0.25.0) is muse's super-resolution modality: a separate MIME tag from `image/generation` because the runtime backbone is different (`StableDiffusionUpscalePipeline`, not `AutoPipelineForText2Image`). Wire shape at `POST /v1/images/upscale` is multipart/form-data (mirroring `/v1/images/edits`), with `image` as the source file plus `model`, `scale`, optional `prompt`, `negative_prompt`, `steps`, `guidance`, `seed`, `n`, and `response_format` as Form fields. Output envelope mirrors `/v1/images/generations`: `{created, data: [{b64_json|url, revised_prompt}]}`. The bundled `stable-diffusion-x4-upscaler` (Apache 2.0, ~3GB, fixed 4x) is the default; the HF resolver plugin (priority 105) sniffs other diffusers-shape upscalers (`model_index.json` + `image-to-image` tag + upscaler-name allowlist). The `supported_scales` capability gates the request `scale` parameter (returns 400 for unsupported values; SD x4 supports `[4]` only). An env-tunable input-side cap (`MUSE_UPSCALE_MAX_INPUT_SIDE`, default 1024) prevents runaway VRAM use on oversized inputs; the cap is read per-request, so changes take effect on the next request, not at supervisor restart. GAN-based upscalers (AuraSR, Real-ESRGAN) need separate non-diffusers runtimes and are deferred to v1.next.

`image/vectorization` (v0.59.0) is muse's raster-to-SVG modality. `POST /v1/images/vectorize` accepts multipart `image` plus optional `model`, `max_new_tokens` (1-7680), `temperature` (0-2), `top_p`, `num_beams` (1-8), `seed`, and `response_format` (`json` or `svg`). JSON returns `{id, object, model, mime_type, svg, source_size, width, height, view_box, seed, usage, metadata}`; raw mode returns `image/svg+xml`, ready to save or pass to an SVG consumer such as Manim. The bundled and curated `starvector-1b-im2svg` model uses the Apache-2.0 official mixed-precision checkpoint (~5.14GB on disk, loaded fp16 with ~7GB working memory, suitable for a 12GB GPU). Supply-chain policy is exact-only: `muse pull` pins HF revision `380ab95d25a8e9ab1dc825debe238b4953ae13b9`, downloads only config/tokenizer/safetensors data, and instantiates Muse's audited inference-only architecture; there is intentionally no broad HF resolver and no `trust_remote_code`. The output boundary accepts a static SVG subset and rejects scripts, event handlers, foreign objects, embedded rasters, remote/data URLs, DTDs/entities, oversized documents, or excessive element nesting. Input long side defaults to 2048 pixels and is configurable with `MUSE_VECTORIZATION_MAX_INPUT_SIDE`. This model is aimed at icons, logos, flat illustrations, and technical diagrams rather than photographs.

`image/segmentation` (v0.26.0) is muse's promptable-segmentation modality. Wire shape at `POST /v1/images/segment` is multipart/form-data: `image` as the source file plus `model`, `mode` (auto/points/boxes/text), `prompt` (text mode), `points` (JSON-encoded `[[x, y], ...]`), `boxes` (JSON-encoded `[[x1, y1, x2, y2], ...]`), `mask_format` (`png_b64` or `rle`), and `max_masks` as Form fields. Output: `{id, model, mode, image_size, masks: [{index, score, mask, bbox, area}]}`. Mode dispatch is capability-gated end-to-end: `supports_automatic`, `supports_point_prompts`, `supports_box_prompts`, `supports_text_prompts`. A request with `mode=text` against a model declaring `supports_text_prompts: False` returns 400 before the runtime is invoked. The bundled `sam2-hiera-tiny` (Apache 2.0, ~40MB, point/box/auto, no text) is the default; curated `sam2-hiera-base-plus` and `sam2-hiera-large` extend the family. The HF resolver plugin (priority 110) sniffs `mask-generation` and `image-segmentation` tags. CLIPSeg is a deferred future: the plugin pattern recognizes it and flips `supports_text_prompts: True`, but the SAM2Runtime backbone needs a CLIPSeg-specific replacement to actually consume the text prompt. The mask format dispatch (`png_b64` for portable / viewable, `rle` for compact / pycocotools-compatible) introduces a precedent: the codec ships pure-Python RLE encode/decode that round-trips internally, with `pycocotools` as an optional faster path that produces output other COCO tooling can decode directly. Axis-order discipline at the wire layer: `image_size` is `[W, H]` (PIL convention); RLE `size` is `[H, W]` (COCO convention); `bbox` is `[x, y, w, h]` (COCO bbox convention).

`image/cv` (v0.37.0) is muse's image-CV-primitives umbrella, hosting three routes on one MIME tag: `POST /v1/images/depth` (depth estimation), `POST /v1/images/keypoints` (pose / keypoint detection), `POST /v1/images/detect` (object detection). Three sibling runtimes: `HFDepthRuntime` over `AutoModelForDepthEstimation`, `HFKeypointRuntime` over `AutoModelForKeypointDetection` (transformers >= 4.46), and `HFObjectDetectionRuntime` over `AutoModelForObjectDetection`. Capability flags `supports_depth` / `supports_keypoints` / `supports_detection` on each model's manifest gate which route accepts it; mismatch returns 400 `wrong_primitive`. The depth route's `response_format` field selects between `png16` (16-bit grayscale PNG, default; quantized into the value range so the full 16-bit precision tracks the actual depth bounds) and `float32` (raw little-endian bytes, fully precise but ~4x payload). The keypoint runtime's v1 ViTPose-style flow takes a single full-image bbox per call; multi-person pose extraction needs an upstream person detector and is deferred. Bundled defaults: `depth-anything-v2-small` (relative depth, ~25M, Apache 2.0), `vitpose-base-simple` (COCO 17-keypoint pose, ~85M, Apache 2.0), `detr-resnet-50` (COCO 80-class detection, ~41M, Apache 2.0; `pip_extras` includes `timm` for the ResNet backbone). Curated additions: `depth-anything-v2-base` (larger), `zoedepth-nyu-kitti` (metric meters), `dpt-large` (classic), `rtdetr-r50vd` (faster detection). The single HF plugin (priority 110) dispatches per primitive via tag (`depth-estimation`, `keypoint-detection`, `object-detection`) with repo-name fallbacks; `metric_depth: True` auto-derives from the repo name (`zoedepth`, `metric`).

`image/ocr` (v0.36.0) is muse's image-to-text modality. Wire shape at `POST /v1/images/ocr` is multipart/form-data: `image` as the source file plus optional `model`, `prompt` (Nougat-style task hint; TrOCR ignores), and `max_new_tokens` (validated to [1, 4096]). Output envelope mirrors `/v1/audio/transcriptions`: `{id, model, text, usage: {completion_tokens}}`. Single runtime `HFVision2SeqRuntime` over `transformers.AutoModelForVision2Seq` + `AutoProcessor`; the processor handles per-family preprocessing (TrOCRProcessor, NougatProcessor) automatically. The bundled `trocr-base-printed` (MIT, 334M, English printed text) is the default; curated additions cover handwriting (`trocr-large-handwritten`, MIT), academic papers with math/LaTeX (`nougat-base`, **CC-BY-NC-4.0**, non-commercial), and math-formula-only (`texteller`, MIT). The HF plugin (priority 110) sniffs `image-to-text`-tagged repos but explicitly excludes `image-text-to-text` (those are VLMs, reserved for the future #97 `image/description` modality). Capability flags `supports_handwritten` and `supports_math` are advisory only (surfaced in `/v1/models` for client-side filtering, not server-enforced). Donut/DocVQA structured-document variants are out of scope for v1; they may land later as a separate `image/document` modality.

`video/generation` (v0.27.0) is muse's narrative-clip modality, the heaviest yet. Wire shape at `POST /v1/video/generations` is JSON-only (no multipart) with `prompt` plus optional `model`, `duration_seconds` (0.5 to 30), `fps` (1 to 60), `size` (WxH string), `seed`, `negative_prompt`, `steps`, `guidance`, `response_format` (`mp4` default, `webm`, or `frames_b64`), and `n` (capped at 2 because each video is heavy). Output envelope mirrors `/v1/images/animations`: `{data: [{b64_json}], model, metadata: {frames, fps, duration_seconds, format, size}}`. Two distinct runtimes ship under the same MIME tag: `WanRuntime` (`diffusers.WanPipeline` or `DiffusionPipeline` fallback) and `CogVideoXRuntime` (`diffusers.CogVideoXPipeline`); the HF resolver dispatches per architecture. The bundled `wan2-1-t2v-1-3b` (Apache 2.0, 5s clips at 832x480) bundles a UMT5-XXL text encoder (~11GB at fp16) alongside its 1.3B transformer, so a full-resident load is ~11-12GB and OOMs a 12GB card; the earlier "~3GB / fits 8GB" claim counted only the transformer and was wrong. Curated additions: `cogvideox-2b` (~6GB at fp16, 6s at 720x480, fits 12GB) and `ltx-video` (~13GB, 30fps at 1216x704, requires 16GB+). The HF plugin (priority 105) sniffs `text-to-video`-tagged repos whose name matches one of `wan`, `cogvideox`, `ltx-video`, `mochi`, or `hunyuan`. LTX/Mochi/Hunyuan currently fall back to `WanRuntime`; their dedicated runtimes ship in v1.next. Distinction from `image/animation`: animation is short looping clips (16 frames @ 8fps = 2s, default `loop=true`, animated WebP), video is narrative clips (5s+, single play, no loop field, mp4). The codec includes vp9 webm with vp8 fallback (when ffmpeg lacks vp9) and an explicit `UnsupportedFormatError` when neither codec is available. All bundled video models declare `device: "cuda"`; CPU inference would take 10 to 30 minutes per clip and isn't a useful default.

CPU offload (v0.52.1) is a per-model capability, not a bool: `capabilities.cpu_offload` is `"model"` (`pipe.enable_model_cpu_offload(device=...)`, whole-component granularity, peak VRAM ~ largest single component), `"sequential"` (`pipe.enable_sequential_cpu_offload(device=...)`, sub-module granularity, fits <=12GB but slower), or absent/`false` (today's plain `.to(device)` move, unchanged default for models that do not set it). Offload and `.to(device)` are mutually exclusive; offload only applies when the resolved device is cuda, mirroring the existing `device != "cpu"` guard. `capabilities.vae_tiling: true` calls `enable_vae_tiling()` plus `enable_vae_slicing()` (best-effort, guarded by `hasattr`) after placement, to cap the VAE-decode spike at higher resolutions. Both `WanRuntime` and `CogVideoXRuntime`, plus the bundled `wan2_1_t2v_1_3b.py` script's own `Model` class (which duplicates the runtime's construction logic rather than aliasing `WanRuntime`, so it needs the same wiring to actually honor the capability), dispatch through one shared helper, `muse.modalities.video_generation.runtimes._offload.place_pipeline`, so all three call sites stay in lockstep. The registry setting `server.video_cpu_offload` (env `MUSE_VIDEO_CPU_OFFLOAD`, opt_str, default unset) globally overrides the per-model capability across every video model: `model`, `sequential`, or `off`/`false`/`none` to force plain placement even on a model that declares offload; unset falls through to the capability. `wan2-1-t2v-1-3b` sets `cpu_offload: "sequential"` and `vae_tiling: true` by default (guarantees load+generate on 8-12GB out of the box) and a corrected `memory_gb: 3.0` (an honest sequential-peak estimate; `muse models probe` self-heals it to the measured value). A bigger-card operator who wants speed over headroom sets `server.video_cpu_offload model` (or `off`).

`3d/generation` (v0.41.0 through v0.45.0) hosts four concrete runtimes under one MIME tag, dispatched per-family via `_family_for(repo_id)` in `model_3d_generation/hf.py`. `TripoSRRuntime` (v0.41.0+) is the bundled default for image-to-3D. `ShapERuntime` (v0.43.0+) wraps `diffusers.ShapEPipeline` for text-to-3D; it follows the deferred-imports pattern (module-level sentinels for `torch`, `ShapEPipeline`, `trimesh`; `_ensure_deps()` populates them lazily). `TRELLISRuntime` (v0.44.0+) wraps Microsoft's TRELLIS SDK via `transformers` AutoPipeline with `trust_remote_code=True` for image-to-3D. `Hunyuan3DRuntime` (v0.45.0+) is the fourth runtime and the first to support BOTH image-to-3D and text-to-3D from one model; it wraps Tencent's `hy3dgen` SDK with `trust_remote_code=True`. Internally, text-to-3D is a two-stage pipeline: text generates an image via `HunyuanDiTPipeline`, then the image is lifted to a mesh via `Hunyuan3DDiTFlowMatchingPipeline`. The `_Family.trust_remote_code` flag flows into the synthesized manifest's `capabilities.trust_remote_code`, which the runtime constructor reads via the kwargs splat. The per-family dispatch adds one `_Family` constant plus one dispatch branch in `_family_for` plus one runtime file under `runtimes/` per new runtime. Shap-E is text-only: the curated `shap-e` entry declares `supports_image_to_3d: False` and `supports_text_to_3d: True`; a request to `/v1/3d/from-image` with `model=shap-e` returns 400 before the runtime is invoked. TRELLIS is image-only: the curated entry (renamed from `trellis-text-image` to `trellis-image` in v0.44.0) declares `supports_text_to_3d: False`. Hunyuan3D-2 declares both flags True. GLB output flows through the existing codec: `trimesh.Trimesh` assembles vertices and faces from the pipeline output, then `mesh.export(file_type="glb")` yields the bytes. A companion `_pip_extras_for(runtime_path)` helper mirrors the dispatch shape so each runtime's venv installs only the deps it needs. After v0.45.0, task #108 is closed. Wonder3D is deferred indefinitely (multi-view plus NeuS reconstruction does not fit muse's one-pipeline-call runtime model).

The package is organized around three plugin surfaces:

- `src/muse/modalities/<mime_name>/`: self-contained wire contract
  (protocol + routes + codec + client). Each modality package exports
  `MODALITY: str` + `build_router: Callable[[registry], APIRouter]`.
  Discovered at runtime by `discover_modalities`.
- `src/muse/models/*.py`: flat directory of drop-in model scripts.
  Each `.py` file declares `MANIFEST: dict` + a `Model` class.
  Discovered at runtime by `discover_models`. Best for one-off models
  with custom code (NV-Embed, Soprano).
- `muse.core.resolvers.*`: URI-addressable model sources. `muse pull
  hf://Qwen/Qwen3-8B-GGUF@q4_k_m` synthesizes a manifest, persists it
  in `catalog.json`, and routes requests through a generic runtime class
  (`LlamaCppModel` for GGUF, `SentenceTransformerModel` for ST embedders).
  Best for uniform model classes where one runtime serves many models.
  See `docs/RESOLVERS.md` and `docs/CHAT_COMPLETION.md`.

`chat/completion` (v0.42.0+) accepts vision-language models via the OpenAI multimodal `messages` shape: `content: [{type: text}, {type: image_url, image_url: {url: ...}}]`. The route's pre-dispatch step (`_decode_image_parts` in `routes.py`) walks `messages`, validates capability flags (`supports_vision`, `supports_multi_image`) on the loaded model, decodes each `image_url` via the existing `decode_image_input` helper (data URLs + `http(s)://` with SSRF guard), and rewrites the part to a muse-internal `{type: image, image: <PIL.Image>}` shape before calling the backend's `chat()` / `chat_stream()`. Capability mismatches return 400 `vision_not_supported` or `too_many_images`; bad images return 400 `invalid_image`. The runtime is `HFVisionLanguageModel` (`muse.modalities.chat_completion.runtimes.transformers_vlm`); bundled default is `smolvlm-256m-instruct` (~500MB, CPU-runnable). Curated extends with SmolVLM-Instruct, Qwen2-VL-2B/7B, and LLaVA-1.5-7B.

A modality-agnostic core (`muse.core`) holds the registry, discovery,
resolver dispatch, HF downloader, per-venv pip install, and FastAPI
app factory.

## Architecture

```
HTTP API (/v1/audio/speech, /v1/images/generations, /v1/images/segment, /v1/images/upscale, /v1/video/generations, /v1/models, /health)
    |
    v
muse.core.server   (FastAPI factory, mounts per-modality routers)
    |
    v
muse.core.registry (ModalityRegistry: {modality: {model_id: Model}})
    |
    v
Modality backends implementing modality-specific protocols
```

### Key modules

- `muse.core.discovery`: scans directories and returns `{model_id:
  DiscoveredModel}` (for model scripts) and `{mime_tag: build_router}`
  (for modality packages). First-found-wins on collisions; script
  errors are logged, never raised. Bundled scripts in the installed
  `muse/models/` tree get their canonical Python import name
  (`muse.models.<stem>`); external scripts get a mangled private name
  to avoid sys.modules collisions.
- `muse.core.catalog.known_models()`: discovery-driven, two-tier cache.
  The script-discovery scan (importlib) is cached for the process
  lifetime (new scripts need a restart); the merged result is memoized
  against catalog.json's (path, mtime_ns), so catalog writes from ANY
  process (admin pull subprocess, operator CLI pull/remove beside a
  running supervisor) are visible on the next call without a restart.
  Projects each script's MANIFEST onto the `CatalogEntry` shape
  the rest of muse consumes (backend_path is synthesized from the
  Model class's `__module__:__name__`). Merges two sources: discovered
  bundled scripts PLUS catalog.json entries that carry a persisted
  `manifest` field (resolver-pulled models). Bundled wins on collision.
  `pull()` dispatches by identifier shape: curated alias > `://` URI
  > bare id. `get_manifest(model_id)` prefers the persisted manifest
  for resolver-pulled models, falls back to the script module's
  MANIFEST. Catalog state lives at `~/.muse/catalog.json` (or
  `MUSE_CATALOG_DIR` env override); writes are atomic (write-then-rename).
- `muse.core.resolvers` + `muse.core.resolvers_hf`: URI -> `ResolvedModel`
  dispatch for `muse pull hf://...`. `HFResolver` sniffs each HF repo
  (`.gguf` siblings -> `chat/completion` / LlamaCppModel; sentence-
  transformers tag -> `embedding/text` / SentenceTransformerModel).
  GGUF `@variant` is required; no magic default. Search implemented for
  both modalities with per-variant deduping (sharded GGUFs don't emit
  one row per shard). Registers `hf://` scheme on import.
- `muse.core.curated`: loads `src/muse/curated.yaml` (hand-edited
  recommendations list). `find_curated(id)` / `expand_curated_pull(id)`.
  Curated entries either alias a bundled script (`bundled: true`) or
  point at a URI; the curated id is preserved as the catalog key even
  when the URI would synthesize a different one, so newbie-friendly ids
  like `qwen3.5-4b-q4` survive end-to-end.
- `muse.core.chat_formats`: loads `src/muse/chat_formats.yaml` (hand-
  edited map from HF repo substring to llama-cpp-python `chat_format`
  string + `supports_tools` flag). Consulted by the HF resolver when
  synthesizing GGUF manifests. First-match-wins; case-insensitive
  substring on `hf_repo`. More-specific patterns must come first.
- `muse.core.registry.ModalityRegistry`: keyed by `(modality, model_id)`.
  First registered model per modality is the default for that modality.
  `register(modality, model, manifest=...)` stores the MANIFEST verbatim;
  `/v1/models` splats `manifest.capabilities` + top-level description
  /license/hf_repo into each entry. No shared protocol base across
  modalities.
- `muse.core.server.create_app(registry, routers)`: builds the FastAPI app
  with shared `/health` and `/v1/models`, mounts per-modality routers, and
  registers the `ModelNotFoundError` exception handler so 404s use the
  OpenAI-style `{"error":{...}}` envelope instead of FastAPI's `{"detail":...}`.
- `muse.core.venv`: venv creation (`create_venv`, `install_into_venv`, `find_free_port`). Each `muse pull` creates `~/.muse/venvs/<model-id>/`; catalog records the `python_path`.
- `muse.cli_impl.worker`: single-worker mode (runs one uvicorn in one venv). Invoked via `muse _worker` (hidden subcommand).
- `muse.cli_impl.gateway`: FastAPI proxy app. Routes by `model` field in request body/query; aggregates `/v1/models` and `/health` across workers.
- `muse.cli_impl.supervisor`: orchestrates workers + gateway. `plan_workers` groups catalog by venv; `spawn_worker` + `wait_for_ready` manage subprocess lifecycle; `run_supervisor` is the entrypoint `muse serve` delegates to.
- `muse.cli_impl.search`: `run_search()` for `muse search`. Thin wrapper over `resolvers.search()` plus log-level quieting so httpx's per-request debug lines don't interleave with the table output.

### Modality conventions

Each modality subpackage (`src/muse/modalities/<mime_name>/`) contains:
- `__init__.py`: exports `MODALITY: str` (MIME-style tag like `"audio/speech"`) and `build_router: Callable[[ModalityRegistry], APIRouter]`. These two are what `discover_modalities` scans for.
- `protocol.py`: Protocol + Result dataclass(es) for this modality
- `routes.py`: defines `build_router(registry) -> APIRouter`
- `client.py`: HTTP client for this modality's endpoints
- `codec.py`: modality-specific encoding (wav/opus for audio; png/jpeg for images; base64 float32 for embeddings; SSE+OpenAI chunk shape for chat)
- `runtimes/` (optional): *generic* runtime classes that serve many models from one implementation. `chat_completion/runtimes/llama_cpp.py:LlamaCppModel` wraps any GGUF; `embedding_text/runtimes/sentence_transformers.py:SentenceTransformerModel` wraps any sentence-transformers repo. Runtime class paths are referenced by resolver-synthesized manifests.
- `backends/` (optional): *private helpers* used by this modality's own model scripts. NOT a plugin surface. Only `audio_speech/backends/` exists (`base.py` with `voices_dir` + `BaseModel`; `transformers.py` with the Narro engine Soprano delegates to).
- `audio_transcription/` was muse's first modality with multipart/form-data uploads (OpenAI Whisper wire shape). `routes.py` handles UploadFile + Form fields inline. As of v0.21.0, `image_generation/` is the second multipart consumer (`/v1/images/edits` + `/v1/images/variations`); v0.25.0 adds `image_upscale/` (`/v1/images/upscale`) as the third. All three implement multipart inline; if a fourth multipart modality lands, factor out to `muse.modalities._common.uploads`.
- `text_classification/` is muse's first modality whose internal MIME tag (`text/classification`) is broader than its primary URL route (`/v1/moderations`). The wire path is OpenAI-specific; v0.35.0 added the second route `/v1/text/classifications` on the same modality tag, sharing dataclasses + protocols. Two runtimes coexist: `HFTextClassifier` (existing, for fine-tuned classifier heads) and `HFZeroShotPipeline` (v0.35.0, wraps `transformers.pipeline("zero-shot-classification")`). Capability flags `supports_classification` and `supports_zero_shot` on the manifest gate which routes / runtime calls each model accepts: a request with `candidate_labels` against a model with `supports_zero_shot=False` returns 400 `zero_shot_not_supported`; the inverse mismatch returns 400 `candidate_labels_required`. The HF plugin's `_resolve` dispatches to either runtime based on tag (`zero-shot-classification`) or repo-name fallback (`zero-shot`, `mnli`, `nli`, `xnli`).

Four distinct concepts worth keeping straight:

| Surface | Who writes it | Purpose |
|---|---|---|
| `muse/models/*.py` | bundled muse + users | public model scripts, one per model, discoverable |
| `modalities/*/runtimes/*.py` | muse internal | generic runtimes, one class serves many models (GGUF, ST) |
| `modalities/*/backends/*.py` | muse internal | private helpers shared inside a modality |
| `muse.core.runtime_helpers` | muse internal | cross-modality utilities every runtime imports: `select_device` (cuda/mps/cpu auto-detect), `dtype_for_name` (string-to-torch.dtype map with `fp16`/`bf16`/`fp32` aliases), `set_inference_mode` (no-grad switch; literal token kept out of caller bodies via shared helper), `LoadTimer` (opt-in load-time logging context). Added v0.31.0; consolidated ~30 per-runtime copies. The meta-test `tests/core/test_runtime_helpers_meta.py` AST-walks every runtime and bundled script to flag re-implementations |

Each model script (`src/muse/models/<id>.py`) contains:
- Top-level `MANIFEST: dict` with required keys `model_id`, `modality`, `hf_repo` and optional `description`, `license`, `pip_extras`, `system_packages`, `capabilities`. Anything else passes through.
- A class named exactly `Model` (tests alias it: `from muse.models.kokoro_82m import Model as KokoroModel`).

Each `Model` class:
- Satisfies the modality's Protocol structurally (no base class required).
- Accepts `hf_repo=`, `local_dir=`, `device=`, `**_` in its constructor (the catalog loader calls with those kwargs; `**_` absorbs future additions).
- Prefers `local_dir` over `hf_repo` when loading weights.
- Defers heavy imports (torch, transformers, diffusers, kokoro, llama_cpp) to inside `__init__` (via an `_ensure_deps()` helper that lazy-imports into module-level sentinels). Tests patch the module-level names directly; the `_ensure_deps` check `if X is None` short-circuits when tests have pre-populated mocks. `muse --help` and `muse pull` must work without any ML deps installed.

### Capability precedence

For resolver-pulled GGUFs, the `chat_format` and `supports_tools` fields
in the catalog's persisted manifest come from layered lookups:

1. MANIFEST `capabilities.chat_format` (user-set explicitly) -- highest
2. `src/muse/chat_formats.yaml` pattern match on `hf_repo` at resolve time
3. None -- falls through to llama-cpp-python's GGUF-metadata autodetection

At `load_backend` time, `manifest.capabilities` is merged into the
runtime constructor's kwargs (caller kwargs win). This lets generic
runtimes like `LlamaCppModel` receive `gguf_file`, `chat_template`,
`context_length`, etc. without the worker layer knowing those keys
exist. The generic runtime also gets `model_id` injected, since one
class serves many models.

### Device placement precedence (v0.48.0+)

Which device a model loads on is resolved in `load_backend`, most
authoritative first:

1. **catalog `device_override`** -- an operator pin set via
   `muse models set-device <id> <auto|cpu|cuda|mps>` (clear with
   `--clear`). Stored as a top-level field on the model's catalog entry
   (NOT in the manifest), read live from `catalog.json` at load time.
   Beats everything below, so an operator can force a cpu-pinned model
   onto cuda (or pin a model to cpu to save VRAM) per deployment without
   editing the bundled script. `set-device <id> auto` un-pins to
   auto-detect.
2. **manifest `capabilities.device` pin** (model-author affinity) -- a
   value other than `"auto"` overrides the `--device` flag. Heavy
   GPU-only models declare `"cuda"`; the lone CPU-only model
   (`supertonic-3`, ONNX) declares `"cpu"`.
3. **`--device` server flag** (`muse serve --device ...`, default
   `auto`), folded into the runtime kwargs.
4. **`"auto"`** -- the runtime's `select_device` picks cuda if available,
   else mps, else cpu.

**Default is `auto`, not `cpu` (v0.48.0).** Every CUDA-safe bundled model
declares `device: "auto"` so it lands on the GPU automatically on a GPU
host and on CPU on a CPU-only host -- generally faster, and safe because
lazy-load + LRU/idle eviction size the live working set to fit VRAM. Only
`supertonic-3` stays pinned `cpu` (its ONNX runtime ignores a cuda
device). `set-device` is catalog state: it takes effect on the model's
**next cold load**, so evict or restart an already-resident worker to
apply it.

**`auto` resolves to a concrete memory POOL consistently across the
control plane (v0.48.0).** An `auto` model that loads on the GPU must be
*sized, admitted, and evicted* against the VRAM pool, not host RAM -- else
the LoadDirector would think it "fits" against terabytes of host RAM while
the GPU OOMs. Three sites share one resolution (a GPU is present iff live
VRAM info is available): `LoadDirector._resolve_pool_device` (admission /
LRU eviction), `supervisor._servability_reason` (boot + request-path
capacity check), and `memory._resolve_auto_side` (the `/v1/admin/memory`
per-model breakdown, which also honors the supervisor `--device` flag
first). The `IdleSweeper` does not resolve independently -- it delegates
to `LoadDirector._free_for_device` so its decision-log readings can't
drift from the director's. `mps` is its own case: it pools against host RAM for sizing
(pynvml is CUDA-only, and MPS is unified memory) but keeps its own `mps`
measurements bucket in `_record_observed_peak`, matching what
`muse models probe` persists. This bug never surfaced in CI because a
CPU-only host degrades `auto`->cpu, which happens to match the buggy
CPU-pool accounting; it only bites on a GPU host.

**GPU-layers pin (v0.56.0, GGUF only).** `muse models set-gpu-layers <id> <N>`
(N >= 0), `--all` (all layers on GPU, i.e. -1; a bare `-1` positional is not
shell-passable through click), or `--clear` writes a top-level catalog
`gpu_layers_override` (the `device_override` pattern): llama.cpp `n_gpu_layers`,
-1 = all layers on GPU, 0 = pure CPU, N > 0 = a static GPU/CPU layer split for
GGUFs bigger than VRAM. Precedence: pin > `capabilities.n_gpu_layers` > runtime
default (-1). Refuses non-GGUF models (other runtimes silently ignore the kwarg).
Takes effect on next cold load; run `muse models probe <id>` after pinning
so admission sizing measures the split's real (smaller) VRAM peak. The
probe honors the pin automatically (it constructs via `load_backend`).
Known limitation: a split model occupies both VRAM and host RAM, but the
director accounts it against its resolved device's pool only -- the
Tier-1 static-offload simplification (automatic offload was evaluated and
rejected; see docs/superpowers/specs/2026-07-08-gpu-layers-pin-design.md).

### No shared supertype across modalities

`AudioResult` and `ImageResult` do NOT share a common base. Streaming semantics
differ (audio chunks are time-ordered and playable immediately; diffusion steps
are progressive refinement of one frame). A `GenerationModel` abstract base
would be a leaky abstraction. Instead, `ModalityRegistry` treats models as
`Any`, and each modality's router + codec knows its own types.

## Process model

`muse serve` is a **supervisor**, not a single process. As of v0.40.0 it
is **lazy by default**: the gateway is healthy as soon as the supervisor
boots, and per-model workers spawn on the first request that names them
(see "Lazy load + LRU eviction" below for the full state machine):

```
User request
    |
    v
muse serve (supervisor, port 8000)              [healthy instantly; zero workers]
  ├── gateway FastAPI app (in-process)
  │    routes by request body `model` field
  │    on cold model: director.acquire() spawns the worker
  │
  └── subprocess per loaded model (spawned lazily):
       ├── worker (port 9001, venv-kokoro)   spawned on first /v1/audio/speech
       ├── worker (port 9002, venv-sd-turbo) spawned on first /v1/images/generations
       └── (other catalog-enabled models stay unloaded until requested)
```

Each pulled model gets its own venv at `~/.muse/venvs/<model-id>/`
with exactly the pip_extras it declares. Workers run the existing
`muse.cli_impl.worker.run_worker` logic via `muse _worker`
(hidden subcommand). The supervisor brings the gateway up immediately,
runs catalog boot validation (stamps an advisory `unservable_reason` on
models that are not sizable at all or whose estimate exceeds device
capacity; the stamp is re-checked live on first request, see "Lazy load
+ LRU eviction" below), then waits. Workers spawn on demand through
`LoadDirector.acquire`; the historical
"poll FIRST worker then promote others" boot dance is gone. Operators
who want eager loading run `muse models warmup <id>` for each model in
their startup script (see "Lazy load + LRU eviction" below).

The gateway extracts `model` from the request body (POST) or query
(GET), calls `director.acquire(model_id)` to ensure the worker is
loaded (cold loads pay one-shot latency; hot loads return in
sub-millisecond), forwards the request, and calls
`director.release(model_id)` on stream-close. `/v1/models` and
`/health` are aggregated across the live (loaded) workers via parallel
httpx calls; entries for catalog-enabled-but-unloaded models still
appear in `/v1/models` with `loaded: false` and the SDK can decide
whether to warm them.

This gives you dep isolation (transformers 4.46 for parler-tts
coexists with transformers 5.x for newer models), crash isolation
(a segfault in one worker does not kill the rest), and a uniform
HTTP surface (clients hit one port, do not care about internal venvs).

The supervisor also runs an auto-restart monitor thread. Every 5
seconds it polls each currently loaded worker's /health and checks for
process death via Popen.poll. After 3 consecutive failures (or
immediate process exit), the monitor terminates the existing process
and respawns it with exponential backoff (1s, 2s, 4s, ..., capped at
30s). After 10 unsuccessful restart attempts the worker is marked
dead; /health reports "degraded" and /v1/models skips its entries.
The monitor reads the dynamic loaded set from `state.director.loaded`
on each tick, so admin-driven loads and evictions show up immediately.

Use `muse models disable <id>` to mark a pulled model as inactive in
the catalog (the supervisor refuses to lazy-load it on request).
`muse models enable <id>` re-enables it. These flip the catalog
`enabled` bit; whether a model is *actually* in memory is a separate
runtime question handled by the LoadDirector (see "Lazy load + LRU
eviction" below). When the supervisor is running and `MUSE_ADMIN_TOKEN`
is set, the same CLI commands route through the admin API; otherwise
they edit the catalog only and take effect on the next request.

## Lazy load + LRU eviction (v0.40.0+)

`muse serve` no longer eager-loads enabled models at boot. The gateway
comes up instantly with zero workers; per-model workers spawn on first
request and are evicted under memory pressure via on-demand LRU. Behind
this is a runtime decoupling: catalog `enabled: true` declares "in
service, may serve requests" but does NOT imply "in memory right now."
A 12GB GPU can have 30 models enabled and serve them all, just not
simultaneously; the live working set is sized to fit current free VRAM
minus a headroom margin.

The orchestrator is `muse.cli_impl.load_director.LoadDirector`,
attached to `SupervisorState`. The gateway wraps every forwarded
request in `director.acquire(model_id)` / `director.release(model_id)`.
`acquire` is a three-phase critical section: under-lock decision (read
loaded set, query live free memory, pick eviction victims if needed) ->
outside-lock load (worker spawn + `/health` poll, possibly with
preceding evictions) -> under-lock commit (insert `LoadEntry`,
increment refcount, update `last_touched_at`). Concurrent acquires for
the *same* cold model collapse to one load via an `asyncio.Future` /
`threading.Event` registered in `in_flight_loads`; concurrent acquires
for *different* cold models proceed in parallel because the load phase
runs outside the lock.

**Acquire runs OFF the event loop (v0.50.3+).** The gateway dispatches
`director.acquire` via `await asyncio.to_thread(...)` under
`asyncio.shield` (`gateway.py:_route_via_director`), NOT as a blocking
synchronous call. A cold load takes tens of seconds (worker spawn +
health poll); calling it synchronously froze the single gateway event
loop and stalled EVERY concurrent request, including ones for
already-hot models (measured: a 37s cold load stalled 6 hot requests
~35s each -- see `examples/concurrent/RESULTS.md`). Moving it off-loop
makes request-path acquires genuinely concurrent, which required three
companion safety fixes the serialized path had masked:
1. **In-flight memory reservation.** `in_flight_loads` now maps to an
   `InFlightLoad(event, memory_gb, pool)` record, and `_decide` debits
   the sum of same-pool in-flight loads (`_reserved_for_pool`) from
   available memory before the fit check (the eviction no-candidates
   re-check nets the same reservations). Without this, two concurrent
   cold loads for different models would both pass the fit check against
   the same live free-VRAM reading and over-commit -> OOM. The
   reservation is deliberately conservative (it may over-reserve while a
   worker is mid-allocation, trading a possible spurious evict/503 for
   guaranteed no-OOM; precise per-load allocation accounting is a
   deferred follow-up).
2. **Cancellation-safe release.** If a request is cancelled (client
   disconnect / timeout) while its off-loop acquire is still running,
   `shield` keeps the acquire alive and a done-callback releases the
   refcount if the acquire later succeeds -- else a bumped refcount leaks
   and pins the model non-evictable forever. `_forward_with_release`'s
   cleanup was likewise widened to `except BaseException` so a cancel on
   the forward leg also releases.
3. **Overlap-gated observed-peak writeback.** `_load_and_commit` only
   writes the self-heal peak when the load was "solo" (the sole in-flight
   load at both endpoints of its free_before..free_after window AND no
   concurrent load/eviction bumped `_inflight_epoch`); an overlapping
   load or eviction pollutes the global free-memory delta the writeback
   infers a peak from.

**Same-model cold-load coalescing (v0.51.0+).** Off-loop acquire fixed
loop-blocking but left a second hazard: a burst of concurrent requests
for the SAME cold model each dispatched their own `to_thread(acquire)`,
and the N-1 that lost the director's singleton-collapse race parked in
`event.wait` INSIDE ThreadPoolExecutor threads, exhausting the default
`min(32, cpu+4)` pool and stalling unrelated hot traffic (measured 11.5s
on the GPU box; loop stayed responsive, `/health` fine). `gateway.py`
`_acquire_coalesced` elects ONE loader per cold model via a per-model
`asyncio.Future` gate in `SupervisorState.cold_load_gates` (a plain dict:
the loader election is await-free, hence atomic on the single loop thread
-- no lock, no two-loader TOCTOU). The loader runs one off-loop acquire;
same-model waiters `await asyncio.shield(gate)` ON THE LOOP (no thread
each). The gate is settled from a done-callback on the acquire future
(NOT the loader body), so a loader cancelled mid-load still resolves its
waiters; waiters `shield` so one cancellation can't poison the group;
waiters PROPAGATE the loader's failure (no retry herd) and on success take
their own refcount via a full re-acquire that re-decides hot-or-cold. The
director is unchanged (its own threading collapse still serves admin
warmup / other callers). Coalescing is same-model only; cross-model
over-admission is still handled by the in-flight reservation above.

Memory accounting is live, not declared. `muse.core.memory_probe`
wraps `pynvml.nvmlDeviceGetMemoryInfo` (GPU) and
`psutil.virtual_memory().available` (CPU). `nvidia-ml-py` is a soft
dep under `[server]`: when missing (CPU-only host, AMD GPU, driver
mismatch) `gpu_free_gb()` returns None and GPU loads either fall back
to a declared `MUSE_GPU_BUDGET_GB` cap or 503 with `unservable_reason`.
Every cold load also captures `free_before - free_after`; if the
observed delta exceeds `measurements.<device>.peak_bytes` it gets
written back via atomic write-then-rename. Estimates self-heal upward
toward reality on every load.

A model's load size comes from a three-tier sizing ladder (in
`supervisor.py:_has_memory_data`), most-honest first:

1. `capabilities.memory_gb` (declared annotation), or
   `measurements.<device>.peak_bytes` from a prior probe -- the
   authoritative numbers.
2. *Weights-on-disk fallback* (v0.47.3): when neither exists, the
   supervisor sizes the model from the bytes already in its `local_dir`
   (sum over the HF snapshot tree, symlinks followed). A pulled model
   that was never probed is therefore sized from its weights and loads
   on demand rather than 503'ing with "no memory estimate." This is the
   common case -- every `muse pull` leaves weights on disk -- so the old
   "run `muse models probe` first" wall is gone. *GGUF exception
   (v0.50.2):* a GGUF snapshot dir routinely holds several quant variants
   of one model (q3/q4/q5/q8/f16), but only the declared
   `capabilities.gguf_file` actually loads. Summing the whole tree would
   OVERestimate wildly (a 4B q4 whose repo ships six quants sums to ~15 GB
   vs its ~2.6 GB weight) and 503 a servable model as "exceeds device
   capacity", so when a specific `gguf_file` is declared the sizer counts
   that one file, falling back to the tree walk only if it is absent on
   disk. This bug stayed hidden until v0.50.1 routed undeclared-device
   models to the cuda pool, where the inflated estimate first exceeded the
   card. *Diffusers/transformers dedup (v0.54.3):* the same over-count bites
   any snapshot that ships redundant encodings of one component -- fp16 AND
   fp32, `.safetensors` AND `.ckpt`/`.bin`, `model-*.safetensors` AND
   `pytorch_model-*.bin` -- plus large non-weight blobs (dataset-attribution
   CSVs, READMEs). Stable Audio Open summed 14.6 GB vs its ~2.6 GB real load
   and 503'd on a 12 GB card. The tree walk now counts only WEIGHT-extension
   files and, per component (same dir + canonical stem, with dtype tags and
   the `model`/`pytorch_model`/`tf_model`/`flax_model` prefix normalized),
   keeps ONE variant -- fp16 over fp32, safetensors over pickle. Counting the
   loaded variant (not the largest) leans the estimate DOWN, the safe
   direction: an under-estimate self-heals via the observed-peak writeback
   and a rare OOM is recovered by the auto-restart monitor, whereas an
   over-estimate 503s a servable model outright.
3. Nothing sizable (no annotation, no probe, no weights on disk): the
   model is flagged `unservable_reason` and 503s before any worker
   spawn.

`backfill_manifest_memory` applies the same ladder to the manifest the
gateway hands `director.acquire`, so the director sizes a never-probed
model from its weights too (not from a default 0.0). A model that DOES
fit (possibly after evicting idle LRU models) loads on demand; the
director runs the live fit/evict decision. A model that cannot fit even
an empty device is caught earlier by the request-path servability
re-check (see below) and 503s at the gateway WITHOUT reaching the
eviction loop -- deferring an impossible model to the director would
only evict the whole idle working set before 503'ing.

Memory-accounting settings (all optional; read via the config registry,
so each can be set as a `MUSE_*` env var OR in `~/.muse/config.yaml` OR
via `muse config set` -- see "Configuration" below and docs/CONFIG.md).
As of v0.52.0 the four budget/headroom knobs are actually wired into the
LoadDirector (before v0.52.0 they were documented but inert):

- `MUSE_GPU_BUDGET_GB` (`server.gpu_budget_gb`): declared cap on GPU
  memory; muse uses `min(declared, live)` when both are available.
- `MUSE_CPU_BUDGET_GB` (`server.cpu_budget_gb`): declared cap on host RAM.
- `MUSE_GPU_HEADROOM_GB` (`server.gpu_headroom_gb`, default `1.0`):
  subtracted from live free VRAM before deciding fit; protects against
  driver allocations and fragmentation.
- `MUSE_CPU_HEADROOM_GB` (`server.cpu_headroom_gb`, default `2.0`):
  subtracted from live free RAM before deciding fit.

`muse models list` shows a five-state status enum:

- `enabled_loaded` (filled circle): catalog-enabled and currently
  resident on a worker.
- `enabled_unloaded` (half circle): catalog-enabled but not resident;
  next request triggers a cold load.
- `disabled` (open circle): catalog-disabled; requests 503 without
  attempting a load.
- `recommended` (star): curated entry not yet pulled.
- `available` (mid-dot): pullable from a resolver but not curated.

The breaking behavior change is cold-start latency on first request
to each model. A 9B GGUF can take 10-30s to load; a 3B diffusion
model 5-15s. `muse models warmup <id>` is the manual pre-load: the
admin endpoint `POST /v1/admin/models/{id}/warmup` runs the load
without bumping refcount, so subsequent requests are hot. Operators
who really want the v0.39 eager-boot semantics put a warmup loop in
their startup script.

`muse pull` runs probe at the end (`--no-probe` opts out for
cross-device pulls, e.g., pulling on a CPU host for later GPU
deployment). This gives freshly-pulled models a measured memory
estimate up front. As of v0.47.3 the estimate is no longer *required*
to serve: even a `--no-probe` pull (or a probe that was skipped) loads
on demand, because the supervisor falls back to sizing the model from
its on-disk weights (tier 2 of the sizing ladder above). Probe still
matters -- it yields a more honest peak than raw weights size, and it
self-heals on every subsequent load -- but it is an optimization, not a
gate. Pulled models still need an explicit `muse models enable <id>` to
start serving traffic, since enabling is a separate operator decision
from "have weights on disk."

Decoupling principle: `enabled` is catalog state; `loaded` is runtime
state. Two new admin operations, `load_model_into_worker` and
`unload_model_from_worker` (in `muse.admin.operations`), perform the
runtime mutation *without* flipping the catalog `enabled` bit. The
LoadDirector calls these on lazy-load and eviction paths so LRU
eviction of model X doesn't accidentally disable X for the next
request. The legacy `enable_model` / `disable_model` ops keep
catalog-flip semantics for explicit operator intent and back the
`muse models enable/disable` CLI verbs.

Catalog format stays backward-compatible. `measurements.<device>.peak_bytes`
gains self-healing semantics (passively updated on every cold load),
and older muse readers ignore unknown fields. `/v1/models` gains
`loaded: bool`, `last_loaded_at: iso8601 | null`, and `unservable_reason:
str | null` per entry, plus the OpenAI-compat `created: int` (a stable
`0`; muse has no per-model creation time) and `owned_by: str` (the HF org
slug from `hf_repo`, else `"muse"`) added in v0.47.4 so strict OpenAI SDK
clients that validate the model-object shape accept it. The OpenAI-shape
envelope is otherwise unchanged. All these fields are written *after* the
`capabilities` splat in `build_model_entry`, so a manifest can never
clobber the authoritative `id`/`object`/`created`/`owned_by`/`loaded`
keys.

The gateway's `/v1/models` lists *every enabled catalog model*, not just
the ones with a live worker (v0.47.3). Loaded workers are aggregated
from their own `/v1/models`; enabled-but-unloaded catalog entries are
appended with `loaded: false` (skipping disabled models and any without
a `python_path`). Both paths render through the shared
`muse.core.server.build_model_entry` helper so the entry shape is
identical whether the model is resident or cold. Before v0.47.3 the
gateway listed only loaded workers, so a freshly-booted lazy supervisor
reported an empty `/v1/models` until the first request warmed a worker
-- a doc-vs-code drift that hid the entire enabled-but-unloaded set from
clients.

Worker-reported `last_loaded_at` is filled in by the gateway, not the
worker (v0.47.4). Per-model workers run outside the supervisor and own no
`LoadDirector`, so their own `/v1/models` reports `last_loaded_at: null`
even for a resident model. The gateway's `_enrich_loaded_at` joins the
real load timestamp from `state.director.loaded[id].loaded_at` (snapshot
under the director lock, rendered via the shared `_format_loaded_at`)
onto each aggregated resident entry before appending the unloaded rows;
it never overwrites an already-non-null value. `/health` is broadened the
same way: it unions enabled-but-unloaded catalog models + modalities into
its `models`/`modalities` sets (via `_unloaded_catalog_entries`) so a
zero-worker lazy supervisor reports the full serviceable surface, not an
empty one.

Servability is re-checked live at request time, not frozen at boot.
Boot and the request-path re-check share ONE verdict function,
`_servability_reason(entry, ...)` (sizing ladder, then a device-capacity
check against live free memory), so the two can never drift.
`validate_catalog_at_boot` stamps `unservable_reason` once on startup;
`revalidate_servability` re-derives the full verdict on the first
request to a stamped model against the LIVE catalog + live free memory.
It clears the stamp only when the model is sizable AND fits: a probe or
weights landing after boot clears a "no memory estimate" stamp, and
freed memory clears a stale "exceeds device capacity" stamp -- both
without a supervisor restart. A GENUINE "exceeds device capacity" stamp
(the model cannot fit even an empty device) is RETAINED, so the gateway
503s `model_unservable` directly instead of routing an impossible model
into the director's eviction loop. So `muse models probe <id>` (which
writes a measurement to the catalog) takes effect on the *next request*.
The mtime-cached `_read_catalog` is the read path; a probe's catalog
write bumps the mtime so the re-check sees fresh measurements.

Lock discipline:

- `state.director.lock` (RLock) guards in-memory mutations of the
  loaded set, in_flight_loads, and the recent_decisions deque. Held
  during decision and commit phases of `acquire`; released around the
  long-running worker spawn / wait.
- Admin operations (`load_model_into_worker`, `disable_model`,
  `unload_model_from_worker`) plan their mutations under `state.lock`,
  release the lock before slow worker spawn or shutdown, then reacquire
  to commit. Same plan-then-execute pattern as the existing admin
  endpoints; lets `acquire` and admin endpoints share the runtime
  without serializing on slow I/O.
- Module-level `_WRITEBACK_LOCK` (in `load_director.py`) serializes
  the read-modify-write on `measurements` across observed-peak writebacks
  for *different* models so catalog.json round-trips stay atomic.

### Idle eviction (v0.40.1+)

In addition to memory-pressure LRU eviction, models can declare a
per-model `capabilities.idle_timeout_seconds` in their manifest. A
background sweeper thread runs every `MUSE_IDLE_SWEEP_INTERVAL_SECONDS`
(default 30) and evicts loaded models whose `last_touched_at` exceeds
the timeout AND whose refcount is 0. This frees memory without
waiting for traffic-driven LRU.

A GLOBAL default idle timeout (`server.idle_timeout_seconds`, env
`MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS`, v0.51.0+) applies to any model that
declares no per-model `idle_timeout_seconds`. Precedence: per-model
`capabilities.idle_timeout_seconds` wins; otherwise the global default;
otherwise never idle-evict. As of v0.52.0 the global default is **600
seconds (10 minutes)**, not off -- an untouched model with refcount 0 is
reclaimed after 10 minutes by default. The value is resolved through the
config registry (lenient: a non-numeric env/file value logs a warning and
falls back to the 600s default, so a bad value cannot crash boot). This
is the knob for "reclaim any idle model after N minutes" on a shared
single-GPU box, where bundled models (which declare no per-model timeout)
would otherwise sit resident until memory pressure.

To DISABLE idle eviction entirely, set `server.idle_timeout_seconds` (or
`MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS`) to `0` (or any value `<= 0`); the
IdleSweeper's `<= 0` guard then never idle-evicts, and only memory
pressure releases a model. Per-model `capabilities.idle_timeout_seconds`
still overrides the global default in both directions.

The sweeper reuses the on-demand eviction's disable_fn primitive,
so the orphan-worker-on-disable-failure remediation (re-insert
LoadEntry) applies uniformly. Idle eviction is logged in
`recent_decisions` with reason="idle_timeout:Ns".

Configuration:
- Per-model: `capabilities.idle_timeout_seconds: <number>` in manifest. Null/absent = fall back to the global default.
- Global default: `server.idle_timeout_seconds` (env `MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS`), default **600** (10 min); set `<= 0` to disable. Applied to models without a per-model timeout.
- Sweep interval: `server.idle_sweep_interval_seconds` (env `MUSE_IDLE_SWEEP_INTERVAL_SECONDS`), default 30.

### Request queueing (v0.55.0+)

Two gateway-side mechanisms (spec docs/superpowers/specs/2026-07-08-request-queueing-design.md), both waiting ON the event loop (never in pool threads):

- **Per-model concurrency cap:** `capabilities.max_concurrency` (else `server.default_max_concurrency`, default 0 = unlimited = no gating). Excess requests park FIFO on a per-model asyncio.Semaphore in `muse.cli_impl.queueing.ConcurrencyGate`. The CPU box's 32B runs with `max_concurrency: 1`.
- **Capacity-wait:** the LoadDirector tags its capacity 503 `retryable=True` when the failure is TRANSIENT -- any of the three in-flight states holds: an in-use loaded model (refcount > 0) blocks eviction, another model's cold load is in flight (its reservation made the fit fail; v0.57.1), or an eviction is mid-flight (`director.evicting`: victim popped from `loaded` but disable_fn still shutting the worker down, so the director looks idle while VRAM is still occupied; v0.57.2). The gateway parks on `CapacityNotifier` (generation asyncio.Event, fired threadsafe by the director on release-to-zero, post-eviction, and load commit/abort) and retries under the shared deadline. Only a truly idle-and-empty device that still cannot fit the model 503s immediately.

One budget covers both: `server.queue_timeout_seconds` (default 300; 0 = today's immediate-503). `server.max_queue_depth` (default 0 = unbounded) fast-fails 503 `queue_full`. Timeout -> 503 `queue_timeout` (message reports actual elapsed wait). Telemetry `request` events carry `queued_ms`; `/v1/admin/memory` and `/v1/telemetry/summary` expose per-model `queue_depth` (the seam a future queue-aware federation router consumes). `/v1/admin/memory` omits queue_depth when no gate is bound (unknown is never rendered as 0), while `/v1/telemetry/summary` always includes it (0 when unbound). Cold-starting models with parked waiters (not yet loaded) are visible too (v0.57.4): the summary lists them in a separate always-present `queued` array, and the admin per-model breakdown includes them as rows marked `loaded: false`. `acquire_slot` returns whether a slot was actually taken and the gateway releases only on True, so a live `server.default_max_concurrency` change can no longer desync the gate's accounting.

## Configuration

All server settings live in ONE declarative registry,
`muse.core.config.SETTINGS` (v0.52.0+). Each setting is one `Setting` row
(`key`, `env`, `type`, `default`, `group`, `help`); env reads, file reads,
`muse config`, and the generated template all derive from that single
list. There is no parallel lookup table. See docs/CONFIG.md for the full
settings inventory.

Resolution precedence (first wins): **CLI-override arg > `MUSE_*` env var >
`~/.muse/config.yaml` > built-in default.** The config file lives at
`<MUSE_CATALOG_DIR or ~/.muse>/config.yaml` (override the whole path with
`MUSE_CONFIG`). The two bootstrap keys (`paths.catalog_dir`,
`paths.config_file`) resolve from env+default ONLY -- the file cannot
redirect the path that locates the file. `config.get(key)` reads env LIVE
on every call and parses the file once (cached); it is LENIENT (a bad
value logs a warning and falls back to the default, never raises), so a
typo in one setting cannot 500 the request path. `muse config set` is the
STRICT path (validates and refuses a bad value).

`muse config` CLI verbs:
- `muse config generate [--force]` -- write a fully-commented `config.yaml`
  from the registry (every setting, its default, its env name).
- `muse config show [--json]` -- every setting's effective value AND source
  (default / file / env). `admin.token` is redacted to `set` / `unset`.
- `muse config path` -- print the resolved config-file path.
- `muse config get <key>` -- print one effective value.
- `muse config set <key> <value>` -- validate and write one value into the
  file (atomic, preserves other keys).
- `muse config unset <key>` -- remove one setting from the file so it falls
  back to env/default (the counterpart to `set`; no override value means
  "use the default", so reverting a key requires removing it).

Scope boundary: `config.yaml` is for SERVER / global settings only.
Per-model state (enable/disable, `device_override` from
`muse models set-device`, probe `measurements`) lives in
`~/.muse/catalog.json` and is edited via `muse models ...`. Shipped
recommendations live in `src/muse/curated.yaml` (package data, not
per-deployment editable). A setting can have both a global default and a
per-model override (idle-timeout is the canonical example: global
`server.idle_timeout_seconds` vs per-model
`capabilities.idle_timeout_seconds`, per-model wins).

## Admin REST API

`muse.admin/` provides eleven endpoints under `/v1/admin/*` for runtime
model control. Admin is closed-by-default: every request returns 503
`admin_disabled` until `MUSE_ADMIN_TOKEN` is set in the supervisor's
environment, after which `Authorization: Bearer <token>` is required.

Endpoints:

| Endpoint | Sync/async | Effect |
|---|---|---|
| `POST /v1/admin/models/{id}/enable` | async (202+job_id) | spawn or restart-in-place |
| `POST /v1/admin/models/{id}/disable` | sync | unload + catalog flip |
| `POST /v1/admin/models/{id}/probe` | async | run `muse models probe` in venv |
| `POST /v1/admin/models/_/pull` | async | run `muse pull <body.identifier>` |
| `DELETE /v1/admin/models/{id}?purge=bool` | sync | refuses 409 if loaded |
| `GET /v1/admin/models/{id}/status` | sync | merged catalog + worker view |
| `GET /v1/admin/workers` | sync | list workers + pid/uptime/restarts |
| `POST /v1/admin/workers/{port}/restart` | sync | SIGTERM; monitor handles bringup |
| `GET /v1/admin/memory` | sync | psutil + pynvml + per-model breakdown (incl. live `refcount` when a director is bound) |
| `GET /v1/admin/jobs/{job_id}` | sync | one job; 404 once reaped |
| `GET /v1/admin/jobs` | sync | recent jobs newest-first |

State management:
- `SupervisorState` (in `muse.cli_impl.supervisor`) is a module-level
  singleton holding the live worker list, device flag, and an RLock.
  `run_supervisor` registers it on boot; admin endpoints reach it via
  `get_supervisor_state()`. The auto-restart monitor reads
  `state.workers` directly so admin mutations show up on the next tick.
- `JobStore` (in `muse.admin.jobs`) is an in-memory map of async jobs
  with 10-minute retention (lazy reap on every list call). Each
  enable/pull/probe spawns a daemon thread tracked by the JobStore;
  `get_default_store().shutdown()` joins them on gateway exit.
- One global RLock guards SupervisorState mutations; per-model locks
  are deferred until contention becomes measurable.

Token leakage rules: the configured token is never echoed in any
error message, log line, or job record. `tests/admin/test_e2e_admin_router.py
::TestAuthEnvelope::test_token_never_appears_in_error_body` is a
regression watchdog.

Admin error envelope: auth failures use the bare OpenAI envelope
`{"error": {code, message, type}}`, matching the route-level admin errors
(which return it directly via `JSONResponse`). The auth dependency
(`verify_admin_token`) raises `HTTPException(detail={"error": ...})`,
which FastAPI's default handler would double-wrap as `{"detail":
{"error": ...}}`; `muse.admin.errors.install_admin_error_handler`
(installed by `build_gateway` right after mounting the admin router)
unwraps any HTTPException whose `detail` is already OpenAI-shaped and
delegates to FastAPI's default for everything else (so plain-string
details elsewhere keep their `{"detail": ...}` shape). Safe-degrading: an
includer that forgets the handler still gets a valid (if double-wrapped)
error, never a 500. `AdminClient` parses both shapes (v0.47.4).

`muse.admin.client.AdminClient` wraps every endpoint. `wait(job_id)`
polls `/jobs/{id}` until done/failed for "fire and block" usage. The
CLI's `muse models enable/disable` falls back to AdminClient when
`MUSE_ADMIN_TOKEN` is set and the supervisor is reachable, otherwise
to the legacy catalog-only mutation with a warning.

## Observability

`muse.observability/` (v0.51.x, on the `feature/observability-dashboard`
branch) is a self-contained telemetry package: a sparse single-table
event model, a fire-and-forget recorder, a per-worker log ring buffer,
a periodic resource sampler, and a `/dashboard` HTML page backed by a
small set of gated JSON/SSE endpoints. It has no dependency on any
other modality; the rest of muse depends on it only through the thin
`record(...)` call.

Import-hygiene contract: `muse.observability/__init__.py` re-exports
`EVENT_COLUMNS`, `event_to_row`, `TelemetryStore`, `TelemetryRecorder`,
`record`, `get_recorder`, `init_recorder`, `reset_recorder`, `Sampler`,
and `LogHub` EAGERLY (all stdlib-only), but re-exports
`build_dashboard_router`, `DASHBOARD_HTML`, `require_dashboard_auth`,
and `check_dashboard_token` LAZILY via a PEP 562 module `__getattr__`,
because `dashboard.py` and `dashboard_auth.py` import fastapi and
sse_starlette at module top. The director, gateway, and supervisor all
import `muse.observability.recorder` from the hot request path, and
importing any submodule runs `__init__.py` first -- an eager dashboard
import there would drag fastapi into `muse --help` / `muse pull`,
violating the project's deferred-imports convention. Regression guard:
`tests/observability/test_public_api.py` re-imports `recorder` in a
clean subprocess and asserts `"fastapi" not in sys.modules`.

**Event model.** One sparse SQLite table, `events`
(`muse.observability.store.TelemetryStore`), with columns
`EVENT_COLUMNS` (`muse.observability.events`): `ts`, `type`, plus every
field any event type might carry (`model_id`, `pool`, `gb`,
`latency_ms`, `status`, `reason`, `cold_load_seconds`, `stream`,
`free_vram_gb`, `free_ram_gb`, `gpu_used_gb`, `loaded_count`,
`in_flight_count`, `modality`) -- unset fields are `NULL`, not one
table per event type. `event_to_row(type, ts, **fields)` builds the
full row and raises `ValueError` on an unknown field name (a typo'd
kwarg fails loud instead of silently writing `NULL` to the wrong
column). Four event types are recorded today: `model_load` and
`model_evict` (from `LoadDirector.acquire` / eviction, `pool` +
`gb`/`cold_load_seconds`), `request` (from the gateway's forwarding
path, `latency_ms`/`status`/`stream`/`modality`), and `sample` (from
`Sampler`, a periodic snapshot of `free_vram_gb`/`free_ram_gb`/
`loaded_count`/`in_flight_count`). `TelemetryStore.series(metric,
since_ts, bucket_seconds)` buckets rows into a fixed set of named
metrics (`request_rate`, `latency`, `vram`, `ram`, `load_evict`); v1
latency is avg+max per bucket, not exact percentiles.

**Fire-and-forget recording.** `muse.observability.recorder.record(type,
**fields)` is meant to be called from every hot path (director,
gateway) and must never block or raise. `TelemetryRecorder` enqueues
onto a bounded `queue.Queue` (`max_queue=10000`) and a daemon flush
thread batches inserts into the store every `flush_interval` (0.5s
default); when the queue is full the event is dropped and
`recorder.dropped` increments rather than blocking the caller --
`/v1/telemetry/summary` surfaces the running drop count so an operator
can tell if the recorder is falling behind. Every call site
(`load_director.py`, `gateway.py`) wraps `record(...)` in its own
try/except so a telemetry regression can never break a real model load,
eviction, or request. `init_recorder(store, enabled=...)` /
`get_recorder()` / `reset_recorder()` manage a module-level singleton;
`enabled=False` (or `telemetry.enabled: false`) swaps in a
`_NoopRecorder` so a disabled deployment pays no queue/thread cost at
all.

**Log capture.** Each worker's stdout is piped into a per-model
`LogHub` (`muse.observability.logs`) ring buffer, byte-bounded by
`telemetry.log_buffer_kb` (KB, not lines), when telemetry is enabled;
`spawn_worker(..., log_hub=state.log_hub)` in the supervisor starts a
daemon reader thread (`_pump_worker_logs`) that tees each line to both
the aggregate supervisor stdout and `hub.append(model_id, line)`. When
telemetry is disabled `log_hub` stays `None` and workers spawn exactly
as before (no reader thread, no behavior change). `LogHub` also fans
out live lines to subscriber queues for the SSE tail; `snapshot(id)`
returns the buffered history for a late-connecting client.

**`/dashboard` + telemetry endpoints.** `build_dashboard_router(state)`
(lazily imported, see above) mounts:
- `GET /dashboard` -- a single self-contained HTML/CSS/JS page
  (`DASHBOARD_HTML`), UN-GATED so it always loads (even with no token
  configured yet) and prompts the browser for a token before hitting
  any gated endpoint, storing it in `sessionStorage`.
- `GET /v1/telemetry/summary` -- gated JSON: currently loaded models
  (`model_id`/`pool`/`gb`/`last_used`), `in_flight` count,
  `dropped_events`, and a federation-forward `node` id (`state.node_url`
  or `state.node_id` if set, else `socket.gethostname()`) so a future
  multi-node aggregator can tell which box a summary came from.
- `GET /v1/telemetry/series?metric=...&window=...` -- gated JSON time
  series from `TelemetryStore.series`; unknown `metric` returns 400
  `invalid_metric`.
- `POST /v1/telemetry/logs-ticket` -- gated (header): mints a
  short-lived ticket (`muse.observability.log_tickets.LogTicketStore`)
  the dashboard exchanges for SSE access without putting the admin
  token in a URL.
- `GET /v1/telemetry/logs/{model_id}` -- SSE tail, authenticated INSIDE
  the handler (not via the `require_dashboard_auth` dependency, since
  `EventSource` sends no custom headers): a valid `?ticket=` OR an
  `Authorization: Bearer <token>` header (for curl-style clients)
  is accepted; `?access_token=<token>` is NOT. Once authenticated, drains
  `hub.snapshot(model_id)` then polls `hub.subscribe(model_id)` every
  250ms until the client disconnects, unsubscribing in a `finally` so a
  disconnect/cancel/exception never leaks a subscriber queue.

**Auth.** `muse.observability.dashboard_auth.check_dashboard_token`
mirrors the admin API's closed-by-default policy: with no `admin.token`
configured, every gated endpoint 503s `dashboard_closed` (same token as
`MUSE_ADMIN_TOKEN`/admin API -- there is no separate dashboard secret).
The admin token is HEADER-ONLY everywhere: a request needs it as
`Authorization: Bearer <token>`; there is no `?access_token=` query-param
fallback. The one place a header alone cannot work is the SSE log
stream (`EventSource` cannot set custom headers), which instead
authenticates with a short-lived, random ticket
(`telemetry.log_ticket_ttl_seconds`, default 60s) minted from the
header-gated `POST /v1/telemetry/logs-ticket`. The ticket is reusable
within its TTL (not single-use, so `EventSource` auto-reconnect keeps
working) and unscoped (it authorizes the logs SSE surface generally,
matching the admin token's own all-or-nothing scope). So the admin
token itself never rides a URL, and a leaked ticket in an access log
is useless once its TTL elapses. Comparison is `secrets.compare_digest`;
the token is never echoed in an error message or log line.

**Config.** Five `telemetry.*` settings (see "Configuration" above and
`docs/CONFIG.md`): `telemetry.enabled` (`MUSE_TELEMETRY_ENABLED`,
default `true`), `telemetry.retention_days` (`MUSE_TELEMETRY_RETENTION_DAYS`,
default `7`), `telemetry.log_buffer_kb` (`MUSE_TELEMETRY_LOG_BUFFER_KB`,
default `64`), `telemetry.sample_interval_seconds`
(`MUSE_TELEMETRY_SAMPLE_INTERVAL_SECONDS`, default `10.0`),
`telemetry.log_ticket_ttl_seconds`
(`MUSE_TELEMETRY_LOG_TICKET_TTL_SECONDS`, default `60.0`). The
supervisor wires the store, recorder, sampler, and log hub together
once at boot (`_init_telemetry`, gated on `telemetry.enabled`) and tears
them down symmetrically on shutdown (`sampler.stop()` joins the sampler
thread; `store.close()` closes the sqlite connection).

### Known limitations (v1)

- **Idle-sweep eviction is not instrumented.** `IdleSweeper` (the
  DEFAULT reclaim path for an untouched model past its idle timeout)
  does not currently call `record("model_evict", ...)`; only
  `LoadDirector`'s memory-pressure eviction path does. The `load_evict`
  series therefore undercounts real evictions on a lightly-loaded box
  where idle timeout, not memory pressure, is doing most of the
  reclaiming.
- **Pool attribution is best-effort.** `summary.loaded[].pool` and the
  `model_evict` event's `pool` field both read `getattr(entry, "pool",
  None)`, but `LoadEntry` has no `pool`/`device` field today, so this is
  currently always `None` in practice. A future `LoadEntry.pool` field
  would need to land before this attribution becomes real.
- **The live log tail can miss a single line emitted in the brief window
  between the initial history snapshot and the SSE subscription.**
  `_stream_model_logs` drains `hub.snapshot(model_id)` (buffered history)
  and then calls `hub.subscribe(model_id)`; a line appended to the hub in
  that narrow window lands in neither the snapshot nor the subscription
  queue and is dropped. Buffered history and every subsequent line are
  unaffected. Not fixed: subscribing before draining the snapshot would
  trade this rare lost line for rare duplicate lines, which is a
  different (not obviously better) trade-off.

## Federation

`muse federate` (branch `feature/federation-coordinator`, see
`docs/superpowers/specs/2026-07-04-federation-design.md` for the full
design) runs a thin coordinator process that fronts a static list of
unmodified muse `serve` nodes so a client sees "one server with all the
GPUs" instead of N separate boxes. This is the second sub-project of
the federation arc; it consumes the per-node observability surface
(`/v1/models`, `/v1/telemetry/summary`) shipped in v0.53.0.

```
                 clients (OpenAI SDK, curl, ...)
                          |
                 [ muse federate ]  :8100        <- new coordinator
                  /       |       \
             node A     node B     node C         <- unmodified muse servers
```

Nodes need NO code changes: they are plain `muse serve` instances. The
coordinator lives in `muse.federation` (node-membership model, async
poll-and-cache registry, pure model-locality router) plus
`muse.cli_impl.federation` (the FastAPI app + `run_coordinator`,
structurally a sibling of the existing single-host gateway in
`muse.cli_impl.gateway`, reusing its `_forward` + SSE-relay machinery
and `extract_model_from_request`).

**Node membership (static, v1).** Nodes come from two merged sources,
CLI entries winning on url collision:
- `--node http://host:8000` or `--node name=http://host:8000` (repeatable).
- A yaml file, `nodes: [{url, name?, token?}, ...]`. Default location is
  `<catalog_dir>/federation.yaml` when it exists and neither `--config`
  nor `federation.config_file` names an explicit path. The optional
  per-node `token` is used ONLY by the coordinator's own outbound poll
  to read that node's gated `/v1/telemetry/summary`; it is never
  exposed to clients. Dynamic membership (registration, gossip,
  health-based ejection) is deferred to v2.

**Node-state refresh.** `muse.federation.registry.NodeRegistry` polls
every node concurrently on a background interval
(`federation.refresh_interval_seconds`, default 3.0s) and caches a
`NodeState` snapshot per node: `GET <node>/v1/models` (reachability +
per-model `loaded`), `GET <node>/health`, and (only if that node has a
configured token) `GET <node>/v1/telemetry/summary` for `in_flight`.
Route handlers read the cache via `.snapshot()`, so routing is a fast
local lookup, never a per-request fan-out. A node whose poll fails is
marked unreachable and skipped by the router until it recovers
(per-node failure isolation: one bad node never aborts the refresh for
the others).

**Routing policy (model-locality, v1).** `muse.federation.router.select_node`
is a pure function over the cached snapshot:
1. Candidates = reachable nodes whose `/v1/models` lists the requested
   model.
2. Prefer loaded: among candidates, those with the model currently
   `loaded` win over those that would have to cold-load it.
3. Tie-break: lowest `in_flight` (from telemetry, if a token is
   configured for that node), else round-robin over the tied set.
4. No candidate -> 404 `model_not_available` (OpenAI-shape).

Deferred to v2: hardware-aware routing (route a big model to the node
whose GPU actually fits it), true queue-depth/latency-aware routing,
sticky sessions, an aggregated cluster dashboard (v1 links out to each
node's own `/dashboard` via `/v1/federation/nodes`), and
coordinator-side inference auth (v1's inference routes are open,
mirroring an unmodified node's own gateway; per-node tokens are used
only for the coordinator's outbound telemetry poll, never exposed to
clients).

**Request forwarding + failover.** The catch-all proxy extracts `model`
from the request body/query (the same logic the single-host gateway
uses), asks `select_node`, and forwards via the shared `_forward`
helper (buffered + SSE relay, unchanged). A forward that fails at
connect/timeout (never mid-stream -- once a `StreamingResponse` starts,
the relay is on its own) retries ONCE against a different candidate for
the same model; if no other candidate exists, or the retry also fails,
the coordinator returns 502 `no_node_available`.

**Aggregated read endpoints**, mounted before the catch-all so they are
never shadowed:
- `GET /v1/models` -- OpenAI-shape union of model ids across reachable
  nodes, each annotated with `loaded` (true if loaded on ANY node) and
  `nodes` (which node names carry it).
- `GET /health` -- `"ok"` if at least one node is reachable, else
  `"degraded"`; per-node reachability/model-count/in_flight breakdown
  in the body.
- `GET /v1/federation/nodes` -- operator view: each node's url/name,
  reachable state, loaded model list, in_flight (if known), and
  last-poll age in seconds.

**Config** (three settings under the `federation.*` group in
`muse.core.config.SETTINGS`; see "Configuration" above and
`docs/CONFIG.md`):
- `federation.refresh_interval_seconds` (`MUSE_FEDERATION_REFRESH_INTERVAL_SECONDS`,
  default `3.0`) -- seconds between background polls of each node.
- `federation.forward_timeout_seconds` (`MUSE_FEDERATION_FORWARD_TIMEOUT_SECONDS`,
  default `300.0`) -- per-request timeout when forwarding to a node
  (long, since generation requests can run for minutes).
- `federation.config_file` (`MUSE_FEDERATION_CONFIG`, default unset) --
  explicit path to the node-list yaml, overriding the
  `<catalog_dir>/federation.yaml` default-if-exists lookup.

**CLI:** `muse federate --port 8100 --node http://192.168.0.204:8000
--node http://192.168.0.50:8000` (or `--config path/to/federation.yaml`).
`run_coordinator` (`muse.cli_impl.federation`) resolves the node list
(`--config` > `federation.config_file` > `<catalog_dir>/federation.yaml`
if it exists), and refuses to start a server -- exiting 2 with a clear
stderr message -- if the merged node list is empty; a coordinator with
zero nodes would otherwise boot healthy and 404 every request, a worse
failure mode than refusing outright. Once nodes resolve, it builds a
`NodeRegistry` + the coordinator app and blocks on
`muse.cli_impl.serve_util.run_uvicorn` (the same bounded-graceful-
shutdown uvicorn wrapper `muse serve` uses). Registry lifecycle is
wired through the coordinator app's ASGI lifespan (an
`asynccontextmanager` passed to `FastAPI(..., lifespan=...)` inside
`build_coordinator`): `registry.start()` runs on startup, launching the
background refresh loop on the SERVER'S OWN running event loop (which
`NodeRegistry.start`'s `asyncio.ensure_future` requires), and
`await registry.aclose()` runs on shutdown. This only fires when the
ASGI lifespan protocol actually executes (a real uvicorn run, or a
`with TestClient(app) as c:` block); a bare `TestClient(app).get(...)`
(the style the existing route tests in `tests/cli_impl/test_federation_app.py`
use) never triggers lifespan, so wiring it into `build_coordinator`
did not disturb those tests' plain `.snapshot()`-only fakes.

Client usage is unchanged OpenAI-compat, just pointed at the
coordinator: `OpenAI(base_url="http://coordinator:8100/v1",
api_key="not-used")`.

## MCP server (Using muse from Claude Desktop)

`muse mcp` runs an MCP (Model Context Protocol) server that exposes
muse's capabilities to LLM clients (Claude Desktop, Cursor, etc.) as
31 structured tools: 11 admin tools (wrap `/v1/admin/*`) and 20
inference tools (one per generation route).

The package layout:

```
src/muse/mcp/
  server.py     MCPServer wrapping mcp.server.lowlevel.Server
  client.py     MuseClient aggregating httpx calls + AdminClient delegation
  binary_io.py  tri-modal binary input resolution (b64 / url / path) + output packers
  tools/
    admin.py             11 admin tools
    inference_text.py    chat / summarize / rerank / classify / embed_text / translate
    inference_image.py   generate / edit / vary / upscale / segment / animation / embed_image
    inference_audio.py   speak / transcribe / music / sfx / embed_audio
    inference_video.py   video generation
src/muse/cli_impl/mcp_server.py  CLI entry (boots stdio or HTTP+SSE transport)
```

Two transport modes:
- **Stdio (default).** For desktop apps that spawn the MCP server as a
  child process. `muse mcp` reads JSON-RPC framing from stdin and
  writes to stdout.
- **HTTP+SSE.** `muse mcp --http --port 8088` runs a Starlette app
  with an `/mcp` mount via `StreamableHTTPSessionManager`. Useful
  for remote / web embedders.

Filter mode pins the tool surface:
- `--filter all` (default): 31 tools.
- `--filter admin`: 11 tools, only useful with `MUSE_ADMIN_TOKEN`.
- `--filter inference`: 20 tools, no admin-token needed.

Auth: `--admin-token` (or `$MUSE_ADMIN_TOKEN`) is forwarded as the
bearer token for admin tool calls. Inference tools don't need a token.

Long-running ops (pull, probe, enable) return a `job_id` and a
`poll_with` hint. The LLM polls `muse_get_jobs` to track progress.

Claude Desktop config example
(`~/Library/Application Support/Claude/claude_desktop_config.json`
on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "muse": {
      "command": "muse",
      "args": ["mcp"],
      "env": {
        "MUSE_SERVER": "http://localhost:8000",
        "MUSE_ADMIN_TOKEN": "your-admin-token-here"
      }
    }
  }
}
```

Inference-only setup (no admin tools, no token needed):

```json
{
  "mcpServers": {
    "muse-inference": {
      "command": "muse",
      "args": ["mcp", "--filter", "inference"],
      "env": {"MUSE_SERVER": "http://localhost:8000"}
    }
  }
}
```

After saving the config, restart Claude Desktop. Tools appear in the
"Search and tools" menu. Test with prompts like:
- "Generate an image of a sunset over a mountain lake."
- "List all enabled models in muse."
- "Transcribe this audio file: /path/to/audio.wav"

Binary I/O conventions:
- Inputs accept three mutually-exclusive fields per slot:
  `<name>_b64` (base64), `<name>_url` (data: or http URL), or
  `<name>_path` (local filesystem). Tool descriptions tell the LLM
  which form is appropriate.
- Outputs return MCP `ImageContent` (image bytes) or `AudioContent`
  (audio bytes) plus a `TextContent` summary. Video bytes return
  inside the JSON envelope (no SDK VideoContent type yet).

## Development commands

```bash
# Install (dev)
pip install -e ".[dev,server,audio,images]"

# Full dev stack (CPU torch) - what CI installs and what preflight expects
pip install -e ".[dev,server,audio,images,embeddings]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Gate the fast lane on a non-drifted venv (used in the release ritual)
python scripts/preflight.py

# Fast lane: unit tests only (excludes slow e2e + integration)
pytest tests/ -q -m "not slow"

# Full lane: fast + slow e2e supervisor test (in-process)
pytest tests/ -q

# Integration tests (opt-in, hit a real muse server)
MUSE_REMOTE_SERVER=http://192.168.0.225:8000 pytest tests/integration/
# Override which chat model the integration suite targets:
MUSE_REMOTE_SERVER=http://192.168.0.225:8000 \
    MUSE_CHAT_MODEL_ID=qwen3.5-9b-q4 pytest tests/integration/

# One modality contract, or one bundled model
pytest tests/modalities/chat_completion/
pytest tests/models/test_kokoro_82m.py

# Single test by name
pytest tests/core/test_resolvers.py::test_register_and_get_resolver -v

# Coverage
pytest tests/ -m "not slow" --cov=muse

# CLI: admin surface (no per-modality verbs; generation is HTTP).
# Built on typer + rich since v0.39.0; `muse --help` and the per-
# subcommand help are auto-generated, color-coded, and respect
# NO_COLOR env. The list / search / refresh outputs auto-detect TTY:
# pretty rich.Table when interactive; plain text (no ANSI, no
# truncation) when piped or redirected, so `muse models list | grep`
# always sees full content. `--json` flag on list / probe / refresh
# emits machine-readable output.
muse --help                                    # top-level help (typer-styled)
muse models list                               # bundled + curated + pulled
muse models list --available --modality chat/completion
muse models list --json                        # deterministic JSON output
muse search qwen3 --modality chat/completion --max-size-gb 10
muse pull qwen3.5-4b-q4                        # curated alias
muse pull hf://unsloth/Qwen3.5-9B-GGUF@q4_k_m  # resolver URI
muse pull kokoro-82m                           # bundled bare id
muse models info <id>
muse models enable <id> / disable <id>
muse models set-device <id> cuda               # operator device pin (overrides manifest device)
muse models set-device <id> --clear            # remove the pin
muse models remove <id>
muse models refresh <id>                       # re-install muse[server,extras] into one venv
muse models refresh --all                      # all pulled venvs, alphabetical
muse models refresh --enabled                  # only enabled venvs
muse models refresh <id> --no-extras           # skip the model's pip_extras (only refresh muse[server])
muse models refresh --all --json               # machine-readable output
muse serve --device cuda
muse mcp                                       # MCP server for LLM clients (stdio mode)
muse mcp --http --port 8088                    # MCP server in HTTP+SSE mode
muse mcp --filter admin                        # only the 11 admin tools
muse mcp --filter inference                    # only the 20 inference tools

# Python clients (HTTP)
python - <<'PY'
from muse.modalities.audio_speech import SpeechClient
from muse.modalities.image_generation import GenerationsClient
from muse.modalities.embedding_text import EmbeddingsClient
from muse.modalities.chat_completion import ChatClient
SpeechClient().infer("hello")                         # WAV bytes; MUSE_SERVER env sets base URL
GenerationsClient().generate("a cat")                 # list[bytes] of PNGs
EmbeddingsClient().embed(["alpha", "beta"])           # list[list[float]]
ChatClient().chat(model="qwen3.5-4b-q4", messages=[{"role": "user", "content": "hi"}])
# or via OpenAI SDK (muse is wire-compatible):
#   OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
PY
```

## Project-specific conventions

- **Deferred imports:** `src/muse/__init__.py` and `src/muse/cli.py` MUST NOT
  import heavy libs (torch, diffusers, transformers). Each backend imports
  its heavy deps at module top-level inside a try/except so import of the
  backend module succeeds even without the deps. Tests mock at the module
  path where the library is imported. `muse --help` and `muse pull` work
  without any ML deps installed; pulling a model installs them on demand.
- **FakeModel-pattern tests:** Server and router tests use plain classes that
  satisfy the modality protocol, no real weights. Backend tests also mock
  heavy libs (see `tests/models/test_sd_turbo.py`).
- **Registry is a singleton at module level** (`muse.core.registry.registry`),
  but tests create their own `ModalityRegistry()` instances to avoid coupling.
- **Audio is float32 in `[-1, 1]`** at the protocol boundary; codec converts
  to int16 PCM at output. Scaling uses `* 32768` + `np.clip` to reach full
  int16 range `[-32768, 32767]`.
- **Images are `Any`** at the protocol boundary; codec normalizes PIL / numpy /
  torch to PIL before encoding.
- **OpenAI error envelopes:** Use `raise ModelNotFoundError(model_id, modality)`
  from `muse.core.errors`, not `HTTPException(detail=...)`. The former gives
  `{"error":{"code","message","type"}}`; the latter gives `{"detail":...}`.
- **Streaming uses producer thread + `asyncio.Queue`**, not `list(generator)`.
  Synthesis chunks must dispatch as they're produced, not after full generation.
- **Env vars are registry settings (v0.52.0+).** Every `MUSE_*` knob is a
  row in the `muse.core.config` registry, so it can equivalently be set as
  an env var, in `~/.muse/config.yaml`, or via `muse config set`. See the
  "Configuration" section and docs/CONFIG.md for the full list. Common ones:
  `MUSE_SERVER` / `client.server_url` (client base URL), `MUSE_CATALOG_DIR`
  / `paths.catalog_dir` (catalog + config.yaml location, defaults
  `~/.muse/`), `MUSE_HOME` / `paths.home` (voices dir base),
  `MUSE_ALLOW_PRIVATE_FETCH` / `fetch.allow_private` (opt-in escape hatch
  for the SSRF guard on `_fetch_http_url`; default off; needed only when
  operators on a trusted network want `image` URLs to reach internal
  services), `MUSE_IMAGE_INPUT_MAX_BYTES` / `limits.image_input_max_bytes`
  (per-request size cap for image uploads and data URLs; default 10485760 /
  10MB; read per-call so changes take effect without a server restart).
- **Auto-restart is always on.** No --no-autorestart flag in this iteration. Workers that can't stay up through 10 restart attempts are marked dead; manual restart via `Ctrl+C` + `muse serve` is required to reset the counter.
- **Enable/disable is catalog state; loaded/unloaded is runtime state (v0.40.0+).** The two are decoupled. `enabled: true` means "in service; allowed to lazy-load." `loaded` (visible via `/v1/models` and `muse models list`'s `enabled_loaded` glyph) means "currently resident on a worker." LRU eviction unloads without disabling; `muse models warmup` loads without bumping refcount. Operator-driven `muse models enable/disable` flips the catalog bit and (when the supervisor is running with `MUSE_ADMIN_TOKEN`) syncs runtime state via `enable_model` / `disable_model`; lazy-load and eviction paths use `load_model_into_worker` / `unload_model_from_worker` which skip the catalog flip.
- **Tool-use asymmetry (known landmine).** llama-cpp-python's `chatml-function-calling` handler parses tool calls *out* of a model's response into structured `tool_calls`, but does NOT format tool *result* messages (role=`tool`) back to the model in a way Qwen's chat template always recognizes. The muse-side contract is correct (verified by `tests/modalities/chat_completion/test_routes_messages_passthrough.py`); the asymmetry is upstream. Larger models (Qwen3.5-9B+) tolerate it in context; smaller models (Qwen3.5-4B) often ignore the tool result and give a generic "I don't have access to tools" reply. Tracked by `tests/integration/test_remote_tools.py::test_observe_tool_result_content_influences_next_response` (xfail-style watchdog). Upstream: [abetlen/llama-cpp-python#2063](https://github.com/abetlen/llama-cpp-python/issues/2063).
- **The `model` field in chat responses is the catalog id**, not the GGUF filesystem path. `LlamaCppModel._dict_to_chat_result` and `_dict_to_chat_chunk` override `response["model"]` with the muse catalog id (not the `resp.get("model") or fallback` pattern that lets llama-cpp's internal `model_path` win). Applies to both non-streaming responses and every streaming chunk.
- **CLI is typer + rich (v0.39.0+).** Per-subcommand parameter binding lives in `src/muse/cli.py` as typer command functions; the heavy logic lives in `src/muse/cli_impl/<command>.py`. New subcommands add a `@app.command(...)` plus a sibling `cli_impl/*.py` module; do not put logic in `cli.py`. Long-form output that goes to a TTY is rendered via `rich.Table` from `cli_impl/console.py`'s shared `Console`; non-TTY output (subprocess, pipe, redirect) is plain text with no ANSI and no truncation, so `muse models list | grep` always sees full content. Status encoding for `models list` and `refresh --all` uses single colored glyphs (`STATUS_STYLE` in `cli_impl/console.py`): `●` enabled, `○` disabled, `★` recommended, `·` available; `✓` ok, `✗` failed for refresh outcomes. `--json` is the canonical machine-readable output across `list` / `probe` / `refresh`.
- **CLI exit codes propagate through `main` (v0.48.0).** Commands signal errors with `raise typer.Exit(<code>)`. `muse.cli:main` runs the app with `standalone_mode=False`, where click handles `typer.Exit` itself and *returns* the code as `app()`'s value rather than raising; `main` now returns that int (None -> 0) so the shipped `muse` binary actually exits nonzero on error. Before v0.48.0 `main` discarded the return and always exited 0; the `python -m muse.cli` subprocess tests hit `app()` in standalone mode (correct exit), so they never caught it. New `main()`-level exit-code regressions belong in `tests/cli_impl/test_set_device_cli.py`.
- **CLI runtime state reads the PUBLIC `/v1/models`, not admin endpoints (v0.47.4).** `muse models list`'s `enabled_loaded` glyph and `muse models info`'s header + worker-status block derive "is this model loaded right now?" from the public `GET /v1/models` `loaded` flag (no `MUSE_ADMIN_TOKEN` required). The shared `muse.cli_impl.runtime_state` module (`fetch_public_models` + `loaded_ids`) centralizes the fetch, failure policy (None = unreachable; both `list` and the `info` "loaded?" predicate treat it identically via `loaded_ids`), and httpx-log quieting. `muse models info` still prefers the admin API first for rich worker pid/uptime/restart detail and falls back to the public loaded state, rendering `loaded (set MUSE_ADMIN_TOKEN for worker pid / uptime / restarts)` instead of faking detail or falsely claiming the supervisor is unreachable. Before v0.47.4 these read admin-only endpoints, so anyone without a token saw every model as `enabled_unloaded` and "supervisor unreachable" even against a live server.
- **Shared image preprocessing** (`muse.core.image_preprocessing`, v0.42.1+): four-tier dispatch ladder for image-side preprocessor selection. Public API: `read_encoder_hints(src)` (read encoder hyperparams from config.json), `DerivedImageProcessor` (synthesize a minimal preprocessor from explicit num_channels / image_size / image_mean / image_std), `build_image_processor(src, *, overrides, model_id)` (the orchestrator: override-first, then AutoImageProcessor, then encoder-hints-derived, then fail loud), and `ImageProcessorError` (structured exception pointing at the override hatch). Manifest hatch: `capabilities.image_processor_overrides: {num_channels, image_size, image_mean?, image_std?}` lets curated entries declare ground-truth preprocessing when AutoImageProcessor's sniff would be wrong (TexTeller is the canonical example: 1-channel grayscale at 448x448). Tier 3 (ViT defaults) was dropped in v0.42.1 because it produced silently wrong output for the very class of repos this fallback targets; failures now raise `ImageProcessorError` pointing at the override hatch. Currently consumed by `image_ocr`; other modalities can adopt as needed.

## Continuous integration

Two GitHub Actions workflows run on every push to `main` and every PR:

- `.github/workflows/tests.yml` runs the fast lane (`pytest -m "not slow"`)
  in a single Python-3.11 job with the full CPU stack
  (`pip install -e ".[dev,server,audio,images,embeddings]"` against the
  `download.pytorch.org/whl/cpu` index). Dependencies float on latest (no
  lockfile), so an upstream breaking release surfaces here on the next push
  rather than shipping silently. Model tests are mocked, so no weights are
  downloaded.
- `.github/workflows/fresh-venv-smoke.yml` (below) validates the per-model
  fresh-venv install/load path.

Locally, run the fast lane through the preflight guard so a drifted venv
fails loudly instead of running a partial suite:

    python scripts/preflight.py            # check deps, then run the lane
    python scripts/preflight.py --check-only

## Fresh-venv smoke test (CI)

Bundled scripts under `src/muse/models/` declare `pip_extras` covering the
deps the runtime source-imports. The dev environment installs broad
extras (`muse[dev,server,audio,images,embeddings]`), so transitive deps
that `from_pretrained` (or sentence-transformers, or diffusers) ALSO
imports at load time happen to be present. Per-model venvs created via
`muse pull <id>` install only `muse[server]` plus the model's declared
`pip_extras`; transitive holes show up as ImportError when the worker
tries to load the model.

The v0.30.0 audit (#110, `tests/models/test_pip_extras_audit.py`)
catches direct-import gaps via AST scan but cannot see transitive
imports that `from_pretrained` triggers. v0.32.0 closes that gap with
a CI workflow:

- `.github/workflows/fresh-venv-smoke.yml` runs on every push to main
  and every PR. Matrix-tests five lightweight models (`kokoro-82m`,
  `dinov2-small`, `bart-large-cnn`, `bge-reranker-v2-m3`,
  `mert-v1-95m`).
- Each job creates a fresh venv, installs only what `muse pull` would
  install, then runs the in-venv probe worker (`muse _probe_worker
  --no-inference`) to verify load. Failure surfaces a structured label
  like `kokoro-82m: FAIL (missing dep: librosa)`.
- Heavy / GPU-only models (sd-turbo, animatediff, stable-audio, wan,
  large LLMs) are deferred until paid runner budget allows.

Local repro: `python scripts/smoke_fresh_venv.py --model_id <id>`.
Use `--json` for machine-readable output.

When you add a new bundled script and the smoke matrix should cover it,
add the id to the matrix in `.github/workflows/fresh-venv-smoke.yml`.
Lightweight models (under ~1 GB on disk, CPU-friendly) are good
candidates; heavier models need GPU runners and are out of scope for
the free-tier CI matrix.

## Benchmarks

`scripts/bench/` holds the reusable performance harness (v0.57.0):
`bench_llm.py` (muse-vs-ollama head-to-head: generation tok/s, streaming
TTFT/inter-token, long-prompt eval, multi-turn prefix-reuse detection,
worker-direct vs gateway split) and `bench_modalities.py` (per-modality
hot latencies against a live server). Reports land in `docs/benchmarks/`.
Protocol lesson from the 2026-07-08 baseline: on thermally-scaled boxes,
only alternating same-window comparisons are trustworthy; cross-window
medians can differ 3x. LlamaCppModel accepts optional perf kwargs
(n_threads, n_threads_batch, n_batch, flash_attn, use_mlock, type_k,
type_v) via manifest capabilities; defaults match llama-cpp-python (and
ollama's thread choice) and shipped untuned because the baseline showed
muse already competitive (non-streaming faster than ollama; streaming
comparison inconclusive on the noisy laptop test box -- muse 5.8-14.3 vs
ollama 7.6-10.9 tok/s in the committed alternating run, with muse's
within-run variance exceeding the delta).

## Memory accounting

Three sources of truth, in order of fidelity:

1. **`muse models probe <id>`** (most honest). Loads the model in
   isolation in its per-model venv, runs a representative inference
   (per-modality default shape from `PROBE_DEFAULTS`), captures peak
   VRAM/RAM via `torch.cuda.max_memory_allocated()` on GPU or process
   RSS on CPU. Persists per-device measurement to `~/.muse/catalog.json`
   under `measurements.<device>`. Default runs inference;
   `--no-inference` is a faster load-only mode that undersells peak.
2. **`capabilities.memory_gb` annotation** (peak-inference estimate).
   Hand-set per-model from architecture knowledge, conservative.
   Used by `muse models list` until probe measurements exist; shown
   with a `~` prefix.
3. **No data**: `-` in the list. Run probe to populate.

`muse models list` picks the most honest available number per row,
tagged GPU or CPU based on `capabilities.device`. The footer aggregates
GPU and CPU separately across enabled models.

`muse models info <id>` shows annotation and probe measurement
side-by-side, including the inference shape that produced the peak
and the date probed.

**Memory is a function of input shape, not a single number.** Whisper
at 30s audio uses different VRAM than at 5min. SDXL-Turbo at 512^2 uses
~half as much as at 1024^2. AnimateDiff at 8 frames uses roughly half
as much as at 16 frames. The `memory_gb` annotation reflects a
typical-shape peak. Probe measures the actual default shape. Future
versions may add `--shape preset=small|medium|large` sweeps to map the
full curve.

Each modality declares its representative inference via a
`PROBE_DEFAULTS = {"shape": ..., "call": lambda m: ...}` dict in its
`__init__.py`. The probe worker imports the modality at run time and
calls the shape-default lambda against the loaded backend.

## Adding a new model (the common case)

Three paths, in order of least-to-most effort:

**1. Curated alias (easiest; a muse-blessed shortcut).** Edit
`src/muse/curated.yaml` to add a friendly id that points at an HF URI
or a bundled script. Users then `muse pull <id>` (no `hf://` prefix
needed). The curated id is preserved as the catalog key even when the
URI would synthesize a different one. See existing entries for shape.

**2. Resolver URI (good for GGUF + sentence-transformers).** No script
needed; let the HF resolver synthesize a manifest:

```bash
muse pull hf://unsloth/Qwen3.5-9B-GGUF@q4_k_m
muse pull hf://sentence-transformers/all-MiniLM-L6-v2
```

The HF resolver sniffs the repo (`.gguf` -> chat/completion via
`LlamaCppModel`; sentence-transformers tag -> embedding/text via
`SentenceTransformerModel`), persists a synthesized manifest in
`~/.muse/catalog.json`. For chat/completion, the resolver also
consults `src/muse/chat_formats.yaml` for the right llama-cpp
`chat_format` string. `muse search <query> --modality chat/completion
--max-size-gb N` helps discover candidates. See `docs/RESOLVERS.md`.

**3. Script path (for one-offs with custom code).** Models that need
non-uniform behavior (NV-Embed's custom `encode` method, Soprano's
Narro engine) get a hand-written script:

1. Write a `.py` file with a `MANIFEST` dict + `Model` class (see
   `docs/MODEL_SCRIPTS.md` for the full schema).
2. Drop it in `~/.muse/models/` or any dir pointed to by `$MUSE_MODELS_DIR`.
3. `muse pull <model_id>` to install deps + download weights.
4. `muse serve` picks it up.

Bundled model scripts live in `src/muse/models/<id>.py`. Adding a
bundled model requires no catalog edits, no registry changes, and no
worker changes: discovery just finds it.

Collision precedence: bundled scripts > resolver-pulled (persisted
manifest). A user pulling `hf://malicious/fake` that claims an
existing bundled id gets shadowed by the bundled script.

## Adding a new modality (rare)

Modalities define wire contracts; most users should NOT need to add one.
If you do:

1. Create `src/muse/modalities/<mime_name>/` (e.g. `audio_transcriptions/`
   for MODALITY `"audio/transcription"`). Use underscores in the dir
   name; the MIME tag has the slash.
2. Write `protocol.py` (Protocol + Result dataclass), `routes.py`
   (with `build_router(registry) -> APIRouter`), `client.py` (HTTP
   client), and `codec.py` (encoding for this modality's output).
3. Export from `__init__.py`: `MODALITY = "audio/transcription"` (the
   MIME string) and `build_router` (the router factory). Also re-export
   the Protocol + Result for user imports.
4. (HF support) write `hf.py` exporting `HF_PLUGIN: dict` (sniff/
   resolve/search + metadata). See `docs/HF_PLUGINS.md` for the
   contract and authoring rules. Loaded via single-file import,
   so no relative imports.
5. Add bundled model scripts under `src/muse/models/` (or rely on
   the resolver alone for uniform-shape modalities).
6. Inside the new runtime / bundled script, import device + dtype
   utilities from `muse.core.runtime_helpers` rather than rolling
   your own. The four utilities (`select_device`, `dtype_for_name`,
   `set_inference_mode`, `LoadTimer`) cover the common needs and the
   meta-test (`tests/core/test_runtime_helpers_meta.py`) AST-walks
   every runtime to flag re-implementations.
7. Add tests under `tests/modalities/<mime_name>/` (route + plugin)
   and `tests/models/test_<new_model>.py`.

No edits to `worker.py`, `catalog.py`, `registry.py`, `server.py`,
or `resolvers_hf.py` are needed: discovery handles the wiring.

No gateway changes are needed either: the gateway routes by the
`model` field in the request body and forwards to whichever worker
loaded that model. New modalities are transparent to the proxy layer.

External escape hatch: dropping a modality subpackage into
`$MUSE_MODALITIES_DIR/<mime_name>/` registers it without forking muse.
Intended for experimentation, not routine extension.

## Test organization

```
tests/
├── core/                 # resolvers, catalog, curated, chat_formats, discovery, registry, server
├── modalities/<name>/    # protocol + codec + routes + client for each modality
├── models/               # one test per bundled model script (fully mocked)
├── cli_impl/             # worker, search, supervisor (in-process e2e marked @pytest.mark.slow)
├── integration/          # opt-in; hits a real muse server via OpenAI SDK
└── test_cli.py           # subprocess-level CLI smoke
```

Fast lane is `-m "not slow"`. The one `@pytest.mark.slow` test in
`cli_impl/test_e2e_supervisor.py` spawns a real supervisor subprocess
(in-process, no network). The `tests/integration/` suite is separately
opt-in via `MUSE_REMOTE_SERVER` env var; fixtures auto-skip when the
server isn't reachable or the required model isn't loaded. The
`MUSE_CHAT_MODEL_ID` env var (default `qwen3.5-4b-q4`) lets the same
integration suite run against any chat model on the target server.

Test naming for integration tests:
- `test_protocol_*`: hard claims muse should always satisfy. Failure is a regression.
- `test_observe_*`: records what a particular model actually did. Usually xfail-style; useful as a watchdog (a passing xfail shows up as XPASS, signaling "upstream fixed something; promote to a hard assertion").
