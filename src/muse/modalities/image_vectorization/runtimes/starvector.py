"""Pinned, remote-code-free runtime for StarVector-1B im2svg."""
from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer,
    dtype_for_name,
    select_device,
    set_inference_mode,
)
from muse.modalities.image_vectorization.protocol import VectorizationResult
from muse.modalities.image_vectorization.svg import validate_static_svg


logger = logging.getLogger(__name__)

OFFICIAL_REPO = "starvector/starvector-1b-im2svg"
PINNED_REVISION = "380ab95d25a8e9ab1dc825debe238b4953ae13b9"

torch: Any = None
np: Any = None
AutoTokenizer: Any = None
StoppingCriteriaList: Any = None
StarVectorConfig: Any = None
StarVectorForCausalLM: Any = None
_IMPORT_ERROR: Exception | None = None


def _ensure_deps() -> None:
    """Import heavyweight inference dependencies only at model load."""
    global torch, np, AutoTokenizer, StoppingCriteriaList
    global StarVectorConfig, StarVectorForCausalLM, _IMPORT_ERROR
    if all((
        torch is not None,
        np is not None,
        AutoTokenizer is not None,
        StoppingCriteriaList is not None,
        StarVectorConfig is not None,
        StarVectorForCausalLM is not None,
    )):
        return
    try:
        import numpy as _np
        import torch as _torch
        from transformers import AutoTokenizer as _tokenizer
        from transformers import StoppingCriteriaList as _criteria_list
        from muse.modalities.image_vectorization.runtimes.starvector_model import (
            StarVectorConfig as _config,
            StarVectorForCausalLM as _model,
        )

        np = _np
        torch = _torch
        AutoTokenizer = _tokenizer
        StoppingCriteriaList = _criteria_list
        StarVectorConfig = _config
        StarVectorForCausalLM = _model
        _IMPORT_ERROR = None
    except Exception as exc:  # noqa: BLE001
        _IMPORT_ERROR = exc
        logger.debug("StarVector dependencies unavailable: %s", exc)


class _StopOnSuffix:
    def __init__(self, suffix: list[int]) -> None:
        self.suffix = list(suffix)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if not self.suffix or input_ids.shape[-1] < len(self.suffix):
            return False
        tail = input_ids[0, -len(self.suffix):].tolist()
        return tail == self.suffix


class StarVectorRuntime:
    """Convert raster images into static SVG with StarVector-1B."""

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str = OFFICIAL_REPO,
        local_dir: str | None = None,
        revision: str = PINNED_REVISION,
        device: str = "auto",
        dtype: str = "fp16",
        max_new_tokens: int = 4096,
        **_: Any,
    ) -> None:
        if hf_repo != OFFICIAL_REPO:
            raise RuntimeError(
                "StarVectorRuntime is exact-only and refuses unreviewed repo "
                f"{hf_repo!r}"
            )
        if revision != PINNED_REVISION:
            raise RuntimeError(
                "StarVectorRuntime requires reviewed revision "
                f"{PINNED_REVISION}; got {revision!r}"
            )
        _ensure_deps()
        if any((
            torch is None,
            np is None,
            AutoTokenizer is None,
            StarVectorConfig is None,
            StarVectorForCausalLM is None,
        )):
            detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
            raise RuntimeError(
                "StarVector dependencies are unavailable; run "
                f"`muse models refresh {model_id}`{detail}"
            )

        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        if self._device == "cpu" and dtype in ("fp16", "float16"):
            logger.warning(
                "StarVector fp16 is not reliable on CPU; loading fp32 instead"
            )
            dtype = "fp32"
        self._dtype = dtype_for_name(dtype, torch_module=torch)
        self._default_max_new_tokens = int(max_new_tokens)
        source = str(Path(local_dir)) if local_dir else hf_repo
        load_kwargs: dict[str, Any] = {
            "local_files_only": local_dir is not None,
        }
        if local_dir is None:
            load_kwargs["revision"] = revision

        with LoadTimer(f"StarVector-1B from {source}", logger):
            config_obj = StarVectorConfig.from_pretrained(source, **load_kwargs)
            _validate_config(config_obj)
            self._tokenizer = AutoTokenizer.from_pretrained(
                source, use_fast=True, **load_kwargs,
            )
            self._model = StarVectorForCausalLM.from_pretrained(
                source,
                config=config_obj,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
                **load_kwargs,
            )
            self._model = self._model.to(self._device)
        set_inference_mode(self._model)

    def vectorize(
        self,
        image: Any,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        num_beams: int = 2,
        seed: int | None = None,
    ) -> VectorizationResult:
        """Generate SVG and reject active or malformed model output."""
        effective_seed = (
            int(seed) if seed is not None else secrets.randbelow(2**31)
        )
        torch.manual_seed(effective_seed)
        if self._device.startswith("cuda"):
            torch.cuda.manual_seed_all(effective_seed)

        source_width, source_height = image.size
        pixel_values = _preprocess_image(
            image, size=int(self._model.config.image_size),
        ).to(device=self._device, dtype=self._model_dtype())

        inner = self._model.model
        with torch.inference_mode():
            vision = inner.image_encoder(pixel_values)
            vision = inner.image_projection(vision)
            prompt = self._tokenizer(
                "<svg", add_special_tokens=False, return_tensors="pt",
            )
            prompt_ids = prompt.input_ids.to(self._device)
            prompt_embeddings = (
                inner.svg_transformer.transformer.transformer.wte(prompt_ids)
            )
            inputs_embeds = torch.cat([vision, prompt_embeddings], dim=1)
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=self._device,
            )

            end_ids = self._tokenizer(
                "</svg>", add_special_tokens=False,
            )["input_ids"]
            criteria = StoppingCriteriaList([_StopOnSuffix(end_ids)])
            do_sample = temperature > 0.0
            generation_kwargs: dict[str, Any] = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": (
                    int(max_new_tokens)
                    if max_new_tokens is not None
                    else self._default_max_new_tokens
                ),
                "do_sample": do_sample,
                "num_beams": int(num_beams),
                "repetition_penalty": 1.0,
                "length_penalty": 1.0,
                "early_stopping": num_beams > 1,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                "stopping_criteria": criteria,
                "use_cache": True,
            }
            if do_sample:
                generation_kwargs["temperature"] = float(temperature)
                generation_kwargs["top_p"] = float(top_p)
            output_ids = inner.svg_transformer.transformer.generate(
                **generation_kwargs,
            )

        # With inputs_embeds, GPTBigCode returns generated ids without the
        # textual "<svg" prompt. Prefix it before decoding, matching the
        # official StarVector generate_im2svg implementation.
        decoded_ids = torch.cat([prompt_ids, output_ids], dim=1)
        raw_svg = self._tokenizer.batch_decode(
            decoded_ids, skip_special_tokens=True,
        )[0]
        info = validate_static_svg(raw_svg)
        return VectorizationResult(
            svg=info.svg,
            model_id=self.model_id,
            source_width=source_width,
            source_height=source_height,
            completion_tokens=int(output_ids.shape[-1]),
            seed=effective_seed,
            width=info.width,
            height=info.height,
            view_box=info.view_box,
            metadata={
                "temperature": float(temperature),
                "top_p": float(top_p),
                "num_beams": int(num_beams),
            },
        )

    def _model_dtype(self):
        try:
            return next(self._model.parameters()).dtype
        except (StopIteration, AttributeError):
            return self._dtype


def _validate_config(config_obj: Any) -> None:
    """Defense in depth against accidentally loading a different family."""
    if getattr(config_obj, "model_type", None) != "starvector":
        raise RuntimeError("checkpoint is not a StarVector model")
    if getattr(config_obj, "image_encoder_type", None) != "clip":
        raise RuntimeError("reviewed checkpoint must use the CLIP image encoder")
    if "starcoder2" in str(
        getattr(config_obj, "starcoder_model_name", "")
    ).lower():
        raise RuntimeError("reviewed checkpoint must use StarCoder v1")
    if int(getattr(config_obj, "hidden_size", 0)) != 2048:
        raise RuntimeError("reviewed checkpoint has unexpected hidden size")
    if int(getattr(config_obj, "num_hidden_layers", 0)) != 24:
        raise RuntimeError("reviewed checkpoint has unexpected layer count")
    expected = {
        "adapter_norm": "batch_norm",
        "image_size": 224,
        "vocab_size": 49156,
        "num_attention_heads": 16,
        "multi_query": True,
        "max_length_train": 8192,
    }
    for name, wanted in expected.items():
        actual = getattr(config_obj, name, None)
        if actual != wanted:
            raise RuntimeError(
                f"reviewed checkpoint has unexpected {name}: "
                f"{actual!r} (expected {wanted!r})"
            )


def _preprocess_image(image: Any, *, size: int):
    """Match StarVector's white-pad, bicubic, CLIP-normalize processor."""
    from PIL import Image

    if "A" in image.getbands() or "transparency" in image.info:
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        image = background
    else:
        image = image.convert("RGB")

    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    resampling = getattr(Image, "Resampling", Image)
    canvas = canvas.resize((size, size), resample=resampling.BICUBIC)

    array = np.asarray(canvas, dtype="float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],
        dtype=tensor.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711],
        dtype=tensor.dtype,
    ).view(1, 3, 1, 1)
    return (tensor - mean) / std
