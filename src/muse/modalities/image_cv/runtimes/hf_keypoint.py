"""HFKeypointRuntime: Hugging Face keypoint and pose detection.

Dispatches from the checkpoint config: ViTPose uses
`transformers.VitPoseForPoseEstimation` (4.48+) and other supported
keypoint families use `AutoModelForKeypointDetection`. Both paths use
`AutoImageProcessor`.

ViTPose-style models expect bounding boxes per pose-extraction target.
v1 of this runtime takes a single full-image bbox per call (the whole
image is treated as the entity to extract pose from). Multi-person
pose extraction needs a person detector first; that pipeline is
deferred to a future v0.X+1 (it's a separate concern from this
runtime).

Inference flow:

  1. processor(image, boxes=[[(0, 0, W, H)]], return_tensors='pt') ->
     batched pixel inputs
  2. model(**inputs) -> outputs
  3. processor.post_process_pose_estimation(outputs, boxes=boxes) ->
     list of dicts per box; Muse applies its own inclusive threshold
  4. Map the dicts into KeypointDetection objects with bbox + score +
     per-keypoint name/x/y/score.

The generic path retains its family-specific keypoint postprocessor and
raw-output fallback. ViTPose requires its pose postprocessor because its
raw output is a heatmap rather than decoded coordinates.

Deferred imports follow the muse pattern.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any

from muse.core.runtime_helpers import (
    LoadTimer, dtype_for_name, select_device, set_inference_mode,
)
from muse.modalities.image_cv.protocol import (
    Keypoint,
    KeypointDetection,
    KeypointResult,
)


logger = logging.getLogger(__name__)


torch: Any = None
AutoConfig: Any = None
AutoModelForKeypointDetection: Any = None
VitPoseForPoseEstimation: Any = None
AutoImageProcessor: Any = None


def _ensure_deps() -> None:
    global torch, AutoConfig, AutoModelForKeypointDetection
    global VitPoseForPoseEstimation, AutoImageProcessor
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("HFKeypointRuntime torch unavailable: %s", e)
    if AutoConfig is None:
        try:
            from transformers import AutoConfig as _c
            AutoConfig = _c
        except Exception as e:  # noqa: BLE001
            logger.debug("HFKeypointRuntime AutoConfig unavailable: %s", e)
    if AutoModelForKeypointDetection is None:
        try:
            from transformers import AutoModelForKeypointDetection as _m
            AutoModelForKeypointDetection = _m
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFKeypointRuntime AutoModelForKeypointDetection unavailable: %s",
                e,
            )
    if VitPoseForPoseEstimation is None:
        try:
            from transformers import VitPoseForPoseEstimation as _v
            VitPoseForPoseEstimation = _v
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFKeypointRuntime VitPoseForPoseEstimation unavailable: %s",
                e,
            )
    if AutoImageProcessor is None:
        try:
            from transformers import AutoImageProcessor as _p
            AutoImageProcessor = _p
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "HFKeypointRuntime AutoImageProcessor unavailable: %s", e,
            )


class HFKeypointRuntime:
    """Generic keypoint detection runtime."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "fp32",
        **_: Any,
    ) -> None:
        _ensure_deps()
        if torch is None:
            raise RuntimeError(
                "torch is not installed; run `muse pull` or install "
                "`torch` into this venv"
            )
        if (
            AutoConfig is None
            or AutoModelForKeypointDetection is None
            or AutoImageProcessor is None
        ):
            raise RuntimeError(
                "transformers is not installed (or too old for "
                "AutoModelForKeypointDetection; need >= 4.46); run "
                "`muse pull` or install `transformers>=4.46.0` into this venv"
            )
        self.model_id = model_id
        self._device = _select_device(device)
        self._dtype = _resolve_dtype(dtype)
        src = local_dir or hf_repo
        with LoadTimer(f"loading keypoint model from {src}", logger):
            config = AutoConfig.from_pretrained(src)
            self._is_vitpose = _is_vitpose_config(config)
            if self._is_vitpose and VitPoseForPoseEstimation is None:
                raise RuntimeError(
                    "this ViTPose checkpoint requires transformers>=4.48.0; "
                    "run `muse pull` or install `transformers>=4.48.0` into "
                    "this venv"
                )
            model_factory = (
                VitPoseForPoseEstimation
                if self._is_vitpose
                else AutoModelForKeypointDetection
            )
            self._processor = AutoImageProcessor.from_pretrained(src)
            self._model = model_factory.from_pretrained(
                src, config=config, torch_dtype=self._dtype,
            )
            self._model = self._model.to(self._device)
        set_inference_mode(self._model)

    def detect_keypoints(
        self, image: Any, *, threshold: float = 0.3,
    ) -> KeypointResult:
        """Detect keypoints in one PIL.Image.

        ViTPose v1 passes a single full-image bbox per call. Its
        post-processing returns one detection's keypoints (the whole image,
        treated as one entity). Multi-person extraction is a future
        enhancement that needs a person detector upstream.
        """
        W, H = image.size
        # ViTPose expects boxes as a list of lists: outer list per
        # image, inner list per detection. One image, one bbox.
        boxes = [[[0.0, 0.0, float(W), float(H)]]]
        processor_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if self._is_vitpose:
            processor_kwargs["boxes"] = boxes
        inputs = self._processor(image, **processor_kwargs)
        inputs = {
            k: (v.to(self._device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        outputs = self._model(**inputs)

        detections: list[KeypointDetection] = []
        processed = None
        if self._is_vitpose:
            post = getattr(
                self._processor, "post_process_pose_estimation", None,
            )
            if not callable(post):
                raise RuntimeError(
                    "the ViTPose image processor is missing "
                    "post_process_pose_estimation; install "
                    "transformers>=4.48.0"
                )
            kwargs = self._adaptive_kwargs(
                post, boxes=boxes,
            )
            if "boxes" not in kwargs:
                raise RuntimeError(
                    "the ViTPose image processor has an incompatible "
                    "post_process_pose_estimation signature; install "
                    "transformers>=4.48.0"
                )
            processed = post(outputs, **kwargs)
        else:
            post = getattr(
                self._processor, "post_process_keypoint_detection", None,
            )
        if not self._is_vitpose and callable(post):
            # Different transformers versions and processor classes accept
            # different optional kwargs. Inspect the signature once and
            # build a kwargs dict containing only what's supported, so a
            # genuine TypeError raised inside the function (a bug) isn't
            # silently swallowed and retried with stale args.
            kwargs = self._adaptive_kwargs(
                post, target_sizes=[(H, W)],
            )
            processed = post(outputs, **kwargs)
        if processed is not None:
            # processed is a list (per image) of lists (per box) of
            # dicts {keypoints, scores, labels}. Pull the first image's
            # first box's results.
            if processed and processed[0]:
                first_image = processed[0]
                # Some processors flatten to a single dict, others to a list.
                first = first_image[0] if isinstance(first_image, list) else first_image
                detection = self._build_detection_from_processed(
                    first, bbox=(0.0, 0.0, float(W), float(H)),
                    threshold=threshold,
                )
                if detection.keypoints:
                    detections.append(detection)
        elif not self._is_vitpose:
            # Fallback: walk the raw outputs. Format is processor-
            # specific, but a common shape is outputs.keypoints of
            # shape (B, N, K, 3) where the last dim is (x, y, score).
            kp_tensor = getattr(outputs, "keypoints", None)
            if kp_tensor is not None and kp_tensor.dim() >= 3:
                detection = self._build_detection_from_raw(
                    kp_tensor[0], bbox=(0.0, 0.0, float(W), float(H)),
                    threshold=threshold,
                )
                if detection.keypoints:
                    detections.append(detection)

        return KeypointResult(
            detections=detections,
            model_id=self.model_id,
            image_size=(W, H),
        )

    @staticmethod
    def _adaptive_kwargs(fn: Any, /, **candidate_kwargs: Any) -> dict[str, Any]:
        """Filter candidate_kwargs to those `fn` actually accepts.

        Returns the subset of candidate_kwargs whose names appear in
        fn's signature parameters or whose presence is implied by a
        **kwargs catch-all (VAR_KEYWORD). When the signature is
        unintrospectable (some C-implemented callables), pass everything
        through.
        """
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return dict(candidate_kwargs)
        params = sig.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(candidate_kwargs)
        return {k: v for k, v in candidate_kwargs.items() if k in params}

    def _build_detection_from_processed(
        self,
        processed: dict,
        *,
        bbox: tuple[float, float, float, float],
        threshold: float,
    ) -> KeypointDetection:
        """Convert a post-processed dict to a KeypointDetection."""
        kps_tensor = processed.get("keypoints")
        scores_tensor = processed.get("scores")
        labels_tensor = processed.get("labels")

        if kps_tensor is None or scores_tensor is None:
            # Defensive default: an empty detection has no confidence,
            # not perfect confidence. The path is unreachable from the
            # current callers but should be sane if a future processor
            # returns malformed dicts.
            return KeypointDetection(bbox=bbox, score=0.0, keypoints=[])

        kps = self._as_list(kps_tensor)
        scores = self._as_list(scores_tensor)
        if kps and not isinstance(kps[0], (list, tuple)):
            kps = [kps]
        labels = (
            self._as_list(labels_tensor) if labels_tensor is not None else None
        )

        # Build an id2label map from the model config when available.
        id2label = self._id2label()

        keypoints: list[Keypoint] = []
        for i, ((x, y), score) in enumerate(zip(kps, scores)):
            if float(score) < threshold:
                continue
            label_idx = labels[i] if labels else i
            name = id2label.get(int(label_idx), str(label_idx))
            keypoints.append(Keypoint(
                name=name, x=float(x), y=float(y), score=float(score),
            ))

        # Use the max keypoint score as the detection-level score for
        # ViTPose-style models that don't emit a separate detection score.
        det_score = float(max(scores)) if scores else 0.0
        return KeypointDetection(bbox=bbox, score=det_score, keypoints=keypoints)

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """Convert tensor/array/scalar postprocessor values to a list."""
        if hasattr(value, "detach"):
            converted = value.detach().cpu().tolist()
        elif isinstance(value, (list, tuple)):
            converted = list(value)
        elif hasattr(value, "tolist"):
            converted = value.tolist()
        else:
            converted = value
        return converted if isinstance(converted, list) else [converted]

    def _build_detection_from_raw(
        self,
        raw_keypoints: Any,
        *,
        bbox: tuple[float, float, float, float],
        threshold: float,
    ) -> KeypointDetection:
        """Fallback: decode keypoints from the raw outputs tensor.

        Accepts shape (N, K, 3) where last dim is (x, y, score), or
        (K, 3) for a single detection.
        """
        t = raw_keypoints
        if t.dim() == 3:
            # (N, K, 3): take the first (and assumed only) detection.
            t = t[0]
        # t is now (K, 3).
        rows = t.detach().cpu().tolist()
        id2label = self._id2label()
        keypoints: list[Keypoint] = []
        scores: list[float] = []
        for i, row in enumerate(rows):
            x, y, score = row[:3]
            if float(score) < threshold:
                continue
            name = id2label.get(i, str(i))
            keypoints.append(Keypoint(
                name=name, x=float(x), y=float(y), score=float(score),
            ))
            scores.append(float(score))
        det_score = max(scores) if scores else 0.0
        return KeypointDetection(bbox=bbox, score=det_score, keypoints=keypoints)

    def _id2label(self) -> dict[int, str]:
        """Pull the integer-keyed id2label from the model config.

        HF configs sometimes ship id2label with string keys (JSON
        round-trip artifact). Coerce keys to int so the runtime's
        lookups work without surprises.
        """
        cfg = getattr(self._model, "config", None)
        if cfg is None:
            return {}
        raw = getattr(cfg, "id2label", None) or {}
        out: dict[int, str] = {}
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        return out


def _select_device(device: str) -> str:
    return select_device(device, torch_module=torch)


def _resolve_dtype(dtype: str):
    return dtype_for_name(dtype, torch_module=torch)


def _is_vitpose_config(config: Any) -> bool:
    """Recognize ViTPose without masking unrelated auto-model failures."""
    model_type = getattr(config, "model_type", None)
    if isinstance(model_type, str) and model_type.lower() == "vitpose":
        return True
    architectures = getattr(config, "architectures", None)
    if not isinstance(architectures, (list, tuple)):
        architectures = ()
    return any(
        isinstance(name, str) and name.lower() == "vitposeforposeestimation"
        for name in architectures
    )
