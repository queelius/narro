"""HFKeypointRuntime: mocked-dep tests."""
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    orig = (
        mod.torch, mod.AutoConfig, mod.AutoModelForKeypointDetection,
        mod.VitPoseForPoseEstimation, mod.AutoImageProcessor,
    )
    yield
    (
        mod.torch, mod.AutoConfig, mod.AutoModelForKeypointDetection,
        mod.VitPoseForPoseEstimation, mod.AutoImageProcessor,
    ) = orig


def _movable():
    t = MagicMock()
    t.to = MagicMock(return_value=t)
    return t


def _wire_runtime(
    mod,
    *,
    processed=None,
    has_post_process=True,
    id2label=None,
    model_type="vitpose",
):
    """Install fakes for config-dispatched models + AutoImageProcessor.

    `processed` is what the family postprocessor returns (a list-of-lists
    structure shaped like the real HF API). When None, the generic runtime
    can fall back to raw outputs.

    `has_post_process` toggles whether the processor exposes
    its expected family postprocessor.
    """
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    fake_torch.float32 = "fp32-sentinel"

    forward_outputs = MagicMock()
    forward_outputs.keypoints = MagicMock()  # only used by fallback path

    config = MagicMock()
    config.model_type = model_type
    config.architectures = []
    config.id2label = id2label or {}
    config_factory = MagicMock()
    config_factory.from_pretrained = MagicMock(return_value=config)
    mod.AutoConfig = config_factory

    model_obj = MagicMock()
    model_obj.return_value = forward_outputs
    model_obj.to = MagicMock(return_value=model_obj)
    model_obj.config = config

    auto_factory = MagicMock()
    auto_factory.from_pretrained = MagicMock(return_value=model_obj)
    vitpose_factory = MagicMock()
    vitpose_factory.from_pretrained = MagicMock(return_value=model_obj)
    mod.AutoModelForKeypointDetection = auto_factory
    mod.VitPoseForPoseEstimation = vitpose_factory

    encoded = {"pixel_values": _movable()}
    processor = MagicMock()
    processor.return_value = encoded

    for attr in (
        "post_process_pose_estimation",
        "post_process_keypoint_detection",
    ):
        if hasattr(processor, attr):
            delattr(processor, attr)
    if has_post_process:
        if processed is None:
            processed = [[]]  # one image, no detections
        attr = (
            "post_process_pose_estimation"
            if model_type == "vitpose"
            else "post_process_keypoint_detection"
        )
        setattr(processor, attr, MagicMock(return_value=processed))

    proc_factory = MagicMock()
    proc_factory.from_pretrained = MagicMock(return_value=processor)
    mod.AutoImageProcessor = proc_factory

    mod.torch = fake_torch
    return processor, model_obj


def _kp_dict(*, kps, scores, labels=None):
    """Build a processed-result dict whose tensors mock to-list."""
    kps_t = MagicMock()
    kps_t.detach.return_value.cpu.return_value.tolist.return_value = kps
    scores_t = MagicMock()
    scores_t.detach.return_value.cpu.return_value.tolist.return_value = scores
    out = {"keypoints": kps_t, "scores": scores_t}
    if labels is not None:
        labels_t = MagicMock()
        labels_t.detach.return_value.cpu.return_value.tolist.return_value = labels
        out["labels"] = labels_t
    return out


def test_detect_keypoints_returns_keypoint_result():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processed = [[
        _kp_dict(
            kps=[[100.0, 50.0], [120.0, 60.0]],
            scores=[0.99, 0.95],
        )
    ]]
    _wire_runtime(
        mod, processed=processed,
        id2label={0: "nose", 1: "left_eye"},
    )
    runtime = mod.HFKeypointRuntime(
        model_id="vp", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (640, 480)
    result = runtime.detect_keypoints(image)

    assert result.model_id == "vp"
    assert result.image_size == (640, 480)
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.bbox == (0.0, 0.0, 640.0, 480.0)
    assert len(det.keypoints) == 2
    assert det.keypoints[0].name == "nose"
    assert det.keypoints[0].x == 100.0
    assert det.keypoints[0].score == 0.99


def test_detect_keypoints_threshold_filter():
    """Keypoints below the threshold get dropped."""
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processed = [[
        _kp_dict(
            kps=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            scores=[0.9, 0.2, 0.8],
        )
    ]]
    _wire_runtime(mod, processed=processed)
    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)
    result = runtime.detect_keypoints(image, threshold=0.5)
    # Only the 0.9 and 0.8 keypoints survive.
    assert len(result.detections) == 1
    assert len(result.detections[0].keypoints) == 2


def test_detect_keypoints_falls_back_to_index_when_no_id2label():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processed = [[
        _kp_dict(kps=[[5.0, 5.0]], scores=[0.99])
    ]]
    _wire_runtime(mod, processed=processed, id2label={})
    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)
    result = runtime.detect_keypoints(image)
    # Index 0 has no label, so name is "0".
    assert result.detections[0].keypoints[0].name == "0"


def test_generic_postprocess_receives_target_size_and_flat_result():
    """Generic postprocessing receives target size and a flat result works."""
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processed_result = [_kp_dict(kps=[[1.0, 1.0]], scores=[0.99])]
    received_target_sizes: list[list[tuple[int, int]]] = []

    def _post(outputs, target_sizes):
        received_target_sizes.append(target_sizes)
        return processed_result

    _wire_runtime(mod, processed=[[]], model_type="superpoint")
    proc_factory = mod.AutoImageProcessor
    proc_obj = proc_factory.from_pretrained.return_value
    proc_obj.post_process_keypoint_detection = _post

    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)
    result = runtime.detect_keypoints(image)
    assert received_target_sizes == [[(10, 10)]]
    assert len(result.detections) == 1


def test_vitpose_postprocess_receives_absolute_boxes_without_target_sizes():
    """ViTPose receives absolute boxes and threshold, but no target_sizes."""
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processed_result = [[_kp_dict(kps=[[1.0, 1.0]], scores=[0.99])]]
    received: dict = {}

    def _post(outputs, **kwargs):
        received.update(kwargs)
        return processed_result

    processor, _model = _wire_runtime(mod, processed=[[]])
    proc_factory = mod.AutoImageProcessor
    proc_obj = proc_factory.from_pretrained.return_value
    proc_obj.post_process_pose_estimation = _post

    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)
    runtime.detect_keypoints(image, threshold=0.42)
    assert received["boxes"] == [[[0.0, 0.0, 10.0, 10.0]]]
    assert "threshold" not in received
    assert "target_sizes" not in received
    processor.assert_called_once_with(
        image,
        boxes=[[[0.0, 0.0, 10.0, 10.0]]],
        return_tensors="pt",
    )


def test_vitpose_config_selects_explicit_model_factory():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    _wire_runtime(mod, processed=[[]])
    config = mod.AutoConfig.from_pretrained.return_value

    mod.HFKeypointRuntime(model_id="m", hf_repo="repo", device="cpu")

    mod.AutoConfig.from_pretrained.assert_called_once_with("repo")
    mod.VitPoseForPoseEstimation.from_pretrained.assert_called_once_with(
        "repo", config=config, torch_dtype="fp32-sentinel",
    )
    mod.AutoModelForKeypointDetection.from_pretrained.assert_not_called()


def test_vitpose_architecture_selects_explicit_model_factory():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    _wire_runtime(mod, processed=[[]], model_type="custom")
    config = mod.AutoConfig.from_pretrained.return_value
    config.architectures = ["VitPoseForPoseEstimation"]

    mod.HFKeypointRuntime(model_id="m", hf_repo="repo", device="cpu")

    mod.VitPoseForPoseEstimation.from_pretrained.assert_called_once()
    mod.AutoModelForKeypointDetection.from_pretrained.assert_not_called()


def test_malformed_architecture_metadata_uses_generic_factory():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    _wire_runtime(mod, processed=[[]], model_type="custom")
    config = mod.AutoConfig.from_pretrained.return_value
    config.architectures = 123

    mod.HFKeypointRuntime(model_id="m", hf_repo="repo", device="cpu")

    mod.AutoModelForKeypointDetection.from_pretrained.assert_called_once()
    mod.VitPoseForPoseEstimation.from_pretrained.assert_not_called()


def test_generic_config_uses_auto_factory_and_omits_preprocess_boxes():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processor, _model = _wire_runtime(
        mod, processed=[[]], model_type="superpoint",
    )
    config = mod.AutoConfig.from_pretrained.return_value
    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="repo", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)

    runtime.detect_keypoints(image)

    mod.AutoModelForKeypointDetection.from_pretrained.assert_called_once_with(
        "repo", config=config, torch_dtype="fp32-sentinel",
    )
    mod.VitPoseForPoseEstimation.from_pretrained.assert_not_called()
    processor.assert_called_once_with(image, return_tensors="pt")


def test_vitpose_scalar_score_and_label_are_normalized():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processed = [[_kp_dict(
        kps=[[5.0, 6.0]], scores=0.91, labels=4,
    )]]
    _wire_runtime(mod, processed=processed, id2label={4: "left_shoulder"})
    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="repo", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)

    result = runtime.detect_keypoints(image)

    assert result.detections[0].score == 0.91
    assert result.detections[0].keypoints[0].name == "left_shoulder"


def test_threshold_equality_is_retained_by_muse():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    processed = [[_kp_dict(kps=[[5.0, 6.0]], scores=[0.3])]]
    _wire_runtime(mod, processed=processed)
    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="repo", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)

    result = runtime.detect_keypoints(image, threshold=0.3)

    assert len(result.detections) == 1
    assert result.detections[0].keypoints[0].score == 0.3


def test_detect_keypoints_internal_typeerror_propagates():
    """A TypeError raised from inside post_process_pose_estimation
    (i.e., a bug in the function body, not a signature mismatch) must
    NOT be silently swallowed. Regression: the old try/except fallback
    would retry with simpler args, masking real bugs."""
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod

    def _post(outputs, boxes):
        raise TypeError("genuine bug inside post-processing")

    _wire_runtime(mod, processed=[[]])
    proc_factory = mod.AutoImageProcessor
    proc_obj = proc_factory.from_pretrained.return_value
    proc_obj.post_process_pose_estimation = _post

    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)
    with pytest.raises(TypeError, match="genuine bug"):
        runtime.detect_keypoints(image)


def test_detect_keypoints_empty_when_no_detections():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    _wire_runtime(mod, processed=[[]])
    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)
    result = runtime.detect_keypoints(image)
    assert result.detections == []


def test_raises_when_transformers_too_old(monkeypatch):
    """Missing generic Transformers primitives surface a clear error."""
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = MagicMock()
    mod.AutoConfig = None
    mod.AutoModelForKeypointDetection = None
    mod.VitPoseForPoseEstimation = None
    mod.AutoImageProcessor = MagicMock()
    with pytest.raises(RuntimeError, match=">= 4.46"):
        mod.HFKeypointRuntime(model_id="m", hf_repo="x", device="cpu")


def test_vitpose_requires_explicit_transformers_model_class(monkeypatch):
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    _wire_runtime(mod)
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.VitPoseForPoseEstimation = None

    with pytest.raises(RuntimeError, match=">=4.48.0"):
        mod.HFKeypointRuntime(model_id="m", hf_repo="x", device="cpu")


def test_vitpose_requires_pose_postprocessor():
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    _wire_runtime(mod, has_post_process=False)
    runtime = mod.HFKeypointRuntime(
        model_id="m", hf_repo="x", device="cpu",
    )
    image = MagicMock()
    image.size = (10, 10)

    with pytest.raises(RuntimeError, match="post_process_pose_estimation"):
        runtime.detect_keypoints(image)


def test_raises_when_torch_not_installed(monkeypatch):
    import muse.modalities.image_cv.runtimes.hf_keypoint as mod
    monkeypatch.setattr(mod, "_ensure_deps", lambda: None)
    mod.torch = None
    mod.AutoConfig = MagicMock()
    mod.AutoModelForKeypointDetection = MagicMock()
    mod.VitPoseForPoseEstimation = MagicMock()
    mod.AutoImageProcessor = MagicMock()
    with pytest.raises(RuntimeError, match="torch is not installed"):
        mod.HFKeypointRuntime(model_id="m", hf_repo="x", device="cpu")
