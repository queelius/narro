from muse.modalities.image_vectorization import (
    MODALITY,
    MODEL_OPTIONAL_PATHS,
    PROBE_DEFAULTS,
    ImageVectorizationModel,
    VectorizationResult,
)


def test_modality_tag():
    assert MODALITY == "image/vectorization"
    assert MODEL_OPTIONAL_PATHS == ("/v1/images/vectorize",)


def test_result_defaults():
    result = VectorizationResult(
        svg="<svg/>",
        model_id="m",
        source_width=10,
        source_height=20,
    )
    assert result.completion_tokens == 0
    assert result.view_box is None
    assert result.metadata == {}


def test_structural_protocol():
    class Backend:
        model_id = "m"

        def vectorize(self, image, **kwargs):
            return VectorizationResult(
                svg="<svg/>", model_id="m",
                source_width=1, source_height=1,
            )

    assert isinstance(Backend(), ImageVectorizationModel)


def test_probe_is_bounded_and_deterministic():
    class Backend:
        def vectorize(self, image, **kwargs):
            self.size = image.size
            self.kwargs = kwargs
            return object()

    backend = Backend()
    assert PROBE_DEFAULTS["call"](backend) is not None
    assert backend.size == (64, 64)
    assert backend.kwargs == {
        "max_new_tokens": 512,
        "temperature": 0.0,
        "num_beams": 1,
        "seed": 0,
    }
