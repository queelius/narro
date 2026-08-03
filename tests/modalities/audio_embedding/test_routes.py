"""Route tests for POST /v1/audio/embeddings.

Uses fully-mocked backends + raw bytes payloads (no real audio
decode) so the route exercises the multipart + envelope path without
needing librosa / ffmpeg.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_embedding import (
    MODALITY,
    AudioEmbeddingResult,
    build_router,
)
from muse.modalities.audio_embedding.codec import base64_to_embedding


def _make_client(backend, manifest=None):
    reg = ModalityRegistry()
    if manifest is None:
        manifest = {"model_id": "mert-v1-95m"}
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app, raise_server_exceptions=False)


def _fake_backend(result):
    backend = MagicMock()
    backend.model_id = "mert-v1-95m"
    backend.embed.return_value = result
    return backend


def _fake_result(*, n_clips=1, dim=768, model_id="mert-v1-95m"):
    return AudioEmbeddingResult(
        embeddings=[[0.1] * dim for _ in range(n_clips)],
        dimensions=dim,
        model_id=model_id,
        n_audio_clips=n_clips,
    )


def test_embeddings_returns_envelope_for_single_input():
    backend = _fake_backend(_fake_result(n_clips=1, dim=4))
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "mert-v1-95m"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == 4


def test_embeddings_returns_envelope_for_batched_input():
    backend = _fake_backend(_fake_result(n_clips=3, dim=4))
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/embeddings",
        files=[
            ("file", ("a.wav", b"AUDIOA", "audio/wav")),
            ("file", ("b.wav", b"AUDIOB", "audio/wav")),
            ("file", ("c.wav", b"AUDIOC", "audio/wav")),
        ],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 3
    indices = [entry["index"] for entry in body["data"]]
    assert indices == [0, 1, 2]


def test_embeddings_default_encoding_is_float():
    backend = _fake_backend(_fake_result(dim=4))
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
    )
    body = r.json()
    assert isinstance(body["data"][0]["embedding"], list)
    assert all(isinstance(x, (int, float)) for x in body["data"][0]["embedding"])


def test_embeddings_base64_encoding_returns_string():
    backend = _fake_backend(_fake_result(dim=4))
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"encoding_format": "base64"},
    )
    body = r.json()
    enc = body["data"][0]["embedding"]
    assert isinstance(enc, str)
    decoded = base64_to_embedding(enc)
    assert len(decoded) == 4


def test_embeddings_400_on_empty_file():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
    assert "empty" in body["error"]["message"].lower()


def test_embeddings_404_on_unknown_model():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "nonexistent-embedder"},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"


def test_embeddings_default_model_resolves_first_registered():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
    )
    assert r.status_code == 200
    assert r.json()["model"] == "mert-v1-95m"


def test_embeddings_invalid_encoding_format_rejected():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"encoding_format": "WRONG"},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"


def test_embeddings_user_field_accepted_and_ignored():
    """OpenAI compat: user field accepted, ignored."""
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"user": "alex@example.com"},
    )
    assert r.status_code == 200


def test_embeddings_payload_too_large(monkeypatch):
    """File exceeding MUSE_AUDIO_EMBEDDINGS_MAX_BYTES returns 413.

    Set the cap to 0 bytes so any non-empty file overflows.
    """
    monkeypatch.setenv("MUSE_AUDIO_EMBEDDINGS_MAX_BYTES", "0")
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"too-big-for-zero-cap", "audio/wav")},
    )
    assert r.status_code == 413
    body = r.json()
    assert body["error"]["code"] == "payload_too_large"


def test_embeddings_rejects_too_many_files(monkeypatch):
    monkeypatch.setattr(
        "muse.modalities.audio_embedding.routes._MAX_FILES", 2,
    )
    backend = _fake_backend(_fake_result(n_clips=3))
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files=[
            ("file", (f"{i}.wav", b"AUDIO", "audio/wav"))
            for i in range(3)
        ],
    )
    assert r.status_code == 413
    backend.embed.assert_not_called()


def test_embeddings_rejects_aggregate_bytes(monkeypatch):
    monkeypatch.setattr(
        "muse.modalities.audio_embedding.routes._MAX_TOTAL_UPLOAD_BYTES", 5,
    )
    backend = _fake_backend(_fake_result(n_clips=2))
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files=[
            ("file", ("a.wav", b"AAA", "audio/wav")),
            ("file", ("b.wav", b"BBB", "audio/wav")),
        ],
    )
    assert r.status_code == 413
    backend.embed.assert_not_called()


def test_embeddings_decoder_error_returns_415():
    """A backend exception matching the decoder-error pattern returns 415."""
    backend = MagicMock()
    backend.model_id = "mert-v1-95m"
    backend.embed.side_effect = RuntimeError(
        "audioread.NoBackendError: no backend available"
    )
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"NOT_REAL_AUDIO", "audio/wav")},
    )
    assert r.status_code == 415
    body = r.json()
    assert body["error"]["code"] == "unsupported_media_type"


def test_embeddings_unrelated_error_not_415():
    """A non-decoder RuntimeError must NOT be silently wrapped as 415."""
    backend = MagicMock()
    backend.model_id = "mert-v1-95m"
    backend.embed.side_effect = RuntimeError("GPU out of memory")
    client = _make_client(backend)

    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
    )
    assert r.status_code != 415, (
        f"non-decoder RuntimeError must not be masked as 415; body={r.text}"
    )


def test_embeddings_returns_model_id_from_backend_result():
    """Response 'model' field is the result.model_id, not the request body."""
    backend = _fake_backend(_fake_result(model_id="custom-id"))
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
    )
    body = r.json()
    assert body["model"] == "custom-id"


def test_embeddings_usage_zero_for_audio_inputs():
    """Audio embedding has no text tokenization; usage stays 0."""
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
    )
    body = r.json()
    assert body["usage"]["prompt_tokens"] == 0
    assert body["usage"]["total_tokens"] == 0


def test_embeddings_404_uses_error_envelope_not_detail():
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "nonexistent"},
    )
    body = r.json()
    assert "detail" not in body
    assert "error" in body


def test_embeddings_400_uses_error_envelope_not_detail():
    """muse error envelope is {"error": {...}}, not FastAPI's {"detail": ...}."""
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"", "audio/wav")},
    )
    body = r.json()
    assert "detail" not in body
    assert "error" in body
    assert "code" in body["error"]
    assert "message" in body["error"]
    assert "type" in body["error"]


def test_embeddings_passes_raw_bytes_to_backend():
    """Backend receives a list of raw bytes, not UploadFile objects."""
    backend = _fake_backend(_fake_result(n_clips=2))
    client = _make_client(backend)
    client.post(
        "/v1/audio/embeddings",
        files=[
            ("file", ("a.wav", b"AUDIOA", "audio/wav")),
            ("file", ("b.wav", b"AUDIOB", "audio/wav")),
        ],
    )
    args, _ = backend.embed.call_args
    audio_list = args[0]
    assert isinstance(audio_list, list)
    assert len(audio_list) == 2
    assert audio_list[0] == b"AUDIOA"
    assert audio_list[1] == b"AUDIOB"


def test_embeddings_data_object_marker():
    """Each entry in `data` carries object='embedding'."""
    backend = _fake_backend(_fake_result(n_clips=2))
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files=[
            ("file", ("a.wav", b"AUDIOA", "audio/wav")),
            ("file", ("b.wav", b"AUDIOB", "audio/wav")),
        ],
    )
    body = r.json()
    for entry in body["data"]:
        assert entry["object"] == "embedding"


def test_embeddings_indices_are_zero_based_and_ordered():
    backend = _fake_backend(_fake_result(n_clips=3))
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files=[
            ("file", ("a.wav", b"AUDIOA", "audio/wav")),
            ("file", ("b.wav", b"AUDIOB", "audio/wav")),
            ("file", ("c.wav", b"AUDIOC", "audio/wav")),
        ],
    )
    body = r.json()
    for i, entry in enumerate(body["data"]):
        assert entry["index"] == i


def test_embeddings_explicit_model_field_resolves():
    """When the request supplies model=X and X is registered, it routes there."""
    backend = _fake_backend(_fake_result())
    client = _make_client(backend)
    r = client.post(
        "/v1/audio/embeddings",
        files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
        data={"model": "mert-v1-95m"},
    )
    assert r.status_code == 200
