"""Tests for the gateway proxy FastAPI app."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from muse.cli_impl.gateway import (
    RequestBodyLimitMiddleware,
    RequestBodyTooLarge,
    _prefetch_worker_response,
    _proxy_headers,
    cache_bounded_request_body,
    extract_model_from_request,
    build_gateway,
    WorkerRoute,
)


class _RawResponse:
    """Minimal httpx response stream double for bounded relay tests."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.chunk_size = None

    def aiter_raw(self, *, chunk_size=None):
        self.chunk_size = chunk_size

        async def _chunks():
            for chunk in self.chunks:
                yield chunk

        return _chunks()


def test_proxy_headers_strip_standard_and_connection_named_hops():
    headers = {
        "Host": "worker.invalid",
        "Connection": "keep-alive, x-private-hop",
        "Keep-Alive": "timeout=5",
        "X-Private-Hop": "do-not-forward",
        "X-End-To-End": "preserve",
    }

    assert _proxy_headers(
        headers, exclude=frozenset({"host"}),
    ) == [(b"X-End-To-End", b"preserve")]


@pytest.mark.asyncio
async def test_worker_response_prefetch_buffers_only_under_limit():
    response = _RawResponse([b"ab", b"cd"])
    content, prefix, iterator = await _prefetch_worker_response(
        response, limit=4,
    )

    assert content == b"abcd"
    assert prefix == []
    assert iterator is None
    assert response.chunk_size == 64 * 1024


@pytest.mark.asyncio
async def test_worker_response_prefetch_hands_off_oversized_body():
    response = _RawResponse([b"abc", b"def", b"ghi"])
    content, prefix, iterator = await _prefetch_worker_response(
        response, limit=4,
    )

    assert content is None
    assert prefix == [b"abc", b"def"]
    assert b"".join([chunk async for chunk in iterator]) == b"ghi"


@pytest.mark.asyncio
async def test_large_non_sse_response_streams_and_releases_after_consumption():
    from fastapi.responses import StreamingResponse
    from muse.cli_impl.gateway import _forward_with_release

    chunks = [b"a" * 600_000, b"b" * 600_000, b"tail"]
    upstream = _RawResponse(chunks)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/octet-stream"}
    mock_response.aiter_raw = upstream.aiter_raw

    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    stream_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_client = MagicMock()
    mock_client.stream.return_value = stream_ctx
    mock_client.aclose = AsyncMock()
    request = MagicMock()
    request.body = AsyncMock(return_value=b"")
    request.headers = {}
    request.method = "GET"
    request.query_params = {}
    director = MagicMock()

    with patch(
        "muse.cli_impl.gateway.httpx.AsyncClient",
        return_value=mock_client,
    ):
        response = await _forward_with_release(
            request,
            "http://127.0.0.1:9001/artifact",
            1.0,
            director=director,
            model_id="artifact-model",
        )

    assert isinstance(response, StreamingResponse)
    director.release.assert_not_called()
    body = b"".join([chunk async for chunk in response.body_iterator])
    assert body == b"".join(chunks)
    director.release.assert_called_once_with("artifact-model")
    stream_ctx.__aexit__.assert_awaited_once()
    mock_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_director_release_when_stream_context_creation_fails():
    from muse.cli_impl.gateway import _forward_with_release

    request = MagicMock()
    request.body = AsyncMock(return_value=b"")
    request.headers = {}
    request.method = "GET"
    request.query_params = {}
    director = MagicMock()
    mock_client = MagicMock()
    mock_client.stream.side_effect = ValueError("invalid worker URL")
    mock_client.aclose = AsyncMock()

    with patch(
        "muse.cli_impl.gateway.httpx.AsyncClient",
        return_value=mock_client,
    ):
        with pytest.raises(ValueError, match="invalid worker URL"):
            await _forward_with_release(
                request,
                "invalid://worker",
                1.0,
                director=director,
                model_id="broken-model",
            )

    director.release.assert_called_once_with("broken-model")
    mock_client.aclose.assert_awaited_once()


class TestExtractModel:
    @pytest.mark.asyncio
    async def test_extracts_model_from_json_body(self):
        """POST with JSON body: model is body['model']."""
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "application/json"}
        request.body = AsyncMock(return_value=b'{"input":"hi","model":"soprano-80m"}')
        model = await extract_model_from_request(request)
        assert model == "soprano-80m"

    @pytest.mark.asyncio
    async def test_returns_none_when_body_has_no_model(self):
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "application/json"}
        request.body = AsyncMock(return_value=b'{"input":"hi"}')
        model = await extract_model_from_request(request)
        assert model is None

    @pytest.mark.asyncio
    async def test_extracts_model_from_query_on_get(self):
        request = MagicMock()
        request.method = "GET"
        request.query_params = {"model": "kokoro-82m"}
        model = await extract_model_from_request(request)
        assert model == "kokoro-82m"

    @pytest.mark.asyncio
    async def test_returns_none_when_get_has_no_query_model(self):
        request = MagicMock()
        request.method = "GET"
        request.query_params = {}
        model = await extract_model_from_request(request)
        assert model is None

    @pytest.mark.asyncio
    async def test_returns_none_when_body_is_invalid_json(self):
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "application/json"}
        request.body = AsyncMock(return_value=b'not json at all')
        model = await extract_model_from_request(request)
        assert model is None

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_content_type(self):
        """text/plain (or any non-JSON, non-multipart) returns None."""
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "text/plain"}
        model = await extract_model_from_request(request)
        assert model is None

    @pytest.mark.asyncio
    async def test_extracts_model_from_multipart_form_body(self):
        """POST with multipart/form-data: model is form['model'].

        OpenAI's audio.transcriptions / audio.translations / images.edits
        / images.variations endpoints all use multipart and put the model
        in a form field, so the gateway must support extraction here.
        """
        # Use a real Starlette Request because request.form() needs the
        # full receive-channel machinery; MagicMock doesn't carry that.
        from starlette.requests import Request

        body = (
            b"--boundary\r\n"
            b'Content-Disposition: form-data; name="model"\r\n\r\n'
            b"whisper-tiny\r\n"
            b"--boundary\r\n"
            b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            b"Content-Type: audio/wav\r\n\r\n"
            b"FAKEWAVBYTES\r\n"
            b"--boundary--\r\n"
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/transcriptions",
            "headers": [
                (b"content-type", b"multipart/form-data; boundary=boundary"),
                (b"content-length", str(len(body)).encode()),
            ],
            "query_string": b"",
        }
        sent = False
        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(scope, receive=receive)
        model = await extract_model_from_request(request)
        assert model == "whisper-tiny"
        upload = request._form.get("file")
        assert upload.file.closed is True

    @pytest.mark.asyncio
    async def test_returns_none_when_multipart_body_has_no_model_field(self):
        """A multipart body without a `model` form field returns None."""
        from starlette.requests import Request

        body = (
            b"--boundary\r\n"
            b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            b"Content-Type: audio/wav\r\n\r\n"
            b"FAKEWAVBYTES\r\n"
            b"--boundary--\r\n"
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/transcriptions",
            "headers": [
                (b"content-type", b"multipart/form-data; boundary=boundary"),
                (b"content-length", str(len(body)).encode()),
            ],
            "query_string": b"",
        }
        sent = False
        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(scope, receive=receive)
        model = await extract_model_from_request(request)
        assert model is None


class TestWorkerRoute:
    def test_worker_route_stores_model_and_url(self):
        r = WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")
        assert r.model_id == "soprano-80m"
        assert r.worker_url == "http://127.0.0.1:9001"


class TestBoundedRequestBody:
    @pytest.mark.asyncio
    async def test_chunked_body_is_bounded_without_content_length(self):
        from starlette.requests import Request

        messages = iter([
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"5678", "more_body": False},
        ])

        async def receive():
            return next(messages)

        request = Request({
            "type": "http",
            "method": "POST",
            "path": "/v1/test",
            "headers": [],
            "query_string": b"",
        }, receive=receive)

        with pytest.raises(RequestBodyTooLarge) as exc_info:
            await cache_bounded_request_body(request, limit=7)
        assert exc_info.value.observed == 8

    @pytest.mark.asyncio
    async def test_bounded_body_is_cached_for_later_forwarding(self):
        from starlette.requests import Request

        sent = False

        async def receive():
            nonlocal sent
            assert sent is False
            sent = True
            return {"type": "http.request", "body": b"payload", "more_body": False}

        request = Request({
            "type": "http",
            "method": "POST",
            "path": "/v1/test",
            "headers": [],
            "query_string": b"",
        }, receive=receive)
        assert await cache_bounded_request_body(request, limit=8) == b"payload"
        assert await request.body() == b"payload"
        assert sent is True

    @pytest.mark.asyncio
    async def test_global_middleware_rejects_chunked_body_before_route(self):
        called = False

        async def downstream(_scope, _receive, _send):
            nonlocal called
            called = True

        messages = iter([
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"5678", "more_body": False},
        ])
        sent = []

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        middleware = RequestBodyLimitMiddleware(downstream, limit=7)
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/admin/example",
                "headers": [],
            },
            receive,
            send,
        )

        assert called is False
        assert sent[0]["status"] == 413
        assert b"request_too_large" in sent[1]["body"]


class TestBuildGateway:
    def test_returns_fastapi_app(self):
        from fastapi import FastAPI
        app = build_gateway([])
        assert isinstance(app, FastAPI)

    def test_gateway_info_endpoint_exposes_routes(self):
        from fastapi.testclient import TestClient
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)
        r = client.get("/_gateway-info")
        assert r.status_code == 200
        data = r.json()
        model_ids = {entry["model_id"] for entry in data["routes"]}
        assert model_ids == {"soprano-80m", "sd-turbo"}


class TestProxy:
    def test_proxy_rejects_oversized_body_before_outbound_http(self):
        routes = [
            WorkerRoute(
                model_id="soprano-80m", worker_url="http://127.0.0.1:9001",
            )
        ]
        app = build_gateway(routes, max_request_body_bytes=48)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            response = client.post(
                "/v1/audio/speech",
                json={"model": "soprano-80m", "input": "x" * 128},
            )

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"
        mock_client_cls.assert_not_called()

    def test_explicit_route_is_also_protected_by_global_body_limit(self):
        app = build_gateway([], max_request_body_bytes=8)
        client = TestClient(app)

        response = client.request("GET", "/v1/models", content=b"x" * 9)

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"

    @pytest.mark.parametrize("model", [[], {}, 1, True, "", " " * 3])
    def test_non_string_or_empty_model_returns_invalid_model(self, model):
        app = build_gateway([
            WorkerRoute(
                model_id="soprano-80m",
                worker_url="http://127.0.0.1:9001",
            ),
        ])
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as outbound:
            response = client.post(
                "/v1/audio/speech",
                json={"model": model, "input": "hi"},
            )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_model"
        outbound.assert_not_called()

    def test_overlong_model_returns_invalid_model(self):
        app = build_gateway([])
        response = TestClient(app).post(
            "/v1/audio/speech",
            json={"model": "m" * 513, "input": "hi"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_model"

    def test_proxy_forwards_post_to_matching_worker(self):
        routes = [WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.aclose = AsyncMock()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.aiter_raw = _RawResponse(
                [b'{"ok": true}'],
            ).aiter_raw

            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=stream_ctx)

            mock_client_cls.return_value = mock_client

            r = client.post("/v1/audio/speech", json={
                "input": "hi", "model": "soprano-80m",
            })

        assert r.status_code == 200
        assert r.json() == {"ok": True}
        # The stream() call should have targeted the worker url
        call_kwargs = mock_client.stream.call_args.kwargs
        call_args = mock_client.stream.call_args.args
        target_url = call_args[1] if len(call_args) > 1 else call_kwargs.get("url")
        assert target_url == "http://127.0.0.1:9001/v1/audio/speech"

    def test_proxy_returns_404_openai_envelope_for_unknown_model(self):
        routes = [WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        r = client.post("/v1/audio/speech", json={
            "input": "hi", "model": "does-not-exist",
        })
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        assert "detail" not in body
        assert body["error"]["code"] == "model_not_found"
        assert "does-not-exist" in body["error"]["message"]

    def test_proxy_forwards_multipart_to_matching_worker(self):
        """Regression: extracting model from a multipart body must NOT
        consume the receive stream. Without `await request.body()`
        before `await request.form()`, _forward's later body() raises
        RuntimeError("Stream consumed") and the request fails as 500.

        Saw this live on v0.13.1 against /v1/audio/transcriptions.
        """
        routes = [WorkerRoute(model_id="whisper-tiny", worker_url="http://127.0.0.1:9099")]
        app = build_gateway(routes)
        client = TestClient(app)

        captured_body = {}

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.aclose = AsyncMock()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.aiter_raw = _RawResponse(
                [b'{"text":"hello"}'],
            ).aiter_raw

            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)

            def _capture_stream(method, url, **kwargs):
                captured_body["body"] = kwargs.get("content")
                captured_body["url"] = url
                return stream_ctx

            mock_client.stream = MagicMock(side_effect=_capture_stream)
            mock_client_cls.return_value = mock_client

            r = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", b"FAKEWAV", "audio/wav")},
                data={"model": "whisper-tiny"},
            )

        # Must not be 500 or 400 - the multipart body must have been
        # parsed for routing AND forwarded with its bytes intact.
        assert r.status_code == 200, f"got {r.status_code}: {r.text}"
        assert r.json() == {"text": "hello"}
        assert captured_body["url"] == "http://127.0.0.1:9099/v1/audio/transcriptions"
        # The forwarded body must contain the multipart payload, not be empty
        forwarded = captured_body["body"]
        assert forwarded, "forwarded body is empty (stream was consumed before forward)"
        assert b"whisper-tiny" in forwarded
        assert b"FAKEWAV" in forwarded


    def test_proxy_returns_400_when_model_not_specified(self):
        """POST without a model field: 400 (client must provide routing info)."""
        routes = [WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        r = client.post("/v1/audio/speech", json={"input": "hi"})
        assert r.status_code == 400
        body = r.json()
        assert "error" in body
        assert body["error"]["code"] == "model_required"


class TestAggregation:
    def test_v1_models_aggregates_across_workers(self):
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            def make_resp(data):
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {"object": "list", "data": data}
                return r

            responses_by_url = {
                "http://127.0.0.1:9001/v1/models": make_resp([
                    {"id": "soprano-80m", "modality": "audio/speech", "object": "model"},
                ]),
                "http://127.0.0.1:9002/v1/models": make_resp([
                    {"id": "sd-turbo", "modality": "image/generation", "object": "model"},
                ]),
            }

            async def fake_get(url, **kwargs):
                return responses_by_url[url]

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()["data"]
        ids = {m["id"] for m in data}
        assert ids == {"soprano-80m", "sd-turbo"}

    def test_v1_models_skips_unreachable_workers(self):
        """If a worker is down, its models are omitted (not a 500)."""
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9999"),  # down
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {"object": "list", "data": [
                {"id": "soprano-80m", "modality": "audio/speech", "object": "model"},
            ]}

            async def fake_get(url, **kwargs):
                if "9001" in url:
                    return r_ok
                raise httpx.ConnectError("connection refused", request=None)

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/v1/models")
        assert r.status_code == 200
        ids = {m["id"] for m in r.json()["data"]}
        assert ids == {"soprano-80m"}

    def test_health_aggregates_worker_status(self):
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            def make_resp(payload):
                r = MagicMock(status_code=200)
                r.json.return_value = payload
                return r

            responses = {
                "http://127.0.0.1:9001/health": make_resp({
                    "status": "ok", "modalities": ["audio/speech"], "models": ["soprano-80m"],
                }),
                "http://127.0.0.1:9002/health": make_resp({
                    "status": "ok", "modalities": ["image/generation"], "models": ["sd-turbo"],
                }),
            }

            async def fake_get(url, **kwargs):
                return responses[url]

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert set(body["modalities"]) == {"audio/speech", "image/generation"}
        assert set(body["models"]) == {"soprano-80m", "sd-turbo"}

    def test_v1_models_survives_worker_returning_non_json_body(self):
        """A worker that answers 200 with a non-JSON body (garbage, a
        truncated response, whatever) must degrade to "contributes
        nothing", not 500 the whole aggregated /v1/models for every
        client. Regression for the gather-blackout finding: r.json()
        raising json.JSONDecodeError (a ValueError) used to propagate
        past the httpx-only except and through a bare asyncio.gather.
        """
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {"object": "list", "data": [
                {"id": "soprano-80m", "modality": "audio/speech", "object": "model"},
            ]}
            r_bad = MagicMock(status_code=200)
            r_bad.json.side_effect = ValueError("Expecting value: line 1 column 1")

            async def fake_get(url, **kwargs):
                if "9001" in url:
                    return r_ok
                return r_bad

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/v1/models")

        assert r.status_code == 200
        ids = {m["id"] for m in r.json()["data"]}
        assert ids == {"soprano-80m"}

    def test_health_survives_worker_returning_non_json_body(self):
        """Mirror of the /v1/models regression above, for /health."""
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {
                "status": "ok", "modalities": ["audio/speech"], "models": ["soprano-80m"],
            }
            r_bad = MagicMock(status_code=200)
            r_bad.json.side_effect = ValueError("not json")

            async def fake_get(url, **kwargs):
                if "9001" in url:
                    return r_ok
                return r_bad

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/health")

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "degraded"
        assert "soprano-80m" in body["models"]
        assert "sd-turbo" not in body["models"]

    def test_health_degraded_when_any_worker_down(self):
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {
                "status": "ok", "modalities": ["audio/speech"], "models": ["soprano-80m"],
            }

            async def fake_get(url, **kwargs):
                if "9001" in url:
                    return r_ok
                raise httpx.ConnectError("down", request=None)

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/health")
        body = r.json()
        assert body["status"] == "degraded"
        assert "sd-turbo" not in body["models"]


class TestStreaming:
    def test_sse_stream_is_relayed_chunk_by_chunk(self):
        """A `stream: true` response (text/event-stream) must pass through."""
        routes = [WorkerRoute("soprano-80m", "http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        chunks = [b"data: chunk1\n\n", b"data: chunk2\n\n", b"event: done\ndata: \n\n"]

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.aclose = AsyncMock()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/event-stream"}

            async def aiter_raw():
                for c in chunks:
                    yield c
            mock_response.aiter_raw = aiter_raw
            mock_response.aclose = AsyncMock()
            mock_response.aread = AsyncMock(return_value=b"".join(chunks))

            # stream() is an async context manager
            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=stream_ctx)

            mock_client_cls.return_value = mock_client

            r = client.post("/v1/audio/speech", json={
                "input": "hi", "model": "soprano-80m", "stream": True,
            })

        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        # All chunks received in order
        assert b"data: chunk1" in r.content
        assert b"data: chunk2" in r.content
        assert b"event: done" in r.content


class TestAsyncClientLifecycle:
    """v0.34.0 finding #9: gateway must close httpx.AsyncClient when
    stream open fails so file descriptors don't leak under flaky workers."""

    def test_stream_open_failure_aclose_client(self):
        """If client.stream(...).__aenter__ raises, AsyncClient.aclose
        MUST be awaited so the connection-pool slot is released."""
        import httpx
        from muse.cli_impl import gateway as gateway_mod

        routes = [WorkerRoute("soprano-80m", "http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.aclose = AsyncMock()

            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(
                side_effect=httpx.ConnectError("worker died"),
            )
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=stream_ctx)
            mock_client_cls.return_value = mock_client

            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "soprano-80m"},
            )

            # Transport failures are normalized instead of surfacing a bare
            # FastAPI 500.
            assert r.status_code == 502
            assert r.json()["error"]["code"] == "worker_unavailable"
            # Critical: aclose must have been awaited exactly once so
            # the AsyncClient does not leak its connection pool.
            mock_client.aclose.assert_awaited_once()

    def test_stream_open_timeout_maps_504(self):
        routes = [WorkerRoute("soprano-80m", "http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.aclose = AsyncMock()
            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(
                side_effect=httpx.ReadTimeout("worker stalled"),
            )
            mock_client.stream = MagicMock(return_value=stream_ctx)
            mock_client_cls.return_value = mock_client

            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "soprano-80m"},
            )

        assert r.status_code == 504
        assert r.json()["error"]["code"] == "worker_timeout"
        mock_client.aclose.assert_awaited_once()

    def test_cleanup_failure_does_not_mask_transport_error(self):
        routes = [WorkerRoute("soprano-80m", "http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app, raise_server_exceptions=False)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.aclose = AsyncMock(side_effect=RuntimeError("close failed"))
            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(
                side_effect=httpx.ConnectError("worker died"),
            )
            mock_client.stream = MagicMock(return_value=stream_ctx)
            mock_client_cls.return_value = mock_client

            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "soprano-80m"},
            )

        assert r.status_code == 502
        assert r.json()["error"]["code"] == "worker_unavailable"


class TestAdminMount:
    """Verify /v1/admin/* lands on the admin router with auth enforced."""

    def test_admin_path_without_token_returns_503(self, tmp_path, monkeypatch):
        from muse.admin.auth import ADMIN_TOKEN_ENV
        from muse.core import config

        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
        monkeypatch.setenv("MUSE_CONFIG", str(tmp_path / "absent-config.yaml"))
        config.reset_config()
        try:
            app = build_gateway([])
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/v1/admin/workers")
            assert r.status_code == 503
            # v0.47.4: bare OpenAI envelope, not the double-wrapped
            # {"detail": {"error": ...}} the default handler would produce.
            body = r.json()
            assert body["error"]["code"] == "admin_disabled"
            assert "detail" not in body
        finally:
            config.reset_config()

    def test_admin_path_with_token_passes_auth(self, monkeypatch):
        from muse.admin.auth import ADMIN_TOKEN_ENV
        from muse.cli_impl.supervisor import (
            SupervisorState,
            clear_supervisor_state,
            set_supervisor_state,
        )
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "tok")
        clear_supervisor_state()
        set_supervisor_state(SupervisorState(workers=[], device="cpu"))
        try:
            app = build_gateway([])
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get(
                "/v1/admin/workers",
                headers={"Authorization": "Bearer tok"},
            )
            assert r.status_code == 200
            assert r.json() == {"workers": []}
        finally:
            clear_supervisor_state()

    def test_admin_path_with_wrong_token_returns_403(self, monkeypatch):
        from muse.admin.auth import ADMIN_TOKEN_ENV
        monkeypatch.setenv(ADMIN_TOKEN_ENV, "tok")
        app = build_gateway([])
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get(
            "/v1/admin/workers",
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 403

    def test_inference_proxy_still_works_after_admin_mount(self):
        """Regression: /v1/* with body['model'] still hits the proxy."""
        # Use the proxy with no actual workers; it should return 404
        # model_not_found, NOT some admin-route shadow.
        app = build_gateway([])
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post(
            "/v1/audio/speech",
            json={"input": "hi", "model": "ghost"},
        )
        assert r.status_code == 404
        body = r.json()
        # The proxy uses {"error": {...}} (not detail)
        assert body["error"]["code"] == "model_not_found"


class TestAggregationTimeout:
    """/v1/models and /health build a per-worker httpx.AsyncClient to fan
    out GET requests. Before this fix that client's timeout was hardcoded
    to 5.0, ignoring build_gateway's own configurable timeout knob, so a
    >5s (but otherwise healthy) worker was silently dropped from both
    endpoints with no way for an operator to loosen it.
    """

    def test_v1_models_default_aggregation_timeout_is_5s(self):
        """Unchanged default: no aggregation_timeout given -> 5.0s, same
        as before this fix."""
        routes = [WorkerRoute("soprano-80m", "http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {"object": "list", "data": []}

            async def fake_get(url, **kwargs):
                return r_ok

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            client.get("/v1/models")

        mock_client_cls.assert_called_once_with(timeout=5.0)

    def test_v1_models_uses_configured_aggregation_timeout(self):
        routes = [WorkerRoute("soprano-80m", "http://127.0.0.1:9001")]
        app = build_gateway(routes, aggregation_timeout=12.5)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {"object": "list", "data": []}

            async def fake_get(url, **kwargs):
                return r_ok

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            client.get("/v1/models")

        mock_client_cls.assert_called_once_with(timeout=12.5)

    def test_health_uses_configured_aggregation_timeout(self):
        routes = [WorkerRoute("soprano-80m", "http://127.0.0.1:9001")]
        app = build_gateway(routes, aggregation_timeout=12.5)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {
                "status": "ok", "modalities": [], "models": [],
            }

            async def fake_get(url, **kwargs):
                return r_ok

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            client.get("/health")

        mock_client_cls.assert_called_once_with(timeout=12.5)

    def test_aggregation_timeout_reads_config_when_not_passed(self, monkeypatch):
        """The `None` sentinel (build_gateway's default) resolves through
        muse.core.config, so MUSE_AGGREGATION_TIMEOUT_SECONDS works too."""
        monkeypatch.setenv("MUSE_AGGREGATION_TIMEOUT_SECONDS", "9.0")
        routes = [WorkerRoute("soprano-80m", "http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {"object": "list", "data": []}

            async def fake_get(url, **kwargs):
                return r_ok

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            client.get("/v1/models")

        mock_client_cls.assert_called_once_with(timeout=9.0)
