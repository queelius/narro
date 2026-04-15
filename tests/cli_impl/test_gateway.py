"""Tests for the gateway proxy FastAPI app."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from muse.cli_impl.gateway import (
    extract_model_from_request,
    build_gateway,
    WorkerRoute,
)


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
    async def test_returns_none_when_content_type_not_json(self):
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "multipart/form-data"}
        model = await extract_model_from_request(request)
        assert model is None


class TestWorkerRoute:
    def test_worker_route_stores_model_and_url(self):
        r = WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")
        assert r.model_id == "soprano-80m"
        assert r.worker_url == "http://127.0.0.1:9001"


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
            mock_response.aread = AsyncMock(return_value=b'{"ok": true}')

            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=stream_ctx)

            mock_client_cls.return_value = mock_client

            r = client.post("/v1/audio/speech", json={
                "input": "hi", "model": "soprano-80m",
            })

        assert r.status_code == 200
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
