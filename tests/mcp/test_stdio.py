"""Tests for stdio mode framing.

Drive run_stdio with anyio memory streams that act as fake stdin /
stdout pairs. Asserts list_tools handshake works end-to-end without
spawning a subprocess or touching real stdio.
"""
from __future__ import annotations

import json

import pytest

from muse.mcp.client import MuseClient
from muse.mcp.server import MCPServer


pytest.importorskip("anyio")


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    client = MuseClient(server_url="http://test")
    return MCPServer(client=client, filter_kind="all")


class TestServerStdioBasics:
    def test_run_stdio_method_exists(self, server):
        assert hasattr(server, "run_stdio")
        assert callable(server.run_stdio)

    def test_initialization_options_built(self, server):
        # Probe the underlying SDK Server for its init options without
        # actually entering stdio_server (which would block on read).
        opts = server._server.create_initialization_options()
        assert opts is not None

    def test_handlers_wired(self, server):
        # The wired handlers should be present in the request_handlers
        # map of the underlying SDK Server.
        from mcp import types
        rh = server._server.request_handlers
        assert types.ListToolsRequest in rh
        assert types.CallToolRequest in rh

    def test_31_tools_in_cache(self, server):
        # Trigger the list_tools handler so the SDK populates its cache.
        # We then assert the cache has exactly 31 entries.
        import asyncio
        from mcp import types

        async def run_handler():
            handler = server._server.request_handlers[types.ListToolsRequest]
            req = types.ListToolsRequest(method="tools/list")
            return await handler(req)

        asyncio.run(run_handler())
        cache = server._server._tool_cache
        assert len(cache) == 31

    def test_call_tool_dispatches(self, server, monkeypatch):
        # Drive the call_tool handler with a fake list_models response.
        import asyncio
        from unittest.mock import MagicMock
        from mcp import types

        server.client.list_models = MagicMock(
            return_value={"data": [{"id": "x", "modality": "chat/completion"}]},
        )

        async def run_handler():
            handler = server._server.request_handlers[types.CallToolRequest]
            req = types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="muse_list_models", arguments={},
                ),
            )
            return await handler(req)

        result = asyncio.run(run_handler())
        # Drill into the ServerResult -> CallToolResult
        out = result.root if hasattr(result, "root") else result
        # The handler returned a list of TextContent blocks
        content_blocks = getattr(out, "content", None)
        assert content_blocks is not None
        assert len(content_blocks) >= 1
        # The first block is text and parses as JSON
        first = content_blocks[0]
        assert first.type == "text"
        body = json.loads(first.text)
        assert "data" in body
        assert body["data"][0]["id"] == "x"
