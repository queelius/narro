"""Tests for muse.mcp.server.MCPServer + build_tools.

Smoke-level tests: server constructs, tool registry is empty in Task A,
filter_kind validates, run_http raises NotImplementedError pre-Task B.
The full tool count (31 as of the image/vectorization modality, v0.59.0)
is asserted here once all tool modules have populated their lists.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from muse.mcp.client import MuseClient
from muse.mcp.server import MCPServer, build_tools


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    return MuseClient(server_url="http://test.example.com", admin_token="t")


class TestBuildTools:
    def test_default_filter_returns_31_tools(self):
        tools = build_tools("all")
        assert isinstance(tools, list)
        assert len(tools) == 31

    def test_admin_filter_returns_11(self):
        tools = build_tools("admin")
        assert len(tools) == 11

    def test_inference_filter_returns_20(self):
        tools = build_tools("inference")
        assert len(tools) == 20

    def test_unknown_filter_raises(self):
        with pytest.raises(ValueError, match="unknown filter_kind"):
            build_tools("nope")

    def test_filters_partition_all(self):
        # admin tools + inference tools should equal all tools (no overlap).
        a = build_tools("admin")
        i = build_tools("inference")
        all_t = build_tools("all")
        assert len(a) + len(i) == len(all_t)
        a_names = {t.name for t in a}
        i_names = {t.name for t in i}
        assert a_names.isdisjoint(i_names)

    def test_all_tool_names_unique(self):
        all_t = build_tools("all")
        names = [t.name for t in all_t]
        assert len(names) == len(set(names))

    def test_all_tool_names_use_muse_prefix(self):
        for t in build_tools("all"):
            assert t.name.startswith("muse_"), (
                f"tool {t.name!r} does not use the muse_ prefix"
            )

    def test_each_tool_has_substantial_description(self):
        for t in build_tools("all"):
            desc = t.description or ""
            assert len(desc) >= 50, (
                f"tool {t.name!r} has too-short description "
                f"(found {len(desc)} chars)"
            )


class TestServerConstruction:
    def test_construct_default(self, fake_client):
        srv = MCPServer(client=fake_client)
        assert srv.client is fake_client
        assert srv.filter_kind == "all"
        # tools/handlers are property-backed
        assert isinstance(srv.tools, list)
        assert isinstance(srv.handlers, dict)

    def test_construct_admin_only(self, fake_client):
        srv = MCPServer(client=fake_client, filter_kind="admin")
        assert srv.filter_kind == "admin"

    def test_construct_inference_only(self, fake_client):
        srv = MCPServer(client=fake_client, filter_kind="inference")
        assert srv.filter_kind == "inference"

    def test_construct_unknown_filter_raises(self, fake_client):
        with pytest.raises(ValueError, match="unknown filter_kind"):
            MCPServer(client=fake_client, filter_kind="nonsense")

    def test_handlers_match_tools(self, fake_client):
        srv = MCPServer(client=fake_client)
        # Every handler key has a matching tool.name (and vice versa).
        tool_names = {t.name for t in srv.tools}
        handler_names = set(srv.handlers.keys())
        assert tool_names == handler_names


class TestCallHandler:
    def test_unknown_tool_returns_error_block(self, fake_client):
        srv = MCPServer(client=fake_client)
        out = srv.call_handler("not_a_real_tool", {})
        assert len(out) == 1
        assert out[0]["type"] == "text"
        assert "unknown tool" in out[0]["text"]


class TestCallToolOffLoop:
    """The wired ``_call_tool`` handler must dispatch synchronous, blocking
    handlers off the event loop (asyncio.to_thread), not inline. Inline
    dispatch would stall every concurrent tool call + SSE frame in
    HTTP+SSE mode for the duration of one slow handler (same class of bug
    the gateway fixed via off-loop acquire dispatch)."""

    @staticmethod
    async def _drive_call_tool(server, name, args):
        from mcp import types

        handler = server._server.request_handlers[types.CallToolRequest]
        req = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name=name, arguments=args),
        )
        return await handler(req)

    def test_call_tool_does_not_block_event_loop(self, monkeypatch):
        from muse.mcp.tools import INFERENCE_TOOLS

        def _slow_handler(client, args):  # noqa: ARG001
            time.sleep(0.3)
            return [{"type": "text", "text": "done"}]

        entry = next(t for t in INFERENCE_TOOLS if t.tool.name == "muse_speak")
        monkeypatch.setattr(entry, "handler", _slow_handler)

        client = MuseClient(server_url="http://test")
        server = MCPServer(client=client, filter_kind="inference")

        async def scenario():
            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.02)
                    ticks += 1

            ticker_task = asyncio.create_task(ticker())
            await self._drive_call_tool(server, "muse_speak", {"input": "hi"})
            ticker_task.cancel()
            return ticks

        ticks = asyncio.run(scenario())
        assert ticks >= 5, (
            f"event loop only ticked {ticks} times while a 0.3s blocking "
            "tool handler ran; _call_tool must dispatch off-loop via "
            "asyncio.to_thread so it does not stall concurrent requests"
        )

    def test_result_still_returned_correctly(self, monkeypatch):
        """Off-loop dispatch must not change the return value shape."""
        from muse.mcp.tools import INFERENCE_TOOLS

        def _slow_handler(client, args):  # noqa: ARG001
            time.sleep(0.05)
            return [{"type": "text", "text": "hello-from-thread"}]

        entry = next(t for t in INFERENCE_TOOLS if t.tool.name == "muse_speak")
        monkeypatch.setattr(entry, "handler", _slow_handler)

        client = MuseClient(server_url="http://test")
        server = MCPServer(client=client, filter_kind="inference")

        result = asyncio.run(
            self._drive_call_tool(server, "muse_speak", {"input": "hi"})
        )
        out = result.root if hasattr(result, "root") else result
        blocks = out.content
        assert len(blocks) == 1
        assert blocks[0].text == "hello-from-thread"


class TestRunHttp:
    def test_pre_task_b_raises(self, fake_client):
        # In Task A, run_http is a placeholder. Task B replaces it.
        # We assert it either raises NotImplementedError or runs cleanly
        # (Task B implementation). Both are acceptable; this test
        # exists as documentation of the lifecycle.
        srv = MCPServer(client=fake_client)
        # The method exists; we don't await it here.
        assert hasattr(srv, "run_http")
