"""muse mcp: MCP server bridging muse capabilities to LLM clients.

`muse mcp` exposes 31 tools (11 admin + 20 inference) over the Model
Context Protocol. Stdio mode is the default (for desktop apps); HTTP+SSE
mode is available for remote / web embedders.

See docs/superpowers/specs/2026-04-28-muse-mcp-design.md for the full
spec; CLAUDE.md "Using muse from Claude Desktop" for the config example.
"""
from muse.mcp.server import MCPServer, build_tools

__all__ = ["MCPServer", "build_tools"]
