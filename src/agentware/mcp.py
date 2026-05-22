"""Optional MCP tool loading via langchain-mcp-adapters."""

from __future__ import annotations

from typing import Any, Dict, List


async def load_mcp_tools(servers: Dict[str, Dict[str, Any]]) -> List[Any]:
    """Load LangChain tools from one or more MCP servers.

    Requires: ``pip install agentware[mcp]``

    Args:
        servers: Config dict passed to ``MultiServerMCPClient``, e.g.::

            {
                "math": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["/path/to/math_server.py"],
                },
                "api": {
                    "transport": "streamable_http",
                    "url": "http://localhost:8000/mcp",
                },
            }

    Returns:
        List of LangChain-compatible tools for use with ``BaseLangGraphAgent(tools=...)``.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        raise ImportError(
            "MCP support requires: pip install agentware[mcp]"
        ) from e

    async with MultiServerMCPClient(servers) as client:
        return await client.get_tools()
