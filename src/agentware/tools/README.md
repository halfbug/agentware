# Tools

Shared tool loaders used by any agent.

## MCP

```python
from agentware.tools import load_mcp_tools

tools = await load_mcp_tools({
    "server": {
        "transport": "stdio",
        "command": "python",
        "args": ["my_mcp_server.py"],
    },
})
```

Requires `pip install agentware[mcp]`.

Pass returned tools to `BaseLangGraphAgent(tools=...)` or `DeepResearchOrchestrator(search_tools=...)`.
