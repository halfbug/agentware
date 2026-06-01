# Project structure

```
agentware/
├── README.md                 # User guide (install, use, extend)
├── docs/
│   └── PROJECT_STRUCTURE.md  # This file
├── examples/
│   ├── README.md
│   ├── react/                # BaseLangGraphAgent examples
│   └── deep_research/        # Deep research examples
├── tests/
│   ├── core/
│   └── agents/
│       └── deep_research/
├── pyproject.toml
└── src/agentware/
    ├── __init__.py           # Public API (re-exports)
    │
    ├── core/                 # ★ Shared by ALL agents
    │   ├── README.md
    │   ├── base.py           # BaseLangGraphAgent
    │   ├── state.py
    │   ├── types.py
    │   ├── tokens.py
    │   ├── checkpointer.py
    │   └── shared/           # Multi-agent state helpers
    │       └── variables.py
    │
    ├── agents/               # ★ One folder per agent
    │   ├── README.md
    │   ├── react/            # Docs for standard ReAct pattern
    │   │   └── README.md
    │   └── deep_research/
    │       ├── README.md
    │       ├── agent.py
    │       ├── state.py
    │       ├── subgraphs.py
    │       └── types.py
    │
    ├── tools/                # MCP and shared tool loaders
    │   ├── README.md
    │   └── mcp.py
    │
    └── integrations/         # FastAPI, etc.
        ├── README.md
        └── fastapi.py
```

## Import paths

| What | Import |
|------|--------|
| Base agent | `from agentware import BaseLangGraphAgent` |
| Core internals | `from agentware.core import BaseLangGraphAgent` |
| Deep research | `from agentware.agents.deep_research import DeepResearchOrchestrator` |
| MCP tools | `from agentware.tools import load_mcp_tools` |
| FastAPI | `from agentware.integrations.fastapi import create_chat_router` |

## Adding an agent

1. `src/agentware/agents/<name>/` with `README.md`, `agent.py`, `__init__.py`
2. Use only `agentware.core` (+ `tools` / `integrations` as needed)
3. `examples/<name>/` and `tests/agents/<name>/`
4. Export from `agents/__init__.py` and `agentware/__init__.py`
