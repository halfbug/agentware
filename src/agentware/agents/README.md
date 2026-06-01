# Agents

Each agent lives in its **own folder** with code, tests, examples, and a README.

## Available agents

| Agent | Folder | Description |
|-------|--------|-------------|
| **ReAct assistant** | [react/](react/) | Single agent with tools — subclass `BaseLangGraphAgent` |
| **Deep research** | [deep_research/](deep_research/) | Multi-subgraph orchestrator with search tools |

## Adding a new agent

1. Create `src/agentware/agents/<your_agent>/`
2. Add:
   - `__init__.py` — public exports
   - `agent.py` — main class (usually extends `BaseLangGraphAgent`)
   - `state.py` — TypedDict state (optional)
   - `README.md` — purpose, graph diagram, usage, extension guide
3. Register exports in `agents/__init__.py` and top-level `agentware/__init__.py`
4. Add `examples/<your_agent>/` and `tests/agents/<your_agent>/`

Depend only on `agentware.core` (and `agentware.tools` if you need MCP). Do not import other agents directly.
