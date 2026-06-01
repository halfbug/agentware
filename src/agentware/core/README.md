# Core

Shared foundation used by **every** agent in Agentware. Import from `agentware.core` or the top-level `agentware` package.

## Modules

| Module | Purpose |
|--------|---------|
| `base.py` | `BaseLangGraphAgent` — ReAct graph, invoke/ainvoke, checkpointing, token tracking |
| `state.py` | `BaseAgentState` — default `messages` + `llm_calls` |
| `types.py` | `AgentRunResult` — content, token_usage, messages |
| `tokens.py` | `TokenUsage`, `aggregate_token_usage()` |
| `checkpointer.py` | `build_checkpointer()` — memory or MongoDB |
| `shared/` | `merge_shared_variables`, `set_shared_value` — for multi-agent subgraphs |

## Building a new agent

1. Subclass `BaseLangGraphAgent` (or compose it).
2. Optionally extend `BaseAgentState` for extra state keys.
3. Override `_build_graph()` for custom workflows; use `core.shared` if subgraphs share a variable array.
4. Override `_to_result()` if your agent returns extra fields.

See [agents/react/README.md](../agents/react/README.md) for the standard tool-calling agent pattern.
