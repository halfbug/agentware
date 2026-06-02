# ReAct assistant agent

The standard **single-agent** pattern: one LLM, optional tools, ReAct loop (agent → tools → agent).

Use `ReactAgent` from this folder (a thin wrapper over `BaseLangGraphAgent`) when you want an explicit agent package path.

## When to use

- Chat assistants with function calling
- Support bots with search/API tools
- Any single-threaded tool-using agent

## Quick start

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from agentware.agents.react import ReactAgent

@tool
def search(q: str) -> str:
    """Search docs."""
    return f"Results for {q}"

agent = ReactAgent(
    system_prompt="You are helpful.",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search],
)

result = agent.invoke("How do I reset my password?", thread_id="user-1")
print(result.content, result.token_usage.total_tokens)
```

## Extending

| Hook | Use for |
|------|---------|
| `enrich_tool_args()` | Inject user_id, tenant, scopes into tool calls |
| `prepare_messages()` | Custom message formatting |
| `state_schema` | Extra state fields on the graph |
| `_build_graph()` | Completely custom graph (rare) |

## Examples

- [examples/react/custom_agent.py](../../../../examples/react/custom_agent.py)
- [examples/react/fastapi_app.py](../../../../examples/react/fastapi_app.py)

## Core dependency

All behavior comes from [core/README.md](../../core/README.md) — `BaseLangGraphAgent`, checkpointing, tokens.
