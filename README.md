# Agentware

Extensible **LangGraph** multi-agent library: a base agent class you subclass, injectable prompts and tools (manual or MCP), per-message **token totals**, and optional **FastAPI** integration.

## Install

```bash
# Core
pip install agentware

# With OpenAI, MCP tools, MongoDB checkpointing, FastAPI
pip install agentware[openai,mcp,mongodb,fastapi]
```

From source (development):

```bash
git clone https://github.com/YOUR_USERNAME/agentware.git
cd agentware
pip install -e ".[dev,openai]"
```

## Quick start

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from agentware import BaseLangGraphAgent

@tool
def search(q: str) -> str:
    """Search docs."""
    return f"Results for {q}"

agent = BaseLangGraphAgent(
    system_prompt="You are a helpful assistant.",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search],
)

result = agent.invoke("How do I reset my password?", thread_id="user-42")
print(result.content)
print(result.token_usage.total_tokens)  # tokens for this turn
```

## Extend the base class

Match your existing pattern: subclass, override `enrich_tool_args`, custom state:

```python
from typing import Any, Dict, List, Optional
from agentware import BaseLangGraphAgent, BaseAgentState

class AssistantState(BaseAgentState):
    publication_ids: Optional[List[str]]

class AIAssistantAgent(BaseLangGraphAgent):
    state_schema = AssistantState

    def enrich_tool_args(self, tool_name, args, state):
        if tool_name == "content_search":
            args["publication_ids_array"] = state.get("publication_ids") or self.publication_ids
        return args
```

See [examples/custom_agent.py](examples/custom_agent.py).

## Token usage

Each `invoke` / `ainvoke` returns `AgentRunResult` with `token_usage`:

| Field | Meaning |
|-------|---------|
| `input_tokens` | Sum of prompt tokens across LLM steps |
| `output_tokens` | Sum of completion tokens |
| `total_tokens` | Sum of totals |
| `llm_calls` | Number of agent→LLM node executions |

Usage is read from `AIMessage.usage_metadata` (standard in LangChain 0.3+). Ensure your chat model returns usage (OpenAI/Anthropic integrations do by default).

## MCP tools

```python
import asyncio
from agentware import BaseLangGraphAgent
from agentware.mcp import load_mcp_tools

async def main():
    tools = await load_mcp_tools({
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        },
    })
    agent = BaseLangGraphAgent(
        system_prompt="You can use MCP tools.",
        llm=...,  # your ChatOpenAI / ChatAnthropic
        tools=tools,
    )
    result = await agent.ainvoke("List files in /tmp")
    print(result.token_usage)

asyncio.run(main())
```

Requires `pip install agentware[mcp]`.

## FastAPI

```python
from fastapi import FastAPI
from agentware.integrations.fastapi import create_chat_router

app = FastAPI()
app.include_router(create_chat_router(lambda: my_agent))

# POST /agent/chat
# {"message": "hello", "thread_id": "user-1"}
# → {"content": "...", "total_tokens": 42, ...}
```

Run: `uvicorn examples.fastapi_app:app --reload`

## Checkpointing

- Default: in-memory (`MemorySaver`)
- MongoDB: pass `mongo_uri` and `mongo_database`, or inject a custom `checkpointer=`

```python
agent = BaseLangGraphAgent(
    system_prompt="...",
    llm=llm,
    mongo_uri="mongodb://localhost:27017",
    mongo_database="agentware",
)
```

Use the same `thread_id` per conversation to restore state.

## Publish to PyPI (pip install from GitHub release)

### 1. Prepare GitHub repo

1. Create repo `https://github.com/YOUR_USERNAME/agentware`
2. Update `pyproject.toml`: `authors`, `[project.urls]` Homepage/Repository
3. Push code:

```bash
git init
git add .
git commit -m "Initial release: LangGraph base agent framework"
git remote add origin git@github.com:YOUR_USERNAME/agentware.git
git push -u origin main
```

### 2. Create PyPI account and API token

1. Register at [pypi.org](https://pypi.org)
2. Account → API tokens → “Add API token” (scope: entire account or project `agentware`)
3. Save the token (starts with `pypi-`)

### 3. Build and upload

```bash
pip install build twine
python -m build
twine upload dist/*
# Username: __token__
# Password: <your pypi API token>
```

After upload, anyone can install:

```bash
pip install agentware
```

### 4. Install directly from GitHub (before PyPI)

```bash
pip install git+https://github.com/YOUR_USERNAME/agentware.git
# specific tag:
pip install git+https://github.com/YOUR_USERNAME/agentware.git@v0.1.0
```

### 5. Versioning and releases

1. Bump version in `pyproject.toml` and `src/agentware/__init__.py`
2. Tag: `git tag v0.1.0 && git push origin v0.1.0`
3. Create GitHub Release from the tag
4. Rebuild and `twine upload` for each release

### 6. Optional: TestPyPI first

```bash
twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ agentware
```

## Project layout

```
agentware/
├── pyproject.toml          # package metadata & dependencies
├── src/agentware/
│   ├── base.py             # BaseLangGraphAgent
│   ├── state.py            # BaseAgentState
│   ├── tokens.py           # Token aggregation
│   ├── types.py            # AgentRunResult
│   ├── mcp.py              # MCP tool loader
│   └── integrations/
│       └── fastapi.py
├── examples/
└── tests/
```

## Development

```bash
pip install -e ".[dev,openai]"
pytest
ruff check src tests
```

## License

MIT
