# Agentware

Build LangGraph agents with a small, extensible base class. Pass your own **system prompt** and **tools** (hand-written or from MCP servers), get **token usage per message**, and plug into **FastAPI** or any Python app.

**Requirements:** Python 3.10+

## Project layout

```
src/agentware/
├── core/              # Shared base class, tokens, checkpointing (all agents use this)
├── agents/            # One folder per agent (each has its own README)
│   ├── react/         # Standard ReAct assistant (BaseLangGraphAgent)
│   └── deep_research/ # Multi-subgraph research orchestrator
├── tools/             # MCP and shared tool loaders
└── integrations/      # FastAPI, etc.
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for full details and how to add new agents.

---

## Install

**From PyPI** (when published):

```bash
pip install agentware
```

**From GitHub:**

```bash
pip install git+https://github.com/halfbug/agentware.git
```

**Optional extras** — install only what you need:

```bash
# OpenAI or Anthropic models
pip install agentware[openai]
pip install agentware[anthropic]

# MCP tools, MongoDB conversation memory, FastAPI helper
pip install agentware[mcp]
pip install agentware[mongodb]
pip install agentware[fastapi]

# Everything
pip install agentware[openai,mcp,mongodb,fastapi]
```

Set your model API key in the environment (e.g. `OPENAI_API_KEY` for OpenAI).

---

## Use

### Minimal agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from agentware import BaseLangGraphAgent

@tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Results for: {query}"

agent = BaseLangGraphAgent(
    system_prompt="You are a helpful assistant.",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search_docs],
)

result = agent.invoke(
    "How do I reset my password?",
    thread_id="user-42",  # same id = same conversation
)

print(result.content)
print(result.token_usage.total_tokens)
```

### Async

```python
result = await agent.ainvoke("Hello", thread_id="user-42")
```

### Response fields

Each call returns an `AgentRunResult`:

| Field | Description |
|-------|-------------|
| `content` | Final assistant reply (text) |
| `token_usage.input_tokens` | Prompt tokens (this turn) |
| `token_usage.output_tokens` | Completion tokens (this turn) |
| `token_usage.total_tokens` | Total tokens (this turn) |
| `token_usage.llm_calls` | How many times the LLM ran in the loop |
| `messages` | Full message history including tool messages |

Token counts are summed across all LLM steps in one user message (including tool-call rounds).

### Constructor options

| Argument | Description |
|----------|-------------|
| `system_prompt` | System message prepended on each LLM call |
| `llm` | Any LangChain `BaseChatModel` (OpenAI, Anthropic, etc.) |
| `tools` | List of `@tool` functions or LangChain tools |
| `checkpointer` | Custom LangGraph checkpointer (advanced) |
| `mongo_uri`, `mongo_database` | Persist conversations in MongoDB |

---

## Tools

### Your own tools

Use LangChain’s `@tool` decorator and pass them in the constructor:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

agent = BaseLangGraphAgent(
    system_prompt="...",
    llm=llm,
    tools=[get_weather],
)
```

### MCP servers

Load tools from Model Context Protocol servers:

```python
import asyncio
from langchain_openai import ChatOpenAI
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
        system_prompt="You can use the provided tools.",
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=tools,
    )
    result = await agent.ainvoke("What files are in /tmp?")
    print(result.content)

asyncio.run(main())
```

Requires `pip install agentware[mcp]`.

---

## Conversation memory

Use a stable `thread_id` per user or chat session so the agent remembers prior turns:

```python
result = agent.invoke("What did I just ask?", thread_id="session-abc")
```

By default, history is kept in memory. For production persistence:

```python
agent = BaseLangGraphAgent(
    system_prompt="...",
    llm=llm,
    mongo_uri="mongodb://localhost:27017",
    mongo_database="myapp",
)
```

Requires `pip install agentware[mongodb]`.

---

## FastAPI

Expose your agent as an HTTP API:

```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from agentware import BaseLangGraphAgent
from agentware.integrations.fastapi import create_chat_router

def make_agent():
    return BaseLangGraphAgent(
        system_prompt="You are a helpful assistant.",
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[],
    )

app = FastAPI()
app.include_router(create_chat_router(make_agent))
```

**Request** — `POST /agent/chat`:

```json
{
  "message": "Hello",
  "thread_id": "user-1"
}
```

**Response:**

```json
{
  "content": "Hi! How can I help?",
  "input_tokens": 12,
  "output_tokens": 8,
  "total_tokens": 20,
  "llm_calls": 1
}
```

Run locally:

```bash
pip install agentware[fastapi,openai]
uvicorn myapp:app --reload
```

---

## Deep research (multi-agent subgraphs)

`DeepResearchOrchestrator` **extends** `BaseLangGraphAgent`, so you get the same `invoke` / `ainvoke` API, **token usage**, **conversation history** (`thread_id` + checkpointer), and `enrich_tool_args` — with a multi-subgraph research workflow on top.

It runs three LangGraph **subgraphs** that share the same state as the parent graph:

```
START → init → plan (subgraph) → research (subgraph) → synthesize (subgraph) → END
```

| Subgraph | Role |
|----------|------|
| **plan** | Breaks the topic into `research_queries` |
| **research** | ReAct loop over your **search engine tools**; fills `findings` and `sources` |
| **synthesize** | Writes the final report from accumulated findings |

### Shared state

Parent and subgraphs all use `DeepResearchState`:

- **Lists** (append across steps): `research_queries`, `findings`, `sources`, `messages`
- **Shared array**: `shared_variables` — `[{"key": "...", "value": ...}, ...]` merged by key across graphs

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from agentware import DeepResearchOrchestrator, SharedVariable

@tool
def web_search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"

@tool
def news_search(query: str) -> str:
    """Search news."""
    return f"News for {query}"

orchestrator = DeepResearchOrchestrator(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    search_tools=[web_search, news_search],  # one or more search tools
    max_research_rounds=5,
)

result = orchestrator.research(
    "Latest advances in quantum error correction",
    thread_id="session-1",
    shared_variables=[
        SharedVariable(key="locale", value="en"),
        SharedVariable(key="depth", value="deep"),
    ],
)

print(result.content)              # final report
print(result.research_queries)     # planned sub-queries
print(result.findings)             # raw research notes
print(result.token_usage.total_tokens)
print(result.shared_variables)     # includes phase: complete
```

Async: `await orchestrator.aresearch(topic, ...)` or `await orchestrator.ainvoke(topic, ...)`.

Pass MCP-loaded search tools the same way as manual tools. Override `enrich_tool_args` on a subclass, or pass an `enrich_tool_args` callback, to inject API keys or tenant into search tool calls.

See [examples/deep_research/run.py](examples/deep_research/run.py) and [src/agentware/agents/deep_research/README.md](src/agentware/agents/deep_research/README.md).

---

## Extend

Subclass `BaseLangGraphAgent` when you need custom behavior, extra graph state, or tool arguments injected at runtime.

### Inject extra tool arguments

Override `enrich_tool_args` to add context (user id, tenant, publication ids, etc.):

```python
from typing import Any, Dict, List, Optional
from agentware import BaseLangGraphAgent, BaseAgentState

class AIAssistantAgent(BaseLangGraphAgent):
    def __init__(self, system_prompt: str, publication_ids: Optional[List[str]] = None, **kwargs):
        super().__init__(system_prompt=system_prompt, tools=[...], **kwargs)
        self.publication_ids = publication_ids or []

    def enrich_tool_args(self, tool_name: str, args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "content_search":
            args["publication_ids_array"] = state.get("publication_ids") or self.publication_ids
        return args
```

### Custom graph state

Extend `BaseAgentState` and set `state_schema` on your class:

```python
from typing import List, Optional
from agentware import BaseAgentState, BaseLangGraphAgent

class AssistantState(BaseAgentState):
    publication_ids: Optional[List[str]]

class AIAssistantAgent(BaseLangGraphAgent):
    state_schema = AssistantState

    # Pass state on invoke:
    # agent.invoke("...", extra_state={"publication_ids": ["pub-1"]})
```

### Other hooks

| Method | When to override |
|--------|----------------|
| `prepare_messages(messages)` | Change how messages are built before the LLM |
| `enrich_tool_args(...)` | Add runtime fields to tool calls |
| `_build_graph()` | Custom LangGraph nodes and edges (advanced) |

A full working example is in [examples/react/custom_agent.py](examples/react/custom_agent.py).

---

## License

MIT — see [LICENSE](LICENSE).
