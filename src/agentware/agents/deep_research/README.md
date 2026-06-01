# Deep research agent

Multi-agent **orchestrator** that plans sub-queries, runs search tools, and synthesizes a report. Extends `BaseLangGraphAgent` with LangGraph **subgraphs** and **shared state**.

## Graph

```
START → init → plan (subgraph) → research (subgraph) → synthesize (subgraph) → END
```

| Subgraph | Role |
|----------|------|
| `plan` | Topic → `research_queries` |
| `research` | ReAct loop over `search_tools` → `findings`, `sources` |
| `synthesize` | Final report from findings |

Parent and subgraphs share `DeepResearchState` keys (`research_queries`, `findings`, `shared_variables`, …).

## Quick start

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from agentware.agents.deep_research import DeepResearchOrchestrator, SharedVariable

@tool
def web_search(query: str) -> str:
    """Search the web."""
    ...

orchestrator = DeepResearchOrchestrator(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    search_tools=[web_search],
)

result = orchestrator.research(
    "Impact of RAG on enterprise search",
    shared_variables=[SharedVariable(key="locale", value="en")],
)
print(result.content)
```

## Files

| File | Purpose |
|------|---------|
| `agent.py` | `DeepResearchOrchestrator` |
| `state.py` | `DeepResearchState`, `SharedVariable`, `ResearchFinding` |
| `subgraphs.py` | `build_plan_subgraph`, `build_research_subgraph`, `build_synthesize_subgraph` |
| `types.py` | `DeepResearchResult` |

## Extending

- Override `enrich_tool_args` on a subclass (or pass a callback) for search API keys / tenant.
- Customize prompts: `planner_prompt`, `researcher_prompt`, `synthesizer_prompt` in constructor.
- Add subgraphs: subclass and override `_build_graph()`.
- Use `shared_variables` for cross-subgraph config without changing `DeepResearchState`.

## Inherited from core

- `invoke` / `ainvoke` / `research` / `aresearch`
- Token usage via `result.token_usage`
- Conversation history via `thread_id` + checkpointer

## Example

[examples/deep_research/run.py](../../../../examples/deep_research/run.py)
