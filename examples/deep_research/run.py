"""Deep research orchestrator with search tools and shared subgraph state."""

from __future__ import annotations

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agentware import DeepResearchOrchestrator, SharedVariable


@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    return f"[web] Results for '{query}': Example article about {query}."


@tool
def news_search(query: str) -> str:
    """Search recent news articles."""
    return f"[news] Headlines for '{query}': Breaking story and analysis."


@tool
def academic_search(query: str) -> str:
    """Search academic papers and journals."""
    return f"[papers] Scholarly sources for '{query}': Smith et al. (2024)."


def main() -> None:
    orchestrator = DeepResearchOrchestrator(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        search_tools=[web_search, news_search, academic_search],
        max_research_rounds=4,
    )

    result = orchestrator.research(
        "Impact of retrieval-augmented generation on enterprise search",
        thread_id="research-session-1",
        shared_variables=[
            SharedVariable(key="locale", value="en"),
            SharedVariable(key="depth", value="deep"),
        ],
    )

    print(result.content)
    print("\n--- Queries ---")
    for q in result.research_queries:
        print(f"  - {q}")
    print("\n--- Findings ---")
    for f in result.findings:
        print(f"  [{f.get('query', '')}] {str(f.get('content', ''))[:120]}...")
    print(f"\nTokens: {result.token_usage.total_tokens}")
    print("Shared:", {v["key"]: v["value"] for v in result.shared_variables})


if __name__ == "__main__":
    main()
