"""Tests for deep research orchestrator and shared state."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agentware.agents.deep_research import DeepResearchOrchestrator
from agentware.core.shared import merge_shared_variables, set_shared_value


@tool
def mock_search(query: str) -> str:
    """Mock search."""
    return f"data about {query}"


def _llm_with_responses(responses: list) -> MagicMock:
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.side_effect = responses
    return llm


def test_merge_shared_variables():
    a = [{"key": "phase", "value": "started"}]
    b = set_shared_value("phase", "planned") + set_shared_value("count", 3)
    merged = merge_shared_variables(a, b)
    by_key = {m["key"]: m["value"] for m in merged}
    assert by_key["phase"] == "planned"
    assert by_key["count"] == 3


def test_deep_research_orchestrator_requires_tools():
    llm = MagicMock()
    try:
        DeepResearchOrchestrator(llm=llm, search_tools=[])
    except ValueError as e:
        assert "search_tools" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_deep_research_full_pipeline():
    plan_response = AIMessage(
        content='["sub query one", "sub query two"]',
        usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    )
    research_with_tools = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "mock_search",
                "args": {"query": "sub query one"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
    research_done = AIMessage(
        content="Gathered enough data.",
        usage_metadata={"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
    )
    synth_response = AIMessage(
        content="# Final Report\n\nSummary of findings.",
        usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
    )

    llm = _llm_with_responses(
        [plan_response, research_with_tools, research_done, synth_response]
    )

    orchestrator = DeepResearchOrchestrator(
        llm=llm,
        search_tools=[mock_search],
        max_research_rounds=3,
    )

    result = orchestrator.research(
        "AI agents in production",
        shared_variables=[{"key": "tenant", "value": "acme"}],
    )

    assert "Final Report" in result.content
    assert len(result.research_queries) >= 1
    assert result.token_usage.total_tokens > 0
    keys = {v["key"] for v in result.shared_variables}
    assert "phase" in keys
    assert "tenant" in keys
