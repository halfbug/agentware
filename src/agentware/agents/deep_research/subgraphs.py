"""Planner, researcher, and synthesizer subgraphs for deep research."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentware.agents.deep_research.state import DeepResearchState, ResearchFinding
from agentware.core.shared import get_shared_value, set_shared_value

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are a research planner. Given a research topic, produce 3-6 focused search queries.
Return ONLY a JSON array of strings, e.g. ["query 1", "query 2"].
Do not include markdown or explanation."""

RESEARCHER_PROMPT = """You are a deep research agent. Use the provided search tools to investigate the topic thoroughly.
For each search tool call, gather concrete facts, quotes, and URLs when available.
When you have enough evidence, respond with a brief summary (no tool calls)."""

SYNTHESIZER_PROMPT = """You are a research synthesizer. Write a comprehensive, well-structured report
using ONLY the findings and sources provided. Include sections, bullet points where helpful, and cite sources.
End with a short 'Sources' section listing URLs or source names mentioned in the findings."""


def _parse_query_list(text: str) -> List[str]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            pass
    lines = [ln.strip("-•* \t") for ln in text.splitlines() if ln.strip()]
    return [ln for ln in lines if len(ln) > 3][:6]


def build_plan_subgraph(
    llm: BaseChatModel,
    *,
    planner_prompt: str = PLANNER_PROMPT,
) -> CompiledStateGraph:
    """Subgraph: turn topic into ``research_queries`` and update ``shared_variables``."""

    def plan_node(state: DeepResearchState) -> Dict[str, Any]:
        topic = state.get("research_topic") or get_shared_value(
            state.get("shared_variables"), "research_topic", ""
        )
        if not topic:
            last = state.get("messages", [])
            for msg in reversed(last):
                if isinstance(msg, HumanMessage):
                    topic = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

        response = llm.invoke(
            [
                SystemMessage(content=planner_prompt),
                HumanMessage(content=f"Research topic:\n{topic}"),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        queries = _parse_query_list(content)

        return {
            "research_topic": topic,
            "research_queries": queries,
            "shared_variables": set_shared_value("phase", "planned")
            + set_shared_value("query_count", len(queries)),
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    builder = StateGraph(DeepResearchState)
    builder.add_node("plan", plan_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", END)
    return builder.compile()


def build_research_subgraph(
    llm: BaseChatModel,
    search_tools: Sequence[Union[BaseTool, Any]],
    *,
    researcher_prompt: str = RESEARCHER_PROMPT,
    enrich_tool_args: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None,
) -> CompiledStateGraph:
    """Subgraph: ReAct loop using search-engine tools; appends ``findings`` and ``sources``."""

    tools = list(search_tools)
    tool_map = {t.name: t for t in tools}
    _enrich = enrich_tool_args or (lambda _n, a, _s: a)

    def _queries_text(state: DeepResearchState) -> str:
        queries = state.get("research_queries") or []
        topic = state.get("research_topic", "")
        if not queries:
            return topic
        return f"Topic: {topic}\n\nInvestigate these queries:\n" + "\n".join(
            f"- {q}" for q in queries
        )

    def research_agent_node(state: DeepResearchState) -> Dict[str, Any]:
        llm_with_tools = llm.bind_tools(tools) if tools else llm
        messages = [
            SystemMessage(content=researcher_prompt),
            HumanMessage(content=_queries_text(state)),
        ]
        prior = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage, ToolMessage))]
        if prior:
            messages.extend(prior[-8:])

        response = llm_with_tools.invoke(messages)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
            "shared_variables": set_shared_value("phase", "researching"),
        }

    def research_tools_node(state: DeepResearchState) -> Dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": []}

        results: List[ToolMessage] = []
        new_sources: List[str] = []
        new_findings: List[ResearchFinding] = []

        for tool_call in last.tool_calls:
            name = tool_call["name"]
            tool = tool_map.get(name)
            if tool is None:
                content = f"Tool '{name}' not found"
            else:
                try:
                    args = _enrich(name, dict(tool_call["args"]), state)
                    content = str(tool.invoke(args))
                except Exception as e:
                    logger.exception("Search tool %s failed", name)
                    content = f"Error: {e}"

            query = str(tool_call["args"].get("query", tool_call["args"].get("q", name)))
            new_findings.append(
                ResearchFinding(query=query, content=content[:8000], sources=[name])
            )
            new_sources.append(name)
            results.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))

        return {
            "messages": results,
            "findings": new_findings,
            "sources": new_sources,
            "research_round": state.get("research_round", 0) + 1,
        }

    def should_continue(state: DeepResearchState) -> Literal["tools", "finalize"]:
        last = state["messages"][-1]
        max_rounds = state.get("max_research_rounds", 5)
        if state.get("research_round", 0) >= max_rounds:
            return "finalize"
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "finalize"

    def finalize_node(state: DeepResearchState) -> Dict[str, Any]:
        last = state["messages"][-1]
        summary = ""
        if isinstance(last, AIMessage):
            summary = last.content if isinstance(last.content, str) else str(last.content)
        elif state.get("findings"):
            summary = "\n\n".join(f["content"][:500] for f in state["findings"][:5])

        finding = ResearchFinding(
            query=state.get("research_topic", "summary"),
            content=summary or "No summary produced.",
            sources=list(state.get("sources") or []),
        )
        return {
            "findings": [finding],
            "shared_variables": set_shared_value("phase", "researched"),
        }

    builder = StateGraph(DeepResearchState)
    builder.add_node("research_agent", research_agent_node)
    builder.add_node("research_tools", research_tools_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "research_agent")
    builder.add_conditional_edges(
        "research_agent",
        should_continue,
        {"tools": "research_tools", "finalize": "finalize"},
    )
    builder.add_edge("research_tools", "research_agent")
    builder.add_edge("finalize", END)
    return builder.compile()


def build_synthesize_subgraph(
    llm: BaseChatModel,
    *,
    synthesizer_prompt: str = SYNTHESIZER_PROMPT,
) -> CompiledStateGraph:
    """Subgraph: produce final report from accumulated findings."""

    def synthesize_node(state: DeepResearchState) -> Dict[str, Any]:
        topic = state.get("research_topic", "")
        findings = state.get("findings") or []
        blocks = []
        for i, f in enumerate(findings, 1):
            src = ", ".join(f.get("sources") or [])
            blocks.append(
                f"### Finding {i}: {f.get('query', '')}\n{f.get('content', '')}\nSources: {src}"
            )
        evidence = "\n\n".join(blocks) or "No findings collected."

        response = llm.invoke(
            [
                SystemMessage(content=synthesizer_prompt),
                HumanMessage(
                    content=f"Topic: {topic}\n\nFindings:\n{evidence}\n\nWrite the final report."
                ),
            ]
        )
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))

        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
            "shared_variables": set_shared_value("phase", "complete"),
        }

    builder = StateGraph(DeepResearchState)
    builder.add_node("synthesize", synthesize_node)
    builder.add_edge(START, "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()
