"""Deep research orchestrator — extends BaseLangGraphAgent with subgraph workflow."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentware.agents.deep_research.state import DeepResearchState, SharedVariable
from agentware.agents.deep_research.subgraphs import (
    build_plan_subgraph,
    build_research_subgraph,
    build_synthesize_subgraph,
)
from agentware.agents.deep_research.types import DeepResearchResult
from agentware.core.base import BaseLangGraphAgent
from agentware.core.shared import set_shared_value
from agentware.core.types import AgentRunResult


class DeepResearchOrchestrator(BaseLangGraphAgent):
    """Multi-agent deep research using LangGraph subgraphs with shared state.

    Inherits from :class:`~agentware.base.BaseLangGraphAgent`:

    - ``invoke`` / ``ainvoke`` — run with ``thread_id`` for conversation checkpointing
    - ``_to_result`` — token usage aggregation and message history
    - ``enrich_tool_args`` — inject context into search tool calls
    - MongoDB / in-memory checkpointer

    Parent graph flow::

        START → init → plan (subgraph) → research (subgraph) → synthesize (subgraph) → END

    Args:
        llm: LangChain chat model for plan, research, and synthesis.
        search_tools: Search engine tools (stored as ``tools`` on the base class).
        max_research_rounds: Max tool-call rounds inside the research subgraph.
    """

    state_schema = DeepResearchState

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        search_tools: Sequence[Union[BaseTool, Any]],
        max_research_rounds: int = 5,
        planner_prompt: Optional[str] = None,
        researcher_prompt: Optional[str] = None,
        synthesizer_prompt: Optional[str] = None,
        checkpointer: Optional[Any] = None,
        mongo_uri: Optional[str] = None,
        mongo_database: Optional[str] = None,
        enrich_tool_args: Optional[
            Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
        ] = None,
    ) -> None:
        if not search_tools:
            raise ValueError("search_tools must be a non-empty list of search tools")

        super().__init__(
            system_prompt="",  # subgraphs use their own prompts
            llm=llm,
            tools=list(search_tools),
            checkpointer=checkpointer,
            mongo_uri=mongo_uri,
            mongo_database=mongo_database,
        )
        self.max_research_rounds = max_research_rounds
        self._planner_prompt = planner_prompt
        self._researcher_prompt = researcher_prompt
        self._synthesizer_prompt = synthesizer_prompt
        self._enrich_tool_args_callback = enrich_tool_args

        self._plan_subgraph: Optional[CompiledStateGraph] = None
        self._research_subgraph: Optional[CompiledStateGraph] = None
        self._synthesize_subgraph: Optional[CompiledStateGraph] = None

    @property
    def search_tools(self) -> List[Any]:
        """Alias for ``tools`` — search engines used in the research subgraph."""
        return self.tools

    def enrich_tool_args(
        self,
        tool_name: str,
        args: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._enrich_tool_args_callback is not None:
            return self._enrich_tool_args_callback(tool_name, args, state)
        return super().enrich_tool_args(tool_name, args, state)

    def _plan_subgraph_compiled(self) -> CompiledStateGraph:
        if self._plan_subgraph is None:
            kw: Dict[str, Any] = {}
            if self._planner_prompt:
                kw["planner_prompt"] = self._planner_prompt
            self._plan_subgraph = build_plan_subgraph(self.llm, **kw)
        return self._plan_subgraph

    def _research_subgraph_compiled(self) -> CompiledStateGraph:
        if self._research_subgraph is None:
            kw: Dict[str, Any] = {"enrich_tool_args": self.enrich_tool_args}
            if self._researcher_prompt:
                kw["researcher_prompt"] = self._researcher_prompt
            self._research_subgraph = build_research_subgraph(
                self.llm, self.tools, **kw
            )
        return self._research_subgraph

    def _synthesize_subgraph_compiled(self) -> CompiledStateGraph:
        if self._synthesize_subgraph is None:
            kw: Dict[str, Any] = {}
            if self._synthesizer_prompt:
                kw["synthesizer_prompt"] = self._synthesizer_prompt
            self._synthesize_subgraph = build_synthesize_subgraph(self.llm, **kw)
        return self._synthesize_subgraph

    @property
    def plan_subgraph(self) -> CompiledStateGraph:
        return self._plan_subgraph_compiled()

    @property
    def research_subgraph(self) -> CompiledStateGraph:
        return self._research_subgraph_compiled()

    @property
    def synthesize_subgraph(self) -> CompiledStateGraph:
        return self._synthesize_subgraph_compiled()

    def _build_graph(self) -> StateGraph:
        """Override base ReAct graph with plan → research → synthesize subgraphs."""
        builder = StateGraph(self.state_schema)

        builder.add_node("plan", self._plan_subgraph_compiled())
        builder.add_node("research", self._research_subgraph_compiled())
        builder.add_node("synthesize", self._synthesize_subgraph_compiled())

        def init_shared(state: DeepResearchState) -> Dict[str, Any]:
            return {
                "shared_variables": set_shared_value("phase", "started"),
                "research_round": 0,
            }

        builder.add_node("init", init_shared)
        builder.add_edge(START, "init")
        builder.add_edge("init", "plan")
        builder.add_edge("plan", "research")
        builder.add_edge("research", "synthesize")
        builder.add_edge("synthesize", END)
        return builder

    def _build_initial_state(
        self,
        user_message: str,
        *,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = super()._build_initial_state(user_message, extra_state=extra_state)
        state.update(
            {
                "research_topic": user_message,
                "research_queries": [],
                "findings": [],
                "sources": [],
                "shared_variables": state.get("shared_variables", []),
                "research_round": 0,
                "max_research_rounds": self.max_research_rounds,
            }
        )
        return state

    def _to_result(self, final_state: Dict[str, Any]) -> DeepResearchResult:
        base: AgentRunResult = super()._to_result(final_state)
        return DeepResearchResult.from_agent_result(
            base,
            research_topic=final_state.get("research_topic", ""),
            research_queries=list(final_state.get("research_queries") or []),
            findings=list(final_state.get("findings") or []),
            sources=list(final_state.get("sources") or []),
            shared_variables=list(final_state.get("shared_variables") or []),
        )

    def research(
        self,
        topic: str,
        *,
        thread_id: str = "default",
        extra_state: Optional[Dict[str, Any]] = None,
        shared_variables: Optional[List[SharedVariable]] = None,
    ) -> DeepResearchResult:
        """Run deep research (alias for :meth:`invoke` with research-specific state)."""
        merged = dict(extra_state or {})
        if shared_variables is not None:
            merged["shared_variables"] = list(shared_variables)
        return self.invoke(topic, thread_id=thread_id, extra_state=merged)

    async def aresearch(
        self,
        topic: str,
        *,
        thread_id: str = "default",
        extra_state: Optional[Dict[str, Any]] = None,
        shared_variables: Optional[List[SharedVariable]] = None,
    ) -> DeepResearchResult:
        """Async deep research (alias for :meth:`ainvoke`)."""
        merged = dict(extra_state or {})
        if shared_variables is not None:
            merged["shared_variables"] = list(shared_variables)
        return await self.ainvoke(topic, thread_id=thread_id, extra_state=merged)
