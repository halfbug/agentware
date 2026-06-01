"""Extensible base LangGraph agent."""

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from agentware.core.checkpointer import build_checkpointer
from agentware.core.state import BaseAgentState
from agentware.core.tokens import TokenUsage, aggregate_token_usage
from agentware.core.types import AgentRunResult

logger = logging.getLogger(__name__)


class BaseLangGraphAgent(ABC):
    """Extensible LangGraph agent with tools, checkpointing, and token tracking.

    Subclass to add custom state fields, graph nodes, or tool argument injection.

    Example::

        class SupportAgent(BaseLangGraphAgent):
            def enrich_tool_args(self, tool_name, args, state):
                if tool_name == "content_search":
                    args["publication_ids"] = self.publication_ids
                return args
    """

    state_schema: Type[TypedDict] = BaseAgentState

    def __init__(
        self,
        *,
        system_prompt: str = "",
        llm: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Union[BaseTool, Any]]] = None,
        checkpointer: Optional[Any] = None,
        mongo_uri: Optional[str] = None,
        mongo_database: Optional[str] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools: List[Any] = list(tools or [])
        self._checkpointer = checkpointer
        self._mongo_uri = mongo_uri
        self._mongo_database = mongo_database
        self._compiled_graph: Optional[CompiledStateGraph] = None

    # ------------------------------------------------------------------
    # Extension hooks
    # ------------------------------------------------------------------

    def enrich_tool_args(
        self,
        tool_name: str,
        args: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Override to inject context into tool calls (e.g. user_id, tenant)."""
        return args

    def prepare_messages(self, messages: List[AnyMessage]) -> List[AnyMessage]:
        """Override to transform messages before LLM invoke."""
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        return [SystemMessage(content=self.system_prompt)] + non_system

    # ------------------------------------------------------------------
    # Graph construction (override for custom workflows)
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(self.state_schema)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")
        return workflow

    def _build_checkpointer(self) -> Any:
        if self._checkpointer is not None:
            return self._checkpointer
        self._checkpointer = build_checkpointer(
            mongo_uri=self._mongo_uri,
            mongo_database=self._mongo_database,
        )
        return self._checkpointer

    @property
    def graph(self) -> CompiledStateGraph:
        if self._compiled_graph is None:
            self._compiled_graph = self._build_graph().compile(
                checkpointer=self._build_checkpointer()
            )
        return self._compiled_graph

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _agent_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm is None:
            return {
                "messages": [AIMessage(content="LLM not configured.")],
                "llm_calls": state.get("llm_calls", 0) + 1,
            }

        messages = self.prepare_messages(list(state["messages"]))
        llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm

        try:
            response = llm_with_tools.invoke(messages)
            if not isinstance(response, AIMessage):
                response = AIMessage(content=str(response))
        except Exception as e:
            logger.exception("LLM invocation failed")
            response = AIMessage(content=f"Error: {e}")

        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def _tool_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}

        tool_map = {t.name: t for t in self.tools}
        results: List[ToolMessage] = []

        for tool_call in last_message.tool_calls:
            name = tool_call["name"]
            tool = tool_map.get(name)
            if tool is None:
                content = f"Tool '{name}' not found"
            else:
                try:
                    args = dict(tool_call["args"])
                    args = self.enrich_tool_args(name, args, state)
                    result = tool.invoke(args)
                    content = str(result)
                except Exception as e:
                    logger.exception("Tool %s failed", name)
                    content = f"Error executing '{name}': {e}"

            results.append(
                ToolMessage(content=content, tool_call_id=tool_call["id"])
            )

        return {"messages": results}

    def _should_continue(self, state: Dict[str, Any]) -> Literal["continue", "end"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "continue"
        return "end"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _build_initial_state(
        self,
        user_message: str,
        *,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build graph input state. Override in subclasses for orchestration fields."""
        state: Dict[str, Any] = {
            "messages": [HumanMessage(content=user_message)],
            "llm_calls": 0,
        }
        if extra_state:
            state.update(extra_state)
        return state

    def invoke(
        self,
        user_message: str,
        *,
        thread_id: str = "default",
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> AgentRunResult:
        """Run the agent synchronously for one user turn."""
        input_state = self._build_initial_state(user_message, extra_state=extra_state)
        final = self.graph.invoke(input_state, config=self._config(thread_id))
        return self._to_result(final)

    async def ainvoke(
        self,
        user_message: str,
        *,
        thread_id: str = "default",
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> AgentRunResult:
        """Run the agent asynchronously for one user turn."""
        input_state = self._build_initial_state(user_message, extra_state=extra_state)
        final = await self.graph.ainvoke(input_state, config=self._config(thread_id))
        return self._to_result(final)

    def _to_result(self, final_state: Dict[str, Any]) -> AgentRunResult:
        messages: List[AnyMessage] = list(final_state.get("messages", []))
        llm_calls = int(final_state.get("llm_calls", 0))

        content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        token_usage = aggregate_token_usage(messages, llm_calls=llm_calls)
        return AgentRunResult(
            content=content,
            token_usage=token_usage,
            messages=messages,
            raw_state=final_state,
        )
