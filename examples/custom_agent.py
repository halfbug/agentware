"""Example: extend BaseLangGraphAgent with custom tool args and state."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from agentware import BaseLangGraphAgent, BaseAgentState


class AssistantState(BaseAgentState):
    publication_ids: Optional[List[str]]


@tool
def support_search(query: str) -> str:
    """Search support documentation."""
    return f"Support results for: {query}"


@tool
def content_search(query: str, publication_ids_array: Optional[List[str]] = None) -> str:
    """Search published content."""
    pubs = publication_ids_array or []
    return f"Content for '{query}' in publications {pubs}"


class AIAssistantAgent(BaseLangGraphAgent):
    state_schema = AssistantState

    def __init__(
        self,
        system_prompt: str,
        publication_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            system_prompt=system_prompt,
            tools=[support_search, content_search],
            **kwargs,
        )
        self.publication_ids = publication_ids or []
        self.user_id = user_id

    def enrich_tool_args(
        self,
        tool_name: str,
        args: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if tool_name == "content_search":
            args["publication_ids_array"] = state.get("publication_ids") or self.publication_ids
        return args


if __name__ == "__main__":
    agent = AIAssistantAgent(
        system_prompt="You are a helpful assistant.",
        llm=ChatOpenAI(model="gpt-4o-mini"),
        publication_ids=["pub-1", "pub-2"],
    )
    result = agent.invoke(
        "Find articles about onboarding",
        thread_id="user-123",
        extra_state={"publication_ids": ["pub-1"]},
    )
    print(result.content)
    print(f"Tokens: {result.token_usage.total_tokens} (calls: {result.token_usage.llm_calls})")
