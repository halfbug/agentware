"""Default LangGraph state schema for agents."""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict


class BaseAgentState(TypedDict, total=False):
    """Default state passed through the agent graph.

    Extend this TypedDict in your own module for agent-specific fields, e.g.:

        class MyState(BaseAgentState):
            publication_ids: Optional[List[str]]
    """

    messages: Annotated[List[AnyMessage], operator.add]
    llm_calls: int
