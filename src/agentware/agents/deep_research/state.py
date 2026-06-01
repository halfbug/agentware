"""State schema shared by the deep-research parent graph and all subgraphs."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, NotRequired, TypedDict

from langchain_core.messages import AnyMessage

from agentware.core.shared import merge_shared_variables
from agentware.core.state import BaseAgentState


class SharedVariable(TypedDict):
    """One entry in the shared state array (visible to parent + subgraphs)."""

    key: str
    value: Any


class ResearchFinding(TypedDict):
    """A single research note gathered by the research subgraph."""

    query: str
    content: str
    sources: NotRequired[List[str]]


class DeepResearchState(BaseAgentState):
    """Parent and subgraph state — keys listed here are shared automatically.

    The ``shared_variables`` array lets you pass arbitrary key/value pairs
    between the main graph and subgraphs without changing this TypedDict.
    """

    research_topic: str
    research_queries: Annotated[List[str], operator.add]
    findings: Annotated[List[ResearchFinding], operator.add]
    sources: Annotated[List[str], operator.add]
    shared_variables: Annotated[List[SharedVariable], merge_shared_variables]
    research_round: int
    max_research_rounds: int
