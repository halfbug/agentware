"""Agentware — extensible LangGraph multi-agent framework."""

from agentware.agents.deep_research import (
    DeepResearchOrchestrator,
    DeepResearchResult,
    DeepResearchState,
    SharedVariable,
)
from agentware.core import (
    AgentRunResult,
    BaseAgentState,
    BaseLangGraphAgent,
    TokenUsage,
    aggregate_token_usage,
)
from agentware.tools import load_mcp_tools

__all__ = [
    "AgentRunResult",
    "BaseAgentState",
    "BaseLangGraphAgent",
    "DeepResearchOrchestrator",
    "DeepResearchResult",
    "DeepResearchState",
    "SharedVariable",
    "TokenUsage",
    "aggregate_token_usage",
    "load_mcp_tools",
]

__version__ = "0.1.0"
