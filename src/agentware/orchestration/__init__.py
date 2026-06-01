"""Backward compatibility — use ``agentware.agents.deep_research``."""

from agentware.agents.deep_research import (
    DeepResearchOrchestrator,
    DeepResearchResult,
    DeepResearchState,
    SharedVariable,
)

__all__ = [
    "DeepResearchOrchestrator",
    "DeepResearchResult",
    "DeepResearchState",
    "SharedVariable",
]
