"""Built-in agents — each subpackage is a self-contained agent implementation."""

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
