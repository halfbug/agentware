"""Result types for orchestrated multi-agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from agentware.agents.deep_research.state import ResearchFinding, SharedVariable
from agentware.core.types import AgentRunResult


@dataclass
class DeepResearchResult(AgentRunResult):
    """Outcome of a deep-research orchestration run."""

    research_topic: str = ""
    research_queries: List[str] = field(default_factory=list)
    findings: List[ResearchFinding] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    shared_variables: List[SharedVariable] = field(default_factory=list)

    @classmethod
    def from_agent_result(
        cls,
        result: AgentRunResult,
        *,
        research_topic: str = "",
        research_queries: List[str] | None = None,
        findings: List[ResearchFinding] | None = None,
        sources: List[str] | None = None,
        shared_variables: List[SharedVariable] | None = None,
    ) -> DeepResearchResult:
        return cls(
            content=result.content,
            token_usage=result.token_usage,
            messages=result.messages,
            raw_state=result.raw_state,
            research_topic=research_topic,
            research_queries=research_queries or [],
            findings=findings or [],
            sources=sources or [],
            shared_variables=shared_variables or [],
        )
