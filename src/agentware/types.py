"""Public result types returned by agent invocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core.messages import AnyMessage

from agentware.tokens import TokenUsage


@dataclass
class AgentRunResult:
    """Outcome of a single agent invocation (one user turn)."""

    content: str
    """Final assistant text shown to the user."""

    token_usage: TokenUsage
    """Total tokens consumed for this turn (all LLM calls in the graph loop)."""

    messages: List[AnyMessage] = field(default_factory=list)
    """Full message list after the run (includes tool messages)."""

    raw_state: Optional[dict[str, Any]] = None
    """Optional final graph state for advanced consumers."""
