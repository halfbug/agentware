"""ReAct-style single-agent implementation."""

from agentware.core.base import BaseLangGraphAgent


class ReactAgent(BaseLangGraphAgent):
    """Thin named wrapper over BaseLangGraphAgent.

    Use this class when you want an explicit agent type under
    ``agentware.agents.react`` while preserving the base behavior.
    """
