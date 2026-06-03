"""Typer CLI helpers for running agents from the terminal."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from agentware.core.base import BaseLangGraphAgent
from agentware.core.types import AgentRunResult


class AgentCLI:
    """Typer CLI wrapper for running agents.

    Usage::

        from agentware.integrations.typer_cli import AgentCLI

        agent_cli = AgentCLI(agent_factory=lambda: my_agent)
        agent_cli.cli()
    """

    def __init__(self, agent_factory: Callable[[], BaseLangGraphAgent]):
        """Initialize the CLI with an agent factory.

        Args:
            agent_factory: Callable that returns a BaseLangGraphAgent instance.
        """
        self._agent_factory = agent_factory
        self._agent: Optional[BaseLangGraphAgent] = None

    def get_agent(self) -> BaseLangGraphAgent:
        """Get or create the agent."""
        if self._agent is None:
            self._agent = self._agent_factory()
        return self._agent

    async def run_agent(
        self,
        message: str,
        thread_id: str = "default",
        extra_state: Optional[dict] = None,
        verbose: bool = False,
    ) -> AgentRunResult:
        """Run the agent with the given message.

        Args:
            message: The input message for the agent.
            thread_id: Thread ID for conversation context.
            extra_state: Extra state to pass to the agent.
            verbose: Whether to print verbose output.

        Returns:
            AgentRunResult with the response and token usage.
        """
        agent = self.get_agent()
        if verbose:
            print(f"[*] Running agent with message: {message}")
        result = await agent.ainvoke(
            message, thread_id=thread_id, extra_state=extra_state
        )
        return result

    def cli(self) -> None:
        """Create and return a Typer app.

        Returns:
            A Typer app instance.
        """
        try:
            import typer
        except ImportError as e:
            raise ImportError(
                "Typer integration requires: pip install agentware[typer]"
            ) from e

        app = typer.Typer(help="Agent CLI")

        @app.command()
        def chat(
            message: str = typer.Argument(..., help="Message to send to the agent"),
            thread_id: str = typer.Option(
                "default", "--thread-id", "-t", help="Thread ID for conversation context"
            ),
            verbose: bool = typer.Option(
                False, "--verbose", "-v", help="Print verbose output"
            ),
        ) -> None:
            """Run the agent with a message and display the response."""
            result = asyncio.run(
                self.run_agent(message, thread_id=thread_id, verbose=verbose)
            )

            # Display response
            print("\n[Agent Response]")
            print(result.content)

            # Display token usage if available
            if result.token_usage:
                print("\n[Token Usage]")
                u = result.token_usage
                print(f"  Input:  {u.input_tokens}")
                print(f"  Output: {u.output_tokens}")
                print(f"  Total:  {u.total_tokens}")
                if u.llm_calls > 0:
                    print(f"  LLM Calls: {u.llm_calls}")

        return app


def create_cli(
    agent_factory: Callable[[], BaseLangGraphAgent],
) -> Any:
    """Create a Typer CLI app for an agent.

    Requires: ``pip install agentware[typer]``

    Usage::

        from agentware.integrations.typer_cli import create_cli

        app = create_cli(lambda: my_agent)

        if __name__ == "__main__":
            app()

    Args:
        agent_factory: Callable that returns a BaseLangGraphAgent instance.

    Returns:
        A Typer app instance.
    """
    cli_handler = AgentCLI(agent_factory)
    return cli_handler.cli()
