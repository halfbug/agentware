"""Example Typer CLI app for running an agent."""

from langchain_openai import ChatOpenAI

from agentware import BaseLangGraphAgent
from agentware.integrations.typer_cli import create_cli

SYSTEM_PROMPT = "You are a concise helpful assistant."


def make_agent() -> BaseLangGraphAgent:
    return BaseLangGraphAgent(
        system_prompt=SYSTEM_PROMPT,
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[],
    )


app = create_cli(make_agent)

if __name__ == "__main__":
    app()

# Usage:
#   python examples/react/typer_cli.py "Hello, how are you?"
#   python examples/react/typer_cli.py "What is 2+2?" --thread-id my-thread
#   python examples/react/typer_cli.py "Tell me a joke" -t thread1 -v
