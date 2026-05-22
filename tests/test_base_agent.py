from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agentware import BaseLangGraphAgent


@tool
def echo(text: str) -> str:
    """Echo input."""
    return text


def test_invoke_without_tools():
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(
        content="Hello!",
        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    )

    agent = BaseLangGraphAgent(
        system_prompt="Test",
        llm=mock_llm,
        tools=[],
    )
    result = agent.invoke("Hi", thread_id="t1")
    assert result.content == "Hello!"
    assert result.token_usage.total_tokens == 5
    assert result.token_usage.llm_calls == 1
