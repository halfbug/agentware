from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from agentware.agents.react import ReactAgent


def test_react_agent_uses_base_behavior():
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(
        content="Hello from react",
        usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )

    agent = ReactAgent(system_prompt="You are helpful", llm=mock_llm, tools=[])
    result = agent.invoke("Hi", thread_id="react-1")

    assert result.content == "Hello from react"
    assert result.token_usage.total_tokens == 5
