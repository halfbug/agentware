from langchain_core.messages import AIMessage

from agentware.core.tokens import TokenUsage, aggregate_token_usage, usage_from_message


def test_usage_from_message_usage_metadata():
    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    u = usage_from_message(msg)
    assert u is not None
    assert u.total_tokens == 15


def test_aggregate_token_usage():
    messages = [
        AIMessage(content="a", usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}),
        AIMessage(content="b", usage_metadata={"input_tokens": 4, "output_tokens": 5, "total_tokens": 9}),
    ]
    u = aggregate_token_usage(messages, llm_calls=2)
    assert u.input_tokens == 5
    assert u.output_tokens == 7
    assert u.total_tokens == 12
    assert u.llm_calls == 2
