"""Token usage extraction and aggregation from LangChain messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from langchain_core.messages import AIMessage, AnyMessage


@dataclass(frozen=True)
class TokenUsage:
    """Aggregated token counts for one agent run (one user message)."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
        )


def _usage_from_metadata(meta: Mapping[str, Any] | None) -> TokenUsage | None:
    if not meta:
        return None
    inp = meta.get("input_tokens") or meta.get("prompt_tokens") or 0
    out = meta.get("output_tokens") or meta.get("completion_tokens") or 0
    total = meta.get("total_tokens") or (inp + out)
    if not inp and not out and not total:
        return None
    return TokenUsage(
        input_tokens=int(inp),
        output_tokens=int(out),
        total_tokens=int(total),
    )


def usage_from_message(message: AIMessage) -> TokenUsage | None:
    """Read token usage from a single AIMessage (usage_metadata or response_metadata)."""
    if getattr(message, "usage_metadata", None):
        return _usage_from_metadata(message.usage_metadata)

    response_meta = getattr(message, "response_metadata", None) or {}
    token_usage = response_meta.get("token_usage") or response_meta.get("usage")
    if token_usage:
        return _usage_from_metadata(token_usage)

    return None


def aggregate_token_usage(
    messages: Sequence[AnyMessage],
    *,
    llm_calls: int = 0,
) -> TokenUsage:
    """Sum token usage across all AIMessages in a run."""
    total = TokenUsage(llm_calls=llm_calls)
    for msg in messages:
        if isinstance(msg, AIMessage):
            part = usage_from_message(msg)
            if part:
                total = total + part
    if total.total_tokens == 0 and (total.input_tokens or total.output_tokens):
        total = TokenUsage(
            input_tokens=total.input_tokens,
            output_tokens=total.output_tokens,
            total_tokens=total.input_tokens + total.output_tokens,
            llm_calls=total.llm_calls,
        )
    return total
