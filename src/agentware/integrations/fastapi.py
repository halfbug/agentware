"""FastAPI helpers for serving agents."""

from __future__ import annotations

from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, Field

from agentware.core.base import BaseLangGraphAgent
from agentware.core.types import AgentRunResult


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    extra_state: Optional[dict[str, Any]] = None


class ChatResponse(BaseModel):
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0

    @classmethod
    def from_result(cls, result: AgentRunResult) -> ChatResponse:
        u = result.token_usage
        return cls(
            content=result.content,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            total_tokens=u.total_tokens,
            llm_calls=u.llm_calls,
        )


def create_chat_router(
    agent_factory: Callable[[], BaseLangGraphAgent],
    *,
    prefix: str = "/agent",
    tags: Optional[list[str]] = None,
) -> Any:
    """Create a FastAPI APIRouter with POST ``/chat`` endpoint.

    Requires: ``pip install agentware[fastapi]``

    Usage::

        from fastapi import FastAPI
        from agentware.integrations.fastapi import create_chat_router

        app = FastAPI()
        app.include_router(create_chat_router(lambda: my_agent))
    """
    try:
        from fastapi import APIRouter
    except ImportError as e:
        raise ImportError(
            "FastAPI integration requires: pip install agentware[fastapi]"
        ) from e

    router = APIRouter(prefix=prefix, tags=tags or ["agent"])
    _agent: Optional[BaseLangGraphAgent] = None

    def get_agent() -> BaseLangGraphAgent:
        nonlocal _agent
        if _agent is None:
            _agent = agent_factory()
        return _agent

    @router.post("/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse:
        agent = get_agent()
        result = await agent.ainvoke(
            body.message,
            thread_id=body.thread_id,
            extra_state=body.extra_state,
        )
        return ChatResponse.from_result(result)

    return router
