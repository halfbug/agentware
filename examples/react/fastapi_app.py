"""Example FastAPI app serving an agent."""

from fastapi import FastAPI
from langchain_openai import ChatOpenAI

from agentware import BaseLangGraphAgent
from agentware.integrations.fastapi import create_chat_router

SYSTEM_PROMPT = "You are a concise helpful assistant."


def make_agent() -> BaseLangGraphAgent:
    return BaseLangGraphAgent(
        system_prompt=SYSTEM_PROMPT,
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[],
    )


app = FastAPI(title="Agentware Example")
app.include_router(create_chat_router(make_agent))

# Run: uvicorn examples.react.fastapi_app:app --reload
