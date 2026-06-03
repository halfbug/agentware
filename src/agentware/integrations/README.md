# Integrations

Framework adapters for serving agents.

## FastAPI

Serve agents over HTTP with a chat endpoint.

```python
from fastapi import FastAPI
from agentware.integrations.fastapi import create_chat_router

app = FastAPI()
app.include_router(create_chat_router(lambda: my_agent))
```

Works with any `BaseLangGraphAgent` subclass, including `DeepResearchOrchestrator`.

Requires `pip install agentware[fastapi]`.

Run: `uvicorn examples.react.fastapi_app:app --reload`

## Typer CLI

Run agents directly from the terminal with Typer commands.

```python
from agentware.integrations.typer_cli import create_cli

app = create_cli(lambda: my_agent)

if __name__ == "__main__":
    app()
```

Works with any `BaseLangGraphAgent` subclass.

Requires `pip install agentware[typer]`.

Usage:
```bash
# Basic usage
python examples/react/typer_cli.py "Hello, how are you?"

# With thread ID for conversation context
python examples/react/typer_cli.py "What is 2+2?" --thread-id my-thread

# Verbose output showing token usage
python examples/react/typer_cli.py "Tell me a joke" --verbose
```
