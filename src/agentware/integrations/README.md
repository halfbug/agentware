# Integrations

Framework adapters for serving agents over HTTP.

## FastAPI

```python
from fastapi import FastAPI
from agentware.integrations.fastapi import create_chat_router

app = FastAPI()
app.include_router(create_chat_router(lambda: my_agent))
```

Works with any `BaseLangGraphAgent` subclass, including `DeepResearchOrchestrator`.

Requires `pip install agentware[fastapi]`.
