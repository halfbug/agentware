"""Checkpointer factory — memory by default, optional MongoDB."""

from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver


def build_checkpointer(
    *,
    mongo_uri: Optional[str] = None,
    mongo_database: Optional[str] = None,
    mongo_collection: Optional[str] = None,
) -> Any:
    """Return a LangGraph checkpointer.

    Uses MongoDB when ``mongo_uri`` and ``mongo_database`` are set and
    ``langgraph-checkpoint-mongodb`` is installed; otherwise MemorySaver.
    """
    if mongo_uri and mongo_database:
        try:
            from langgraph.checkpoint.mongodb import MongoDBSaver
            from pymongo import MongoClient

            client = MongoClient(mongo_uri)
            kwargs: dict[str, Any] = {}
            if mongo_collection:
                kwargs["collection_name"] = mongo_collection
            return MongoDBSaver(client, mongo_database, **kwargs)
        except ImportError as e:
            raise ImportError(
                "MongoDB checkpointer requires: pip install agentware[mongodb]"
            ) from e
    return MemorySaver()
