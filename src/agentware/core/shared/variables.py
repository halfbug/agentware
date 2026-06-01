"""Reducers for shared state arrays passed between parent graph and subgraphs."""

from __future__ import annotations

from typing import Any, Dict, List


def merge_shared_variables(
    existing: List[Dict[str, Any]] | None,
    new: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """Merge shared variables by key; later updates overwrite same key."""
    by_key: Dict[str, Dict[str, Any]] = {}
    for item in existing or []:
        key = item.get("key")
        if key is not None:
            by_key[str(key)] = dict(item)
    for item in new or []:
        key = item.get("key")
        if key is not None:
            by_key[str(key)] = dict(item)
    return list(by_key.values())


def get_shared_value(
    shared_variables: List[Dict[str, Any]] | None,
    key: str,
    default: Any = None,
) -> Any:
    """Read a value from the shared_variables array."""
    for item in shared_variables or []:
        if item.get("key") == key:
            return item.get("value", default)
    return default


def set_shared_value(key: str, value: Any) -> List[Dict[str, Any]]:
    """Build a state update fragment for one shared variable."""
    return [{"key": key, "value": value}]
