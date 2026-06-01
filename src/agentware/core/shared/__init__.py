"""Shared state utilities for multi-agent orchestration."""

from agentware.core.shared.variables import (
    get_shared_value,
    merge_shared_variables,
    set_shared_value,
)

__all__ = ["get_shared_value", "merge_shared_variables", "set_shared_value"]
